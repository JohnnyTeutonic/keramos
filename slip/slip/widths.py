"""Width checking: every value's width and every tensor's shape, solved by unification.

Values flow as [T, width] (IR_SPEC 3.6), so a value has one unknown, its width, and a tensor one
unknown per dimension (IR_SPEC 3.7: [out, in], the reverse of the GGUF's ne). Each op says which of
those unknowns are equal, or equal to a number: model.d and model.vocab, an attribute product
(heads * head_dim), a split's sizes, a tensor's shape in the weights when they are given. Two numbers
meeting in one unknown is a width mismatch, reported at the node that brought them together, with
where each number came from. Nothing here adds a rule: each op's widths are its IR_SPEC section's.

An op this module does not model (recurrence, depth_route) writes values of unknown width, which
constrain nothing; an attribute that differs between layers (per_layer) is left out.
"""

class Dim:
    __slots__ = ("parent", "value", "why")

    def __init__(self, value=None, why=None):
        self.parent = self
        self.value = value
        self.why = why


def find(d):
    while d.parent is not d:
        d.parent = d.parent.parent
        d = d.parent
    return d


class Conflict(Exception):
    pass


def fmt(shape):
    return "[" + ", ".join("?" if v is None else str(v) for v in shape) + "]"


class Widths:
    def __init__(self, doc, dims=None):
        """dims: tensor name -> shape [out, in, ...] (numpy order), from the weights, or None."""
        self.doc = doc
        self.dims = dims
        self.tensors = {}
        self.errors = []
        self.deferred = []
        m = doc.get("model", {})
        self.d = self.const(m.get("d"), "model.d")
        self.vocab = self.const(m.get("vocab"), "model.vocab")
        self.L = doc.get("block", {}).get("repeat")

    # unknowns
    def const(self, v, why):
        return Dim(v, why) if isinstance(v, int) and not isinstance(v, bool) else Dim()

    def unify(self, a, b, what):
        ra, rb = find(a), find(b)
        if ra is rb:
            return
        if ra.value is not None and rb.value is not None and ra.value != rb.value:
            raise Conflict(f"{what}: {ra.value} ({ra.why}) against {rb.value} ({rb.why})")
        if ra.value is None:
            ra, rb = rb, ra
        rb.parent = ra

    def fix(self, d, v, why, what):
        if v is None:
            return
        self.unify(d, Dim(v, why), what)

    # attributes
    def attr(self, n, key, default=None):
        v = n.get(key, default)
        if isinstance(v, dict) and "per_layer" in v:
            vals = v["per_layer"]
            v = vals[0] if vals and all(x == vals[0] for x in vals) else None
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        return v

    # tensors
    def tensor(self, name, rank, what):
        if not isinstance(name, str):
            return [Dim() for _ in range(rank)]
        if name not in self.tensors:
            dims = [Dim() for _ in range(rank)]
            self.tensors[name] = dims
            shape = self._file_shape(name)
            if shape is not None:
                if len(shape) != rank:
                    raise Conflict(f"{what}: {name} has {len(shape)} dimensions in the weights, {fmt(shape)}; "
                                   f"the op reads {rank}")
                for d, v in zip(dims, shape):
                    d.value, d.why = v, f"{name} is {fmt(shape)} in the weights"
        dims = self.tensors[name]
        if len(dims) != rank:
            raise Conflict(f"{what}: {name} is read with {rank} dimensions here and {len(dims)} elsewhere")
        return dims

    def _file_shape(self, name):
        if self.dims is None:
            return None
        if "{i}" not in name:
            return self.dims.get(name)
        shapes = [self.dims.get(name.replace("{i}", str(i))) for i in range(self.L or 0)]
        shapes = [s for s in shapes if s is not None]
        return shapes[0] if shapes and all(s == shapes[0] for s in shapes) else None

    def shape(self, name):
        return [find(d).value for d in self.tensors[name]]

    # values
    def width(self, env, name):
        v = env.get(name)
        if isinstance(v, Dim):
            return v
        if v is None and name not in env:
            d = Dim()
            env[name] = d
            return d
        return Dim()  # a list where a tensor is read, or a value of unknown kind

    # the document
    def run(self):
        env = {}
        for sect in ("embed", "block", "head"):
            s = self.doc.get(sect)
            if not isinstance(s, dict):
                continue
            before = dict(env)
            self.read_first = set()
            self.written = set()
            for idx, n in enumerate(s.get("nodes", [])):
                self._idx = idx
                try:
                    self.node(sect, n, env)
                except Conflict as e:
                    self.errors.append((sect, idx, str(e)))
            try:
                if sect == "embed" and isinstance(env.get("x"), Dim):
                    self.unify(env["x"], self.d, "the residual stream x the embed section writes and model.d")
                if sect == "block":  # what an application reads from the one before it
                    for name in self.read_first:
                        if isinstance(before.get(name), Dim) and isinstance(env.get(name), Dim):
                            self.unify(env[name], before[name],
                                       f"{name} as one application writes it and the next reads it")
                if sect == "head" and isinstance(env.get("logits"), Dim):
                    self.unify(env["logits"], self.vocab, "the logits the head writes and model.vocab")
            except Conflict as e:
                self.errors.append((sect, None, str(e)))
        for sect, idx, check in self.deferred:
            msg = check()
            if msg:
                self.errors.append((sect, idx, msg))
        return self.errors

    def node(self, sect, n, env):
        op = n.get("op")
        ins = n.get("in", [])
        out = n.get("out")
        outs = out if isinstance(out, list) else [out]
        for name in ins:
            if name not in self.written:
                self.read_first.add(name)
        W = [self.width(env, i) if isinstance(i, str) else Dim() for i in ins]
        prior = {o: env.get(o) for o in outs}
        f = getattr(self, "op_" + str(op), None)
        res = f(n, W, sect) if f else None
        if res is None:
            res = [Dim() for _ in outs]
        elif not isinstance(res, list) or (len(outs) == 1 and not isinstance(out, list)):
            res = [res]
        for o, r in zip(outs, res):
            env[o] = r
            self.written.add(o)
        apply = n.get("apply", True)
        if apply is not True:  # where the node does not run, its outputs keep their earlier values
            for o, r in zip(outs, res):
                if isinstance(prior[o], Dim) and isinstance(r, Dim):
                    self.unify(r, prior[o], f"{o} as this node writes it and as it stands where the node "
                                            "is not applied")

    # ops (IR_SPEC section 5 and 6)
    def _name(self, n):
        return f"{n.get('op')} writing {n.get('out')}"

    def op_embedding(self, n, W, sect):
        w = self.tensor(n.get("weight"), 2, self._name(n))
        if n.get("state_weight"):
            s = self.tensor(n["state_weight"], 2, self._name(n))
            self.unify(s[1], w[1], f"{self._name(n)}: the widths of weight and state_weight")
        return w[1]

    def op_pos_embedding(self, n, W, sect):
        w = self.tensor(n.get("weight"), 2, self._name(n))
        self.unify(W[0], w[1], f"{self._name(n)}: the width of {n['in'][0]} and of the position table")
        return W[0]

    def op_linear(self, n, W, sect):
        what = self._name(n)
        w = self.tensor(n.get("weight"), 2, what)
        if n.get("tied_to"):
            t = self.tensor(n["tied_to"], 2, what)
            self.unify(w[0], t[0], f"{what}: rows of weight and of tied_to")
            self.unify(w[1], t[1], f"{what}: columns of weight and of tied_to")
        self.unify(W[0], w[1], f"{what}: the width of its input {n['in'][0]} and the weight's in-dimension")
        if n.get("bias"):
            b = self.tensor(n["bias"], 1, what)
            self.unify(b[0], w[0], f"{what}: the bias length and the weight's out-dimension")
        return w[0]

    def _per_feature(self, n, W, keys):
        for k in keys:
            if n.get(k):
                t = self.tensor(n[k], 1, self._name(n))
                self.unify(t[0], W[0], f"{self._name(n)}: the length of {k} and the width of {n['in'][0]}")
        return W[0]

    def op_scale(self, n, W, sect):
        return self._per_feature(n, W, ("weight", "bias"))

    def op_layernorm(self, n, W, sect):
        return self._per_feature(n, W, ("weight", "bias"))

    def op_rmsnorm(self, n, W, sect):
        g = self.attr(n, "group")
        if g is None:
            return self._per_feature(n, W, ("weight",))
        if n.get("weight"):
            t = self.tensor(n["weight"], 1, self._name(n))
            self.fix(t[0], g, f"group {g}", f"{self._name(n)}: the weight's length and the group")
        x, name = W[0], n["in"][0]
        idx = self._idx

        def check():
            v = find(x).value
            if v is not None and v % g:
                return f"{self._name(n)}: group {g} does not divide the width of {name}, {v} ({find(x).why})"
        self.deferred.append((sect, idx, check))
        return W[0]

    def _same(self, n, W, sect):
        return W[0] if W else None

    op_gelu = op_silu = op_sigmoid = op_relu = op_affine = _same

    def op_add(self, n, W, sect):
        self.unify(W[0], W[1], f"{self._name(n)}: the widths of {n['in'][0]} and {n['in'][1]}")
        return W[0]

    op_mul = op_sub = op_add

    def op_expr(self, n, W, sect):
        for j in range(1, len(W)):
            self.unify(W[0], W[j], f"{self._name(n)}: the widths of {n['in'][0]} and {n['in'][j]}")
        return W[0] if W else None

    def op_mul_col(self, n, W, sect):
        self.fix(W[1], 1, "mul_col's column input", f"{self._name(n)}: the width of {n['in'][1]}")
        return W[0]

    def op_rms_row(self, n, W, sect):
        return Dim(1, "rms_row's output")

    def op_split(self, n, W, sect):
        sizes, g = n.get("sizes"), self.attr(n, "groups", 1)
        if not isinstance(sizes, list) or not all(isinstance(s, int) for s in sizes) or g is None:
            return None
        why = f"split sizes {sizes}" + (f" in {g} groups" if g != 1 else "")
        self.fix(W[0], g * sum(sizes), why, f"{self._name(n)}: the width of {n['in'][0]}")
        return [Dim(g * s, why) for s in sizes]

    def _heads(self, n, h, hd, label):
        if h is None or hd is None:
            return Dim()
        return Dim(h * hd, f"{label} {h} x head_dim {hd} of {n.get('op')}")

    def op_rope(self, n, W, sect):
        h, hd = self.attr(n, "heads"), self.attr(n, "head_dim")
        kh = self.attr(n, "kv_heads", n.get("heads"))
        what = self._name(n)
        nr = self.attr(n, "n_rot", n.get("head_dim"))
        ff = n.get("freq_factors")
        for name in ([ff] if isinstance(ff, str) else [ff.get("short"), ff.get("long")] if isinstance(ff, dict) else []):
            t = self.tensor(name, 1, what)
            if nr is not None:
                self.fix(t[0], nr // 2, f"n_rot {nr} / 2", f"{what}: the length of {name}")
        if n.get("rope_heads") == "first":
            return [W[0], W[1]]
        self.unify(W[0], self._heads(n, h, hd, "heads"), f"{what}: the width of {n['in'][0]}")
        self.unify(W[1], self._heads(n, kh, hd, "kv_heads"), f"{what}: the width of {n['in'][1]}")
        return [W[0], W[1]]

    def op_attention(self, n, W, sect):
        h, hd, kh = self.attr(n, "heads"), self.attr(n, "head_dim"), self.attr(n, "kv_heads")
        what = self._name(n)
        q, kv = self._heads(n, h, hd, "heads"), self._heads(n, kh, hd, "kv_heads")
        self.unify(W[0], q, f"{what}: the width of {n['in'][0]}")
        self.unify(W[1], kv, f"{what}: the width of {n['in'][1]}")
        self.unify(W[2], kv, f"{what}: the width of {n['in'][2]}")
        pb = n.get("position_bias")
        if isinstance(pb, dict) and pb.get("weight"):
            t = self.tensor(pb["weight"], 2, what)
            self.fix(t[1], h, "heads", f"{what}: the columns of the position bias table")
        return W[0]

    def op_linear_attention(self, n, W, sect):
        h, hd = self.attr(n, "heads"), self.attr(n, "head_dim")
        what = self._name(n)
        for j in range(3):
            self.unify(W[j], self._heads(n, h, hd, "heads"), f"{what}: the width of {n['in'][j]}")
        return W[0]

    def op_mla_attention(self, n, W, sect):
        H, r, dn, dr, dv = (self.attr(n, k) for k in ("heads", "kv_rank", "qk_nope", "qk_rope", "v_head_dim"))
        what = self._name(n)
        if None in (H, r, dn, dr, dv):
            return None
        want = [(H * dn, f"heads {H} x qk_nope {dn}"), (H * dr, f"heads {H} x qk_rope {dr}"),
                (r, f"kv_rank {r}"), (dr, f"qk_rope {dr}")]
        for j, (v, why) in enumerate(want):
            self.fix(W[j], v, why, f"{what}: the width of {n['in'][j]}")
        for key, shape in (("k_b", (H, r, dn)), ("v_b", (H, dv, r))):
            t = self.tensor(n.get(key), 3, what)
            for d, v in zip(t, shape):
                self.fix(d, v, f"{key} [heads, ...] from the attributes", f"{what}: the shape of {n.get(key)}")
        return Dim(H * dv, f"heads {H} x v_head_dim {dv} of mla_attention")

    def op_moe_ffn(self, n, W, sect):
        what = self._name(n)
        E = self.attr(n, "n_experts")
        r = self.tensor(n.get("router"), 2, what)
        self.fix(r[0], E, "n_experts", f"{what}: the router's rows")
        self.unify(r[1], W[0], f"{what}: the router's columns and the width of {n['in'][0]}")
        ex = n.get("experts") or {}
        g, u, dn = (self.tensor(ex.get(k), 3, what) for k in ("gate", "up", "down"))
        for t, k in ((g, "gate"), (u, "up"), (dn, "down")):
            self.unify(t[0], r[0], f"{what}: the expert count of {k} and the router's rows")
        self.unify(g[2], W[0], f"{what}: gate's in-dimension and the width of {n['in'][0]}")
        self.unify(u[2], W[0], f"{what}: up's in-dimension and the width of {n['in'][0]}")
        self.unify(g[1], u[1], f"{what}: the hidden widths of gate and up")
        self.unify(dn[2], g[1], f"{what}: down's in-dimension and the hidden width")
        self.unify(dn[1], W[0], f"{what}: down's out-dimension and the width of {n['in'][0]}")
        if n.get("selection_bias"):
            b = self.tensor(n["selection_bias"], 1, what)
            self.unify(b[0], r[0], f"{what}: the selection bias length and the expert count")
        return W[0]

    def op_list(self, n, W, sect):
        return ("list", W)

    def op_append(self, n, W, sect):
        return ("list", None)

    def op_depth_attend(self, n, W, sect):
        what = self._name(n)
        out = Dim()
        for j in range(1, len(W)):
            self.unify(W[j], out, f"{what}: the width of {n['in'][j]}")
        if n.get("weight"):
            t = self.tensor(n["weight"], 1, what)
            self.unify(t[0], out, f"{what}: the pseudo-query's length and the width of its sources")
        return out

    def op_ssm_scan(self, n, W, sect):
        what, N = self._name(n), self.attr(n, "n")
        self.fix(W[0], N, "n of ssm_scan", f"{what}: the width of {n['in'][0]}")
        for key, rank in (("a", 2), ("b", 1), ("c", 1), ("d", 1)):
            t = self.tensor(n.get(key), rank, what)
            want = [N, N] if key == "a" else [1] if key == "d" else [N]
            for d, v in zip(t, want):
                self.fix(d, v, "n of ssm_scan" if v == N else "ssm_scan's d", f"{what}: the shape of {n.get(key)}")
        return W[0]

    def op_conv1d_causal(self, n, W, sect):
        what, C, K = self._name(n), self.attr(n, "channels"), self.attr(n, "kernel")
        self.fix(W[0], C, "channels of conv1d_causal", f"{what}: the width of {n['in'][0]}")
        t = self.tensor(n.get("weight"), 2, what)
        self.fix(t[0], C, "channels", f"{what}: the rows of {n.get('weight')}")
        self.fix(t[1], K, "kernel", f"{what}: the columns of {n.get('weight')}")
        if n.get("bias"):
            b = self.tensor(n["bias"], 1, what)
            self.fix(b[0], C, "channels", f"{what}: the length of {n['bias']}")
        return W[0]

    def op_selective_scan(self, n, W, sect):
        what, D, N = self._name(n), self.attr(n, "channels"), self.attr(n, "d_state")
        for j, (v, why) in enumerate(((D, "channels"), (D, "channels"), (N, "d_state"), (N, "d_state"))):
            if j < len(W):
                self.fix(W[j], v, f"{why} of selective_scan", f"{what}: the width of {n['in'][j]}")
        a = self.tensor(n.get("a"), 2, what)
        self.fix(a[0], D, "channels", f"{what}: the rows of {n.get('a')}")
        self.fix(a[1], N, "d_state", f"{what}: the columns of {n.get('a')}")
        if n.get("d"):
            t = self.tensor(n["d"], 1, what)
            self.fix(t[0], D, "channels", f"{what}: the length of {n['d']}")
        return W[0]


def check(doc, dims=None):
    """(errors, solver): errors are (section, node index or None, message)."""
    w = Widths(doc, dims)
    w.run()
    return w.errors, w


def gguf_dims(path, tools):
    """Tensor name -> shape in numpy order ([out, in]), from a GGUF's tensor infos."""
    tensors = tools.read_gguf(path)[1]
    return {t[0]: [int(x) for x in reversed(t[1])] for t in tensors}
