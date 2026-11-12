#!/usr/bin/env python3
"""A minimal conformance adapter: the smallest engine that speaks IR_SPEC section 11.3.

An example for implementers (conformance/IMPLEMENTING.md), not an engine. It answers `hello`, runs the
three EXACT elementwise ops `add`, `sub` and `mul` through `run_node`, dequantises the three plain storage
types (F32, F16, BF16) and replies `skip` to everything else, so the suite reports those cases SKIP, never
PASS. Standard library only: it imports nothing from keramos, so it can be copied and grown into the adapter
of a real engine (replace the handlers' bodies with calls into the engine).

    python tools/ir_conformance.py --adapter "cmd:python3 adapters/minimal/minimal_adapter.py" --json results.json

--spec R.S declares another specification revision in `hello` (to see how the runner treats a mismatch).
"""
import json
import math
import struct
import sys

ENGINE = "minimal-example"
SPEC = "1.6"  # the revision this adapter implements (IR_SPEC section 2.6); the runner compares it with its suite's


class Skip(Exception):
    """Not implemented: the reply is `skip`, decided before anything is computed (IR_SPEC 11.3, Skips)."""


# ---------------------------------------------------------------- tensors (IR_SPEC 11.2)

def decode(t):
    """(shape, [float]) of a tensor {"shape", "f32"}: 8 hex digits per value, the binary32 bit pattern."""
    raw = bytes.fromhex(t["f32"])
    n = len(raw) // 4
    if n != math.prod(t["shape"]):
        raise ValueError(f"shape {t['shape']} does not hold {n} values")
    return list(t["shape"]), list(struct.unpack(f">{n}f", raw))


def f32_bits(x):
    """The binary32 bytes (big-endian) of a double, rounded to nearest even; overflow rounds to infinity."""
    try:
        return struct.pack(">f", x)
    except OverflowError:  # the value rounds past the largest binary32
        return struct.pack(">f", math.copysign(math.inf, x))


def encode(shape, values):
    return {"shape": list(shape), "f32": b"".join(f32_bits(v) for v in values).hex()}


# ---------------------------------------------------------------- methods

def hello(params):
    return {"engine": ENGINE, "version": "0.1 (example)", "backend": "cpu", "exact": True,
            "profiles": ["core"], "methods": sorted(HANDLERS), "spec": SPEC}


# One f32 operation each: the exact result of two binary32 values is computed in binary64 and rounded once
# to binary32 (binary64 holds enough bits for that double rounding to be innocuous for +, - and *), so the
# results are the IEEE f32 results bit for bit, signed zeros included (IR_SPEC 4.1, 4.2, 5.10).
ELEMENTWISE = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b}


def run_node(params):
    node = params["node"]
    fn = ELEMENTWISE.get(node.get("op"))
    if fn is None:
        raise Skip(f"{ENGINE} has no kernel for op {node.get('op')!r}")
    unknown = [k for k in node if k not in ("op", "in", "out") and not k.startswith("_")]
    if unknown:  # an attribute this engine does not implement: decline before computing
        raise Skip(f"{ENGINE} does not implement {node['op']} attributes {unknown}")
    if params.get("state"):
        raise Skip(f"{ENGINE} keeps no state")
    a_name, b_name = node["in"]
    (sa, a), (sb, b) = decode(params["inputs"][a_name]), decode(params["inputs"][b_name])
    if sa != sb:  # rule 37: no broadcasting; a request the engine cannot honour is an error, not a skip
        raise ValueError(f"{node['op']}: shapes {sa} and {sb} differ")
    return {"outputs": {node["out"]: encode(sa, [fn(x, y) for x, y in zip(a, b)])}, "state": {}}


def dequantize(params):
    kind, n, raw = params["type"], params["n"], bytes.fromhex(params["bytes"])
    if kind == "F32":  # the bits as stored, little-endian
        bits = [raw[i:i + 4][::-1] for i in range(0, 4 * n, 4)]
    elif kind == "BF16":  # the 16 bits as the high half of a binary32 (IR_SPEC 4.7)
        bits = [raw[i:i + 2][::-1] + b"\0\0" for i in range(0, 2 * n, 2)]
    elif kind == "F16":  # binary16 to binary32 is exact, subnormals included
        bits = [f32_bits(x) for x in struct.unpack(f"<{n}e", raw[:2 * n])]
    else:
        raise Skip(f"{ENGINE} does not dequantise {kind}")
    if len(bits) != n or any(len(x) != 4 for x in bits):
        raise ValueError(f"{len(raw)} bytes do not hold {n} {kind} values")
    return {"values": {"shape": [n], "f32": b"".join(bits).hex()}}


HANDLERS = {"hello": hello, "run_node": run_node, "dequantize": dequantize}


# ---------------------------------------------------------------- transport (IR_SPEC 11.3)

def reply(req):
    rid = req.get("id") if isinstance(req, dict) else None
    try:
        handler = HANDLERS.get(req["method"])
        if handler is None:  # check, open, forward, digest, render_chat, ...: not built yet
            raise Skip(f"{ENGINE} does not implement {req['method']}")
        return {"id": rid, "result": handler(req.get("params") or {})}
    except Skip as e:
        return {"id": rid, "skip": str(e)}
    except Exception as e:  # the method failed: the case fails; the process keeps serving
        return {"id": rid, "error": f"{type(e).__name__}: {e}"}


def main(argv):
    global SPEC
    if "--spec" in argv:
        SPEC = argv[argv.index("--spec") + 1]
    for line in sys.stdin:  # one request per line, one reply per line, in order; logs go to stderr
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except ValueError as e:
            out = {"id": None, "error": f"unreadable request: {e}"}
        else:
            out = reply(req)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
