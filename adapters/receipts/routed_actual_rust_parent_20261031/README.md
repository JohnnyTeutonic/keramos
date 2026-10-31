# Actual pre-fix Rust failure on the new routed LongRoPE case

The untouched Rust engine at `9c69143c8e33b10a4d442543b31d7a14ff9baf46`
**fails** `trace/routed_longrope_session_length`: step 0 logits have scaled worst
error **2.03e-04**, above the unchanged fixture tolerance **1e-05**. The supplied
corrected Rust binary **passes** the same case. Runner exit codes are respectively
1 and 0; neither run skips the case. These are executions of native Rust binaries,
independent of the Python semantic mutation test.

Both runs use the exact same current runner, helper modules and fixture archive
from `a4a4409928e81234d01f7d3257be6fbcfef5b15f`. Only the engine executable changes.
The old engine chooses the LongRoPE factor table from the greatest active position;
the later fix `1fd77ed` retains the full session length across gathered depth rows.
This receipt establishes sensitivity to the actual old Rust implementation, not
that every preceding engine version or every possible routed input fails.

The old source was extracted to a new ext4 directory using `git archive`, without
editing source files or overwriting an existing build. A first build of only
`cleanroom/rust` failed because `schema.rs` includes `../../ir.schema.json` at
compile time. That failure log is preserved. The retry adds only the unmodified
schema from the **same old commit**. Cargo generates a dependency-free lockfile;
there are no crate dependencies, downloads, source patches or tolerance changes.

The successful offline release build used the CPU-budget wrapper, session
`keramos`, one core, Cargo `--jobs 1`, `RUSTFLAGS="-C codegen-units=1"` and a
1.5 GiB address-space cap (`ulimit -v 1572864`) inherited by the build tree. It
finished in 12.98 seconds. The pre-existing corrected executable was supplied by
the parent task and was not rebuilt for this paired check; its complete SHA-256
is pinned in `audit.json` alongside the rebuilt old executable's hash.

`audit.json` records source/module/model/fixture hashes, compiler versions, complete
native binary hashes, results and exit codes. `old.json` / `old.log` and
`corrected.json` / `corrected.log` retain the ordinary runner receipts. The shell
scripts record the exact archive/build/replay commands. No historical receipt,
paper text, fixture or production source was changed by this audit.
