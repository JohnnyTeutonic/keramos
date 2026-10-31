set -e
ROOT=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
AUD=/home/jonat/keramos_routed_actual_parent_20261031
mkdir -p "$AUD/old" "$AUD/current"
git -C "$ROOT/keramos" rev-parse 9c69143 > "$AUD/OLD_COMMIT"
git -C "$ROOT/keramos" rev-parse HEAD > "$AUD/CURRENT_COMMIT"
git -C "$ROOT/keramos" archive 9c69143 cleanroom/rust | tar -x -C "$AUD/old"
git -C "$ROOT/keramos" archive HEAD tools conformance | tar -x -C "$AUD/current"
python3 "$ROOT/scripts/cpu_budget.py" run --session keramos --job "Actual pre-fix Rust routed LongRoPE proof" --cores 1 --resume "Offline cargo release build in $AUD/old/cleanroom/rust" -- bash -c 'ulimit -v 1572864; export RUSTFLAGS="-C codegen-units=1"; export CARGO_TARGET_DIR="$1/target"; /home/jonat/.cargo/bin/cargo build --offline --release --jobs 1 --manifest-path "$1/old/cleanroom/rust/Cargo.toml"' bash "$AUD" >"$AUD/build.log" 2>&1
