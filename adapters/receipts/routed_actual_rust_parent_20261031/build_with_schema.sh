set -e
ROOT=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
AUD=/home/jonat/keramos_routed_actual_parent_20261031
git -C "$ROOT/keramos" archive 9c69143 ir.schema.json | tar -x -C "$AUD/old"
python3 "$ROOT/scripts/cpu_budget.py" run --session keramos --job "Actual pre-fix Rust routed proof with required schema" --cores 1 --resume "Offline cargo build $AUD" -- bash -c 'ulimit -v 1572864; export RUSTFLAGS="-C codegen-units=1"; export CARGO_TARGET_DIR="$1/target"; /home/jonat/.cargo/bin/cargo build --offline --release --jobs 1 --manifest-path "$1/old/cleanroom/rust/Cargo.toml"' bash "$AUD" >"$AUD/build_with_schema.log" 2>&1
