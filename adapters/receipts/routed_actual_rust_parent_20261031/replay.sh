set -u
AUD=/home/jonat/keramos_routed_actual_parent_20261031
mkdir -p "$AUD/runs"
for label in old corrected; do
 if [ "$label" = old ]; then binary="$AUD/target/release/kcr"; else binary=/home/jonat/cleanroom_build/target_codex_20261030/release/kcr; fi
 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 "$AUD/current/tools/ir_conformance.py" --adapter "cmd:$binary --conformance" --only trace/routed_longrope_session_length --json "$AUD/runs/$label.json" > "$AUD/runs/$label.log" 2>&1
 result=$?
 printf '%s\n' "$result" > "$AUD/runs/$label.exit"
done
