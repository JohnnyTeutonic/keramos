set -e
ROOT=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
AUD=/home/jonat/keramos_trace_receipt_audit_20261031
SCRIPT="$ROOT/keramos/adapters/receipts/trace_score_impact_20261031/replay.py"
mkdir -p "$AUD/runs"
for mode in historical_runner parent_runner fixed_runner; do
 extra=()
 case "$mode" in
 historical_runner) runner="$AUD/fixtures/tools/ir_conformance.py" ;;
 parent_runner) runner="$AUD/original/tools/ir_conformance.py"; extra=(--only trace/pk_) ;;
 fixed_runner) runner="$AUD/fixed/tools/ir_conformance.py" ;;
 esac
 python3 "$ROOT/scripts/cpu_budget.py" run --session paperkiln --job "Keramos trainer receipt $mode" --cores 1 --resume "Replay exclusive receipt under $AUD/runs" -- python3 "$SCRIPT" --runner "$runner" --fixtures "$AUD/fixtures/conformance" --adapter "$AUD/paperkiln/tools/ir_conformance_mtstudio.py" --binary "$AUD/build/mtstudio" --output "$AUD/runs/$mode.json" "${extra[@]}" >"$AUD/runs/$mode.log" 2>&1
done
python3 "$ROOT/keramos/adapters/receipts/trace_score_impact_20261031/summarize.py" --root "$ROOT" --runs "$AUD/runs" --output "$AUD/runs/impact.json"
