#!/bin/bash
# Retest keramos's ORT GenAI reproductions against the builder at onnxruntime-genai main after #2884 and #2866.
set -u
export CUDA_VISIBLE_DEVICES=
K=$(cd "$(dirname "$0")" && pwd)
W=~/ort_keramos/genai_main
OUT=${OUT:-$HOME/ort_keramos/retest_main.txt}
if [ ! -d "$W/.git" ]; then
  git clone -q --filter=blob:none --sparse https://github.com/microsoft/onnxruntime-genai.git "$W"
  (cd "$W" && git sparse-checkout set src/python/py/models)
fi
cd "$W" && git fetch -q origin main && git checkout -q --detach origin/main
B="$W/src/python/py/models/builder.py"
{
  echo "main $(git -C "$W" rev-parse HEAD) $(git -C "$W" log -1 --format=%cI)"
  echo "contains 25809d6 (#2884): $(git -C "$W" merge-base --is-ancestor 25809d6f0d09f7aadb3e0a202a8b1ac625962db5 HEAD && echo yes || echo no)"
  echo "contains 5e4a263 (#2866): $(git -C "$W" merge-base --is-ancestor 5e4a2630421a0221ff1c2e3c74193fe1b2c0b3dc HEAD && echo yes || echo no)"
  for py in ~/ort_keramos/venv/bin/python ~/ort_keramos/venv_tf4/bin/python; do
    tag=$($py -c 'import transformers;print("tf"+transformers.__version__)')
    # ort1_repro.py always calls the installed builder, so #2788 is retested through gemma3.py under transformers 5
    for r in ort2_repro.py ort3_repro.py gemma3.py mistral.py; do
      echo "##### $r $tag main"
      cd "$(mktemp -d)" && timeout 900 $py -B "$K/$r" "$B" 2>&1 | grep -v -e Warning -e warn | tail -n 25
    done
  done
} > "$OUT" 2>&1
echo DONE >> "$OUT"
