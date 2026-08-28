#!/usr/bin/env bash
cd ~/code/EG-HRDF
for zm in none independent hier; do
  .venv/bin/python -u scripts/eval_hrdf.py --ckpt "output/ws3_b4_${zm}/hrdf_stream_latest.pth" \
    --config chair --split val --n-gen 64 --n-ref 128 \
    --budgets 1.0 0.75 0.5 0.25 0.1 --out "output/eval_ws3_${zm}.json" \
    > "output/eval_ws3_${zm}.log" 2>&1
done
echo EVALS_DONE
