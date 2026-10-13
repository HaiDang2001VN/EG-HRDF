#!/usr/bin/env bash
cd ~/code/EG-HRDF
for v in all_hier chair_direct chair_hash chair_text; do
  ckpt="output/full_${v}/hrdf_stream_latest.pth"
  case $v in
    all_hier) cfg="chair" ;;
    *) cfg="chair" ;;
  esac
  n_ref=128
  if [ "$v" = "all_hier" ]; then mode="all"; else mode="category"; fi
  .venv/bin/python -u scripts/eval_hrdf.py --ckpt "$ckpt" --config "$cfg" --split val \
    --mode "$mode" --n-gen 64 --n-ref $n_ref --budgets 1.0 0.75 0.5 0.25 0.1 \
    --out "output/eval_full_${v}.json" > "output/eval_full_${v}.log" 2>&1
done
echo FULL_EVALS_DONE
