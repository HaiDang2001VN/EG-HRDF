#!/usr/bin/env bash
set -e
cd ~/code/EG-HRDF
for zm in none independent hier; do
  .venv/bin/python -u -m eg_hrdf.training.train --config chair --steps 20000 --log-every 1000 \
    --reservoir 8 --batch-triples 4 --branch 4 --depth 3 --arch perceiver \
    --z-mode "$zm" --z-dim 32 --lambda-hier 0.1 \
    --out-dir "output/ws3_b4_${zm}" > "output/ws3_b4_${zm}.log" 2>&1
done
echo WS3_ALL_DONE
