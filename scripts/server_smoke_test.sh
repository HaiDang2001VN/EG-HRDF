#!/usr/bin/env bash
# Smoke tests for bm87: run after clone/pull with the hrdf conda env active.
set -e
cd "$(dirname "$0")/.."

echo "== [1/4] EG-HRDF core unit tests (no CUDA required) =="
python -m pytest tests/ -q

echo "== [2/4] EG-HRDF synthetic train smoke =="
python train_hrdf.py --synthetic --max-shapes 8 --epochs 2 --blocks-per-shape 32 --depth 5 --npoints 512 --eval-every 2

echo "== [3/4] CUDA check + PVCNN backend build =="
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import modules.functional.backend; print('pvcnn backend OK')"

echo "== [4/4] PSF original train (1 epoch, small batch) =="
python train_flow.py --category chair --bs 8 --niter 1 --distribution_type single --workers 2

echo "ALL SMOKE TESTS PASSED"
