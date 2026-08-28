#!/usr/bin/env bash
# WS5: PSF baseline on the bounded export (equal data), then test_flow metrics.
# Run AFTER scripts/export_psf_format.py completes for chair.
set -e
cd ~/code/EG-HRDF
export CUDA_HOME=$HOME/cuda-12.1 PATH=$HOME/cuda-12.1/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.6" PYTORCH_NVCC=$HOME/nvcc-wrapper

# PSF train_flow expects ./ShapeNetCore.v2.PC15k/<synset>/<split>/<id>.npy
mkdir -p ShapeNetCore.v2.PC15k
ln -sfn ~/code/EG-HRDF/data/psf_export/03001627 ShapeNetCore.v2.PC15k/03001627

.venv/bin/python -u train_flow.py --category chair --bs 48 --niter 500 \
  --distribution_type single --workers 4 --dataroot ./ShapeNetCore.v2.PC15k/ \
  > output/psf_train_flow.log 2>&1
echo PSF_FLOW_DONE

CKPT=$(ls -t output/train_flow/epoch_*.pth | head -1)
.venv/bin/python -u test_flow.py --category chair --model "$CKPT" --step 1 \
  > output/psf_test_flow.log 2>&1
echo PSF_TEST_DONE
