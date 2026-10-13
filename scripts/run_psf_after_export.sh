#!/usr/bin/env bash
# Chains the PSF baseline after the bounded export completes (polls file count).
cd ~/code/EG-HRDF
export CUDA_HOME=$HOME/cuda-12.1 PATH=$HOME/cuda-12.1/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.6" PYTORCH_NVCC=$HOME/nvcc-wrapper

while true; do
  N=$(ls data/psf_export/03001627/train 2>/dev/null | wc -l)
  if pgrep -f 'psf_[f]ormat' > /dev/null; then :; else break; fi
  sleep 60
done
echo "export finished (${N} files), starting PSF baseline"
bash scripts/run_psf_baseline.sh
