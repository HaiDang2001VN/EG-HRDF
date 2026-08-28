#!/usr/bin/env bash
# torch 2.x compatibility patches for the PSF CUDA extensions
# (THC headers removed in torch >= 1.11; Tensor::data<T> removed in torch >= 1.13)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

emd="$ROOT/metrics/PyTorchEMD/cuda/emd_kernel.cu"
if [ -f "$emd" ]; then
  sed -i.bak -e 's|#include <THC/THC.h>|#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>|' "$emd"
  sed -i.bak 's/THCudaCheck(/AT_CUDA_CHECK(/g' "$emd"
  sed -i.bak 's/\.data</.data_ptr</g' "$emd"
  sed -i.bak 's/x\.type()\.is_cuda()/x.is_cuda()/' "$emd"
  if ! grep -q "define CHECK_EQ" "$emd"; then
    sed -i.bak 's|#define CHECK_INPUT(x)|#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), #a " must equal " #b)\n#define CHECK_INPUT(x)|' "$emd"
  fi
  rm -f "$emd.bak"
  echo "patched: metrics/PyTorchEMD/cuda/emd_kernel.cu"
fi

for f in "$ROOT"/metrics/ChamferDistancePytorch/chamfer3D/chamfer3D.cu \
         "$ROOT"/metrics/ChamferDistancePytorch/chamfer3D/chamfer_cuda.cpp; do
  if [ -f "$f" ]; then
    sed -i.bak 's/\.data</.data_ptr</g' "$f"
    rm -f "$f.bak"
    echo "patched: ${f#$ROOT/}"
  fi
done

echo "done"
