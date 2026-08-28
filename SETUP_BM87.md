# BM87 Server Setup (EG-HRDF / PSF modernized stack)

Target GPU: RTX 3070 (sm_86, 8 GB). Environment manager: **uv** (no conda).
GPU works via driver only (535.154.05 → CUDA ≤ 12.2 → use cu121 wheels); torch
wheels carry the CUDA runtime, so nvidia-smi/NVML may be broken without breaking
CUDA compute.

## 0. GPU access caveat (resolved 2026-08-28)

`/dev/nvidia0` was missing (fixed by admin). NVML still inaccessible
("Can't initialize NVML" warning) — harmless for torch.

## 1. Environment (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv not installed
cd ~/code/EG-HRDF
uv python install 3.10
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python -r requirements.txt pytest transformers==4.44.2 gdown
```

transformers must be 4.44.2 (5.x rejects torch < 2.5).
HF token for the gated captions dataset lives in `~/.bashrc` as `HF_TOKEN`.

## 2. CUDA toolkit for extension builds (no root)

The 4 GB runfile stalls on this network; use the **redist components** (~46 MB):

```bash
mkdir -p ~/cuda-dl && cd ~/cuda-dl
curl -s https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.1.1.json -o redist.json
for f in cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.1.105-archive.tar.xz \
         cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.1.105-archive.tar.xz \
         cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.1.109-archive.tar.xz; do
  curl -sC - -O https://developer.download.nvidia.com/compute/cuda/redist/$f
done
mkdir -p extracted && for f in *.tar.xz; do tar xf $f -C extracted; done
mkdir -p ~/cuda-12.1
cp -a extracted/cuda_nvcc-linux-x86_64-12.1.105-archive/. ~/cuda-12.1/
cp -a extracted/cuda_cudart-linux-x86_64-12.1.105-archive/include/. ~/cuda-12.1/include/
cp -a extracted/cuda_cudart-linux-x86_64-12.1.105-archive/lib/. ~/cuda-12.1/lib64/
mkdir -p ~/cuda-12.1/include ~/cuda-12.1/lib64
cp -a extracted/cuda_cccl-linux-x86_64-12.1.109-archive/include/. ~/cuda-12.1/include/

# ATen headers need cusparse/cublas/etc from the pip nvidia packages:
SP=$HOME/code/EG-HRDF/.venv/lib/python3.10/site-packages/nvidia
for p in cusparse cublas cusolver curand cufft nccl nvtx nvjitlink; do
  cp -as $SP/$p/include/* ~/cuda-12.1/include/ 2>/dev/null
  cp -as $SP/$p/lib/* ~/cuda-12.1/lib64/ 2>/dev/null
done
```

### Host compiler problem (Gentoo gcc-15 vs nvcc 12.1)

System gcc is 15 (unsupported headers: `_FloatN`, bfloat16 literals). Install a
conda-forge gcc-11 toolchain via micromamba (no conda needed):

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
~/bin/micromamba create -y -p ~/gcc11 -c conda-forge \
    gcc_impl_linux-64=11 gxx_impl_linux-64=11 sysroot_linux-64=2.17
```

Wrapper that injects the host compiler:

```bash
cat > ~/nvcc-wrapper <<'EOF'
#!/bin/bash
exec /home/hlhdang/cuda-12.1/bin/nvcc -ccbin /home/hlhdang/gcc11/bin/x86_64-conda-linux-gnu-g++ -allow-unsupported-compiler "$@"
EOF
chmod +x ~/nvcc-wrapper
```

## 3. Build the three CUDA extensions

```bash
cd ~/code/EG-HRDF
export CUDA_HOME=$HOME/cuda-12.1 PATH=$HOME/cuda-12.1/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.6" PYTORCH_NVCC=$HOME/nvcc-wrapper
bash patches/apply_torch2_patches.sh          # THC -> ATen, .data<T> -> data_ptr, CHECK_EQ
python -c "import modules.functional.backend"  # _pvcnn_backend
(cd metrics/PyTorchEMD && ../../.venv/bin/python setup.py install)
(cd metrics/ChamferDistancePytorch/chamfer3D && ../../../.venv/bin/python setup.py install)
```

## 4. Data

- **EG-HRDF**: streamed from HF (`EPFL-IVRL/ShapeNetSDF`), nothing stored except
  `data/text_cache*/` (captions + CLIP embeddings).
- **PSF baseline**: bounded export (16K-point clouds in PC15k layout):
  `python scripts/export_psf_format.py --config chair --splits train val --out data/psf_export`

## 5. Smoke tests

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python train_hrdf.py --synthetic --max-shapes 8 --epochs 2 --device cpu
```

## 6. Git workflow

Local Mac: implement + mock test in the `vision` conda env, commit & push to origin
(`HaiDang2001VN/EG-HRDF`). On bm87: `git pull` and run.
