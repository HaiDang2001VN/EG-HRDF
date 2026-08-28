# BM87 Server Setup (EG-HRDF / PSF modernized stack)

Target GPU: RTX 3070 (sm_86, 8 GB) — batch sizes for PSF/PVCNN must be scaled down
accordingly. Environment manager: **uv** (conda is not used on this server).

## 0. GPU status caveat (2026-08)

`nvidia-smi` fails with "Insufficient Permissions" and torch reports
"No CUDA GPUs are available" because `/dev/nvidia0` is missing and
`/usr/bin/nvidia-modprobe` (setuid root:video) is not executable by user
`hlhdang` (not in the `video` group). An admin must fix this, e.g.:

```bash
sudo nvidia-modprobe -u -c=0     # or simply reboot the machine
sudo usermod -aG video hlhdang   # then re-login
```

Until then everything below runs on CPU (EG-HRDF core tests do not need CUDA).

## 1. Environment (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv not installed
cd ~/code/EG-HRDF
uv python install 3.10
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python -r requirements.txt pytest
```

Driver 535.154.05 supports up to CUDA 12.2 → use cu121 wheels (cu124 will install
but `torch.cuda.is_available()` stays False).

Verify:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. CUDA extensions (only after GPU access is fixed)

Needs nvcc 12.1 (`conda install -c nvidia/label/cuda-12.1.1 cuda-nvcc cuda-toolkit`
inside a throwaway env, or the `nvidia-cuda-nvcc-cu12==12.1.105` pip package), then:

```bash
export CUDA_HOME=<nvcc prefix>
export TORCH_CUDA_ARCH_LIST="8.6"
.venv/bin/python -c "import modules.functional.backend"   # _pvcnn_backend
bash patches/apply_torch2_patches.sh                       # THC/.data<T> fixes
cd metrics/PyTorchEMD && ../../.venv/bin/python setup.py install && cd -
cd metrics/ChamferDistancePytorch/chamfer3D && ../../../.venv/bin/python setup.py install && cd -
```

## 3. Data

Download PointFlow's ShapeNetCore.v2.PC15k (see https://github.com/stevenygd/PointFlow)
and extract into `data/` (git-ignored).

## 4. Smoke tests

```bash
.venv/bin/python -m pytest tests/ -q                                   # CPU, no extensions
.venv/bin/python train_hrdf.py --synthetic --max-shapes 8 --epochs 3 --device cpu
# after GPU fix + extension builds:
bash scripts/server_smoke_test.sh
```

## 5. Git workflow

Local Mac: implement + mock test in the `vision` conda env, commit & push to origin
(`HaiDang2001VN/EG-HRDF`). Then on bm87:

```bash
cd ~/code/EG-HRDF && git pull
```
