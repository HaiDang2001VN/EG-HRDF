# BM87 Server Setup (EG-HRDF / PSF modernized stack)

Target: single RTX 3090/4090 (24 GB, sm_86/sm_89). PSF's original pin (torch 1.4 +
CUDA 10.1) cannot run on these GPUs, so we use torch 2.x and rebuild the three CUDA
extensions. nvidia-smi may be unavailable on bm87 due to root restrictions; use
`python -c "import torch; print(torch.cuda.is_available())"` instead.

## 1. Conda environment

```bash
conda create -n hrdf python=3.10 -y
conda activate hrdf
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

If `torch.cuda.is_available()` is False, ask the admin which CUDA driver is exposed,
then pick the matching wheel index (cu118/cu121/cu124).

## 2. CUDA extensions

Set arch flags for the local GPU before building:

```bash
export TORCH_CUDA_ARCH_LIST="8.6"    # 3090/A5000; use "8.9" for 4090, "8.6;8.9" for both
```

### 2.1 PVCNN backend (needed by model/pvcnn_generation.py)

```bash
python -c "import modules.functional.backend"   # compiles _pvcnn_backend on first import
```

If `torch.utils.cpp_extension.load` fails on ninja or g++, install:
`conda install -y ninja` and ensure `g++ >= 9`.

### 2.2 PyTorchEMD (needed by metrics/evaluation_metrics.py)

```bash
cd metrics/PyTorchEMD
python setup.py install
```

The original code uses deprecated THC headers and `Tensor::data<T>()`. Apply the
in-repo torch 2.x patch **before** building (run once from the repo root):

```bash
bash patches/apply_torch2_patches.sh
cd metrics/PyTorchEMD && python setup.py install
```

### 2.3 Chamfer3D (needed by test_flow.py / evaluation metrics)

```bash
bash patches/apply_torch2_patches.sh   # idempotent; also fixes .data<T>() here
cd metrics/ChamferDistancePytorch/chamfer3D
python setup.py install
```

Note: EG-HRDF core (`eg_hrdf/`, `train_hrdf.py`) needs **no** compiled extensions;
a CPU-only box can run all `tests/`.

## 3. Data

Download the PointFlow ShapeNet point clouds (ShapeNetCore.v2.PC15k) and extract:

```bash
mkdir -p data && cd data
# follow https://github.com/stevenygd/PointFlow (dataset download section)
unzip ShapeNetCore.v2.PC15k.zip
```

## 4. Smoke tests (in order)

```bash
bash scripts/server_smoke_test.sh
```

which runs:

1. `python -m pytest tests/ -q` (EG-HRDF core, CPU or GPU)
2. `python train_hrdf.py --synthetic --max-shapes 8 --epochs 2 --blocks-per-shape 32 --depth 5 --npoints 512` (EG-HRDF core)
3. `python -c "import modules.functional.backend"` (PVCNN CUDA backend build)
4. `python train_flow.py --category chair --bs 8 --niter 1 --distribution_type single --workers 2` (PSF original, 1 epoch)

## 5. Git workflow

Local Mac: implement + mock test in `vision` conda env, commit & push to origin
(`HaiDang2001VN/EG-HRDF`). Then on bm87:

```bash
ssh bm87
cd ~/code && git clone https://github.com/HaiDang2001VN/EG-HRDF.git   # first time
cd EG-HRDF && git pull
conda activate hrdf && bash scripts/server_smoke_test.sh
```
