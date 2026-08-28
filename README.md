# PSF → EG-HRDF

This repository implements **EG-HRDF** (Entropy-Guided Hierarchical Rectified Density Flow):
information-adaptive hierarchical point-cloud generation, built on top of the PSF
(straight-flow point cloud generation) codebase.

- `eg_hrdf/` — EG-HRDF core (pure PyTorch, no CUDA extensions required):
  adaptive octree representation (`octree.py`), local simplex rectified density flow
  (`flow.py`), FiLM density network (`network.py`), entropy-budgeted adaptive scheduler
  (`scheduler.py`), hierarchical stochastic latents (`hier_latent.py`), spatial hash
  context (`hash_context.py`), density-aware chamfer (`metrics.py`).
- `train_hrdf.py` — Stage A/B training entry point (synthetic smoke mode: `--synthetic`).
- `tests/` — unit tests (mass conservation, flow objective, scheduler budget, DCD).
- `SETUP_BM87.md` — server environment guide (torch 2.x modernization for RTX 3090/4090).
- `patches/apply_torch2_patches.sh` — torch 2.x fixes for the PSF CUDA extensions.

Quick start (CPU is enough for the EG-HRDF core):

```bash
python -m pytest tests/ -q
python train_hrdf.py --synthetic --max-shapes 12 --epochs 3 --blocks-per-shape 64
```

Everything below is the original PSF documentation.

# PSF
This is the official code of
> **[Fast Point Cloud Generation with Straight Flows](https://arxiv.org/abs/2212.01747)** \
> Lemeng Wu, Dilin Wang, Chengyue Gong, Xingchao Liu, Yunyang Xiong, Rakesh Ranjan, Raghuraman Krishnamoorthi, Vikas Chandra, Qiang Liu

<p align="center">
  <img src="assets/teaser.png" width="90%"/>
</p>

## About This Code:
Now we release code for training and inference. Some works are still in progress including pretrained checkpoint.

# Requirements:
This code is largely build based on [PVD](https://github.com/alexzhou907/PVD).
Make sure at least the following environments are installed (newer version may also works, we test in the below environments).

```
python==3.8
pytorch==1.4.0
torchvision==0.5.0
cudatoolkit==10.1
matplotlib==2.2.5
tqdm==4.32.1
open3d==0.9.0
trimesh=3.7.12
scipy==1.5.1
```

We also need to install pytorch3D for Chamfer Distance Loss, we recommend to follow the offical
install guideline [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

Install PyTorchEMD by
```
cd metrics/PyTorchEMD
python setup.py install
cp build/**/emd_cuda.cpython-36m-x86_64-linux-gnu.so .
```


## Data

We use the data follow PVD and PointFlow, which can be downloaded [here](https://github.com/stevenygd/PointFlow). Extract and put the data in ./data/ folder/


## Train:

First Stage, train the flow model. We do not add EMA here for a simple and quick converge as illustration.
```bash
$ python train_flow.py --category car|chair|airplane
```
Assume the checkpoint is saved as flow_checkpoint.pth (you can find it in the ./output/train_flow/ )

Second Stage, straight the flow, first sample the data pairs. We provide a single GPU version, in practice, we use
multiGPU to speed up.

```bash
$ python sample_flow.py --category car|chair|airplane --model flow_checkpoint.pth
```
Then run the reflow procedure:

```bash
$ python train_reflow.py --category car|chair|airplane --model flow_checkpoint.pth
```
Assume the checkpoint is saved as reflow_checkpoint.pth (you can find it in the ./output/train_reflow/ )

Third Stage, distill the flow.
```bash
$ python train_distill.py --category car|chair|airplane --model reflow_checkpoint.pth
```

Assume the checkpoint is saved as distill_checkpoint.pth (you can find it in the ./output/train_distill/ )


## Test:

```bash
$ python test_flow.py --category car|chair|airplane --model {flow|reflow|distill}_checkpoint.pth --step 1|20|50|100|500|1000
```

You can adjust the step in this test code. For flow, reflow model, we can still expect a good few-step generation.

## Reference

```
@InProceedings{Wu_2023_CVPR,
    author    = {Wu, Lemeng and Wang, Dilin and Gong, Chengyue and Liu, Xingchao and Xiong, Yunyang and Ranjan, Rakesh and Krishnamoorthi, Raghuraman and Chandra, Vikas and Liu, Qiang},
    title     = {Fast Point Cloud Generation With Straight Flows},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {9445-9454}
}
```

## Acknowledgement:
This code is built based on [PVD](https://github.com/alexzhou907/PVD). Thanks for their great code repo!
