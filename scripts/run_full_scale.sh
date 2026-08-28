#!/usr/bin/env bash
# Full-scale quick-iteration runs (100% data, scaled batch, modest steps).
# GPU: RTX 3070 8GB. Higher-scale reruns deferred until results are positive.
set -e
cd ~/code/EG-HRDF
export CUDA_HOME=$HOME/cuda-12.1 PATH=$HOME/cuda-12.1/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.6" PYTORCH_NVCC=$HOME/nvcc-wrapper

# Quick-iteration hyperparameters (GPU-saturating batch, small model, 20k steps)
COMMON="--branch 4 --depth 3 --arch perceiver --steps 20000 --log-every 2000 \
  --reservoir 12 --batch-triples 32 --lr 2e-4 --gamma 0.5 --lambda-hier 0.1"

# 1) chair 100% z-hier (primary)
.venv/bin/python -u -m eg_hrdf.training.train --config chair --split train \
  --z-mode hier --z-dim 32 $COMMON --out-dir output/full_chair_hier \
  > output/full_chair_hier.log 2>&1
echo CHAIR_HIER_DONE

# 2) freestyle all-50-categories z-hier (thesis target)
.venv/bin/python -u -m eg_hrdf.training.train --mode all --split train \
  --z-mode hier --z-dim 32 $COMMON --out-dir output/full_all_hier \
  > output/full_all_hier.log 2>&1
echo ALL_HIER_DONE

# 3) direct-occupancy ablation (chair)
.venv/bin/python -u -m eg_hrdf.training.train --config chair --split train \
  --z-mode hier --z-dim 32 --flow-mode direct $COMMON --out-dir output/full_chair_direct \
  > output/full_chair_direct.log 2>&1
echo DIRECT_DONE

# 4) hash-context ablation (chair)
.venv/bin/python -u -m eg_hrdf.training.train --config chair --split train \
  --z-mode hier --z-dim 32 --use-hash --hash-neighbors 6 $COMMON --out-dir output/full_chair_hash \
  > output/full_chair_hash.log 2>&1
echo HASH_DONE

# 5) text-conditioned (chair, CLIP cache)
.venv/bin/python -u -m eg_hrdf.training.train --config chair --split train \
  --z-mode hier --z-dim 32 --text-embeddings data/text_cache $COMMON \
  --out-dir output/full_chair_text > output/full_chair_text.log 2>&1
echo TEXT_DONE

echo FULL_SCALE_ALL_DONE
