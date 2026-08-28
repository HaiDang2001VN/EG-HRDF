#!/usr/bin/env bash
# Full-stage mock test (3-5 samples per stage). All stages must PASS before any
# full-scale run. Run on bm87 with the hrdf venv active.
set -uo pipefail
cd ~/code/EG-HRDF
mkdir -p output/mock
PASS=0; FAIL=0

check() {  # check <name> <exit-code>
  if [ "$2" -eq 0 ]; then echo "PASS: $1"; PASS=$((PASS+1)); else echo "FAIL: $1"; FAIL=$((FAIL+1)); fi
}

echo "=== M1: streaming + octree (3 objects, with_sdf) ==="
.venv/bin/python -u - <<'EOF' && check M1_streaming_octree 0 || check M1_streaming_octree 1
import sys; sys.path.insert(0, ".")
from eg_hrdf.data import ShapeNetSDFObjectStream
from eg_hrdf.octree import build_flat_hierarchy
s = ShapeNetSDFObjectStream(config="chair", split="train", n_points=2048, with_sdf=True, max_objects=3)
n = 0
for rec in iter(s):
    t = build_flat_hierarchy(rec["points"], max_depth=3, branch=4)
    assert t.check_mass_conservation()
    assert "sdf" in rec and rec["sdf"].shape == (2048,)
    n += 1
assert n == 3, n
print("M1 ok: 3 objects, trees conserve mass, sdf present")
EOF

echo "=== M2: captions join + CLIP cache ==="
.venv/bin/python -u - <<'EOF' && check M2_text_cache 0 || check M2_text_cache 1
import sys, os, json; sys.path.insert(0, ".")
from eg_hrdf.data.caption_index import ShapeNetCaptionIndex
from eg_hrdf.data.join_validator import load_category_ids, validate_join
caps = ShapeNetCaptionIndex.load()
ids = load_category_ids("chair")[:5]
rep = validate_join(caps, ids, min_rate=1.0)
assert rep.match_rate == 1.0
emb = __import__("numpy").load("data/text_cache/caption_embeddings.npy")
assert emb.shape[1] == 512 and len(emb) > 0
print("M2 ok: join 100%, cache", emb.shape)
EOF

echo "=== M3: EG-HRDF training mock (60 steps, branch=4, z-hier + text) ==="
.venv/bin/python -u -m eg_hrdf.training.train --config chair --steps 60 --log-every 30 \
  --reservoir 3 --batch-triples 2 --branch 4 --depth 3 --arch perceiver --z-mode hier --z-dim 32 \
  --text-embeddings data/text_cache --out-dir output/mock/m3 > output/mock/m3.log 2>&1 \
  && check M3_training 0 || check M3_training 1
.venv/bin/python -u - <<'EOF' && true
import torch, math
ck = torch.load("output/mock/m3/hrdf_stream_latest.pth", map_location="cpu", weights_only=False)
assert ck["net"] is not None
print("M3 ckpt ok")
EOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo "=== M4: direct-occupancy ablation mock (40 steps) ==="
.venv/bin/python -u -m eg_hrdf.training.train --config chair --steps 40 --log-every 20 \
  --reservoir 3 --batch-triples 2 --branch 4 --depth 3 --arch perceiver --z-mode none \
  --flow-mode direct --out-dir output/mock/m4 > output/mock/m4.log 2>&1 \
  && check M4_direct 0 || check M4_direct 1

echo "=== M5: hash-context mock (40 steps, 6-neighbors) ==="
.venv/bin/python -u -m eg_hrdf.training.train --config chair --steps 40 --log-every 20 \
  --reservoir 3 --batch-triples 2 --branch 4 --depth 3 --arch perceiver --z-mode none \
  --use-hash --hash-neighbors 6 --out-dir output/mock/m5 > output/mock/m5.log 2>&1 \
  && check M5_hash 0 || check M5_hash 1

echo "=== M6: generation + budget eval mock (4 gen / 8 ref) ==="
.venv/bin/python -u scripts/eval_hrdf.py --ckpt output/mock/m3/hrdf_stream_latest.pth \
  --config chair --split val --n-gen 4 --n-ref 8 --budgets 1.0 0.25 \
  --out output/mock/eval_m6.json > output/mock/m6.log 2>&1 \
  && check M6_eval_run 0 || check M6_eval_run 1
.venv/bin/python -u - <<'EOF' && check M6_eval_sane 0 || check M6_eval_sane 1
import json, math
r = json.load(open("output/mock/eval_m6.json"))
rows = r["results"]
assert len(rows) == 2
assert rows[1]["mean_evaluated"] < rows[0]["mean_evaluated"] * 0.6
for row in rows:
    for m in ("MMD-CD", "COV", "1-NNA-CD", "JSD", "MMD-DCD"):
        assert math.isfinite(row[m]), (m, row)
print("M6 ok: K scales with rho, metrics finite")
EOF

echo "=== M7: semantic eval mock (2 clouds) ==="
.venv/bin/python -u - <<'EOF' && check M7_semantic 0 || check M7_semantic 1
import sys, json; sys.path.insert(0, ".")
import numpy as np
from eg_hrdf.evaluation import CLIPScorer, semantic_scores
caps = json.load(open("data/text_cache/captions_filtered.json"))
ids = list(caps)[:2]
scorer = CLIPScorer(device="cpu")
rng = np.random.default_rng(0)
out = []
for mid in ids:
    z = 2 * rng.random(1024) - 1
    r = np.sqrt(np.maximum(1 - z * z, 0))
    pts = np.stack([r * np.cos(2 * np.pi * rng.random(1024)), r * np.sin(2 * np.pi * rng.random(1024)), z], 1).astype(np.float32)
    s = semantic_scores(pts, caps[mid], scorer, image_size=64)
    assert all(np.isfinite(v) for v in s.values())
    out.append(s)
print("M7 ok:", out)
EOF

echo "=== M8: PSF loop mock (export 5 -> train 2 epochs -> test_flow) ==="
rm -rf /tmp/psf_mock ~/psf_mock_root
.venv/bin/python -u scripts/export_psf_format.py --config chair --splits train val \
  --out /tmp/psf_mock --max-objects 5 > output/mock/m8_export.log 2>&1 \
  && check M8a_export 0 || check M8a_export 1
mkdir -p ~/psf_mock_root/ShapeNetCore.v2.PC15k
ln -sfn /tmp/psf_mock/chair ~/psf_mock_root/ShapeNetCore.v2.PC15k/chair
.venv/bin/python -u train_flow.py --category chair --bs 2 --niter 2 --workers 0 --saveIter 1 \
  --distribution_type single --dataroot ~/psf_mock_root/ShapeNetCore.v2.PC15k/ \
  > output/mock/m8_train.log 2>&1 \
  && check M8b_psf_train 0 || check M8b_psf_train 1
CKPT=$(ls -t output/train_flow/*/epoch_*.pth 2>/dev/null | head -1)
echo "ckpt: $CKPT"
if [ -n "$CKPT" ]; then
  .venv/bin/python -u test_flow.py --category chair --bs 4 --workers 0 --step 1 \
    --dataroot ~/psf_mock_root/ShapeNetCore.v2.PC15k/ --model "$CKPT" \
    > output/mock/m8_test.log 2>&1 \
    && check M8c_psf_test 0 || check M8c_psf_test 1
else
  echo "FAIL: M8c_psf_test (no ckpt)"; FAIL=$((FAIL+1))
fi

echo "=== M9: entropy-validity mock (3 objects) ==="
.venv/bin/python -u scripts/entropy_validity.py --config chair --n-objects 3 --depth 6 --branch 4 \
  --out output/mock/entropy_m9.json > output/mock/m9.log 2>&1 \
  && check M9_entropy 0 || check M9_entropy 1
.venv/bin/python -u - <<'EOF' && check M9_entropy_sane 0 || check M9_entropy_sane 1
import json, math
r = json.load(open("output/mock/entropy_m9.json"))
assert math.isfinite(r["auc_entropy"])
assert 0.0 < r["positive_rate"] < 1.0
print("M9 ok: auc_entropy=%.3f pos=%.3f" % (r["auc_entropy"], r["positive_rate"]))
EOF

echo ""
echo "================ MOCK SUMMARY: PASS=$PASS FAIL=$FAIL ================"
if [ "$FAIL" -eq 0 ]; then echo "ALL STAGES VERIFIED — full-scale runs may proceed"; else echo "DO NOT START FULL RUNS — fix failing stages"; exit 1; fi
