import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eg_hrdf.data.shapenet_sdf_stream import ShapeNetSDFObjectStream, _model_seed


N = 262144


def _fake_row(model_id: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    def arr():
        a = rng.uniform(-1, 1, size=(N, 4)).astype(np.float32)
        a[:, 3] = rng.uniform(-0.1, 0.1, size=N).astype(np.float32)
        return a
    return {"model_id": model_id, "uniform": arr(), "groundtruth": arr()}


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parquet")
    rows = [_fake_row("aaa" * 10 + "1", 0), _fake_row("bbb" * 10 + "2", 1),
            _fake_row("ccc" * 10 + "3", 2), _fake_row("ddd" * 10 + "4", 3)]
    n = len(rows)
    inner = {}
    for col in ("uniform", "groundtruth"):
        flat = np.concatenate([r[col].reshape(-1) for r in rows])
        fixed = pa.FixedSizeListArray.from_arrays(pa.array(flat), 4)
        inner[col] = pa.ListArray.from_arrays(
            pa.array(np.arange(n + 1, dtype=np.int32) * N), fixed
        )
    table = pa.table({"model_id": pa.array([r["model_id"] for r in rows]),
                      "uniform": inner["uniform"],
                      "groundtruth": inner["groundtruth"]})
    path = tmp / "shard0.parquet"
    pq.write_table(table, path, row_group_size=2)
    ids = [r["model_id"] for r in rows]
    return str(path), ids


def test_stream_yields_all_objects(shard_dir):
    path, ids = shard_dir
    stream = ShapeNetSDFObjectStream(shard_urls=[("chair", path)], n_points=4096)
    got = list(iter(stream))
    assert len(got) == len(ids)
    assert [g["model_id"] for g in got] == ids
    assert got[0]["points"].shape == (4096, 3)
    assert got[0]["points"].min() >= -1.0 and got[0]["points"].max() <= 1.0


def test_stream_deterministic_subsample(shard_dir):
    path, _ = shard_dir
    s1 = list(iter(ShapeNetSDFObjectStream(shard_urls=[("chair", path)], n_points=1024)))
    s2 = list(iter(ShapeNetSDFObjectStream(shard_urls=[("chair", path)], n_points=1024)))
    for a, b in zip(s1, s2):
        assert np.array_equal(a["points"], b["points"])


def test_stream_with_sdf(shard_dir):
    path, _ = shard_dir
    stream = ShapeNetSDFObjectStream(shard_urls=[("chair", path)], n_points=512, with_sdf=True)
    rec = next(iter(stream))
    assert rec["sdf_points"].shape == (512, 3)
    assert rec["sdf"].shape == (512,)


def test_model_seed_stable():
    assert _model_seed("abc") == _model_seed("abc")
    assert _model_seed("abc") != _model_seed("abd")


def test_prefetch_bounded(shard_dir):
    path, ids = shard_dir
    stream = ShapeNetSDFObjectStream(shard_urls=[("chair", path)], n_points=256, prefetch=1)
    it = iter(stream)
    first = next(it)
    assert first["model_id"] == ids[0]
    stream.close()
