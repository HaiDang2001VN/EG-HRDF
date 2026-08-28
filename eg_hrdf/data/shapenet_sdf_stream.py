import hashlib
import os
import logging
import queue
import threading
from enum import Enum
from typing import Iterator, List, Optional, Tuple

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ID = "EPFL-IVRL/ShapeNetSDF"
HF_RESOLVE = "https://huggingface.co/datasets/EPFL-IVRL/ShapeNetSDF/resolve/main"
POINTS_PER_ROW = 262144

logger = logging.getLogger("eg_hrdf.stream")


class StreamMode(Enum):
    CATEGORY = "category"
    ALL = "all"


def _model_seed(model_id: str) -> int:
    return int.from_bytes(hashlib.sha1(model_id.encode()).digest()[:8], "big")


def _list_shards(categories: List[str], split: str) -> List[Tuple[str, str]]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    out = []
    for cat in categories:
        prefix = f"data/{cat}/"
        for f in files:
            if f.startswith(prefix) and f"{split}-" in f and f.endswith(".parquet"):
                out.append((cat, f))
    out.sort()
    return out


def _rows_from_column(column) -> List[np.ndarray]:
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if isinstance(arr, pa.ListArray):
        offsets = np.asarray(arr.offsets, dtype=np.int64)
        flat = np.asarray(arr.values.flatten(), dtype=np.float32)
        rows = []
        for i in range(arr.offset, arr.offset + len(arr)):
            s, e = offsets[i] * 4, offsets[i + 1] * 4
            rows.append(flat[s:e].reshape(-1, 4))
        return rows
    values = column.to_numpy(zero_copy_only=False)
    rows = []
    for v in values:
        a = np.asarray(v, dtype=np.float32)
        rows.append(a.reshape(-1, 4) if a.ndim == 1 else a)
    return rows


class ShapeNetSDFObjectStream:
    """Streams one ShapeNet object at a time from sharded Parquet over HTTP.

    Yields dicts: {model_id, category, points (n_points,3) float32 [, sdf (n_points,)]}.
    Only the requested Parquet columns are read (groundtruth for training, uniform
    for SDF diagnostics); row-groups are decoded incrementally so no shard is
    fully materialized.
    """

    def __init__(
        self,
        config: str = "chair",
        split: str = "train",
        mode: StreamMode = StreamMode.CATEGORY,
        n_points: int = 16384,
        with_sdf: bool = False,
        prefetch: int = 2,
        max_retries: int = 3,
        seed: int = 0,
        max_objects: int = 0,
        shard_urls: Optional[List[Tuple[str, str]]] = None,
    ):
        self.config = config
        self.split = split
        self.mode = mode
        self.n_points = n_points
        self.with_sdf = with_sdf
        self.prefetch = prefetch
        self.max_retries = max_retries
        self.rng = np.random.default_rng(seed)
        self.max_objects = max_objects
        if shard_urls is not None:
            self._shards = shard_urls
        elif mode == StreamMode.CATEGORY:
            self._shards = _list_shards([config], split)
        else:
            from huggingface_hub import hf_hub_download
            import json

            p = hf_hub_download(REPO_ID, repo_type="dataset", filename="meta/all/labels.json")
            cats = sorted(json.load(open(p))["category_to_filename"].keys())
            self._shards = _list_shards(cats, split)
        self._queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=prefetch)
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self.failures: List[Tuple[str, str, int]] = []

    def _open_shard(self, url: str):
        f = fsspec.open(url, "rb").open()
        return f, pq.ParquetFile(f)

    @staticmethod
    def _shard_url(path: str) -> str:
        if path.startswith(("http://", "https://", "file:")) or os.path.exists(path):
            return path
        return f"{HF_RESOLVE}/{path}"

    def _decode_row_groups(self, cat: str, path: str):
        url = self._shard_url(path)
        columns = ["model_id", "groundtruth"] + (["uniform"] if self.with_sdf else [])
        attempt = 0
        while attempt <= self.max_retries:
            try:
                f, pf = self._open_shard(url)
                try:
                    for rg in range(pf.metadata.num_row_groups):
                        tbl = pf.read_row_group(rg, columns=columns)
                        ids = tbl.column("model_id").to_pylist()
                        gt = _rows_from_column(tbl.column("groundtruth"))
                        uni = _rows_from_column(tbl.column("uniform")) if self.with_sdf else None
                        for i in range(len(ids)):
                            yield self._make_record(cat, ids[i], gt[i], uni[i] if uni is not None else None)
                finally:
                    f.close()
                return
            except Exception as exc:
                attempt += 1
                self.failures.append((path, repr(exc), attempt))
                logger.warning("shard %s failed (attempt %d/%d): %r", path, attempt, self.max_retries + 1, exc)
        logger.error("shard %s dropped after %d retries", path, self.max_retries + 1)

    def _make_record(self, cat: str, model_id: str, gt: np.ndarray, uni: Optional[np.ndarray]) -> dict:
        gt = np.asarray(gt, dtype=np.float32)
        if gt.ndim == 1:
            gt = gt.reshape(-1, 4)
        k = min(self.n_points, len(gt))
        rng = np.random.default_rng(_model_seed(model_id))
        sel = rng.choice(len(gt), size=k, replace=False)
        points = gt[sel, :3]
        record = {"model_id": model_id, "category": cat, "points": points}
        if uni is not None:
            uni = np.asarray(uni, dtype=np.float32)
            if uni.ndim == 1:
                uni = uni.reshape(-1, 4)
            record["sdf_points"] = uni[sel, :3]
            record["sdf"] = uni[sel, 3]
        return record

    def _worker(self):
        count = 0
        try:
            for cat, path in self._shards:
                if self.max_objects and count >= self.max_objects:
                    break
                for record in self._decode_row_groups(cat, path):
                    if self.max_objects and count >= self.max_objects:
                        break
                    self._queue.put(record)
                    count += 1
        except Exception as exc:
            logger.exception("stream worker crashed: %r", exc)
        finally:
            self._queue.put(None)

    def __iter__(self) -> Iterator[dict]:
        if not self._started:
            self._started = True
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        return self

    def __next__(self) -> dict:
        item = self._queue.get()
        if item is None:
            raise StopIteration
        return item

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(None)
