import time

import fsspec
import pyarrow.parquet as pq

url = "https://huggingface.co/datasets/EPFL-IVRL/ShapeNetSDF/resolve/main/data/chair/train-00000-of-00064.parquet"

t0 = time.time()
with fsspec.open(url, "rb") as f:
    pf = pq.ParquetFile(f)
    md = pf.metadata
    print("shard opened in %.1fs" % (time.time() - t0))
    print("rows:", md.num_rows, "| row groups:", md.num_row_groups)
    rg0 = md.row_group(0)
    print("rg0 rows:", rg0.num_rows, "| total byte size: %.1f MB" % (rg0.total_byte_size / 1e6))
    print("columns:", [md.row_group(0).column(j).path_in_schema for j in range(rg0.num_columns)])

    t1 = time.time()
    tbl = pf.read_row_group(0, columns=["model_id", "groundtruth"])
    t2 = time.time()
    print("rg0 read (model_id+groundtruth): %.1fs" % (t2 - t1))
    ids = tbl.column("model_id").to_pylist()
    gt = tbl.column("groundtruth").to_numpy(zero_copy_only=False)
    print("ids:", ids[:3])
    import numpy as np
    g0 = np.asarray(gt[0], dtype=np.float32)
    print("gt0:", g0.shape, g0.dtype, "| xyz range: [%.2f, %.2f]" % (g0[:, 0].min(), g0[:, 0].max()))
