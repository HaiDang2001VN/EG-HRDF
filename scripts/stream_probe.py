import time
import resource

import numpy as np

from datasets import load_dataset

t0 = time.time()
ds = load_dataset("EPFL-IVRL/ShapeNetSDF", "chair", split="train", streaming=True)
it = iter(ds)
row = next(it)
t1 = time.time()
gt = np.asarray(row["groundtruth"], dtype=np.float32)
print("model_id:", row["model_id"])
print("groundtruth:", gt.shape, gt.dtype, "| xyz range: [%.2f, %.2f]" % (gt[:, 0].min(), gt[:, 0].max()))
print("sdf abs max: %.3f" % abs(gt[:, 3]).max())
print("first-row latency: %.1fs | peak RSS: %.0f MB" % (t1 - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
t2 = time.time()
row2 = next(it)
print("second row: %.1fs | id: %s" % (time.time() - t2, row2["model_id"]))
