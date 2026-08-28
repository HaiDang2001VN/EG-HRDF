import threading
from typing import List, Optional

import numpy as np
import torch

from ..data.shapenet_sdf_stream import ShapeNetSDFObjectStream
from ..octree import build_flat_hierarchy
from ..octree.builder import FlatOctree
from ..octree.coordinates import encode_coords, octant_offsets


class _TreeEntry:
    __slots__ = ("model_id", "category", "tree", "level_index", "leaf_index", "leaf_mass", "n_points", "branch")

    def __init__(self, model_id: str, category: str, tree: FlatOctree):
        self.model_id = model_id
        self.category = category
        self.tree = tree
        self.branch = tree.branch
        self.n_points = tree.n_points
        self.level_index = {}
        for depth, level in tree.levels.items():
            self.level_index[depth] = {
                int(k): i for i, k in enumerate(encode_coords(level.coords, depth, self.branch))
            }
        leaf_depth = tree.max_depth
        self.leaf_index = {int(k): i for i, k in enumerate(
            encode_coords(tree.leaf_coords, leaf_depth, self.branch))}
        self.leaf_mass = tree.leaf_counts.astype(np.float64) / tree.n_points

    def node_mass(self, depth: int, cell: np.ndarray) -> Optional[float]:
        if depth >= self.tree.max_depth:
            i = self.leaf_index.get(int(encode_coords(np.asarray(cell)[None], depth, self.branch)[0]))
            return None if i is None else float(self.leaf_mass[i])
        idx = self.level_index.get(depth)
        if idx is None:
            return None
        i = idx.get(int(encode_coords(np.asarray(cell)[None], depth, self.branch)[0]))
        return None if i is None else float(self.tree.levels[depth].mass[i])


class TripleReservoir:
    """Bounded reservoir of recently streamed objects, converted to flat trees."""

    def __init__(self, stream: ShapeNetSDFObjectStream, size: int = 8, depth: int = 6,
                 branch: int = 2, seed: int = 0):
        self.stream = stream
        self.size = size
        self.depth = depth
        self.branch = branch
        self.rng = np.random.default_rng(seed)
        self.entries: List[_TreeEntry] = []
        self._seen = 0
        self._lock = threading.Lock()
        self._fill_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _background_fill(self):
        for record in iter(self.stream):
            if self._stop.is_set():
                break
            tree = build_flat_hierarchy(record["points"], max_depth=self.depth, branch=self.branch)
            entry = _TreeEntry(record["model_id"], record["category"], tree)
            with self._lock:
                self._seen += 1
                if len(self.entries) < self.size:
                    self.entries.append(entry)
                else:
                    j = int(self.rng.integers(0, self._seen))
                    if j < self.size:
                        self.entries[j] = entry

    def start(self):
        self._fill_thread = threading.Thread(target=self._background_fill, daemon=True)
        self._fill_thread.start()

    def stop(self):
        self._stop.set()
        self.stream.close()

    def ready(self, min_entries: int = 1) -> bool:
        return len(self.entries) >= min_entries

    def sample_entry(self) -> _TreeEntry:
        with self._lock:
            return self.entries[int(self.rng.integers(0, len(self.entries)))]

    @property
    def seen(self) -> int:
        return self._seen


class TripleBatcher:
    """Samples (parent, b^3 children, grandchild masses) triples from a reservoir."""

    def __init__(self, reservoir: TripleReservoir, min_depth: int = 0, seed: int = 0):
        self.reservoir = reservoir
        self.branch = reservoir.branch
        self.min_depth = min_depth
        self.n_children = self.branch ** 3
        self._offsets = octant_offsets(self.branch)
        self.rng = np.random.default_rng(seed)

    def sample_triple(self) -> dict:
        entry = self.reservoir.sample_entry()
        tree = entry.tree
        b = self.branch
        max_parent_depth = tree.max_depth - 2
        d = int(self.rng.integers(self.min_depth, max_parent_depth + 1))
        level = tree.levels[d]
        pi = int(self.rng.integers(0, len(level.coords)))
        parent_cell = level.coords[pi]
        parent_mass = float(level.mass[pi])
        parent_p1 = level.occupancy[pi]

        child_cells = b * parent_cell[None, :] + self._offsets
        child_depth = d + 1
        child_p1 = np.zeros((self.n_children, self.n_children), dtype=np.float64)
        child_mass = np.zeros(self.n_children, dtype=np.float64)
        grandchild_mass = np.zeros((self.n_children, self.n_children), dtype=np.float64)
        child_centers = np.zeros((self.n_children, 3), dtype=np.float64)
        for o in range(self.n_children):
            cc = child_cells[o]
            m = entry.node_mass(child_depth, cc)
            if m is None:
                continue
            child_mass[o] = m
            if child_depth < tree.max_depth:
                ci = entry.level_index[child_depth].get(int(encode_coords(cc[None], child_depth, b)[0]))
                if ci is not None:
                    cl = tree.levels[child_depth]
                    child_p1[o] = cl.occupancy[ci]
                    child_centers[o] = cl.centers[ci]
            else:
                child_centers[o] = (cc + 0.5) / (b ** child_depth) * 2 - 1
            gc_cells = b * cc[None, :] + self._offsets
            for oo in range(self.n_children):
                gm = entry.node_mass(child_depth + 1, gc_cells[oo])
                if gm is not None:
                    grandchild_mass[o, oo] = gm

        parent_center = (parent_cell + 0.5) / (b ** d) * 2 - 1
        parent_depth_frac = d / max(tree.max_depth, 1)
        child_depth_frac = child_depth / max(tree.max_depth, 1)
        return {
            "model_id": entry.model_id,
            "parent_depth": d,
            "parent_cell": parent_cell,
            "child_depth": child_depth,
            "child_cell": child_cells,
            "parent_p1": parent_p1,
            "parent_mass": parent_mass,
            "parent_e": np.concatenate([parent_center, [parent_depth_frac, parent_mass]]),
            "child_p1": child_p1,
            "child_mass": child_mass,
            "child_e": np.concatenate(
                [child_centers, np.full((self.n_children, 1), child_depth_frac), child_mass[:, None]], axis=1
            ),
            "grandchild_mass": grandchild_mass,
        }

    def sample_batch(self, batch_triples: int) -> dict:
        items = [self.sample_triple() for _ in range(batch_triples)]

        def stack(key):
            return torch.as_tensor(np.stack([it[key] for it in items]).astype(np.float32))

        return {
            "model_ids": [it["model_id"] for it in items],
            "parent_depth": torch.tensor([it["parent_depth"] for it in items], dtype=torch.long),
            "parent_cell": torch.tensor(np.stack([it["parent_cell"] for it in items]), dtype=torch.long),
            "child_depth": torch.tensor([it["child_depth"] for it in items], dtype=torch.long),
            "child_cell": torch.tensor(np.stack([it["child_cell"] for it in items]), dtype=torch.long),
            "parent_p1": stack("parent_p1"),
            "parent_mass": stack("parent_mass"),
            "parent_e": stack("parent_e"),
            "child_p1": stack("child_p1"),
            "child_mass": stack("child_mass"),
            "child_e": stack("child_e"),
            "grandchild_mass": stack("grandchild_mass"),
        }
