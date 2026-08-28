"""Multi-resolution spatial hash grid context (learned feature tables)."""

import torch
import torch.nn as nn

NEIGHBOR_OFFSETS_6 = torch.tensor(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.long
)
NEIGHBOR_OFFSETS_18 = torch.cat([
    NEIGHBOR_OFFSETS_6,
    torch.tensor([[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
                  [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
                  [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]], dtype=torch.long),
])
NEIGHBOR_OFFSETS_26 = torch.stack(
    torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2), torch.arange(-1, 2), indexing="ij"),
    dim=-1,
).reshape(-1, 3)
NEIGHBOR_OFFSETS_26 = NEIGHBOR_OFFSETS_26[NEIGHBOR_OFFSETS_26.abs().sum(-1) > 0].long()

_OFFSETS = {6: NEIGHBOR_OFFSETS_6, 18: NEIGHBOR_OFFSETS_18, 26: NEIGHBOR_OFFSETS_26}


class SpatialHashContext(nn.Module):
    def __init__(
        self,
        n_levels: int = 2,
        table_size: int = 1024,
        feat_dim: int = 16,
        out_dim: int = 32,
        n_neighbors: int = 6,
        branch: int = 2,
    ):
        super().__init__()
        assert n_neighbors in _OFFSETS, f"n_neighbors must be one of {sorted(_OFFSETS)}"
        self.n_levels = n_levels
        self.table_size = table_size
        self.n_neighbors = n_neighbors
        self.branch = branch
        self.register_buffer("offsets", _OFFSETS[n_neighbors])
        self.tables = nn.ModuleList([nn.Embedding(table_size, feat_dim) for _ in range(n_levels)])
        for table in self.tables:
            nn.init.normal_(table.weight, std=0.1)
        self.mlp = nn.Sequential(
            nn.Linear(n_levels * n_neighbors * feat_dim, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    @staticmethod
    def hash_cell(cell: torch.Tensor, table_size: int) -> torch.Tensor:
        x, y, z = cell[..., 0].long(), cell[..., 1].long(), cell[..., 2].long()
        h = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
        return h % table_size

    def forward(self, cell: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """cell: (B, 3) int; depth: (B,) long/int tensor (per-row).

        Returns (B, out_dim) context features.
        """
        feats = []
        for level in range(self.n_levels):
            scale = (self.branch ** (depth + level)).clamp_min(1).float()[:, None]
            scaled = (cell.float() / scale).long()
            neighbors = scaled[:, None, :] + self.offsets[None].to(cell.device)
            idx = self.hash_cell(neighbors, self.table_size)
            feats.append(self.tables[level](idx).reshape(cell.shape[0], -1))
        return self.mlp(torch.cat(feats, dim=-1))
