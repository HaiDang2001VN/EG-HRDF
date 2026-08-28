"""Multi-resolution spatial hash grid context (learned feature tables)."""

import torch
import torch.nn as nn

NEIGHBOR_OFFSETS_27 = torch.stack(
    torch.meshgrid(
        torch.arange(-1, 2),
        torch.arange(-1, 2),
        torch.arange(-1, 2),
        indexing="ij",
    ),
    dim=-1,
).reshape(-1, 3)


class SpatialHashContext(nn.Module):
    def __init__(
        self,
        n_levels: int = 2,
        table_size: int = 1024,
        feat_dim: int = 16,
        out_dim: int = 32,
        n_neighbors: int = 27,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.table_size = table_size
        self.n_neighbors = n_neighbors
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

    def forward(self, cell: torch.Tensor, depth: int, max_depth: int) -> torch.Tensor:
        feats = []
        for level in range(self.n_levels):
            scale = max(2 ** (depth + level), 1)
            scaled = (cell.float() / scale).long()
            offsets = NEIGHBOR_OFFSETS_27.to(cell.device)
            neighbors = scaled[:, None, :] + offsets[None]
            idx = self.hash_cell(neighbors, self.table_size)
            feats.append(self.tables[level](idx).reshape(cell.shape[0], -1))
        return self.mlp(torch.cat(feats, dim=-1))
