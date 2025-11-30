from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCN(nn.Module):
    """Relational GCN with simple linear relation-specific aggregators."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_relations: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_relations = num_relations
        self.dropout = dropout

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(RelationAggregator(hidden_dim, hidden_dim, num_relations))
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        h = x
        for layer in self.layers:
            if isinstance(layer, RelationAggregator):
                h = layer(h, adjs)
            else:
                h = layer(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out(h)


class RelationAggregator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.rels = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)])
        self.self_loop = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.self_loop.weight)

    def forward(self, x: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        out = self.self_loop(x)
        for idx, adj in enumerate(adjs):
            sparse_adj = adj if adj.layout == torch.sparse_coo else adj.to_sparse_coo()
            out += self.rels[idx](torch.sparse.mm(sparse_adj, x))
        return out
