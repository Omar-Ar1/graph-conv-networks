from __future__ import annotations

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) implementation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.6,
        activation: Optional[nn.Module] = None,
        add_self_loops: bool = True,
        use_skip_connections: bool = True,
    ) -> None:
        super().__init__()
        
        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        self.dropout = dropout
        self.activation = activation if activation is not None else nn.ELU()
        self.use_skip_connections = use_skip_connections
        
        self.layers = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        
        # Input layer
        # Multi-head attention: output dim is hidden_dim * heads
        self.layers.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, add_self_loops=add_self_loops))
        
        if use_skip_connections:
             if input_dim != hidden_dim * heads:
                 self.skip_projections.append(nn.Linear(input_dim, hidden_dim * heads))
             else:
                 self.skip_projections.append(nn.Identity())
        else:
             self.skip_projections.append(nn.Identity())

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, add_self_loops=add_self_loops))
            
            if use_skip_connections:
                if hidden_dim * heads != hidden_dim * heads:
                    self.skip_projections.append(nn.Linear(hidden_dim * heads, hidden_dim * heads))
                else:
                    self.skip_projections.append(nn.Identity())
            else:
                self.skip_projections.append(nn.Identity())

        # Output layer
        # Typically average heads for the last layer in classification
        self.layers.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout, add_self_loops=add_self_loops))
        
        if use_skip_connections:
            if hidden_dim * heads != output_dim:
                self.skip_projections.append(nn.Linear(hidden_dim * heads, output_dim))
            else:
                self.skip_projections.append(nn.Identity())
        else:
            self.skip_projections.append(nn.Identity())

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = x
        
        for i, (layer, skip_proj) in enumerate(zip(self.layers, self.skip_projections)):
            h_in = h
            h = layer(h, edge_index)
            
            if i < len(self.layers) - 1:
                h = self.activation(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            if self.use_skip_connections:
                h = h + skip_proj(h_in)
                
        return h

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()
        for skip in self.skip_projections:
            if hasattr(skip, "reset_parameters"):
                skip.reset_parameters()
