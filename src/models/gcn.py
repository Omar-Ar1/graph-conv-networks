from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import add_self_loops, to_dense_adj, degree
try:
    from torch_sparse import SparseTensor, matmul
except ImportError:
    SparseTensor = None
    matmul = None
from entmax import entmax15 as entmax

@dataclass
class GCNConfig:
    """Configuration container for the Spectral GCN."""
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int = 2
    dropout: float = 0.1
    order: int = 1
    add_self_loops: bool = True
    activation: Optional[nn.Module] = None
    use_skip_connections: bool = True
    use_layer_norm: bool = False
    use_graph_norm: bool = False
    sparse_mode: bool = False  # Use sparse operations for efficiency
    adaptive_order: bool = False  # Learn order per layer
    cache_adj: bool = True  # Cache adjacency matrix (disable for inductive/mini-batch)


class AdaptiveOrderModule(nn.Module):
    """Learn optimal Chebyshev order per layer with attention mechanism."""
    
    def __init__(self, max_order: int = 10):
        super().__init__()
        self.max_order = max_order
        # Learnable attention weights over different orders
        self.order_attention = nn.Parameter(torch.ones(max_order + 1) / (max_order + 1))
        
    def forward(self, polynomial_features: List[Tensor]) -> Tensor:
        """
        Args:
            polynomial_features: List of features from T_0, T_1, ..., T_k
        Returns:
            Weighted combination of polynomial features
        """
        attention_weights = entmax(self.order_attention[:len(polynomial_features)], dim=0)
        result = sum(w * feat for w, feat in zip(attention_weights, polynomial_features))
        return result
    
    def get_effective_order(self) -> float:
        """Return the effective order based on attention weights."""
        weights = F.softmax(self.order_attention, dim=0)
        orders = torch.arange(len(weights), dtype=weights.dtype, device=weights.device)
        return (weights * orders).sum().item()


def scale_sparse(tensor: SparseTensor, scalar: float) -> SparseTensor:
    return tensor.set_value(tensor.storage.value() * scalar)

class GCN(nn.Module):
    """
    Enhanced spectral GCN with:
    - Skip connections (ResNet-style)
    - Layer/Batch normalization
    - Sparse operations for efficiency
    - Adaptive Chebyshev order (optional)
    - Over-smoothing monitoring
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        order: int = 2,
        activation: Optional[nn.Module] = None,
        add_self_loops: bool = True,
        use_skip_connections: bool = True,
        use_layer_norm: bool = True,
        use_graph_norm: bool = False,
        sparse_mode: bool = True,

        adaptive_order: bool = False,
        cache_adj: bool = True,
    ) -> None:
        super().__init__()
        
        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        self.config = GCNConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            order=order,
            add_self_loops=add_self_loops,
            activation=activation,
            use_skip_connections=use_skip_connections,
            use_layer_norm=use_layer_norm,
            use_graph_norm=use_graph_norm,
            sparse_mode=sparse_mode,
            adaptive_order=adaptive_order,
            cache_adj=cache_adj,
        )
        
        self.activation = activation if activation is not None else nn.ReLU()
        self.dropout = dropout
        self.order = order
        self.add_self_loops = add_self_loops
        self.use_skip_connections = use_skip_connections
        self.sparse_mode = sparse_mode
        self.adaptive_order = adaptive_order
        self.cache_adj = cache_adj
        
        # Cached adjacency
        self._cached_adj: Optional[Tensor] = None
        self._cached_sparse_adj: Optional[SparseTensor] = None
        
        # Over-smoothing metrics tracking
        self.layer_similarities: List[float] = []

        # Build layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        
        # Adaptive order modules
        if adaptive_order:
            self.order_modules = nn.ModuleList()
        
        in_features = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(in_features, hidden_dim))
            
            # Normalization
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif use_graph_norm:
                self.norms.append(GraphNorm(hidden_dim))
            else:
                self.norms.append(nn.Identity())
            
            # Skip connection projection
            if use_skip_connections:
                self.skip_projections.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.skip_projections.append(nn.Identity())
            
            # Adaptive order
            if adaptive_order:
                self.order_modules.append(AdaptiveOrderModule(max_order=order))
            
            in_features = hidden_dim
        
        # Output layer
        self.layers.append(nn.Linear(in_features, output_dim))
        self.norms.append(nn.Identity())
        self.skip_projections.append(nn.Linear(in_features, output_dim))
        
        if adaptive_order:
            self.order_modules.append(AdaptiveOrderModule(max_order=order))

    def forward(self, x: Tensor, edge_index: Tensor, return_embeddings: bool = False) -> Tensor | Tuple[Tensor, List[Tensor]]:
        """
        Forward pass with optional embedding extraction.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            return_embeddings: If True, return (output, layer_embeddings)
        """
        # Compute adjacency (cached)
        if self.sparse_mode:
            adjacency = self._get_sparse_adjacency(edge_index, num_nodes=x.size(0), device=x.device)
        else:
            adjacency = self._normalized_adjacency(edge_index, num_nodes=x.size(0), device=x.device)
        
        # Compute Chebyshev polynomials
        if self.order > 0:
            if self.adaptive_order:
                poly_features = self._compute_polynomial_features(adjacency, x)
            else:
                if self.sparse_mode:
                    adjacency = self._apply_chebyshev_polynomial_sparse(adjacency)
                else:
                    adjacency = self._apply_chebyshev_polynomial(adjacency)

        h_0 = self.input_projection(x) 
        h = h_0
        embeddings = []
        self.layer_similarities = []
        
        for layer_idx, (layer, norm, skip_proj) in enumerate(
            zip(self.layers, self.norms, self.skip_projections)
        ):
            h_in = h
            
            # Graph convolution
            if self.adaptive_order:
                # Apply different orders and combine with attention
                h_polys = [poly_feat @ h for poly_feat in poly_features]
                h = self.order_modules[layer_idx](h_polys)
            else:
                if self.sparse_mode and isinstance(adjacency, SparseTensor):
                    h = matmul(adjacency, h)
                else:
                    h = adjacency @ h
            
            # Apply layer
            h = layer(h)
            
            # Normalization
            h = norm(h)
            
            # Activation and dropout (except last layer)
            if layer_idx < len(self.layers) - 1:
                h = self.activation(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                
            # Skip connection
            if self.use_skip_connections:
                h = h + skip_proj(h_in)
                
            # Track over-smoothing
            if self.training and layer_idx < len(self.layers) - 1:
                similarity = self._compute_node_similarity(h)
                self.layer_similarities.append(similarity)
            
            if return_embeddings:
                embeddings.append(h.detach())
        
        if return_embeddings:
            return h, embeddings
        return h

    def _get_sparse_adjacency(self, edge_index: Tensor, num_nodes: int, device: torch.device) -> SparseTensor:
        """Compute normalized adjacency using sparse operations."""
        if self.cache_adj and self._cached_sparse_adj is not None:
            return self._cached_sparse_adj
        
        edge_index = edge_index.to(device)
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Compute degree
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        
        # Normalize: D^(-1/2) A D^(-1/2)
        value = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        adj = SparseTensor(
            row=row, col=col, value=value,
            sparse_sizes=(num_nodes, num_nodes)
        )
        
        if self.cache_adj:
            self._cached_sparse_adj = adj
        return adj

    def _normalized_adjacency(self, edge_index: Tensor, num_nodes: int, device: torch.device) -> Tensor:
        """Dense normalized adjacency (for small graphs)."""
        if self.cache_adj and self._cached_adj is not None:
            return self._cached_adj
        
        formatted_edge_index = edge_index.to(device)
        if self.add_self_loops:
            formatted_edge_index, _ = add_self_loops(formatted_edge_index, num_nodes=num_nodes)
        
        adj_dense = to_dense_adj(formatted_edge_index, max_num_nodes=num_nodes).squeeze(0)
        degree_mat = adj_dense.sum(dim=1)
        degree_inv_sqrt = degree_mat.pow(-0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        
        normalized_adj = degree_inv_sqrt.view(-1, 1) * adj_dense * degree_inv_sqrt.view(1, -1)
        if self.cache_adj:
            self._cached_adj = normalized_adj
        return normalized_adj

    def _apply_chebyshev_polynomial(self, adjacency: Tensor) -> Tensor:
        """Apply Chebyshev polynomial to dense adjacency."""
        if self.order <= 0:
            return torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype)

        identity = torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype)
        if self.order == 1:
            return identity + adjacency

        t0 = identity
        t1 = adjacency
        result = t0 + t1
        
        for _ in range(2, self.order + 1):
            t2 = 2 * adjacency @ t1 - t0
            result += t2
            t0, t1 = t1, t2
        
        return result

    def _apply_chebyshev_polynomial_sparse(self, adjacency: SparseTensor) -> SparseTensor:
        """Apply Chebyshev polynomial to sparse adjacency."""
        if self.order <= 0:
            return SparseTensor.eye(adjacency.size(0), device=adjacency.device())

        identity = SparseTensor.eye(adjacency.size(0), device=adjacency.device())
        if self.order == 1:
            return identity + adjacency

        t0 = identity
        t1 = adjacency
        result = t0 + t1
        
        for _ in range(2, self.order + 1):
            A = scale_sparse(matmul(adjacency, t1), 2)
            B = scale_sparse(t0, -1)        # instead of t0.mul(-1)
            t2 = A.add(B)

            result = result + t2
            t0, t1 = t1, t2
        
        return result

    def _compute_polynomial_features(self, adjacency, x: Tensor) -> List[Tensor]:
        """Compute features for each polynomial order separately (for adaptive order)."""
        poly_features = []
        
        if self.sparse_mode and isinstance(adjacency, SparseTensor):
            identity = SparseTensor.eye(adjacency.size(0), device=adjacency.device())
            poly_features.append(identity)
            poly_features.append(adjacency)
            
            if self.order >= 2:
                t0 = identity
                t1 = adjacency
                for _ in range(2, self.order + 1):
                    A = scale_sparse(matmul(adjacency, t1), 2)
                    B = scale_sparse(t0, -1)        
                    t2 = A.add(B)
                    poly_features.append(t2)
                    t0, t1 = t1, t2
        else:
            identity = torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype)
            poly_features.append(identity)
            poly_features.append(adjacency)
            
            if self.order >= 2:
                t0 = identity
                t1 = adjacency
                for _ in range(2, self.order + 1):
                    t2 = 2 * adjacency @ t1 - t0
                    poly_features.append(t2)
                    t0, t1 = t1, t2
        
        return poly_features

    def _compute_node_similarity(self, h: Tensor) -> float:
        """Compute average cosine similarity between node features (over-smoothing metric)."""
        h_norm = F.normalize(h, p=2, dim=1)
        similarity_matrix = h_norm @ h_norm.t()
        # Average off-diagonal elements
        n = h.size(0)
        total_sim = similarity_matrix.sum().item() - n  # subtract diagonal
        avg_sim = total_sim / (n * (n - 1)) if n > 1 else 0.0
        return avg_sim

    def get_over_smoothing_metrics(self) -> dict:
        """Return over-smoothing metrics from last forward pass."""
        return {
            "layer_similarities": self.layer_similarities,
            "mean_similarity": sum(self.layer_similarities) / len(self.layer_similarities) if self.layer_similarities else 0.0
        }

    def get_effective_orders(self) -> List[float]:
        """Get effective Chebyshev orders per layer (for adaptive mode)."""
        if not self.adaptive_order:
            return [self.order] * len(self.layers)
        return [module.get_effective_order() for module in self.order_modules]

    def reset_parameters(self) -> None:
        """Reset all learnable parameters."""
        torch.manual_seed(42)  # set a fixed seed here
        self._cached_adj = None
        self._cached_sparse_adj = None
        for layer in self.layers:
            if hasattr(layer, "weight") and layer.weight is not None:
                if layer.weight.dim() >= 2:
                    nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        
        for skip in self.skip_projections:
            if hasattr(skip, "weight") and skip.weight is not None:
                if skip.weight.dim() >= 2:
                    nn.init.xavier_uniform_(skip.weight)
            if hasattr(skip, "bias") and skip.bias is not None:
                nn.init.zeros_(skip.bias)   


    def reset_cache(self) -> None:
        """Reset cached adjacency matrices."""
        self._cached_adj = None
        self._cached_sparse_adj = None

