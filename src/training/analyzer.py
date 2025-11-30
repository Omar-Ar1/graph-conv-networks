"""
Analysis tools for understanding GCN behavior:
- Over-smoothing analysis
- Spectral analysis
- Receptive field visualization
- Feature evolution tracking
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import to_dense_adj, get_laplacian
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class OverSmoothingAnalyzer:
    """Analyze over-smoothing in GCN layers."""
    
    @staticmethod
    def compute_mad(features: Tensor) -> float:
        """
        Compute Mean Average Distance (MAD) between node features.
        Lower MAD indicates more over-smoothing.
        """
        n = features.size(0)
        if n <= 1:
            return 0.0
        
        # Pairwise distances
        dists = torch.cdist(features, features, p=2)
        # Average (excluding diagonal)
        mad = (dists.sum() - dists.trace()).item() / (n * (n - 1))
        return mad
    
    @staticmethod
    def compute_dirichlet_energy(features: Tensor, edge_index: Tensor) -> float:
        """
        Compute Dirichlet energy: measures smoothness of features over graph.
        Lower energy = smoother features = more over-smoothing.
        """
        row, col = edge_index
        energy = ((features[row] - features[col]) ** 2).sum().item()
        return energy / edge_index.size(1)
    
    @staticmethod
    def compute_node_similarity_variance(features: Tensor) -> float:
        """
        Compute variance of pairwise cosine similarities.
        Lower variance = nodes becoming more similar = over-smoothing.
        """
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = features_norm @ features_norm.t()
        
        # Get upper triangle (excluding diagonal)
        n = features.size(0)
        mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        similarities = similarity_matrix[mask]
        
        return similarities.var().item()
    
    @staticmethod
    def compute_inter_intra_mad(features: Tensor, labels: Tensor) -> Dict[str, float]:
        """
        Compute Inter-class and Intra-class Mean Average Distance (MAD).
        
        Args:
            features: Node features [num_nodes, hidden_dim]
            labels: Node labels [num_nodes]
            
        Returns:
            Dictionary with 'intra_mad', 'inter_mad', and 'mad_ratio'
        """
        n = features.size(0)
        if n <= 1:
            return {"intra_mad": 0.0, "inter_mad": 0.0, "mad_ratio": 0.0}
            
        # Pairwise distances
        dists = torch.cdist(features, features, p=2)
        
        # Masks for same and different classes
        labels = labels.view(-1, 1)
        same_class_mask = torch.eq(labels, labels.T)
        diff_class_mask = ~same_class_mask
        
        # Remove diagonal from same_class_mask
        same_class_mask.fill_diagonal_(False)
        
        # Compute Intra-class MAD
        num_intra = same_class_mask.sum().item()
        intra_mad = dists[same_class_mask].sum().item() / num_intra if num_intra > 0 else 0.0
        
        # Compute Inter-class MAD
        num_inter = diff_class_mask.sum().item()
        inter_mad = dists[diff_class_mask].sum().item() / num_inter if num_inter > 0 else 0.0
        
        ratio = inter_mad / intra_mad if intra_mad > 1e-6 else 0.0
        
        return {
            "intra_mad": intra_mad,
            "inter_mad": inter_mad,
            "mad_ratio": ratio
        }
    
    @staticmethod
    def analyze_layer_embeddings(
        embeddings: List[Tensor],
        edge_index: Tensor,
        labels: Optional[Tensor] = None,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        Comprehensive analysis of embeddings across layers.
        
        Args:
            embeddings: List of feature tensors from each layer
            edge_index: Graph edge index
            layer_names: Optional names for layers
            
        Returns:
            Dictionary with metrics per layer
        """
        if layer_names is None:
            layer_names = [f"Layer {i}" for i in range(len(embeddings))]
        
        metrics = {
            "mad": [],
            "dirichlet_energy": [],
            "similarity_variance": [],
            "feature_norm": [],
            "intra_mad": [],
            "inter_mad": [],
            "mad_ratio": [],
            "layer_names": layer_names
        }
        
        for emb in embeddings:
            metrics["mad"].append(
                OverSmoothingAnalyzer.compute_mad(emb)
            )
            metrics["dirichlet_energy"].append(
                OverSmoothingAnalyzer.compute_dirichlet_energy(emb, edge_index)
            )
            metrics["similarity_variance"].append(
                OverSmoothingAnalyzer.compute_node_similarity_variance(emb)
            )
            metrics["feature_norm"].append(
                emb.norm(dim=1).mean().item()
            )
            
            if labels is not None:
                class_mad = OverSmoothingAnalyzer.compute_inter_intra_mad(emb, labels)
                metrics["intra_mad"].append(class_mad["intra_mad"])
                metrics["inter_mad"].append(class_mad["inter_mad"])
                metrics["mad_ratio"].append(class_mad["mad_ratio"])
        
        return metrics
    
    @staticmethod
    def plot_over_smoothing_metrics(metrics: Dict[str, List[float]], suptitle: Optional[str] = None) -> Figure:
        """Create comprehensive visualization of over-smoothing metrics."""
        """Create comprehensive visualization of over-smoothing metrics."""
        plt.style.use("seaborn-v0_8-whitegrid")
        
        # Determine layout based on available metrics
        has_class_metrics = len(metrics.get("intra_mad", [])) > 0
        
        if has_class_metrics:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=140)
            configs = [
                ("mad", "Mean Average Distance (MAD)", "MAD"),
                ("dirichlet_energy", "Dirichlet Energy", "Energy"),
                ("similarity_variance", "Similarity Variance", "Variance"),
                ("feature_norm", "Average Feature Norm", "Norm"),
                ("intra_mad", "Intra-class MAD", "Intra MAD"),
                ("inter_mad", "Inter-class MAD", "Inter MAD"),
                # ("mad_ratio", "Inter/Intra Ratio", "Ratio"), # Optional 7th plot or replace one
            ]
            # Adjust configs to fit 6 subplots or use 3x3
            # Let's stick to 6 key metrics for 2x3
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
            configs = [
                ("mad", "Mean Average Distance (MAD)", "MAD"),
                ("dirichlet_energy", "Dirichlet Energy", "Energy"),
                ("similarity_variance", "Similarity Variance", "Variance"),
                ("feature_norm", "Average Feature Norm", "Norm"),
            ]
            
        layer_indices = list(range(len(metrics["layer_names"])))
        x_labels = metrics["layer_names"]
        palettes = plt.get_cmap("tab10").colors
        
        for i, (ax, (metric_key, title, y_label)) in enumerate(zip(axes.ravel(), configs)):
            if metric_key in metrics and metrics[metric_key]:
                values = metrics[metric_key]
                color = palettes[i % len(palettes)]
                ax.plot(layer_indices, values, marker="o", linewidth=2.5, markersize=6, color=color)
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.set_ylabel(y_label)
                ax.set_xticks(layer_indices)
                ax.set_xticklabels(x_labels, rotation=25, ha="right")
                ax.grid(True, linestyle="--", alpha=0.4)
        
        fig.suptitle("Over-Smoothing Analysis Across Layers" if not suptitle else suptitle, fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig


class SpectralAnalyzer:
    """Analyze spectral properties of graphs and GCN filters."""
    
    @staticmethod
    def compute_graph_laplacian_spectrum(
        edge_index: Tensor,
        num_nodes: int,
        normalized: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors of graph Laplacian.
        
        Returns:
            eigenvalues: Array of eigenvalues (sorted)
            eigenvectors: Matrix of eigenvectors
        """
        # Get Laplacian
        edge_index_lap, edge_weight = get_laplacian(
            edge_index, 
            normalization='sym' if normalized else None,
            num_nodes=num_nodes
        )
        
        # Convert to dense
        L = to_dense_adj(
            edge_index_lap, 
            edge_attr=edge_weight,
            max_num_nodes=num_nodes
        )[0]
        
        # Compute spectrum
        eigenvalues, eigenvectors = eigh(L.cpu().numpy())
        
        return eigenvalues, eigenvectors
    
    @staticmethod
    def compute_chebyshev_response(
        eigenvalues: np.ndarray,
        order: int
    ) -> np.ndarray:
        """
        Compute Chebyshev polynomial response across spectrum.
        
        Args:
            eigenvalues: Graph Laplacian eigenvalues (scaled to [-1, 1])
            order: Chebyshev polynomial order
            
        Returns:
            Response values for each eigenvalue
        """
        # Scale eigenvalues to [-1, 1] for Chebyshev polynomials
        lambda_max = eigenvalues.max()
        scaled_eigs = 2 * eigenvalues / lambda_max - 1
        
        # Compute Chebyshev polynomials
        T = np.zeros((len(eigenvalues), order + 1))
        T[:, 0] = 1  # T_0(x) = 1
        if order >= 1:
            T[:, 1] = scaled_eigs  # T_1(x) = x
        
        for k in range(2, order + 1):
            T[:, k] = 2 * scaled_eigs * T[:, k-1] - T[:, k-2]
        
        # Sum all orders (as in the model)
        response = T.sum(axis=1)
        return response
    
    @staticmethod
    def plot_spectral_analysis(
        edge_index: Tensor,
        num_nodes: int,
        orders: List[int] = [1, 2, 3, 6, 10]
    ) -> Figure:
        """
        Visualize spectral properties and Chebyshev filter responses.
        """
        eigenvalues, _ = SpectralAnalyzer.compute_graph_laplacian_spectrum(
            edge_index, num_nodes
        )
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=140)
        
        idx = np.arange(len(eigenvalues))
        scatter = axes[0].scatter(
            idx,
            eigenvalues,
            c=eigenvalues,
            cmap="viridis",
            s=35,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.3
        )
        axes[0].set_title("Graph Laplacian Spectrum", fontweight="bold")
        axes[0].set_xlabel("Eigenvalue Index")
        axes[0].set_ylabel("Eigenvalue")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        
        cmap = plt.get_cmap("tab10")
        for idx_order, order in enumerate(orders):
            response = SpectralAnalyzer.compute_chebyshev_response(eigenvalues, order)
            axes[1].plot(
                eigenvalues,
                response,
                label=f"Order {order}",
                linewidth=2.2,
                color=cmap(idx_order % len(cmap.colors))
            )
        
        axes[1].set_title("Chebyshev Filter Frequency Response", fontweight="bold")
        axes[1].set_xlabel("Eigenvalue λ")
        axes[1].set_ylabel("Filter Response")
        axes[1].legend(frameon=False)
        axes[1].grid(True, linestyle="--", alpha=0.4)
        
        fig.suptitle("Spectral Analysis of Graph and GCN Filters", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig
    
    @staticmethod
    def analyze_spectral_gap(eigenvalues: np.ndarray) -> Dict[str, float]:
        """
        Analyze spectral gap and connectivity.
        
        Returns:
            Dictionary with spectral properties
        """
        # Sort eigenvalues
        eigs_sorted = np.sort(eigenvalues)
        
        # Spectral gap (difference between first two non-zero eigenvalues)
        non_zero_eigs = eigs_sorted[eigs_sorted > 1e-6]
        spectral_gap = non_zero_eigs[1] - non_zero_eigs[0] if len(non_zero_eigs) > 1 else 0
        
        return {
            "min_eigenvalue": float(eigs_sorted[0]),
            "max_eigenvalue": float(eigs_sorted[-1]),
            "spectral_gap": float(spectral_gap),
            "num_zero_eigenvalues": int((eigs_sorted < 1e-6).sum()),
            "algebraic_connectivity": float(non_zero_eigs[0]) if len(non_zero_eigs) > 0 else 0
        }


class ReceptiveFieldAnalyzer:
    """Analyze receptive fields of GCN layers."""
    
    @staticmethod
    def compute_k_hop_neighbors(
        edge_index: Tensor,
        num_nodes: int,
        k: int
    ) -> Dict[int, set]:
        """
        Compute k-hop neighbors for each node.
        
        Returns:
            Dictionary mapping node_id -> set of neighbors within k hops
        """
        # Build adjacency list
        adj = {i: set() for i in range(num_nodes)}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj[src].add(dst)
            adj[dst].add(src)
        
        # BFS to find k-hop neighbors
        k_hop = {}
        for node in range(num_nodes):
            visited = {node}
            current_level = {node}
            
            for _ in range(k):
                next_level = set()
                for n in current_level:
                    next_level.update(adj[n] - visited)
                visited.update(next_level)
                current_level = next_level
            
            k_hop[node] = visited
        
        return k_hop
    
    @staticmethod
    def analyze_receptive_fields(
        edge_index: Tensor,
        num_nodes: int,
        max_k: int = 10
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze receptive field sizes for different k values.
        
        Returns:
            Dictionary mapping k -> {mean, std, min, max} receptive field size
        """
        results = {}
        
        for k in range(1, max_k + 1):
            k_hop = ReceptiveFieldAnalyzer.compute_k_hop_neighbors(
                edge_index, num_nodes, k
            )
            
            sizes = [len(neighbors) for neighbors in k_hop.values()]
            
            results[k] = {
                "mean": np.mean(sizes),
                "std": np.std(sizes),
                "min": np.min(sizes),
                "max": np.max(sizes),
                "median": np.median(sizes)
            }
        
        return results
    
    @staticmethod
    def plot_receptive_field_growth(
        receptive_field_stats: Dict[int, Dict[str, float]],
        title: Optional[str] = None,
    ) -> Figure:
        """Plot receptive field size vs. number of hops."""
        k_values = sorted(receptive_field_stats.keys())
        means = [receptive_field_stats[k]["mean"] for k in k_values]
        stds = [receptive_field_stats[k]["std"] for k in k_values]
        
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
        ax.errorbar(
            k_values,
            means,
            yerr=stds,
            fmt="-o",
            color="#1f77b4",
            linewidth=2.5,
            markersize=6,
            ecolor="0.4",
            capsize=4
        )
        ax.fill_between(
            k_values,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color="#1f77b4",
            alpha=0.12
        )
        ax.set_title("Receptive Field Growth with Number of Hops" if not title else title, fontweight="bold")
        ax.set_xlabel("Number of Hops (k)")
        ax.set_ylabel("Receptive Field Size (# nodes)")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        return fig


class HomophilyAnalyzer:
    """Analyze homophily properties of the graph."""

    @staticmethod
    def compute_node_homophily(edge_index: Tensor, labels: Tensor) -> float:
        """
        Compute node homophily: fraction of neighbors with the same label.
        """
        src, dst = edge_index
        # Check if source and destination have the same label
        same_label = labels[src] == labels[dst]
        # Count matches
        num_matches = same_label.sum().item()
        num_edges = edge_index.size(1)
        
        return num_matches / num_edges if num_edges > 0 else 0.0

    @staticmethod
    def compute_edge_homophily(edge_index: Tensor, labels: Tensor) -> float:
        """
        Compute edge homophily: fraction of edges connecting nodes of the same class.
        (Same as node homophily for undirected graphs if defined this way, 
         but often distinguished in literature. Here we use the standard definition).
        """
        return HomophilyAnalyzer.compute_node_homophily(edge_index, labels)

    @staticmethod
    def compute_class_homophily(edge_index: Tensor, labels: Tensor, num_classes: int) -> Dict[int, float]:
        """
        Compute homophily per class.
        """
        src, dst = edge_index
        class_homophily = {}
        
        for c in range(num_classes):
            # Edges where source node is class c
            mask = labels[src] == c
            if mask.sum() == 0:
                class_homophily[c] = 0.0
                continue
                
            # Among these edges, how many connect to class c?
            same_class = labels[dst][mask] == c
            class_homophily[c] = same_class.sum().item() / mask.sum().item()
            
        return class_homophily

    @staticmethod
    def compute_adjusted_homophily(edge_index: Tensor, labels: Tensor) -> float:
        """
        Compute adjusted homophily (accounting for class imbalance).
        """
        homophily = HomophilyAnalyzer.compute_node_homophily(edge_index, labels)
        
        # Compute expected homophily (if edges were random)
        num_nodes = labels.size(0)
        class_counts = torch.bincount(labels)
        class_probs = class_counts.float() / num_nodes
        expected_homophily = (class_probs ** 2).sum().item()
        
        # Adjusted homophily
        return (homophily - expected_homophily) / (1 - expected_homophily)
