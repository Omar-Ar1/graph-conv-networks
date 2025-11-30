from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch.nn.functional import softmax

from ..training import Trainer


def visualize_embeddings(trainer: Trainer, dataset_name: str) -> None:
    data = trainer.data_handler.get_data("full")
    test_mask = data["test_mask"]
    embeddings = trainer.model(data["x"], data["edge_index"]).detach().cpu().numpy()
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings[test_mask])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=data["y"][test_mask].cpu(),
        cmap="jet",
        s=30,
        alpha=0.8,
        edgecolor="k",
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Node Classes", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Embedding Dimension 1", fontsize=12)
    plt.ylabel("Embedding Dimension 2", fontsize=12)
    plt.title(f"Node Embedding Visualization for {dataset_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_confidence_distribution(trainer: Trainer, dataset_name: str) -> None:
    data = trainer.data_handler.get_data("full")
    test_mask = data["test_mask"]
    trainer.model.eval()
    with torch.no_grad():
        out = trainer.model(data["x"], data["edge_index"])
        confidences = softmax(out[test_mask], dim=1).max(dim=1).values.cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, color="skyblue", edgecolor="black", alpha=0.8)
    plt.xlabel("Prediction Confidence", fontsize=12)
    plt.ylabel("Number of Nodes", fontsize=12)
    plt.title(f"Prediction Confidence Distribution for {dataset_name}", fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.axvline(x=0.5, color="red", linestyle="--", label="Threshold (0.5)")
    plt.legend(fontsize=10)
    mean_confidence = confidences.mean()
    std_confidence = confidences.std()
    plt.text(
        0.7,
        plt.ylim()[1] * 0.8,
        f"Mean: {mean_confidence:.2f}\nStd Dev: {std_confidence:.2f}",
        fontsize=10,
        color="black",
        bbox=dict(facecolor="white", alpha=0.6),
    )
    plt.tight_layout()
    plt.show()


def node_degree_distribution(trainer: Trainer, dataset_name: str) -> None:
    data = trainer.data_handler.get_data("full")
    node_degrees = torch.bincount(data["edge_index"][0])
    node_degrees = node_degrees.cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(node_degrees, bins=20, color="skyblue", edgecolor="black", alpha=0.8, log=True)
    plt.xlabel("Node Degree", fontsize=12)
    plt.ylabel("Frequency (Log Scale)", fontsize=12)
    plt.title(f"Log-Scaled Degree Distribution for {dataset_name}", fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def test_acc_chebychev(test_accuracies, chebyshev_orders, dataset_name: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(
        chebyshev_orders,
        test_accuracies,
        marker="o",
        linestyle="-",
        color="royalblue",
        linewidth=2,
        markersize=8,
        label="Test Accuracy",
    )
    plt.title(f"Test Accuracy vs. Chebyshev Order for {dataset_name}", fontsize=16, fontweight="bold")
    plt.xlabel("Chebyshev Polynomial Order", fontsize=14)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
