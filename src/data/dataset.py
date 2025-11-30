from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Union

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import NELL, Planetoid, PPI
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected


Split = Literal["train", "val", "test", "full"]


@dataclass
class DataConfig:
    """Configuration for dataset loading."""

    name: str
    device: str = "cpu"
    batch_size: int = 2
    root_dir: Path = Path("data")

    def normalized_name(self) -> str:
        return self.name.lower()


class GraphDataset:
    """Dataset loader providing a unified interface for full-graph and batch workloads."""

    FULL_GRAPH_DATASETS = {"cora", "pubmed", "nell", "citeseer"}
    MULTI_GRAPH_DATASETS = {"ppi"}

    def __init__(self, name: str, device: str = "cpu", batch_size: int = 2, root_dir: Path | None = None):
        self.config = DataConfig(name=name, device=device, batch_size=batch_size, root_dir=root_dir or Path("data"))
        self.config.root_dir.mkdir(parents=True, exist_ok=True)

        self.name = self.config.normalized_name()
        self.device = self.config.device

        self.task_type: Literal["classification", "multilabel_classification"]
        self.num_classes: int
        self.num_features: int

        self.dataset: Union[Planetoid, NELL, PPI, None] = None
        self.data: Dict[str, torch.Tensor] | Data | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        self._load_dataset()

    def _load_dataset(self) -> None:
        if self.name in self.FULL_GRAPH_DATASETS:
            self._load_full_graph_dataset()
        elif self.name == "ppi":
            self._load_ppi_dataset()
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported dataset: {self.name}")

    def _load_full_graph_dataset(self) -> None:
        root = self.config.root_dir / self.name
        dataset = Planetoid(root=str(root), name=self.name) if self.name != "nell" else NELL(root=str(root))
        self.dataset = dataset
        data = dataset[0]
        data.edge_index = to_undirected(data.edge_index)
        self.data = {
            "x": data.x.to(self.device),
            "edge_index": data.edge_index.to(self.device),
            "y": data.y.to(self.device),
            "train_mask": data.train_mask,
            "val_mask": data.val_mask,
            "test_mask": data.test_mask,
        }
        self.task_type = "classification"
        self.num_classes = dataset.num_classes
        self.num_features = dataset.num_features

    def _load_ppi_dataset(self) -> None:
        root = self.config.root_dir / "ppi"
        dataset_kwargs = {"root": str(root)}
        self.train_loader = DataLoader(PPI(split="train", **dataset_kwargs), batch_size=self.config.batch_size)
        self.val_loader = DataLoader(PPI(split="val", **dataset_kwargs), batch_size=self.config.batch_size)
        self.test_loader = DataLoader(PPI(split="test", **dataset_kwargs), batch_size=self.config.batch_size)

        sample_batch = next(iter(self.train_loader))
        self.task_type = "multilabel_classification"
        self.num_classes = sample_batch.y.shape[1]
        self.num_features = sample_batch.x.shape[1]

    def get_data(self, split: Split) -> Union[Dict[str, torch.Tensor], DataLoader]:
        """Return the requested split (full graph dict or PPI loader)."""
        if self.name in self.FULL_GRAPH_DATASETS:
            if split != "full":
                raise ValueError("Mask-level access is handled inside the trainer for full-graph datasets.")
            assert isinstance(self.data, dict)
            return self.data
        loader = getattr(self, f"{split}_loader", None)
        if loader is None:
            raise ValueError(f"Split '{split}' is not available for dataset '{self.name}'.")
        return loader
