"""Core GRM project package exposing primary entry points."""

from .data import GraphDataset
from .models import GCN, GAT, RGCN
from .training import Trainer, TrainerConfig

__all__ = ["GraphDataset", "GCN", "GAT", "RGCN", "Trainer", "TrainerConfig"]
