from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import wandb

from ..data import GraphDataset
from ..models import GCN


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerConfig:
    dataset_name: str = "Cora"
    n_epochs: int = 200
    lr: float = 0.01
    optimizer: str = "adam"
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    tchebychev_order: int = 1
    device: str = "cpu"
    verbose: bool = True
    
    # Early stopping
    early_stopping: bool = False
    patience: int = 50
    min_delta: float = 1e-4
    
    # Learning rate scheduling
    lr_scheduler: Optional[str] = "plateau"  # "plateau", "cosine", None
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    
    # model features
    use_skip_connections: bool = True
    use_layer_norm: bool = False
    use_graph_norm: bool = False
    sparse_mode: bool = True
    adaptive_order: bool = False
    
    # Experiment tracking
    log_dir: Optional[str] = None
    save_best_model: bool = True
    checkpoint_dir: Optional[str] = Path("../checkpoints") 
    use_wandb: bool = False
    wandb_project: str = "gcn-project"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None

    # Data
    batch_size: int = 32
    root_dir: Optional[str] = Path("../data")  


@dataclass
class MetricTracker:
    history: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "train_loss": [], 
            "val_loss": [], 
            "train_acc": [], 
            "val_acc": [],
            "learning_rates": [],
            "over_smoothing": []
        }
    )

    def log(
        self, 
        train_loss: float, 
        val_loss: float, 
        train_acc: float, 
        val_acc: float,
        lr: Optional[float] = None,
        over_smoothing: Optional[float] = None
    ) -> None:
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)
        if lr is not None:
            self.history["learning_rates"].append(lr)
        if over_smoothing is not None:
            self.history["over_smoothing"].append(over_smoothing)

    def __getitem__(self, item: str) -> List[float]:
        return self.history[item]
    
    def get_best_val_acc(self) -> Tuple[int, float]:
        """Return (epoch, best_val_acc)."""
        if not self.history["val_acc"]:
            return -1, 0.0
        best_epoch = int(np.argmax(self.history["val_acc"]))
        best_acc = self.history["val_acc"][best_epoch]
        return best_epoch, best_acc


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 50, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric: float) -> bool:
        score = metric if self.mode == "max" else -metric
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


class Trainer:
    """
    Enhanced trainer with:
    - Early stopping
    - Learning rate scheduling
    - Better logging
    - Model checkpointing
    - Over-smoothing tracking
    """

    def __init__(self, config: Optional[TrainerConfig] = None, **overrides) -> None:
        base_config = config or TrainerConfig()
        if overrides:
            try:
                base_config = replace(base_config, **overrides)
            except TypeError as exc:
                raise ValueError(f"Unknown trainer configuration override: {exc}") from exc

        self.config = base_config
        self.data_handler = GraphDataset(name=self.config.dataset_name, device=self.config.device, batch_size=self.config.batch_size, root_dir=self.config.root_dir)
        self.metrics = MetricTracker()
        self._uses_full_graph = self.data_handler.task_type == "classification"
        self._is_multilabel = self.data_handler.task_type == "multilabel_classification"

        # Setup checkpoint directory
        if self.config.checkpoint_dir:
            self.checkpoint_path = Path(self.config.checkpoint_dir) / self.config.dataset_name
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_path = None

        self.model = self._build_model()
        self.criterion = self._build_loss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Early stopping
        self.early_stopping = None
        if self.config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                mode="max"
            )
        
        self.best_val_acc = 0.0
        self.best_model_state = None

        logger.info(f"Initialized trainer for {self.config.dataset_name}")
        logger.info(f"Model: {sum(p.numel() for p in self.model.parameters())} parameters")

        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_name,
                config=self.config.__dict__
            )
            wandb.watch(self.model, log="all")

    def _build_model(self) -> GCN:
        model = GCN(
            input_dim=self.data_handler.num_features,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.data_handler.num_classes,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            order=self.config.tchebychev_order,
            use_skip_connections=self.config.use_skip_connections,
            use_layer_norm=self.config.use_layer_norm,
            use_graph_norm=self.config.use_graph_norm,
            sparse_mode=self.config.sparse_mode,
            adaptive_order=self.config.adaptive_order,
            cache_adj=self._uses_full_graph,
        ).to(self.config.device)
        model.reset_parameters()
        return model

    def _build_loss(self) -> nn.Module:
        if self._is_multilabel:
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def _build_optimizer(self) -> optim.Optimizer:
        opt_name = self.config.optimizer.lower()
        params = self.model.parameters()
        
        if opt_name == "adam":
            return optim.Adam(
                params,
                lr=self.config.lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
        elif opt_name == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
        elif opt_name == "sgd":
            return optim.SGD(
                params, 
                lr=self.config.lr, 
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer '{self.config.optimizer}'.")

    def _build_scheduler(self) -> Optional[object]:
        if self.config.lr_scheduler is None:
            return None
        
        if self.config.lr_scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=self.config.lr_scheduler_factor,
                patience=self.config.lr_scheduler_patience,
            )
        elif self.config.lr_scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.n_epochs,
                eta_min=self.config.lr * 0.01
            )
        else:
            raise ValueError(f"Unsupported scheduler '{self.config.lr_scheduler}'.")

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        if self._uses_full_graph:
            return self._train_on_full_graph()
        train_loader: DataLoader = self.data_handler.train_loader
        return self._run_loader(train_loader, training=True)

    def _train_on_full_graph(self) -> Tuple[float, float]:
        data = self.data_handler.get_data("full")
        self.optimizer.zero_grad()
        logits = self.model(data["x"], data["edge_index"])
        loss = self.criterion(logits[data["train_mask"]], data["y"][data["train_mask"]])
        loss.backward()
        self.optimizer.step()
        acc = self._calculate_accuracy(logits[data["train_mask"]], data["y"][data["train_mask"]])
        return float(loss.item()), acc

    def _run_loader(self, loader: DataLoader, training: bool) -> Tuple[float, float]:
        total_loss = total_correct = 0.0
        total_samples = 0

        for batch in loader:
            batch = self._prepare_batch(batch)
            if training:
                self.optimizer.zero_grad()

            logits = self.model(batch.x, batch.edge_index)
            target = batch.y.float() if self._is_multilabel else batch.y
            loss = self.criterion(logits, target)

            if training:
                loss.backward()
                self.optimizer.step()

            batch_weight = target.shape[0]
            total_loss += loss.item() * batch_weight
            total_correct += self._calculate_accuracy(logits, batch.y) * batch_weight
            total_samples += batch_weight

        mean_loss = total_loss / max(total_samples, 1)
        mean_acc = total_correct / max(total_samples, 1)
        return float(mean_loss), float(mean_acc)

    def _prepare_batch(self, batch: Batch) -> Batch:
        batch.edge_index = to_undirected(batch.edge_index)
        return batch.to(self.config.device)

    def evaluate(self, split: str = "val") -> Tuple[float, float]:
        self.model.eval()
        if self._uses_full_graph:
            data = self.data_handler.get_data("full")
            mask = data[f"{split}_mask"]
            with torch.no_grad():
                logits = self.model(data["x"], data["edge_index"])
                loss = self.criterion(logits[mask], data["y"][mask])
                acc = self._calculate_accuracy(logits[mask], data["y"][mask])
            return float(loss.item()), acc

        loader: DataLoader = getattr(self.data_handler, f"{split}_loader")
        with torch.no_grad():
            return self._run_loader(loader, training=False)

    def _calculate_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        if self._is_multilabel:
            preds = (torch.sigmoid(logits) > 0.5).float()
            return float((preds == labels).float().mean().item())
        predicted_classes = logits.argmax(dim=1)
        return float((predicted_classes == labels).float().mean().item())

    def train(self) -> MetricTracker:
        progress = tqdm(range(self.config.n_epochs), disable=not self.config.verbose)
        
        for epoch in progress:
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate("val")
            
            # Get over-smoothing metrics
            over_smoothing_metric = None
            if hasattr(self.model, 'get_over_smoothing_metrics'):
                metrics = self.model.get_over_smoothing_metrics()
                over_smoothing_metric = metrics.get('mean_similarity', None)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.metrics.log(
                train_loss, val_loss, train_acc, val_acc,
                lr=current_lr,
                over_smoothing=over_smoothing_metric
            )

            if self.config.use_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "lr": current_lr,
                    "over_smoothing": over_smoothing_metric
                })
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if self.config.save_best_model:
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
            
            # Progress bar
            desc = f"Epoch {epoch + 1} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}"
            if over_smoothing_metric is not None:
                desc += f" | Smooth: {over_smoothing_metric:.3f}"
            progress.set_description(desc)
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_acc):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        # Restore best model
        if self.config.save_best_model and self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.config.device) for k, v in self.best_model_state.items()})
            logger.info(f"Restored best model with val_acc={self.best_val_acc:.4f}")
        
        return self.metrics

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Save full training checkpoint."""
        if path is None:
            if self.checkpoint_path is None:
                raise ValueError("No checkpoint path configured")
            path = self.checkpoint_path / "checkpoint.pt"
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics.history,
            "config": self.config,
            "best_val_acc": self.best_val_acc,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.metrics.history = checkpoint["metrics"]
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Checkpoint loaded from {path}")

    def get_model_info(self) -> dict:
        """Return detailed model information."""
        info = {
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "num_layers": self.config.num_layers,
            "hidden_dim": self.config.hidden_dim,
        }
        
        if self.config.adaptive_order and hasattr(self.model, 'get_effective_orders'):
            info["effective_orders"] = self.model.get_effective_orders()
        
        return info