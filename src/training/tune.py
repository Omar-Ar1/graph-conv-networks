import optuna
import torch
import logging
from .trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)

def objective(trial: optuna.Trial, dataset_name: str, device: str, n_epochs: int = 100) -> float:
    # Define hyperparameter search space
    config = TrainerConfig(
        dataset_name=dataset_name,
        device=device,
        n_epochs=n_epochs,
        verbose=False,
        
        # Hyperparameters to tune
        lr=trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        dropout=trial.suggest_float("dropout", 0.1, 0.8),
        num_layers=trial.suggest_int("num_layers", 2, 5),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        tchebychev_order=trial.suggest_int("tchebychev_order", 1, 5),
        
        # Fixed settings for tuning
        early_stopping=True,
        patience=20,
    )
    
    try:
        trainer = Trainer(config)
        metrics = trainer.train()
        _, best_val_acc = metrics.get_best_val_acc()
        return best_val_acc
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return 0.0

def run_tuning(
    dataset_name: str = "Cora",
    n_trials: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, dataset_name, device), 
        n_trials=n_trials
    )
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
