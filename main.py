import argparse
import torch
import sys
import os

# Add the current directory to path so we can import the package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training import Trainer, TrainerConfig, run_tuning

def train(args):
    config = TrainerConfig(
        dataset_name=args.dataset,
        n_epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        tchebychev_order=args.order,
        device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        early_stopping=True,
    )
    
    trainer = Trainer(config)
    metrics = trainer.train()
    
    best_epoch, best_acc = metrics.get_best_val_acc()
    print(f"Training completed. Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")

def tune(args):
    run_tuning(
        dataset_name=args.dataset,
        n_trials=args.trials,
        device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

def main():
    parser = argparse.ArgumentParser(description="GCN Project CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train GCN model")
    train_parser.add_argument("--dataset", type=str, default="Cora", help="Dataset name")
    train_parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    train_parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    train_parser.add_argument("--layers", type=int, default=2, help="Number of layers")
    train_parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    train_parser.add_argument("--order", type=int, default=2, help="Chebyshev order")
    train_parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    train_parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    train_parser.add_argument("--wandb_project", type=str, default="gcn-project", help="WandB project name")
    train_parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name")
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Tune hyperparameters")
    tune_parser.add_argument("--dataset", type=str, default="Cora", help="Dataset name")
    tune_parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    tune_parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "tune":
        tune(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
