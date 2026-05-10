# Import necessary libraries
import argparse
import random
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kan.KANLayer import KANLayer
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import set_seed


def get_args():
    """
    Parse command line arguments for training parameters.
    
    Returns:
        args: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Double-KAN Training Parameters')
    
    # Training parameters
    parser.add_argument('--mode', type=str, default='train', help='Mode: train, predict_set, grid_search')
    parser.add_argument('--model_name', type=str, default='double_kan', help='Model name for saving')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    # Model architecture parameters
    parser.add_argument('--n_input', type=int, default=28*28, help='Input dimension')
    parser.add_argument('--n_hidden', type=int, default=22, help='Hidden layer size')
    parser.add_argument('--n_output', type=int, default=10, help='Output dimension')
    parser.add_argument('--spline_order', type=int, default=3, help='Spline order k')
    parser.add_argument('--grid_size', type=int, default=5, help='Grid size for splines')
    
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=8e-4, help='Weight decay')
    parser.add_argument('--gamma', type=float, default=0.8, help='Learning rate scheduler gamma')
    parser.add_argument('--drop_out', type=float, default=0.0, help='Dropout rate')
    
    # Dataset parameters
    parser.add_argument('--ds_name', type=str, default='mnist', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--model_path', type=str, default='output/model.pth', help='Path to save model')
    parser.add_argument('--n_examples', type=int, default=0, help='Number of examples to use (0 for all)')
    
    # Experimental parameters
    parser.add_argument('--note', type=str, default='full', help='Experiment notes')
    parser.add_argument('--n_part', type=float, default=0.0, help='Fraction of data to use')
    parser.add_argument('--func_list', type=str, default='sin,cos', help='Function list for FC-KAN')
    parser.add_argument('--combined_type', type=str, default='quadratic', help='Combined type')
    parser.add_argument('--basis_function', type=str, default='sin', help='Basis function for SKAN')
    
    # Experiment setup
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4,10,20,30,40,50', help='Comma-separated seeds for multiple runs')
    parser.add_argument('--print_every', type=int, default=10, help='Print progress every N epochs')
    
    return parser.parse_args()


class DoubleKAN(nn.Module):
    """
    DoubleKAN implementation.
    
    Args:
        layers_hidden (list): List of hidden layer sizes.
    """
    
    def __init__(self, layers_hidden):
        super(DoubleKAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(KANLayer(in_features, out_features, k=3))

    def forward(self, x):
        """
        Forward pass through the KAN network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x


def run_once(seed, args):
    """
    Run a single training experiment with specified seed and arguments.
    
    Args:
        seed (int): Random seed for reproducibility.
        args (argparse.Namespace): Command line arguments.
        
    Returns:
        tuple: (train_accuracy, test_accuracy, recall, f1_score)
    """
    set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL.Image to Tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Dataset Loading
    train_dataset = datasets.MNIST(
        root="dataset", train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        root="dataset", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model Initialization
    model = DoubleKAN([args.n_input, args.n_hidden, args.n_output]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training Loop
    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()
        y_true_train, y_pred_train = [], []
        
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            x_train = torch.flatten(x_train, 1)

            optimizer.zero_grad()
            output = model(x_train)
            loss = F.cross_entropy(output, y_train)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_train.size(0)

            # Calculate training accuracy
            pred = output.argmax(dim=1)
            y_true_train.extend(y_train.cpu().numpy())
            y_pred_train.extend(pred.cpu().numpy())

        epoch_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(y_true_train, y_pred_train)

        # Print progress every N epochs
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Final Training Accuracy
    final_train_acc = train_acc

    # Testing Phase
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            x_val = torch.flatten(x_val, 1)

            output = model(x_val)
            pred = output.argmax(dim=1)

            y_true.extend(y_val.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # Calculate evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return final_train_acc, acc, f1


def print_stats(name, values):
    """
    Print statistical summary of experimental results.
    
    Args:
        name (str): Name of the metric.
        values (list): List of metric values.
    """
    mean = np.mean(values)
    std = np.std(values)
    print(f"{name}: {mean:.4f} ± {std:.4f}")


def main():
    """
    Main function to run multiple experiments with different seeds using command line arguments.
    """
    # Parse command line arguments
    args = get_args()
    
    # Parse seeds from comma-separated string
    seeds = [int(seed.strip()) for seed in args.seeds.split(',')]
    
    print(f"Starting Double-KAN training with the following parameters:")
    print(f"  Model: {args.model_name}")
    print(f"  Architecture: [{args.n_input}, {args.n_hidden}, {args.n_output}]")
    print(f"  Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}, Weight Decay: {args.wd}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {args.device}")
    print("-" * 50)

    # Initialize result containers
    train_accs, val_accs, f1s = [], [], []
    times = []

    # Run experiments with different seeds
    for seed in seeds:
        print(f"\n===== Seed {seed} =====")
        start = time.time()

        train_acc, val_acc, recall, f1 = run_once(seed, args)

        end = time.time()

        # Store results
        val_accs.append(val_acc)
        train_accs.append(train_acc)
        f1s.append(f1)
        times.append(end - start)

    # Print final results
    print("\n===== Final Results =====")
    print_stats("Train Acc", train_accs)
    print_stats("Test Acc", val_accs)
    print_stats("F1", f1s)
    print(f"Average training time per seed: {np.mean(times):.2f}s")
    
    # Save results to CSV if needed
    if args.note:
        import pandas as pd
        results_df = pd.DataFrame({
            'seed': seeds,
            'train_acc': train_accs,
            'val_acc': val_accs,
            'f1': f1s,
            'time': times
        })
        results_df.to_csv(f'results_{args.model_name}_{args.note}.csv', index=False)
        print(f"Results saved to results_{args.model_name}_{args.note}.csv")


if __name__ == "__main__":
    main()