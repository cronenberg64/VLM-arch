"""
Batch benchmarking script to run multiple models and datasets
"""
import subprocess
import sys
import pandas as pd
from pathlib import Path

# Models to benchmark
MODELS = [
    'convnext_v2_tiny',
    'efficientnetv2_s',
    'resnet50',
    'vit_base',
    'deit_small',
    'swin_tiny',
    'mobilenetv3'
]

# Datasets to test
DATASETS = ['cifar10', 'cifar100']

# Subset sizes to test (None means full dataset)
SUBSETS = [None, 5000, 10000, 50000]

def run_benchmark(model, dataset='cifar10'):
    """Run benchmark for a specific model"""
    cmd = f"python scripts/benchmark.py model={model} dataset={dataset}"
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0

def run_training(model, dataset='cifar10', subset=None, epochs=100):
    """Run training for a specific configuration"""
    cmd = f"python scripts/train.py model={model} dataset={dataset} training.epochs={epochs}"
    if subset:
        cmd += f" dataset.subset={subset}"
    
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0

def collect_benchmark_results():
    """Collect all benchmark CSV files into a single table"""
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found")
        return
    
    dfs = []
    for csv_file in results_dir.glob("benchmark_*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('Params (M)')
        combined.to_csv("results/leaderboard_benchmark.csv", index=False)
        print("\nBenchmark Leaderboard:")
        print(combined.to_string(index=False))
    else:
        print("No benchmark results found")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['benchmark', 'train', 'collect'], default='benchmark')
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    
    if args.mode == 'benchmark':
        print(f"Benchmarking {len(args.models)} models...")
        for model in args.models:
            run_benchmark(model, args.dataset)
        collect_benchmark_results()
    
    elif args.mode == 'train':
        print(f"Training {len(args.models)} models...")
        for model in args.models:
            run_training(model, args.dataset, args.subset, args.epochs)
    
    elif args.mode == 'collect':
        collect_benchmark_results()
