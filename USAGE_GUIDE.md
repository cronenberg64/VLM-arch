# Complete Usage Guide - VLM Backbone Benchmarking

This guide provides detailed instructions on how to use the benchmarking framework, train models, and analyze results.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Running Benchmarks](#running-benchmarks)
3. [Training Models](#training-models)
4. [Analyzing Results](#analyzing-results)
5. [Understanding the Codebase](#understanding-the-codebase)
6. [Advanced Usage](#advanced-usage)

---

## Quick Start

### Installation
```bash
# Activate your virtual environment
# Windows:
.venv\Scripts\Activate.ps1

# Linux/Mac:
source .venv/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### Run Your First Benchmark
```bash
# Benchmark a single model
python scripts/benchmark.py model=convnext_v2_tiny

# Benchmark all models
python scripts/run_batch.py --mode benchmark
```

---

## Running Benchmarks

### Single Model Benchmark

To benchmark a specific model:
```bash
python scripts/benchmark.py model=MODEL_NAME dataset=DATASET_NAME
```

**Available Models:**
- `convnext_v2_tiny` - ConvNeXt V2 Tiny
- `efficientnetv2_s` - EfficientNet V2 Small
- `resnet50` - ResNet-50
- `vit_base` - Vision Transformer Base
- `deit_small` - DeiT Small
- `swin_tiny` - Swin Transformer Tiny
- `mobilenetv3` - MobileNet V3

**Available Datasets:**
- `cifar10` - CIFAR-10 (10 classes, 60k images)
- `cifar100` - CIFAR-100 (100 classes, 60k images)

**Example:**
```bash
python scripts/benchmark.py model=vit_base dataset=cifar10
```

### Batch Benchmarking

To benchmark all models at once:
```bash
python scripts/run_batch.py --mode benchmark
```

**Options:**
```bash
# Benchmark specific models only
python scripts/run_batch.py --mode benchmark --models convnext_v2_tiny resnet50 vit_base

# Benchmark on CIFAR-100
python scripts/run_batch.py --mode benchmark --dataset cifar100
```

### Understanding Benchmark Metrics

The benchmark measures:

| Metric | Description | Unit |
|--------|-------------|------|
| **Params (M)** | Number of trainable parameters | Millions |
| **FLOPs (G)** | Floating point operations per forward pass | Billions |
| **Latency (ms)** | Average inference time per image | Milliseconds |
| **Throughput (img/s)** | Images processed per second | Images/sec |
| **Peak Memory (MB)** | GPU memory usage during inference | Megabytes |

### Viewing Results

Results are saved to:
- **Individual**: `results/benchmark_MODEL_NAME.csv`
- **Leaderboard**: `results/leaderboard_benchmark.csv`

```bash
# View leaderboard
cat results/leaderboard_benchmark.csv

# Or collect and display results
python scripts/run_batch.py --mode collect
```

---

## Training Models

### Basic Training

Train a model on the full dataset:
```bash
python scripts/train.py model=MODEL_NAME dataset=DATASET_NAME
```

**Example:**
```bash
python scripts/train.py model=convnext_v2_tiny dataset=cifar10
```

### Training with Different Dataset Sizes

This is the **core focus** of the project - testing small-data regimes!

```bash
# Train with 5,000 samples (5k subset)
python scripts/train.py model=convnext_v2_tiny dataset=cifar10 dataset.subset=5000

# Train with 10,000 samples
python scripts/train.py model=convnext_v2_tiny dataset=cifar10 dataset.subset=10000

# Train with 50,000 samples
python scripts/train.py model=convnext_v2_tiny dataset=cifar10 dataset.subset=50000

# Train with full dataset (50,000 for CIFAR-10 train set)
python scripts/train.py model=convnext_v2_tiny dataset=cifar10
```

### Adjusting Training Hyperparameters

```bash
# Change learning rate
python scripts/train.py model=vit_base dataset=cifar10 training.lr=0.001

# Change batch size
python scripts/train.py model=vit_base dataset=cifar10 training.batch_size=64

# Change number of epochs
python scripts/train.py model=vit_base dataset=cifar10 training.epochs=200

# Combine multiple overrides
python scripts/train.py model=vit_base dataset=cifar10 \
    training.lr=0.001 \
    training.batch_size=64 \
    training.epochs=200 \
    dataset.subset=10000
```

### Disable WandB Logging

If you don't want to use Weights & Biases:
```bash
python scripts/train.py model=convnext_v2_tiny dataset=cifar10 debug=true
```

### Batch Training Experiments

Train multiple models with the same configuration:
```bash
# Train all models on 10k samples for 100 epochs
python scripts/run_batch.py --mode train --subset 10000 --epochs 100

# Train specific models
python scripts/run_batch.py --mode train \
    --models convnext_v2_tiny vit_base \
    --subset 5000 \
    --epochs 50
```

### Sample Efficiency Experiments

To understand how models perform with different amounts of data:

```bash
# Create a shell script for systematic testing
for subset in 5000 10000 50000
do
    for model in convnext_v2_tiny vit_base resnet50
    do
        python scripts/train.py model=$model dataset=cifar10 dataset.subset=$subset training.epochs=100
    done
done
```

---

## Analyzing Results

### Using Jupyter Notebook

1. **Start Jupyter Lab:**
```bash
jupyter lab
```

2. **Open the analysis notebook:**
   - Navigate to `notebooks/analysis.ipynb`

3. **Run the cells** to:
   - Load benchmark results
   - Generate visualizations
   - Compare model performance
   - Create plots for papers/presentations

### What the Notebook Does

The analysis notebook provides:

1. **Leaderboard Table**: View all benchmarked models side-by-side
2. **Params vs FLOPs Plot**: Understand model complexity trade-offs
3. **Latency Comparison**: Bar chart of inference speed
4. **Throughput Comparison**: Bar chart of processing speed
5. **Custom Analysis**: Add your own analysis cells

### Example Analysis Workflow

```python
# In Jupyter notebook
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('../results/leaderboard_benchmark.csv')

# Sort by efficiency (Throughput per Parameter)
df['Efficiency'] = df['Throughput (img/s)'] / df['Params (M)']
df_sorted = df.sort_values('Efficiency', ascending=False)
print(df_sorted[['Model', 'Params (M)', 'Throughput (img/s)', 'Efficiency']])
```

### WandB Dashboard (if enabled)

If you're using WandB for training:
1. Visit [https://wandb.ai](https://wandb.ai)
2. View your project: `vlm-arch-benchmark`
3. Compare runs, view loss curves, and track metrics

---

## Understanding the Codebase

### Project Structure Overview

```
VLM-arch/
├── src/                    # Core source code
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model definitions and factory
│   ├── engine/             # Training loop
│   └── benchmark/          # Profiling and benchmarking tools
├── scripts/                # Entry point scripts
├── configs/                # Hydra configuration files
├── notebooks/              # Jupyter notebooks for analysis
└── results/                # Output directory for results
```

### Core Python Files Explained

#### src/data/

**`datamodule.py`** - Data Pipeline Manager
- **Purpose**: Handles dataset loading, subsampling, and data loader creation
- **Key Features**:
  - Loads CIFAR-10/100 from torchvision
  - Implements stratified subsampling (maintains class balance)
  - Creates train/val/test data loaders
- **Main Class**: `DataModule`
  - `setup()`: Downloads and prepares datasets
  - `_get_stratified_subset_indices()`: Creates balanced subsets
  - `train_dataloader()`, `val_dataloader()`, `test_dataloader()`: Returns data loaders

**`transforms.py`** - Data Augmentation
- **Purpose**: Defines image transformations for training and validation
- **Features**:
  - StandardCIFAR augmentations (RandomCrop, HorizontalFlip)
  - Normalization with CIFAR mean/std
  - Extensible for advanced augmentations (RandAugment, etc.)

---

#### src/models/

**`factory.py`** - Model Factory
- **Purpose**: Unified interface to create any backbone model
- **Key Features**:
  - Uses `timm` library for pre-built architectures
  - Handles model configuration from Hydra configs
  - Supports custom modifications
- **Main Class**: `ModelFactory`
  - `create_model(cfg)`: Instantiates a model based on config
- **Function**: `get_model_info(model)`: Returns parameter count

**How it works:**
```python
# When you run: python scripts/train.py model=vit_base
# The factory reads configs/model/vit_base.yaml and creates:
model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=False,
    num_classes=10,
    drop_path_rate=0.1
)
```

---

#### src/engine/

**`trainer.py`** - Training Engine
- **Purpose**: Implements the complete training loop
- **Key Features**:
  - Mixed precision training (AMP)
  - Gradient accumulation for large batches on small GPUs
  - Gradient clipping
  - Learning rate scheduling (Cosine Annealing)
  - WandB logging integration
- **Main Class**: `Trainer`
  - `__init__()`: Sets up model, optimizer, scheduler
  - `train_epoch()`: Runs one training epoch
  - `validate()`: Evaluates on validation set
  - `fit()`: Main training loop

**Training flow:**
```
for epoch in range(epochs):
    1. train_epoch():
       - Forward pass → Calculate loss
       - Backward pass → Accumulate gradients
       - Optimizer step (with gradient clipping)
    2. validate():
       - Evaluate on validation set
    3. scheduler.step():
       - Update learning rate
    4. Log metrics to WandB
```

---

#### src/benchmark/

**`profiler.py`** - Performance Profiler
- **Purpose**: Measures model efficiency and resource usage
- **Key Features**:
  - Parameter counting
  - FLOPs calculation (using fvcore)
  - Latency measurement (with warmup)
  - Throughput measurement
  - Peak memory tracking (GPU)
- **Main Class**: `Profiler`
  - `count_params()`: Total trainable parameters
  - `count_flops()`: FLOPs for one forward pass
  - `measure_latency()`: Average inference time
  - `measure_throughput()`: Images processed per second
  - `measure_peak_memory()`: Peak GPU memory usage

---

#### scripts/

**`train.py`** - Training Entry Point
- **Purpose**: Main script to train models
- **How it works**:
  1. Uses Hydra to load configurations
  2. Creates DataModule and loads datasets
  3. Creates model using ModelFactory
  4. Initializes Trainer
  5. Runs training loop
  6. Logs to WandB (if enabled)

**Usage pattern:**
```bash
python scripts/train.py \
    model=MODEL_NAME \        # Which model config to use
    dataset=DATASET_NAME \    # Which dataset config to use
    training.PARAM=VALUE      # Override training parameters
```

---

**`benchmark.py`** - Benchmarking Entry Point
- **Purpose**: Profile a single model's performance
- **How it works**:
  1. Loads model configuration
  2. Creates model using ModelFactory
  3. Creates Profiler instance
  4. Measures all metrics (Params, FLOPs, Latency, etc.)
  5. Saves results to CSV
  6. Prints summary

**Output:**
- Individual CSV: `results/benchmark_MODEL_NAME.csv`
- Console output with all metrics

---

**`run_batch.py`** - Batch Processing Utility
- **Purpose**: Automate benchmarking/training across multiple models
- **Modes**:
  - `--mode benchmark`: Benchmark multiple models
  - `--mode train`: Train multiple models
  - `--mode collect`: Collect results into leaderboard
- **How it works**:
  - Iterates through specified models
  - Calls `benchmark.py` or `train.py` via subprocess
  - Aggregates results into leaderboard

**Usage examples:**
```bash
# Benchmark all models
python scripts/run_batch.py --mode benchmark

# Train specific models with 5k samples
python scripts/run_batch.py --mode train \
    --models convnext_v2_tiny vit_base \
    --subset 5000 \
    --epochs 100

# Collect results into leaderboard
python scripts/run_batch.py --mode collect
```

---

#### configs/

**Hydra Configuration System**

The project uses Hydra for flexible configuration management. All settings are defined in YAML files.

**`config.yaml`** - Main Configuration
- Imports all sub-configurations
- Sets global parameters (seed, debug mode)
- Defines output directory structure

**`model/*.yaml`** - Model Configurations
Each file defines a specific architecture:
```yaml
name: vit_base_patch16_224  # Model name in timm
pretrained: false            # Use pretrained weights?
num_classes: ${dataset.num_classes}  # Auto-set from dataset
drop_path_rate: 0.1         # Regularization
```

**`dataset/*.yaml`** - Dataset Configurations
```yaml
name: cifar10               # Dataset identifier
root: data/cifar10          # Storage location
num_classes: 10             # Number of classes
image_size: 32              # Input resolution
subset: null                # Subset size (null = full dataset)
download: true              # Auto-download if missing
```

**`training/default.yaml`** - Training Hyperparameters
```yaml
epochs: 100                 # Training epochs
batch_size: 128             # Batch size
lr: 1e-3                    # Learning rate
weight_decay: 0.05          # L2 regularization
optimizer: adamw            # Optimizer type
scheduler: cosine           # LR schedule
warmup_epochs: 5            # Warmup period
num_workers: 4              # Data loading threads
amp: true                   # Mixed precision training
accumulate_grad_batches: 1  # Gradient accumulation steps
grad_clip: 1.0              # Gradient clipping threshold
```

---

## Advanced Usage

### Custom Model Addition

1. **Create config file**: `configs/model/my_model.yaml`
```yaml
name: my_timm_model_name
pretrained: false
num_classes: ${dataset.num_classes}
```

2. **Use it**:
```bash
python scripts/benchmark.py model=my_model
```

### Custom Dataset Addition

1. **Modify `src/data/datamodule.py`** to add new dataset
2. **Create config**: `configs/dataset/my_dataset.yaml`
3. **Use it**:
```bash
python scripts/train.py model=resnet50 dataset=my_dataset
```

### Experiment Tracking Best Practices

1. **Use descriptive names** when training:
```bash
# WandB will create a run named: "convnextv2_tiny_cifar10_5000"
python scripts/train.py model=convnext_v2_tiny dataset=cifar10 dataset.subset=5000
```

2. **Keep a spreadsheet** of experiments:
   - Model name
   - Dataset size
   - Final accuracy
   - Training time
   - Notes

3. **Save important runs**:
```bash
# Copy outputs directory
cp -r outputs/DATE/TIME my_important_experiment/
```

### Performance Optimization Tips

1. **Batch Size**: Increase as much as GPU memory allows
2. **Num Workers**: Set to number of CPU cores (typically 4-8)
3. **Mixed Precision (AMP)**: Already enabled by default
4. **Gradient Accumulation**: Simulate larger batches:
```bash
python scripts/train.py model=vit_base \
    training.batch_size=32 \
    training.accumulate_grad_batches=4  # Effective batch size: 128
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size or use gradient accumulation
```bash
python scripts/train.py model=vit_base \
    training.batch_size=32 \
    training.accumulate_grad_batches=4
```

### Issue: Slow Training

**Solutions**:
1. Increase `num_workers`: `training.num_workers=8`
2. Use smaller model for testing
3. Reduce dataset size: `dataset.subset=1000`

### Issue: WandB Login Required

**Solution**: Either:
1. Login: `wandb login`
2. Disable: `debug=true`

---

## Next Steps

1. **Run baseline benchmarks** on all models
2. **Train models** with different dataset sizes (5k, 10k, 50k, full)
3. **Analyze results** in Jupyter notebook
4. **Generate visualizations** for your research
5. **Write your paper!**

---

## Questions?

Check the main README.md or review the code comments in each file. Each function is documented with its purpose and usage.
