# Small-Data Visual Backbone Benchmarking & Architecture Exploration for Real-World Robotics Deployment

## Objective
Identify which visual backbone architecture performs best when training a small-data Visual-Language Model (VLM), with a specific focus on robotics and edge deployment scenarios. The study compares CNN, Transformer, and hybrid backbone families under identical training conditions and small dataset constraints.

## Why This Matters
Most academic results assume access to very large datasets, expensive compute, and large batch sizes. However, real robotics teams train models under constraints:
- **<300k training samples**
- **1–4 GPUs** (or edge devices)
- **Tight inference latency, memory, and power limits**

This project fills that gap and produces results directly relevant to real industrial robotics.

## Architectures to Compare
- **CNN**: ConvNeXt-V2 Tiny, EfficientNetV2-S, MobileNetV3, ResNet50
- **Transformer**: ViT-B/16, DeiT-Small, Swin-T
- **Hybrid**: CoAtNet, MetaFormer (to be added)

## Datasets
- CIFAR-10 / CIFAR-100
- Subsets: 5k, 10k, 50k, 100k samples

## Installation

```bash
# Clone the repository
git clone https://github.com/cronenberg64/VLM-arch.git
cd VLM-arch

# Create virtual environment (optional but recommended)
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/Mac: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Benchmark Models
Run performance benchmarks (FLOPs, Params, Latency, Throughput):

```bash
# Single model
python scripts/benchmark.py model=convnext_v2_tiny

# Batch benchmark all models
python scripts/run_batch.py --mode benchmark
```

Results are saved to `results/benchmark_*.csv` and aggregated in `results/leaderboard_benchmark.csv`.

### 2. Train Models
Train a model on CIFAR-10:

```bash
# Full dataset
python scripts/train.py model=convnext_v2_tiny dataset=cifar10

# With subset (5k samples)
python scripts/train.py model=convnext_v2_tiny dataset=cifar10 dataset.subset=5000

# Adjust hyperparameters
python scripts/train.py model=vit_base dataset=cifar10 training.epochs=200 training.batch_size=64
```

### 3. Batch Experiments
Run training across multiple models:

```bash
# Train all models on CIFAR-10 with 10k samples
python scripts/run_batch.py --mode train --subset 10000 --epochs 100
```

### 4. Analyze Results
Open the analysis notebook:

```bash
jupyter lab notebooks/analysis.ipynb
```

## Project Structure

```
VLM-arch/
├── configs/              # Hydra configuration files
│   ├── config.yaml       # Main config
│   ├── model/            # Model configs
│   ├── dataset/          # Dataset configs
│   └── training/         # Training configs
├── src/
│   ├── data/             # Data loading and augmentation
│   ├── models/           # Model factory
│   ├── engine/           # Training engine
│   └── benchmark/        # Profiling utilities
├── scripts/
│   ├── train.py          # Training script
│   ├── benchmark.py      # Benchmarking script
│   └── run_batch.py      # Batch processing utility
├── notebooks/            # Analysis notebooks
└── results/              # Benchmark and training results

```

## Metrics
- **Model Complexity**: Params, FLOPs, Peak Activations
- **Training Behavior**: Convergence Speed, Sensitivity to Batch Size
- **Deployment**: Latency, Throughput, RAM usage (Desktop/Edge)

## Configuration
All settings are managed via Hydra. Key config files:
- `configs/config.yaml`: Main configuration
- `configs/model/*.yaml`: Model architectures
- `configs/dataset/*.yaml`: Dataset settings
- `configs/training/default.yaml`: Training hyperparameters

Override any config from command line:
```bash
python scripts/train.py model=swin_tiny training.lr=0.001 training.batch_size=128
```

## Results & Deliverables
- **Leaderboard table**: Model | Params | FLOPs | Memory | Accuracy | Latency | Throughput
- **Visualizations**: Loss curves, Latency charts, Memory heatmaps
- **CLI tools**: Easy-to-use benchmarking and training scripts
- **Analysis notebooks**: Deep-dive into results

## Contributing
This is a research project. Feel free to:
- Add new architectures to benchmark
- Extend to new datasets
- Improve profiling tools
- Add deployment targets (Jetson, RaspberryPi, etc.)

## License
MIT License - See LICENSE file for details

## Citation
If you use this work, please cite:
```
@misc{vlm-arch-benchmark,
  title={Small-Data VLM Backbone Benchmarking for Robotics},
  author={Your Name},
  year={2025},
  url={https://github.com/cronenberg64/VLM-arch}
}
```
