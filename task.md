# Task List: Small-Data VLM Backbone Benchmarking for Robotics

## Project Setup
- [ ] Initialize Git repository and connect to remote <!-- id: 0 -->
- [ ] Create project structure (src, configs, notebooks, scripts) <!-- id: 1 -->
- [ ] Setup Python environment (requirements.txt / environment.yml) <!-- id: 2 -->
- [ ] Create basic configuration system (Hydra or argparse based) <!-- id: 3 -->

## Data Pipeline
- [ ] Implement dataset wrappers for CIFAR-10/100 <!-- id: 4 -->
- [ ] Implement data subsampling logic (5k, 10k, 50k, 100k, Full) <!-- id: 5 -->
- [ ] Implement standardized augmentations pipeline <!-- id: 6 -->
- [ ] **Bonus**: Implement storage format benchmarks (WebDataset, LMDB, etc.) <!-- id: 7 -->

## Model Architecture Wrappers
- [ ] Create unified interface for backbones <!-- id: 8 -->
- [ ] Integrate CNN Backbones (ConvNeXt-V2, EfficientNetV2, MobileNetV3, RepVGG, DarkNet-53) <!-- id: 9 -->
- [ ] Integrate Transformer Backbones (ViT-B/16, DeiT-Small, Swin-T, PVT) <!-- id: 10 -->
- [ ] Integrate Hybrid Backbones (CoAtNet, MetaFormer) <!-- id: 11 -->

## Training Infrastructure
- [ ] Implement standardized training loop (PyTorch) <!-- id: 12 -->
- [ ] Implement logging (WandB / Tensorboard) for loss and accuracy <!-- id: 13 -->
- [ ] Implement checkpointing and resumption <!-- id: 14 -->
- [ ] Implement gradient accumulation and mixed precision support <!-- id: 15 -->

## Benchmarking & Metrics
- [ ] Implement model complexity profiler (Params, FLOPs) <!-- id: 16 -->
- [ ] Implement inference latency profiler (CPU, GPU) <!-- id: 17 -->
- [ ] Implement memory usage profiler (Peak activations, VRAM) <!-- id: 18 -->
- [ ] Create CLI tool for running benchmarks (`benchmark.py`) <!-- id: 19 -->

## Experiments & Analysis
- [ ] Run baseline benchmarks on CIFAR-10 (All architectures) <!-- id: 20 -->
- [ ] Run sample efficiency experiments (Subsets) <!-- id: 21 -->
- [ ] Run storage/loading efficiency experiments <!-- id: 22 -->
- [ ] Generate Leaderboard table <!-- id: 23 -->
- [ ] Create visualization notebooks (Loss curves, Latency charts, Memory heatmaps) <!-- id: 24 -->

## Documentation & Deliverables
- [ ] Write README.md with project overview and usage <!-- id: 25 -->
- [ ] Create "Investor-style" demo explanation <!-- id: 26 -->
