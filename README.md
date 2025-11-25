# Small-Data Visual Backbone Benchmarking & Architecture Exploration for Real-World Robotics Deployment

## Objective
Identify which visual backbone architecture performs best when training a small-data Visual-Language Model (VLM), with a specific focus on robotics and edge deployment scenarios. The study compares CNN, Transformer, and hybrid backbone families under identical training conditions and small dataset constraints.

## Architectures to Compare
- **CNN**: ConvNeXt-V2 Tiny, EfficientNetV2-S, MobileNetV3, RepVGG-B1g4, DarkNet-53
- **Transformer**: ViT-B/16, DeiT-Small, Swin-T, PVT
- **Hybrid**: CoAtNet, MetaFormer

## Datasets
- CIFAR-10 / CIFAR-100
- Subsets: 5k, 10k, 50k, 100k samples

## Metrics
- **Model Complexity**: Params, FLOPs, Peak Activations
- **Training Behavior**: Convergence Speed, Sensitivity to Batch Size
- **Deployment**: Latency, Throughput, RAM usage (Desktop/Edge)

## Deliverables
- Leaderboard table
- Loss curves and Latency charts
- CLI tool for benchmarking
- Jupyter notebooks for analysis
