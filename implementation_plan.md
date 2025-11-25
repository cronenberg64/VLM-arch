# Implementation Plan - Small-Data VLM Backbone Benchmarking

# Goal Description
Develop a benchmarking framework to evaluate various visual backbone architectures (CNN, Transformer, Hybrid) for small-data robotics applications. The project will compare performance, efficiency, and resource usage across different dataset sizes and hardware constraints.

## User Review Required
> [!IMPORTANT]
> - **Hardware Constraints**: The benchmarking will be designed to run on Desktop GPU, CPU, and ideally Jetson-class hardware. We will simulate edge constraints where physical hardware is unavailable by limiting resources.
> - **Datasets**: Focusing on CIFAR-10/100 as primary proxies for small-data regimes, with subsets (5k, 10k, etc.). ImageNet-1K is optional depending on compute availability.

## Proposed Changes

### Project Structure
#### [NEW] [Structure](file:///d:/Programming Projects/VLM-arch/)
- `src/`: Source code for models, datasets, and training.
- `configs/`: Configuration files (Hydra/OmegaConf).
- `scripts/`: Entry points for training and benchmarking.
- `notebooks/`: Analysis and visualization.

### Dependencies
- **PyTorch**: Core deep learning framework.
- **timm**: For accessing a wide range of pretrained/initialized backbones (ConvNeXt, ViT, etc.).
- **Hydra**: For flexible configuration management.
- **WandB** (optional but recommended): For experiment tracking.
- **fvcore** / **thop**: For FLOPs counting.

### Components

#### [Model Factory](file:///d:/Programming Projects/VLM-arch/src/models/factory.py)
- Unified interface to instantiate models from `timm` or custom implementations.
- Wrappers to ensure consistent input/output shapes for the VLM context (though we are benchmarking backbones primarily).

#### [Data Pipeline](file:///d:/Programming Projects/VLM-arch/src/data/)
- `datamodule.py`: Handles loading CIFAR-10/100.
- `sampler.py`: Implements the subset sampling (5k, 10k, etc.).
- `transforms.py`: Standardized augmentations (RandAugment, Mixup/Cutmix if needed).

#### [Training Engine](file:///d:/Programming Projects/VLM-arch/src/engine/trainer.py)
- Standard training loop with:
    - Gradient accumulation.
    - Mixed precision (AMP).
    - Logging of loss, accuracy, and resource usage.

#### [Benchmarking Tools](file:///d:/Programming Projects/VLM-arch/src/benchmark/)
- `profiler.py`: Measures latency, throughput, and memory.
- `complexity.py`: Measures FLOPs and parameter counts.

## Verification Plan

### Automated Tests
- Unit tests for data loading to ensure correct subset sizes.
- Unit tests for model instantiation to verify all target architectures load correctly.
- Integration test running a single epoch on a tiny dataset to verify the training loop.

### Manual Verification
- Run `benchmark.py` with a lightweight model (e.g., MobileNetV3) to verify CLI functionality.
- Inspect generated logs and plots to ensure metrics are being captured.
