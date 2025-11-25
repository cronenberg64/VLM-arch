# Small-Data VLM Backbone Benchmarking for Robotics

> **Systematic benchmarking of vision backbones under small-data constraints for real-world robotics deployment**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project addresses a critical gap in vision model research: **how do different architectures perform when trained with limited data?**

Most academic benchmarks assume:
- Millions of training samples
- Expensive compute clusters
- Large batch sizes (512+)

**Real robotics teams face:**
- <300k training samples
- 1-4 GPUs or edge devices
- Tight latency, memory, and power constraints

This framework provides **standardized benchmarking** to answer: *Which backbone should a robotics company actually use?*

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/cronenberg64/VLM-arch.git
cd VLM-arch
pip install -r requirements.txt

# Benchmark all models
python scripts/run_batch.py --mode benchmark

# Train with 5k samples (small-data regime)
python scripts/train.py model=convnext_v2_tiny dataset=cifar10 dataset.subset=5000

# Analyze results
jupyter lab notebooks/analysis.ipynb
```

**[Read the Complete Usage Guide](USAGE_GUIDE.md)** for detailed instructions.

---

## What Gets Benchmarked

### Architectures (7 models)
| Type | Models |
|------|--------|
| **CNNs** | ConvNeXt-V2 Tiny, EfficientNetV2-S, MobileNetV3, ResNet50 |
| **Transformers** | ViT-Base, DeiT-Small, Swin-Tiny |
| **Hybrid** | *Coming soon: CoAtNet, MetaFormer* |

### Metrics

**Model Complexity**
- Parameters (M)
- FLOPs (G)
- Peak Memory (MB)

**Training Behavior**
- Convergence speed
- Sample efficiency (5k, 10k, 50k, full)
- Batch size sensitivity

**Deployment**
- Inference latency (ms)
- Throughput (img/s)
- CPU/GPU performance

---

## Project Structure

```
VLM-arch/
├── src/
│   ├── data/          # Dataset loading & subsampling
│   ├── models/        # Model factory (timm integration)
│   ├── engine/        # Training loop (AMP, grad accumulation)
│   └── benchmark/     # Profiling tools
├── scripts/
│   ├── train.py       # Training entry point
│   ├── benchmark.py   # Benchmarking entry point
│   └── run_batch.py   # Batch processing utility
├── configs/           # Hydra configurations
├── notebooks/         # Analysis & visualization
└── results/           # Benchmark outputs
```

---

## Key Features

- **Automatic subsampling** with stratified sampling (maintains class balance)  
- **Unified interface** for 7+ architectures via `timm`  
- **Mixed precision training** (AMP) and gradient accumulation  
- **Comprehensive profiling**: FLOPs, latency, memory, throughput  
- **WandB integration** for experiment tracking  
- **Batch processing** for running multiple experiments  
- **Jupyter notebooks** for analysis and visualization  

---

## Example Results

After running `python scripts/run_batch.py --mode benchmark`:

```
                Model  Params (M)  FLOPs (G)  Latency (ms)  Throughput (img/s)
mobilenetv3_large_100    4.21       0.007        9.89           1133.44
  tf_efficientnetv2_s   20.19       0.062       26.60            531.24
             resnet50   23.53       0.084       11.25            628.88
      convnextv2_tiny   27.87       0.091       22.51            328.17
```

*Your results will vary based on hardware*

---

## Research Workflow

1. **Benchmark**: Profile all models to understand complexity trade-offs
2. **Train**: Run experiments with different data sizes (5k → 50k → full)
3. **Analyze**: Generate plots comparing accuracy vs. data size
4. **Deploy**: Select best model for your robotics hardware constraints

---

## Documentation

- **[Complete Usage Guide](USAGE_GUIDE.md)** - Detailed instructions, code explanations, advanced usage
- **[Implementation Plan](implementation_plan.md)** - Technical design decisions
- **[Task List](task.md)** - Development progress

---

## Use Cases

This framework is ideal for:
- **Robotics researchers** evaluating vision backbones for edge deployment
- **ML engineers** comparing architectures under data constraints
- **Students** learning about model efficiency and benchmarking
- **Companies** selecting models for production systems

---

## Advanced Usage

### Custom Dataset Sizes
```bash
# Compare performance across data regimes
for subset in 5000 10000 50000; do
    python scripts/train.py model=vit_base dataset=cifar10 dataset.subset=$subset
done
```

### Hyperparameter Tuning
```bash
python scripts/train.py model=convnext_v2_tiny \
    training.lr=0.001 \
    training.batch_size=64 \
    training.epochs=200
```

### Add Your Own Model
1. Create `configs/model/my_model.yaml`
2. Run: `python scripts/benchmark.py model=my_model`

See [USAGE_GUIDE.md](USAGE_GUIDE.md#advanced-usage) for more details.

---

## Contributing

Contributions welcome! Ideas:
- Add new architectures (RepVGG, PVT, CoAtNet, etc.)
- Extend to new datasets (ImageNet, custom robotics data)
- Add deployment benchmarks (Jetson, RaspberryPi, Intel Movidius)
- Improve profiling tools (energy consumption, etc.)

---

## License

MIT License - See [LICENSE](LICENSE) for details

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{vlm-arch-benchmark,
  title={Small-Data VLM Backbone Benchmarking for Real-World Robotics},
  author={cronenberg64},
  year={2025},
  url={https://github.com/cronenberg64/VLM-arch}
}
```

---

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/), [timm](https://github.com/huggingface/pytorch-image-models), and [Hydra](https://hydra.cc/)
- Inspired by real-world robotics deployment challenges

---

**Ready to benchmark?** → [Start with the Usage Guide](USAGE_GUIDE.md)
