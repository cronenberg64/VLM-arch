# Experimental Results - Small-Data VLM Backbone Benchmarking

This document tracks all experimental results for comparing vision backbone architectures under small-data constraints.

---

## Table of Contents
1. [Benchmark Results](#benchmark-results) - Model complexity and inference performance
2. [Training Results](#training-results) - Accuracy across different configurations
3. [Sample Efficiency Analysis](#sample-efficiency-analysis) - Performance vs. dataset size
4. [Key Findings](#key-findings) - Conclusions and insights

---

## Benchmark Results

Performance metrics for inference speed and model complexity.

| Model | Params (M) | FLOPs (G) | Latency (ms) | Throughput (img/s) | Peak Memory (MB) |
|-------|-----------|-----------|--------------|-------------------|------------------|
| MobileNetV3 | 4.21 | 0.007 | 9.89 | 1133.44 | 0.0 |
| EfficientNetV2-S | 20.19 | 0.062 | 26.60 | 531.24 | 0.0 |
| ResNet50 | 23.53 | 0.084 | 11.25 | 628.88 | 0.0 |
| ConvNeXt-V2 Tiny | 27.87 | 0.091 | 22.51 | 328.17 | 0.0 |
| ViT-Base | - | - | - | - | - |
| DeiT-Small | - | - | - | - | - |
| Swin-Tiny | - | - | - | - | - |

**Hardware**: NVIDIA GeForce RTX 1660 Ti

---

## Training Results

### CIFAR-10 - Small Data Regime (5,000 samples)

| Model | Train Acc (%) | Val Acc (%) | Train Loss | Val Loss | Training Time | Epochs | Notes |
|-------|--------------|-------------|------------|----------|---------------|--------|-------|
| **ConvNeXt-V2 Tiny** | 99.52 | 51.58 | 0.0132 | 3.3098 | ~50 min | 100 | Heavy overfitting |
| MobileNetV3 | 63.06 | 51.59 | 1.0444 | 1.4148 | ~70 min | 100 | Best Val Acc: 51.59% |
| **ResNet50** | 99.62 | 67.19 | 0.1141 | NaN | ~60 min | 100 | Best performer so far |
| EfficientNetV2-S | - | - | - | - | - | - | - |
| ViT-Base | - | - | - | - | - | - | - |
| DeiT-Small | - | - | - | - | - | - | - |
| Swin-Tiny | - | - | - | - | - | - | - |

**Training Configuration:**
- Batch Size: 128
- Learning Rate: 0.001
- Optimizer: AdamW
- Epochs: 100
- Weight Decay: 0.05

---

### CIFAR-10 - Medium Data Regime (10,000 samples)

| Model | Train Acc (%) | Val Acc (%) | Train Loss | Val Loss | Training Time | Notes |
|-------|--------------|-------------|------------|----------|---------------|-------|
| ConvNeXt-V2 Tiny | - | - | - | - | - | - |
| MobileNetV3 | - | - | - | - | - | - |
| ResNet50 | - | - | - | - | - | - |
| ViT-Base | - | - | - | - | - | - |

---

### CIFAR-10 - Full Dataset (~50,000 samples)

| Model | Train Acc (%) | Val Acc (%) | Train Loss | Val Loss | Training Time | Notes |
|-------|--------------|-------------|------------|----------|---------------|-------|
| ConvNeXt-V2 Tiny | - | - | - | - | - | - |
| MobileNetV3 | - | - | - | - | - | - |
| ResNet50 | - | - | - | - | - | - |
| ViT-Base | - | - | - | - | - | - |

---

## Sample Efficiency Analysis

How does each model's performance scale with dataset size?

### ConvNeXt-V2 Tiny

| Dataset Size | Train Acc (%) | Val Acc (%) | Overfitting Gap | Training Time |
|--------------|--------------|-------------|-----------------|---------------|
| 5,000 | 99.52 | 51.58 | 47.94% | ~50 min |
| 10,000 | - | - | - | - |
| 50,000 (Full) | - | - | - | - |

**Convergence Plot**: (To be added from Jupyter notebook)

### MobileNetV3

| Dataset Size | Train Acc (%) | Val Acc (%) | Overfitting Gap | Training Time |
|--------------|--------------|-------------|-----------------|---------------|
| 5,000 | - | - | - | - |
| 10,000 | - | - | - | - |
| 50,000 (Full) | - | - | - | - |

### ResNet50

| Dataset Size | Train Acc (%) | Val Acc (%) | Overfitting Gap | Training Time |
|--------------|--------------|-------------|-----------------|---------------|
| 5,000 | - | - | - | - |
| 10,000 | - | - | - | - |
| 50,000 (Full) | - | - | - | - |

### ViT-Base

| Dataset Size | Train Acc (%) | Val Acc (%) | Overfitting Gap | Training Time |
|--------------|--------------|-------------|-----------------|---------------|
| 5,000 | - | - | - | - |
| 10,000 | - | - | - | - |
| 50,000 (Full) | - | - | - | - |

---

## Model Comparison Summary

### Efficiency Metrics

| Model | Best Val Acc @ 5k | Params/Accuracy Ratio | Latency/Accuracy Ratio | Overfitting Resistance |
|-------|-------------------|----------------------|------------------------|------------------------|
| ConvNeXt-V2 Tiny | 51.58% | 54.03 | 43.64 | Low (47.94% gap) |
| MobileNetV3 | - | - | - | - |
| ResNet50 | - | - | - | - |
| ViT-Base | - | - | - | - |

**Overfitting Resistance** = (Train Acc - Val Acc). Lower is better.

---

## Key Findings

### 1. Model Performance in Small-Data Regimes

**Observations** (to be filled as experiments complete):
- ConvNeXt-V2 Tiny achieves 51.58% validation accuracy with only 5,000 samples
- Heavy overfitting observed (99.52% train vs 51.58% val)
- [Add more as you complete experiments]

### 2. CNN vs Transformer Performance

**Hypothesis to Test**:
- CNNs should outperform Transformers with <10k samples
- Transformers may catch up at 50k samples

**Results**:
- [To be filled]

### 3. Deployment Recommendations

**For Edge Devices (Jetson/RaspberryPi)**:
- **Best Choice**: [To be determined based on Throughput and Accuracy]
- **Why**: [Reasoning]

**For Small Data (<10k samples)**:
- **Best Choice**: [To be determined]
- **Why**: [Reasoning]

**For Balanced Performance**:
- **Best Choice**: [To be determined]
- **Why**: [Reasoning]

### 4. Sample Efficiency Rankings

**Models ranked by validation accuracy at 5k samples**:
1. ConvNeXt-V2 Tiny: 51.58%
2. [Add others as completed]

**Models ranked by overfitting resistance**:
1. [Lower gap = better generalization]

### 5. Training Time Analysis

**GPU Utilization** (RTX 1660 Ti):
- ConvNeXt-V2 Tiny: ~50 minutes for 100 epochs (5k samples)
- Average speed: ~1.4-1.5 it/s
- [Add others as completed]

---

## Recommended Next Steps

Based on current results:

1. **Immediate**: Train MobileNetV3 and ResNet50 with 5k samples for comparison
2. **Next**: Test ViT-Base to compare CNN vs Transformer on small data
3. **Then**: Run 10k sample experiments for sample efficiency analysis
4. **Finally**: Full 50k dataset experiments

---

## Visualizations

*Note: Run `jupyter lab notebooks/analysis.ipynb` to generate plots*

### Plots to Generate:
1. **Accuracy vs Dataset Size** (sample efficiency curves)
2. **Params vs Accuracy** (model complexity trade-offs)
3. **Latency vs Accuracy** (deployment trade-offs)
4. **Training Loss Curves** (convergence behavior)
5. **Overfitting Analysis** (train vs val gap)

---

## Experiment Log

### Completed Experiments
- [x] Benchmark: MobileNetV3, EfficientNetV2-S, ResNet50, ConvNeXt-V2 Tiny
- [x] Training: ConvNeXt-V2 Tiny @ 5k samples (CIFAR-10)

### In Progress
- [ ] Training: MobileNetV3 @ 5k samples
- [ ] Training: ResNet50 @ 5k samples
- [ ] Training: ViT-Base @ 5k samples

### Planned
- [ ] Training: All models @ 10k samples
- [ ] Training: All models @ 50k samples (full dataset)
- [ ] CIFAR-100 experiments
- [ ] Generate all visualizations

---

## Notes & Observations

### ConvNeXt-V2 Tiny (5k samples)
- Training converged quickly (high training accuracy by epoch 50)
- Severe overfitting despite weight decay (0.05)
- Validation accuracy plateaued around epoch 30 (~51%)
- Suggests need for stronger regularization or more data
- Training was stable, no divergence

### General Notes
- GPU acceleration working perfectly on RTX 1660 Ti
- Mixed precision training (AMP) enabled
- All models using same hyperparameters for fair comparison

---

**Last Updated**: 2025-11-27  
**Hardware**: NVIDIA GeForce RTX 1660 Ti  
**Dataset**: CIFAR-10  
**Framework**: PyTorch 2.5.1+cu121
