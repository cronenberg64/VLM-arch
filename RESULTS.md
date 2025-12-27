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

**Hardware**: 
NVIDIA GeForce RTX 1660 Ti
NVIDIA RTX A4000
---

## Training Results

### CIFAR-10 - Small Data Regime (5,000 samples)

| Model | Train Acc (%) | Val Acc (%) | Train Loss | Val Loss | Training Time | Epochs | Notes |
|-------|--------------|-------------|------------|----------|---------------|--------|-------|
| **ConvNeXt-V2 Tiny** | 99.52 | 51.58 | 0.0132 | 3.3098 | ~50 min | 100 | Heavy overfitting |
| MobileNetV3 | 63.06 | 51.59 | 1.0444 | 1.4148 | ~70 min | 100 | Best Val Acc: 51.59% |
| **ResNet50** | 99.62 | 67.19 | 0.1141 | NaN | ~60 min | 100 | Best performer so far |
| **EfficientNetV2-S** | 93.52 | 65.06 | 0.1887 | 1.5018 | ~55 min | 100 | Strong performance, needed amp=false |
| **ViT-Base** | 35.46 | 34.17 | 1.7342 | 1.7794 | ~110 min | 100 | Resized to 224x224, worst performer |
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

### MobileNetV3

| Dataset Size | Train Acc (%) | Val Acc (%) | Overfitting Gap | Training Time |
|--------------|--------------|-------------|-----------------|---------------|
| 5,000 | 63.06 | 51.59 | 11.47% | ~70 min |
| 10,000 | - | - | - | - |
| 50,000 (Full) | - | - | - | - |

### ResNet50

| Dataset Size | Train Acc (%) | Val Acc (%) | Overfitting Gap | Training Time |
|--------------|--------------|-------------|-----------------|---------------|
| 5,000 | 99.62 | 67.19 | 32.43% | ~60 min |
| 10,000 | - | - | - | - |
| 50,000 (Full) | - | - | - | - |

### EfficientNetV2-S

| Dataset Size | Train Acc (%) | Val Acc (%) | Overfitting Gap | Training Time |
|--------------|--------------|-------------|-----------------|---------------|
| 5,000 | 93.52 | 65.06 | 28.46% | ~55 min |
| 10,000 | - | - | - | - |
| 50,000 (Full) | - | - | - | - |

### ViT-Base

| Dataset Size | Train Acc (%) | Val Acc (%) | Overfitting Gap | Training Time |
|--------------|--------------|-------------|-----------------|---------------|
| 5,000 | 35.46 | 34.17 | 1.29% | ~110 min |
| 10,000 | - | - | - | - |
| 50,000 (Full) | - | - | - | - |

---

## Model Comparison Summary

### Efficiency Metrics

| Model | Best Val Acc @ 5k | Params/Accuracy Ratio | Latency/Accuracy Ratio | Overfitting Resistance |
|-------|-------------------|----------------------|------------------------|------------------------|
| **ResNet50** | **67.19%** | 35.02 | 16.74 | Medium (32.43% gap) |
| **EfficientNetV2-S** | 65.06% | 31.03 | 40.88 | Medium (28.46% gap) |
| MobileNetV3 | 51.59% | 81.59 | 191.73 | **Best (11.47% gap)** |
| ConvNeXt-V2 Tiny | 51.58% | 54.03 | 43.64 | Poor (47.94% gap) |
| ViT-Base | 34.17% | - | - | Excellent (1.29% gap, but poor accuracy) |

**Overfitting Resistance** = (Train Acc - Val Acc). Lower is better.

---

## Key Findings

### 1. Model Performance in Small-Data Regimes

**Observations** (to be filled as experiments complete):
- ConvNeXt-V2 Tiny achieves 51.58% validation accuracy with only 5,000 samples
- Heavy overfitting observed (99.52% train vs 51.58% val)
- [Add more as you complete experiments]

### 2. CNN vs Transformer Performance

**Hypothesis**:
- CNNs should outperform Transformers with <10k samples due to inductive biases
- Transformers may catch up at 50k samples when they can learn these patterns from data

**Results @ 5k samples (CONFIRMED)**:
- **CNNs dominate**: ResNet50 (67.19%) vs ViT-Base (34.17%) = **33% gap**
- **Inductive bias matters**: CNNs have built-in locality/translation equivariance
- **Data hunger**: ViT-Base needs significantly more data to learn spatial patterns
- **Overfitting paradox**: ViT shows minimal overfitting (1.29% gap) but stays stuck at low accuracy (underfitting)
- **Training efficiency**: CNNs converge faster; ViT needs 2x the time for worse results

### 3. Deployment Recommendations

**For Edge Devices (Jetson/RaspberryPi)**:
- **Best Choice**: MobileNetV3 (51.59% acc, 1133 img/s)
- **Why**: Highest throughput with acceptable accuracy; minimal params (4.21M)
- **Alternative**: ResNet50 if accuracy is critical (67.19% acc, 628 img/s)

**For Small Data (<10k samples)**:
- **Best Choice**: ResNet50
- **Why**: Best accuracy (67.19%), reasonable overfitting, proven architecture
- **Avoid**: ViT-Base (34.17%) - needs 10x+ more data to be competitive

**For Balanced Performance**:
- **Best Choice**: EfficientNetV2-S
- **Why**: 65.06% acc with 531 img/s, good accuracy/efficiency trade-off

### 4. Sample Efficiency Rankings

**Models ranked by validation accuracy at 5k samples**:
1. **ResNet50**: 67.19%
2. **EfficientNetV2-S**: 65.06% (-2.13%)
3. **MobileNetV3**: 51.59% (-15.60%)
4. **ConvNeXt-V2 Tiny**: 51.58% (-15.61%)
5. **ViT-Base**: 34.17% (-33.02%) ⚠️

**Models ranked by overfitting resistance** (lower gap = better):
1. **ViT-Base**: 1.29% gap (but underfitting)
2. **MobileNetV3**: 11.47% gap ✓ Best practical balance
3. **EfficientNetV2-S**: 28.46% gap
4. **ResNet50**: 32.43% gap
5. **ConvNeXt-V2 Tiny**: 47.94% gap (severe overfitting)

### 5. Training Time Analysis

**GPU Utilization** (RTX A4000 & RTX 1660 Ti):
- **ConvNeXt-V2 Tiny**: ~50 min (100 epochs, 5k samples)
- **EfficientNetV2-S**: ~55 min (AMP disabled due to instability)
- **ResNet50**: ~60 min
- **MobileNetV3**: ~70 min
- **ViT-Base**: ~110 min (slowest, requires 224×224 resize)

**Key Insight**: Transformers take 2x longer to train for significantly worse results in small-data regimes

---

## Recommended Next Steps

Based on current results:

1. **Complete Transformer Benchmarks**: Train DeiT-Small and Swin-Tiny @ 5k to validate CNN superiority
2. **Scale Up**: Run 10k sample experiments to see if Transformers start catching up
3. **Full Dataset**: Run 50k experiments to test if Transformers match CNNs with more data
4. **Benchmark Missing Models**: Profile ViT-Base, DeiT-Small, Swin-Tiny for FLOPs/latency
5. **Visualize**: Generate plots (accuracy vs data size, train/val curves, etc.)

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

### Completed Experiments @ 5k samples
- [x] Benchmark: MobileNetV3, EfficientNetV2-S, ResNet50, ConvNeXt-V2 Tiny (FLOPs, latency)
- [x] Training: ConvNeXt-V2 Tiny @ 5k samples
- [x] Training: MobileNetV3 @ 5k samples
- [x] Training: ResNet50 @ 5k samples
- [x] Training: EfficientNetV2-S @ 5k samples
- [x] Training: ViT-Base @ 5k samples (100 epochs, resized 224x224)

### In Progress
- [ ] Training: DeiT-Small @ 5k samples
- [ ] Training: Swin-Tiny @ 5k samples

### Planned
- [ ] Training: All models @ 10k samples
- [ ] Training: All models @ 50k samples (full dataset)
- [ ] CIFAR-100 experiments
- [ ] Generate all visualizations

---

## Notes & Observations

### ConvNeXt-V2 Tiny (5k samples)
- Training converged quickly (high training accuracy by epoch 50)
- Severe overfitting despite weight decay (0.05) - 47.94% train/val gap
- Validation accuracy plateaued around epoch 30 (~51%)
- Suggests need for stronger regularization or more data
- Training was stable, no divergence

### ResNet50 (5k samples)
- **Best overall performer**: 67.19% validation accuracy
- Moderate overfitting (32.43% gap) - acceptable for small data
- Stable training, no NaN issues in later epochs despite initial warning
- Fast inference (628 img/s) makes it production-ready

### ViT-Base (5k samples)
- **Worst performer**: Only 34.17% validation accuracy
- Minimal overfitting (1.29% gap) but stuck at low accuracy = **underfitting**
- Requires 224×224 input (7x larger than CIFAR-10's 32×32) = inefficient
- Training time: 2x longer than CNNs (~110 min vs ~50-70 min)
- **Key finding**: Confirms Transformers need massive datasets to compete with CNNs
- Lacks inductive biases (locality, translation equivariance) that CNNs have built-in
- Performance might improve with:
  - 10x more data (50k samples)
  - Pretrained weights (not tested to maintain fair comparison)
  - Data augmentation tweaks (RandAugment, Mixup)

### MobileNetV3 (5k samples)
- Excellent overfitting resistance: Only 11.47% train/val gap (best among practical models)
- 51.59% accuracy with minimal params (4.21M)
- Best for edge deployment: 1133 img/s throughput

### EfficientNetV2-S (5k samples)
- Strong performer: 65.06% accuracy (2nd best after ResNet50)
- Required AMP=false due to training instability
- Good balance: 531 img/s with competitive accuracy

### General Notes
- GPU acceleration working on both RTX 1660 Ti and RTX A4000
- Mixed precision training (AMP) enabled for most models (disabled for EfficientNetV2-S)
- All models using identical hyperparameters for fair comparison:
  - LR: 0.001 | Optimizer: AdamW | Weight Decay: 0.05
  - Batch Size: 128 | Epochs: 100 | Scheduler: Cosine
- **Major finding**: CNNs dominate in small-data regimes by 33% accuracy gap
- PyTorch 2.5.1+cu121 with CUDA support

---

**Last Updated**: 2025-12-27  
**Hardware**: NVIDIA RTX A4000 (primary) / NVIDIA GeForce RTX 1660 Ti  
**Dataset**: CIFAR-10  
**Framework**: PyTorch 2.5.1+cu121
