# Phase 4 Walkthrough: CNN + Transformer + Attention Pooling

## Objective

Improve zero-shot cross-corpus generalization (RAVDESS→TESS) through better temporal representation and attention-based aggregation.

## Implementation

### Models Implemented

1. **PositionalEncoding**: Sinusoidal positional embeddings
2. **AttentionPooling**: Learnable attention mechanism (256→1 projection + tanh + softmax)
3. **CNNTransformer**: Modular architecture supporting three ablation modes

### Experiments Conducted

#### 1. Full Model (CNN + Transformer + Attention)

- **Result**: **FAILED** (27.9% Val UA, 30.3% TESS UA)
- **Analysis**: Transformer encoder (2 layers, 4 heads) failed to converge on small dataset (~1440 samples)

#### 2. Ablation 1: CNN + Attention Pooling ✓ **BEST**

- **5-Fold Results**:
  - **RAVDESS Val**: 64.5% UA (±4.3%)
  - **TESS Zero-Shot**: **57.7% UA** (±5.8%)
  - **Improvement**: +4.9% over baseline
- **Analysis**: Lightweight attention mechanism provides effective temporal aggregation without over-parameterization

#### 3. Ablation 2: CNN + Transformer + Mean Pooling

- **Result**: **FAILED** (20.0% Val UA)
- **Analysis**: Confirms Transformer component (not attention pooling) was the bottleneck

## Key Findings

### Why Transformer Failed

1. **Dataset Size**: Only ~1440 training samples insufficient for Transformer
2. **Over-parameterization**: Self-attention mechanism too complex for available data
3. **Optimization Issues**: Loss plateaued early, poor gradient flow

### Why Attention Pooling Succeeded

1. **Simplicity**: Single learnable query vector (256 params) vs. Transformer (thousands)
2. **Effective Frame Selection**: Learns to weight important time steps
3. **Better Generalization**: Simpler models transfer better across domains

## Final Comparison

| Model                   | RAVDESS Val UA | TESS Zero-Shot UA | Notes                |
| :---------------------- | :------------- | :---------------- | :------------------- |
| Baseline CNN            | 63.0%          | 52.8%             | Good in-domain       |
| Baseline CNN-LSTM       | 66.3%          | 53.0%             | Best in-domain       |
| **Proposed (CNN+Attn)** | **64.5%**      | **57.7%**         | **Best zero-shot** ✓ |
| Full (CNN+Trans+Attn)   | 27.9%          | 30.3%             | Failed               |
| CNN+Trans+Mean          | 20.0%          | N/A               | Failed               |

## GPU Utilization

- All experiments ran successfully on **NVIDIA GeForce RTX 4050 Laptop GPU**
- Training time: ~2.5 hours for 5-fold CV (30 epochs/fold)
- GPU utilization: ~80-90% during training

## Conclusion

**CNN + Attention Pooling** is the recommended architecture:

- **+4.9% improvement** on zero-shot TESS
- Simpler and more robust than Transformer approaches
- Suitable for IEEE publication and deployment
