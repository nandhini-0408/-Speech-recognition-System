# Implementation Plan - Phase 4: Proposed Model (CNN + Transformer)

## Goal

Implement and evaluate a CNN-Transformer model with Attention Pooling for SER.

## Model Architecture (`models.py`)

### Class: `CNNTransformer`

1.  **CNN Feature Extractor**:
    - Same 4-layer CNN backbone as baseline.
    - Output: (Batch, 256, 8, 11) -> Reshape to (Batch, Time=11, Feats=2048)
2.  **Transformer Encoder**:
    - Positional Encoding (Sinusoidal).
    - `nn.TransformerEncoder`
      - d_model: 256 (Project feats 2048 -> 256)
      - nhead: 4
      - dim_feedforward: 512
      - num_layers: 2
      - dropout: 0.1
3.  **Temporal Pooling**:
    - **Attention Pooling**: Learnable query vector to compute weights over time steps.
    - **Ablation (Mean Pooling)**: Simple average over time.
4.  **Classifier**:
    - FC Layer -> 6 Emotions.

## Training Script (`train_proposed.py`)

- Based on `train_baseline.py`.
- **Arguments**:
  - `--ablation`: 'none' (Full Model), 'no_transformer' (CNN+Attn), 'no_attention' (CNN+Trans+Mean).
- **GPU Enforcement**: Explicitly check for 'NVIDIA' in device name.

## Experiments (5-Fold CV)

1.  **Proposed**: CNN + Transformer + Attention Pooling.
2.  **Ablation 1**: CNN + Attention Pooling (No Transformer).
    - Tests if Transformer adds value over simple attention.
3.  **Ablation 2**: CNN + Transformer + Mean Pooling.
    - Tests if Attention Pooling adds value over global average.

## Comparison

- Compare UA/WA/F1 with Phase 3 Baselines.
- Analyze confusion matrices for specific emotions.

## Files

- `models.py`: Update with `CNNTransformer`, `PositionalEncoding`, `AttentionPooling`.
- `train_proposed.py`: New training loop with ablations.
