# Phase 4: Proposed Model Results

## 1. Full Proposed Model (CNN + Transformer + Attention)

- **Status**: Terminated early (Poor convergence).
- **Fold 1 Results**:
  - Gain 26% Train Acc after 20 epochs.
  - Val UA ~28%.
  - TESS ~30%.
- **Observation**: Model struggled to learn. Likely due to Transformer complexity on small dataset or hyperparameter sensitivity.

## 2. Ablation 1: No Transformer (CNN + Attention Pooling)

- **Configuration**: CNN backbone -> Projector (2048->256) -> Attention Pooling -> FC.
- **Status**: In Progress.
- **Run Log**:
  - **Fold 1**:
    - Ep 10: Val UA **54.2%**
    - Ep 15: Val UA **53.8%**
    - Peak UA: **[PENDING]**
