# Phase 3: Baseline Model Training Walkthrough

## 1. Environment Setup

- **GPU Enabled**: Successfully configured Python 3.11 virtual environment to bypass Python 3.14/CUDA incompatibility.
- **Dependencies**: Installed `torch` (CUDA 12.1), `librosa`, `sklearn`.

## 2. Models Implemented

- **BaselineCNN**: 4-layer 2D CNN.
- **BaselineCNNLSTM**: 4-layer CNN + Bidirectional LSTM.

## 3. Training Execution (5-Fold CV)

- **Data**: RAVDESS (Train/Val), TESS (Zero-Shot Test).
- **CNN Results**:
  - Validation UA: **63.0%**
  - Zero-Shot UA: **52.8%**
- **CNN-LSTM Results**:
  - Validation UA: **66.3%**
  - Zero-Shot UA: **~53%** (High variance)

## 4. Conclusion

- **CNN-LSTM** proved superior for in-domain speaker-independent recognition.
- **Zero-shot generalization** remains a challenge (~53%), confirming the difficulty of cross-corpus SER without adaptation.

## Next Steps

- Implement Attention mechanisms.
- Explore Domain Adversarial Training (DANN) for TESS generalization.
