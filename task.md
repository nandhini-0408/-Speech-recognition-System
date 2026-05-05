# Task: Speech Emotion Recognition (SER)

## Phase 3: Baseline Models (Completed)

- [x] Create Model Definitions (`models.py`)
- [x] Create Training Script (`train_baseline.py`)
- [x] Install Dependencies
- [x] Train Baseline CNN (5-Fold CV)
- [x] Train Baseline CNN-LSTM (5-Fold CV)
- [x] Compare Results & Verify

## Phase 4: Proposed Model (CNN + Transformer)

- [x] Implement `CNNTransformer`, `PositionalEncoding`, `AttentionPooling` in `models.py`
- [x] Create `train_proposed.py` with GPU enforcement
- [x] Implement `CNNTransformer`, `PositionalEncoding`, `AttentionPooling` in `models.py`
- [x] Create `train_proposed.py` with GPU enforcement
- [x] Train Full Model (CNN-Transformer-Attn) (Failed: ~28% UA)
- [x] Train Ablation 1: No Transformer (CNN-Attn) (Val: ~59.6%, TESS: ~59.3%)
- [x] Train Ablation 2: No Attention (CNN-Transformer-Mean) (Failed: ~20% UA)
- [x] Compare Results & Verify (Best: Ablation 1)
