# Presentation Cheat Sheet: Cross-Corpus Speech Emotion Recognition

## 1. Project Title & Goal

**Title**: Improving Zero-Shot Cross-Corpus Generalization in Speech Emotion Recognition using Attention Pooling.
**Goal**: Build an AI model that recognizes emotions (Angry, Happy, Sad, etc.) from speech, trained on one dataset (RAVDESS) and tested on a completely different one (TESS) without seeing it before.

## 2. The Problem (Why is this hard?)

- **Domain Shift**: Different datasets have different recording environments, speakers, and microphone qualities.
- **Overfitting**: Models usually memorize the specific characteristics of the training data (e.g., specific actors in RAVDESS).
- **Result**: A model might get 90% accuracy on its training data but fail (<50%) on new, unseen data (like TESS).

## 3. Our Methodology

### Data

- **Training**: RAVDESS (North American English, professional actors).
- **Testing**: TESS (Toronto dialect, different speakers). **Zero-Shot**: The model _never_ saw TESS during training.
- **Features**: Log-Mel Spectrograms (visual representation of sound).

### Models Compared

1.  **Baseline CNN**: Standard image-processing network.
2.  **Baseline CNN-LSTM**: Adds memory (LSTM) to track emotion over time.
3.  **Proposed Model (CNN + Attention Pooling)**: Our novel approach.
    - **CNN Backend**: Extracts features from spectrogram frames.
    - **Attention Pooling**: A smart, lightweight mechanism that "votes" on which time frames are most important for the emotion (e.g., ignoring silence, focusing on the loud shout).

## 4. Key Experiments & Results (The Numbers)

Reference: 5-Fold Cross-Validation (Robust Testing).

| Model                        | RAVDESS Accuracy (Validation) | TESS Accuracy (Zero-Shot Test) |
| :--------------------------- | :---------------------------- | :----------------------------- |
| **Baseline CNN**             | 63.0%                         | 52.8%                          |
| **Baseline CNN-LSTM**        | 66.3% (Best In-Domain)        | 53.0% (Poor Generalization)    |
| **Proposed (CNN+Attention)** | 64.5%                         | **57.7% (+4.9% Improvement)**  |

- **Result**: Our Attention Pooling model generalizes **~5% better** to new data than the industry-standard LSTM.

## 5. What Didn't Work (Meaningful Negative Result)

- We tried adding a **Transformer Encoder** (the tech behind ChatGPT).
- **It Failed**: Accuracy dropped to ~28%.
- **Why?**: Transformers are "data hungry" and complex. Our dataset (~1400 clips) was too small, leading to severe overfitting. Use this to argue that **"Simple is Better"** for small datasets.

## 6. What is Novel? (The "Selling Point")

1.  **Lightweight Selection**: We proved that a simple mathematical attention mechanism (257 parameters) beats a complex LSTM (65,000 parameters) for generalization. It learns _what_ to listen to, not just _how_ to memorize sequences.
2.  **Robust Zero-Shot Performance**: We achieved significant improvement (+~5%) on a completely unseen dataset, addressing the major "Domain Shift" problem in audio AI.
3.  **Efficiency**: Our model is faster and smaller, making it suitable for real-time mobile apps or embedded devices, unlike heavy Transformer models.

## 7. Conclusion

We built a robust, efficient Speech Emotion Recognition model. By removing complex sequence layers (LSTM/Transformer) and replacing them with smart **Attention Pooling**, we forced the model to learn stable, universal emotional cues, resulting in superior performance on unseen speakers and recording environments.
