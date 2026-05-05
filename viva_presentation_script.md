# Viva Presentation Script: Cross-Corpus Speech Emotion Recognition

**Time**: ~10 Minutes
**Speaker**: [Your Name]
**Title**: Improving Zero-Shot Cross-Corpus Generalization in SER using Attention Pooling

---

## 1. Introduction (0:00 - 1:00)

"Good morning everyone. The title of my project is **'Improving Zero-Shot Cross-Corpus Generalization in Speech Emotion Recognition using Attention Pooling'**.

We live in a world where AI assistants like Siri or Alexa can understand _what_ we say, but they often fail to understand _how_ we say it. Emotion recognition is the missing link for truly human-AI interaction.

However, there is a fundamental flaw in current research: **Overfitting**. Most models work perfectly on the specific actors they were trained on (e.g., North American speakers) but fail miserably when tested on new people (e.g., Toronto dialect).

**My project solves this specific problem.** We developed a model that doesn't just memorize actors; it learns universal emotional patterns, allowing it to work on completely new, unseen groups of people. We achieved a **5% improvement** over standard industry baselines like LSTMs specifically in this 'Zero-Shot' scenario."

---

## 2. Problem Statement & Motivation (1:00 - 2:30)

"Let's dive deeper into the problem. This is known as the **Domain Shift** challenge.
In speech processing, every dataset has a unique 'acoustic signature'—the microphone quality, the background noise, and the accent of the speakers.

**The Status Quo:**
Most existing papers train and test on the same dataset (mixing speakers setup). This is easy and yields high accuracy (90%+), but it's unrealistic. Real-world AI encounters strangers every day.

**Our Approach:**
We adopted a strict **Cross-Corpus Zero-Shot Protocol**.

1.  **Training**: We used **RAVDESS** (24 professional actors, Neutral North American accent).
2.  **Testing**: We used **TESS** (Senior female speakers, Toronto dialect).
3.  **Constraint**: The model _never_ saw a single sample from TESS during training. This forces the model to learn actual emotion, not just the recording environment."

---

## 3. Methodology: From Scratch (2:30 - 5:00)

"Now, let me walk you through our pipeline, covering every nook and corner.

**Step 1: Preprocessing & Features**
We didn't feed raw audio into the model. We converted audio into **Log-Mel Spectrograms**. Think of this as an 'image' of the sound (128x188 dimension).

- _Why?_ Spectrograms capture both frequency (pitch) and intensity (loudness) over time, which are critical for distinguishing 'Angry' (loud, high pitch) from 'Sad' (quiet, low pitch).
- _Augmentation_: To make the model robust, we applied Noise Injection, Time Stretching, and Pitch Shifting to the training data _only_.

**Step 2: The Architecture**
We compared three architectures to find the best solution.

- **Baseline 1: Standard CNN**: Treats the spectrogram like an image classification task. Good at finding patterns, but bad at understanding time (temporal dynamics).
- **Baseline 2: CNN-LSTM**: The industry standard. Uses a CNN to find features and an LSTM (memory network) to track how emotion changes over time.
- **Proposed Model: CNN + Attention Pooling**: This is our valid contribution. Instead of a heavy LSTM, we used a **Learnable Attention Mechanism**.
  - _How it works_: In a 3-second audio clip, there might be 1 second of silence and 0.5 seconds of strict emotional shouting. A normal model averages everything. Our Attention mechanism assigns a 'weight' to every time frame, effectively saying 'Listen to the shout, ignore the silence'. It mathematically 'votes' for the most important segments."

---

## 4. The "Novel" Negative Result (5:00 - 6:30)

"Here is where our work gets scientifically interesting and **IEEE-worthy**.
We hypothesized that a **Transformer Encoder** (the architecture behind ChatGPT) would crush the benchmarks because it's the state-of-the-art in text and vision.

**We built a CNN-Transformer model, and it failed.**
It achieved only ~28% accuracy, barely better than random guessing.
**Why is this a 'Result'?**
This proves that **Data Quantity matters more than Model Complexity**. Transformers are 'data hungry'. On our small dataset (~1400 clips), the Transformer overfit massively.
This negative result is crucial: it scientifically proves that **for small SER datasets, Simpler is Better.** You don't need a cannon to kill a mosquito."

---

## 5. Quantitative Results (6:30 - 8:30)

"We ran a rigorous **5-Fold Cross-Validation** to ensure our results weren't just luck. Here is the scorecard:

| Model                    | RAVDESS (Validation) | TESS (Zero-Shot Test) |
| :----------------------- | :------------------- | :-------------------- |
| **Baseline CNN**         | 63.0%                | 52.8%                 |
| **Baseline CNN-LSTM**    | 66.3%                | 53.0%                 |
| **Proposed (Attention)** | 64.5%                | **57.7%**             |

**Key Takeaway**:
While the LSTM was great on the training data (66%), it failed to generalize (53%).
Our Proposed Attention Model improved generalization to **57.7%**.
That is a **+4.9% absolute improvement** on completely unseen data. In the world of signal processing, a 5% jump on a zero-shot task is significant."

---

## 6. Why is this IEEE Level? (8:30 - 9:30)

"If asked 'Why should IEEE accept this?', I have three strong arguments:

1.  **Rigorous Protocol**: We didn't cheat. We used strict speaker independence and cross-corpus testing. Many lower-tier papers mix data, which inflates scores. We established a realistic benchmark.
2.  **Novel Insights on Architecture**: We empirically demonstrated that **Attention Pooling > LSTMs > Transformers** for small-data speech tasks. We challenged the trend of 'just add a Transformer' and showed that lightweight attention is superior for generalization.
3.  **Reproducibility**: We logged every fold, every epoch, and every metric. The study is statistically sound (standard deviations reported) and fully reproducible."

---

## 7. Conclusion (9:30 - 10:00)

"To conclude: We tackled the problem of Domain Shift in Speech Emotion Recognition. We proved that heavy sequence models like LSTMs and Transformers prone to overfitting in low-data regimes. We proposed a lightweight **CNN-Attention** architecture that captures universal emotional cues better, achieving a **~5% improvement** in zero-shot accuracy.

Thank you. I am open to questions."
