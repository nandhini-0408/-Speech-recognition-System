# SER Model Results Summary

## Phase 3: Baselines (5-Fold CV)

| Model                 | Val UA (RAVDESS) | Test UA (TESS Zero-Shot) | Notes                          |
| :-------------------- | :--------------- | :----------------------- | :----------------------------- |
| **Baseline CNN**      | 63.0%            | 52.8%                    | Good in-domain, poor transfer. |
| **Baseline CNN-LSTM** | **66.3%**        | ~53.0%                   | Best in-domain.                |

## Phase 4: Proposed Models & Ablations

| Model                           | Val UA (RAVDESS) | Test UA (TESS Zero-Shot) | Notes                                                              |
| :------------------------------ | :--------------- | :----------------------- | :----------------------------------------------------------------- |
| **Full Model (CNN+Trans+Attn)** | ~28%             | ~30%                     | Failed to converge. Transformer likely too complex for small data. |
| **Ablation 1 (CNN+Attn)**       | 59.6%            | **59.3%**                | **Best Generalization**. Beat baseline zero-shot by +6%.           |
| **Ablation 2 (CNN+Trans+Mean)** | ~20%             | N/A                      | Failed. Confirms Transformer is the bottleneck.                    |

## Key Findings

1.  **Transformer Bottleneck**: Adding a Transformer Encoder (even 2 layers) caused optimization issues on this small dataset (~1440 samples).
2.  **Attention Pooling Effectiveness**: Removing the Transformer but keeping **Attention Pooling** (Ablation 1) yielded the **best zero-shot performance (59.3%)**.
3.  **Generalization**: Ablation 1 improved TESS accuracy by **+6% absolute** over baselines, suggesting that robust frame selection (Attention) without over-parameterized sequence modeling (Transformer) generalizes better.

## Recommendation

- Adopt **CNN + Attention Pooling** (Ablation 1) as the final model for deployment/paper.
- Dropping the Transformer simplifies the model and improves robustness.
