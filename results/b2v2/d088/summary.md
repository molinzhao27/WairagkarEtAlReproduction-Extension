# D-088: confidence-gated voice on the adversarial-base inner-speech decodes (200 held-out trials)

Baseline `gan_read`. Rule (primary arms): extra words lower (paired per-trial p < 0.05) and words heard right down by at most 2.0 points. Movement inside words: attempted held-out = 1.00 by construction of the units (median over islands differs). Extra words and words heard right: per reference word, the raw transcript (change vs baseline, p).

| arm | WER | vs baseline (p) | vs gate (p) | WER sped up | missing words | extra words | heard right | movement | top-1 | mean rank | PER | rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gan_read | 79.9% | – | – | 76.1% | 13.2% | 14.9% | 34.9% | 1.24 | 78.0% | 3.3 | 51.9% | – |
| gate | 78.5% | -2.1 (0.22) | – | 73.9% | 13.6% | 13.3% (-1.8, 0.18) | 34.8% (+0.5, 0.71) | 1.24 | 78.0% | 3.3 | 51.9% | not met |
| gate_10db | 81.2% | +1.8 (0.84) | +3.9 (0.082) | 73.9% | 13.2% | 16.0% (+1.6, 0.83) | 34.8% (-0.0, 0.98) | 1.24 | 78.0% | 3.3 | 51.9% | – |
| gate_40db | 79.3% | -0.4 (0.7) | +1.7 (0.86) | 75.6% | 14.0% | 12.4% (-1.8, 0.1) | 33.1% (-1.3, 0.45) | 1.24 | 78.0% | 3.3 | 51.9% | – |
| gate_centered | 80.0% | -0.0 (0.55) | +2.1 (0.96) | 75.4% | 12.7% | 12.9% (-1.8, 0.076) | 33.0% (-1.7, 0.16) | 1.24 | 78.0% | 3.3 | 51.9% | – |
