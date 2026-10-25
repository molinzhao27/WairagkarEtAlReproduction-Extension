# D-092: pooled inner speech in two stages, other days first, then March (200 held-out March trials)

Baseline `gan_read`. Rule (primary arms): pooled PER >= 3.0 points lower, paired per-trial p < 0.025. Movement inside words: attempted held-out = 1.00 by construction of the units (median over islands differs). Extra words and words heard right: per reference word, the raw transcript (change vs baseline, p).

| arm | WER | vs baseline (p) | vs pool (p) | WER sped up | missing words | extra words | heard right | movement | top-1 | mean rank | PER | words read right | rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gan_read | 79.9% | – | – | 76.1% | 13.2% | 14.9% | 34.9% | 1.24 | 78.0% | 3.3 | 51.9% | 38.0% | – |
| pool | 82.0% | +2.3 (0.63) | – | 78.1% | 13.7% | 14.3% (+0.1, 0.93) | 32.1% (-2.3, 0.38) | 1.23 | 81.0% | 3.2 | 53.9% (+1.8, 0.11) | 34.9% | – |
| stagedft | 84.1% | +4.3 (0.11) | +2.0 (0.62) | 79.3% | 14.6% | 15.2% (+0.8, 0.65) | 31.2% (-3.2, 0.06) | 1.22 | 78.5% | 3.3 | 55.4% (+3.8, 0.00023) | 32.5% | not met |
| stagedfrozen | 81.2% | +2.3 (0.98) | -0.0 (0.86) | 78.5% | 13.5% | 15.6% (+2.2, 0.92) | 34.1% (-0.4, 0.68) | 1.21 | 75.0% | 3.6 | 56.5% (+4.3, 0.00011) | 30.4% | not met |
