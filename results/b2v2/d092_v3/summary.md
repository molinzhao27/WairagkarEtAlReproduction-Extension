# (same-hardware control: baseline gan_read re-run on vast-3) D-092: pooled inner speech in two stages, other days first, then March (200 held-out March trials)

Baseline `gan_read_v3`. Rule (primary arms): pooled PER >= 3.0 points lower, paired per-trial p < 0.025. Movement inside words: attempted held-out = 1.00 by construction of the units (median over islands differs). Extra words and words heard right: per reference word, the raw transcript (change vs baseline, p).

| arm | WER | vs baseline (p) | vs pool (p) | WER sped up | missing words | extra words | heard right | movement | top-1 | mean rank | PER | words read right | rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gan_read_v3 | 82.6% | – | – | 79.0% | 13.3% | 14.0% | 31.3% | 1.23 | 77.0% | 3.4 | 54.0% | 33.8% | – |
| gan_read | 79.9% | -1.4 (0.24) | -2.3 (0.63) | 76.1% | 13.2% | 14.9% (+0.7, 1) | 34.9% (+2.1, 0.24) | 1.24 | 78.0% | 3.3 | 51.9% (-1.4, 0.24) | 38.0% | – |
| pool | 82.0% | +0.9 (0.98) | – | 78.1% | 13.7% | 14.3% (+0.7, 0.98) | 32.1% (-0.2, 0.81) | 1.23 | 81.0% | 3.2 | 53.9% (+0.5, 0.7) | 34.9% | – |
| stagedft | 84.1% | +2.9 (0.41) | +2.0 (0.62) | 79.3% | 14.6% | 15.2% (+1.5, 0.72) | 31.2% (-1.2, 0.75) | 1.22 | 78.5% | 3.3 | 55.4% (+2.4, 0.032) | 32.5% | not met |
| stagedfrozen | 81.2% | +0.9 (0.76) | -0.0 (0.86) | 78.5% | 13.5% | 15.6% (+2.8, 0.51) | 34.1% (+1.7, 0.29) | 1.21 | 75.0% | 3.6 | 56.5% (+2.9, 0.0052) | 30.4% | not met |
