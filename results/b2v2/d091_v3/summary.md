# (same-hardware control: baseline gan_read re-run on vast-3) D-091: the inner-speech read-out with inner speech pooled across four days (200 held-out March trials)

Baseline `gan_read_v3`. Rule (primary arms): pooled PER >= 3.0 points lower, paired per-trial p < 0.05. Movement inside words: attempted held-out = 1.00 by construction of the units (median over islands differs). Extra words and words heard right: per reference word, the raw transcript (change vs baseline, p).

| arm | WER | vs baseline (p) | vs marchonly (p) | WER sped up | missing words | extra words | heard right | movement | top-1 | mean rank | PER | words read right | rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gan_read_v3 | 82.6% | – | – | 79.0% | 13.3% | 14.0% | 31.3% | 1.23 | 77.0% | 3.4 | 54.0% | 33.8% | – |
| gan_read | 79.9% | -1.4 (0.24) | +2.1 (0.73) | 76.1% | 13.2% | 14.9% (+0.7, 1) | 34.9% (+2.1, 0.24) | 1.24 | 78.0% | 3.3 | 51.9% (-1.4, 0.24) | 38.0% | – |
| pool | 82.0% | +0.9 (0.98) | +4.4 (0.074) | 78.1% | 13.7% | 14.3% (+0.7, 0.98) | 32.1% (-0.2, 0.81) | 1.23 | 81.0% | 3.2 | 53.9% (+0.5, 0.7) | 34.9% | not met |
| motor | 79.9% | -1.8 (0.27) | +1.7 (0.39) | 76.7% | 11.5% | 13.3% (-1.0, 0.4) | 33.4% (+0.9, 0.65) | 1.24 | 83.5% | 2.9 | 52.1% (-1.3, 0.25) | 37.4% | – |
| marchonly | 77.0% | -3.5 (0.13) | – | 70.7% | 10.7% | 13.9% (+0.7, 0.64) | 36.7% (+4.1, 0.013) | 1.24 | 85.0% | 3.0 | 48.5% (-4.8, 1.8e-05) | 41.3% | – |
