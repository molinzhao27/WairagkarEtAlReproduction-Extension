# D-089: the inner-speech read-out on the adversarial base (200 held-out trials)

Baseline `gan_read`. Rule (primary arms): pooled PER >= 3.0 points lower, paired per-trial p < 0.0167. Movement inside words: attempted held-out = 1.00 by construction of the units (median over islands differs). Extra words and words heard right: per reference word, the raw transcript (change vs baseline, p).

| arm | WER | vs baseline (p) | vs ens_read (p) | WER sped up | missing words | extra words | heard right | movement | top-1 | mean rank | PER | words read right | rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gan_read | 79.9% | – | – | 76.1% | 13.2% | 14.9% | 34.9% | 1.24 | 78.0% | 3.3 | 51.9% | 38.0% | – |
| hid | 78.7% | -0.3 (0.92) | +3.5 (0.42) | 73.2% | 13.5% | 12.7% (-1.2, 0.5) | 33.9% (-0.8, 0.4) | 1.24 | 80.0% | 3.4 | 53.1% (+1.1, 0.29) | 35.9% | not met |
| ens_read | 76.4% | -3.9 (0.17) | – | 74.9% | 11.9% | 10.9% (-3.7, 0.037) | 34.4% (+0.1, 0.96) | 1.07 | 76.0% | 3.5 | 55.2% (+3.0, 0.00019) | 28.7% | not met |
| ens_hid | 78.2% | -1.8 (0.33) | +2.1 (0.74) | 73.2% | 14.3% | 11.6% (-2.4, 0.12) | 33.3% (-0.7, 0.86) | 1.09 | 78.0% | 3.4 | 54.5% (+2.3, 0.004) | 30.9% | not met |
