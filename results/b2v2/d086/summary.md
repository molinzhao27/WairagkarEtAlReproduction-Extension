# D-086: making inner-speech voice less muddy (200 held-out imagined trials)

Baseline `learned`. Rule (primary arms): pooled WER >= 5.0 points lower, paired p < 0.0083. Movement inside words: attempted held-out = 1.00 by construction of the units (median over islands differs). Extra words and words heard right: per reference word, the raw transcript (change vs baseline, p).

| arm | WER | vs baseline (p) | vs read (p) | WER sped up | missing words | extra words | heard right | movement | top-1 | mean rank | PER | rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| learned | 82.7% | – | – | 86.4% | 30.5% | 6.1% | 24.0% | 0.46 | 74.5% | 3.9 | 59.5% | – |
| boost | 84.1% | +1.5 (0.22) | +3.5 (0.076) | 83.7% | 25.6% | 8.4% (+2.3, 0.057) | 24.5% (+0.5, 0.87) | 0.48 | 74.5% | 3.9 | 59.5% | not met |
| read | 80.3% | -2.0 (0.2) | – | 76.4% | 24.6% | 8.4% (+2.4, 0.31) | 28.3% (+4.1, 0.0031) | 0.47 | 81.5% | 3.3 | 52.1% | not met |
| distill | 76.2% | -5.9 (0.005) | -3.9 (0.14) | 77.6% | 25.4% | 6.4% (+0.6, 0.78) | 30.7% (+6.4, 4.1e-05) | 0.47 | 81.5% | 3.1 | 50.1% | met |
| adv | 76.9% | -5.3 (0.012) | -3.2 (0.26) | 75.3% | 24.4% | 4.7% (-1.3, 0.36) | 28.5% (+4.2, 0.011) | 0.51 | 83.5% | 3.2 | 50.3% | not met |
| gan_read | 79.9% | -2.1 (0.13) | -0.1 (0.86) | 76.1% | 13.2% | 14.9% (+8.7, 3.2e-06) | 34.9% (+10.1, 9.6e-08) | 1.24 | 78.0% | 3.3 | 51.9% | not met |
| all | 84.5% | +3.0 (0.99) | +5.1 (0.35) | 77.7% | 12.8% | 19.7% (+14.4, 6.3e-07) | 34.1% (+9.8, 1.3e-07) | 1.26 | 81.5% | 3.2 | 50.8% | not met |
| all_boost | 81.9% | -0.5 (0.52) | +1.5 (0.82) | 73.8% | 15.1% | 15.6% (+9.9, 7.8e-06) | 33.6% (+9.8, 2e-07) | 1.42 | 81.5% | 3.2 | 50.8% | – |
