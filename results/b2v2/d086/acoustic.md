# D-086 sound-only measures (decoded audio; 200 held-out inner-speech trials per arm)

MCD = mel-cepstral distortion (dB, lower = closer); r = spectral correlation (higher = closer); both DTW-aligned, speech frames only. In brackets: per-trial change vs the baseline (p). Acoustic identification: the own sentence's rank among the block's 50 by MCD (chance top-1 2%, mean rank 25.5).

| arm | MCD to attempted | r to attempted | MCD to clean | r to clean | id. vs attempted: top-1 / mean rank (p) | id. vs clean: top-1 / mean rank (p) |
|---|---|---|---|---|---|---|
| learned | 63.58 | 0.572 | 88.82 | 0.394 | 35.5% / 7.2 (p 0.0001; length only 2.0% / 19.8) | 9.1% / 14.0 (p 0.0001; length only 2.0% / 23.8) |
| boost | 63.63 (+0.05, p 0.47) | 0.570 (-0.00, p 0.97) | 88.87 (+0.05, p 0.25) | 0.393 (-0.00, p 0.58) | 35.0% / 7.3 (p 0.0001; length only 2.5% / 19.8) | 11.6% / 13.7 (p 0.0001; length only 2.0% / 23.8) |
| read | 61.55 (-2.04, p 4.5e-09) | 0.608 (+0.04, p 2e-07) | 87.81 (-1.01, p 0.0032) | 0.432 (+0.04, p 1.3e-06) | 49.0% / 5.2 (p 0.0001; length only 4.5% / 17.8) | 17.2% / 12.0 (p 0.0001; length only 3.0% / 22.3) |
| distill | 59.38 (-4.20, p 4.2e-22) | 0.612 (+0.04, p 7.5e-11) | 84.17 (-4.64, p 1.2e-24) | 0.417 (+0.02, p 0.0004) | 55.0% / 4.8 (p 0.0001; length only 3.0% / 21.6) | 18.7% / 11.5 (p 0.0001; length only 2.0% / 24.6) |
| adv | 61.74 (-1.84, p 1.1e-06) | 0.630 (+0.06, p 3.6e-13) | 88.26 (-0.56, p 0.096) | 0.450 (+0.06, p 4.2e-11) | 55.5% / 4.8 (p 0.0001; length only 5.5% / 16.4) | 17.7% / 11.2 (p 0.0001; length only 3.5% / 21.6) |
| gan_read | 80.61 (+17.02, p 2e-34) | 0.496 (-0.08, p 3.9e-12) | 100.19 (+11.37, p 1.5e-31) | 0.445 (+0.05, p 6.5e-07) | 47.0% / 5.9 (p 0.0001; length only 4.5% / 18.9) | 33.8% / 7.9 (p 0.0001; length only 7.6% / 14.7) |
| all | 81.15 (+17.57, p 1.7e-34) | 0.502 (-0.07, p 3e-12) | 98.47 (+9.65, p 4.6e-28) | 0.459 (+0.06, p 1e-08) | 53.5% / 4.7 (p 0.0001; length only 4.5% / 17.8) | 35.4% / 7.3 (p 0.0001; length only 3.5% / 20.7) |
| all_boost | 84.10 (+20.52, p 1.5e-34) | 0.482 (-0.09, p 4.1e-18) | 100.47 (+11.65, p 1.8e-30) | 0.444 (+0.05, p 1.7e-06) | 49.5% / 4.7 (p 0.0001; length only 5.0% / 18.0) | 35.9% / 7.5 (p 0.0001; length only 4.0% / 20.5) |
| attempted | – | – | 81.55 | 0.606 | – | 65.2% / 2.6 (p 0.0001; length only 5.1% / 18.2) |
