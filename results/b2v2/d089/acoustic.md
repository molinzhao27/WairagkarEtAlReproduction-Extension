# D-089 sound-only measures (decoded audio; 200 held-out inner-speech trials per arm)

MCD = mel-cepstral distortion (dB, lower = closer); r = spectral correlation (higher = closer); both DTW-aligned, speech frames only. In brackets: per-trial change vs the baseline (p). Acoustic identification: the own sentence's rank among the block's 50 by MCD (chance top-1 2%, mean rank 25.5).

| arm | MCD to attempted | r to attempted | MCD to clean | r to clean | id. vs attempted: top-1 / mean rank (p) | id. vs clean: top-1 / mean rank (p) |
|---|---|---|---|---|---|---|
| gan_read | 80.61 | 0.496 | 100.19 | 0.445 | 47.0% / 5.9 (p 0.0001; length only 4.5% / 18.9) | 33.8% / 7.9 (p 0.0001; length only 7.6% / 14.7) |
| hid | 81.60 (+1.00, p 0.022) | 0.490 (-0.01, p 0.35) | 99.96 (-0.22, p 0.49) | 0.454 (+0.01, p 0.16) | 47.0% / 6.5 (p 0.0001; length only 4.5% / 17.1) | 38.9% / 7.9 (p 0.0001; length only 4.0% / 16.6) |
| ens_read | 76.29 (-4.31, p 2.3e-30) | 0.514 (+0.02, p 0.0012) | 97.61 (-2.57, p 3.3e-17) | 0.462 (+0.02, p 9.8e-06) | 53.0% / 5.3 (p 0.0001; length only 2.5% / 20.0) | 34.8% / 7.8 (p 0.0001; length only 8.1% / 14.8) |
| ens_hid | 76.96 (-3.64, p 5.3e-23) | 0.527 (+0.03, p 1.8e-06) | 97.23 (-2.96, p 7.6e-15) | 0.467 (+0.02, p 2.2e-05) | 49.0% / 5.1 (p 0.0001; length only 3.5% / 17.0) | 34.8% / 7.4 (p 0.0001; length only 3.5% / 16.0) |
| attempted | – | – | 81.55 | 0.606 | – | 65.2% / 2.6 (p 0.0001; length only 5.1% / 18.2) |
