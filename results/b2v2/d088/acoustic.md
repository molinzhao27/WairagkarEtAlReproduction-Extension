# D-088 sound-only measures (decoded audio; 200 held-out inner-speech trials per arm)

MCD = mel-cepstral distortion (dB, lower = closer); r = spectral correlation (higher = closer); both DTW-aligned, speech frames only. In brackets: per-trial change vs the baseline (p). Acoustic identification: the own sentence's rank among the block's 50 by MCD (chance top-1 2%, mean rank 25.5).

| arm | MCD to attempted | r to attempted | MCD to clean | r to clean | id. vs attempted: top-1 / mean rank (p) | id. vs clean: top-1 / mean rank (p) |
|---|---|---|---|---|---|---|
| gan_read | 80.61 | 0.496 | 100.19 | 0.445 | 47.0% / 5.9 (p 0.0001; length only 4.5% / 18.9) | 33.8% / 7.9 (p 0.0001; length only 7.6% / 14.7) |
| gate | 79.18 (-1.42, p 5.5e-14) | 0.475 (-0.02, p 0.0014) | 100.16 (-0.02, p 0.66) | 0.414 (-0.03, p 1.7e-07) | 45.5% / 5.7 (p 0.0001; length only 3.0% / 22.9) | 36.4% / 7.4 (p 0.0001; length only 4.0% / 16.8) |
| gate_10db | 79.90 (-0.71, p 8.2e-13) | 0.510 (+0.01, p 0.0022) | 100.23 (+0.04, p 0.93) | 0.439 (-0.01, p 0.051) | 50.0% / 5.7 (p 0.0001; length only 4.0% / 20.5) | 33.8% / 7.9 (p 0.0001; length only 8.1% / 14.7) |
| gate_40db | 79.12 (-1.48, p 1e-07) | 0.453 (-0.04, p 1.5e-08) | 101.52 (+1.34, p 0.0024) | 0.424 (-0.02, p 0.00024) | 43.0% / 6.3 (p 0.0001; length only 2.0% / 24.6) | 34.8% / 8.2 (p 0.0001; length only 3.5% / 20.2) |
| gate_centered | 79.57 (-1.04, p 6.3e-09) | 0.450 (-0.05, p 8.6e-11) | 100.83 (+0.64, p 0.1) | 0.401 (-0.04, p 7.6e-11) | 49.0% / 6.0 (p 0.0001; length only 2.0% / 23.1) | 35.4% / 8.2 (p 0.0001; length only 2.5% / 17.1) |
| attempted | – | – | 81.55 | 0.606 | – | 65.2% / 2.6 (p 0.0001; length only 5.1% / 18.2) |
