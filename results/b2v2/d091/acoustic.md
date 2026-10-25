# D-091 sound-only measures (decoded audio; 200 held-out inner-speech trials per arm)

MCD = mel-cepstral distortion (dB, lower = closer); r = spectral correlation (higher = closer); both DTW-aligned, speech frames only. In brackets: per-trial change vs the baseline (p). Acoustic identification: the own sentence's rank among the block's 50 by MCD (chance top-1 2%, mean rank 25.5).

| arm | MCD to attempted | r to attempted | MCD to clean | r to clean | id. vs attempted: top-1 / mean rank (p) | id. vs clean: top-1 / mean rank (p) |
|---|---|---|---|---|---|---|
| gan_read | 80.61 | 0.496 | 100.19 | 0.445 | 47.0% / 5.9 (p 0.0001; length only 4.5% / 18.9) | 33.8% / 7.9 (p 0.0001; length only 7.6% / 14.7) |
| pool | 80.71 (+0.10, p 0.94) | 0.496 (+0.00, p 0.87) | 100.54 (+0.35, p 0.13) | 0.456 (+0.01, p 0.4) | 49.5% / 5.5 (p 0.0001; length only 3.0% / 20.1) | 31.8% / 7.7 (p 0.0001; length only 5.6% / 16.1) |
| motor | 79.97 (-0.64, p 0.42) | 0.497 (+0.00, p 0.81) | 99.60 (-0.59, p 0.21) | 0.452 (+0.01, p 0.37) | 48.5% / 5.1 (p 0.0001; length only 3.0% / 19.8) | 33.3% / 7.2 (p 0.0001; length only 5.1% / 15.8) |
| marchonly | 80.81 (+0.20, p 0.58) | 0.507 (+0.01, p 0.1) | 99.56 (-0.62, p 0.14) | 0.466 (+0.02, p 0.035) | 50.0% / 5.4 (p 0.0001; length only 7.0% / 14.9) | 38.9% / 7.2 (p 0.0001; length only 4.0% / 17.0) |
| attempted | – | – | 81.55 | 0.606 | – | 65.2% / 2.6 (p 0.0001; length only 5.1% / 18.2) |
