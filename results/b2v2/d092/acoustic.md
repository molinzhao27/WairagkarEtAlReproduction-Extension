# D-092 sound-only measures (decoded audio; 200 held-out inner-speech trials per arm)

MCD = mel-cepstral distortion (dB, lower = closer); r = spectral correlation (higher = closer); both DTW-aligned, speech frames only. In brackets: per-trial change vs the baseline (p). Acoustic identification: the own sentence's rank among the block's 50 by MCD (chance top-1 2%, mean rank 25.5).

| arm | MCD to attempted | r to attempted | MCD to clean | r to clean | id. vs attempted: top-1 / mean rank (p) | id. vs clean: top-1 / mean rank (p) |
|---|---|---|---|---|---|---|
| gan_read | 80.61 | 0.496 | 100.19 | 0.445 | 47.0% / 5.9 (p 0.0001; length only 4.5% / 18.9) | 33.8% / 7.9 (p 0.0001; length only 7.6% / 14.7) |
| pool | 80.71 (+0.10, p 0.94) | 0.496 (+0.00, p 0.87) | 100.54 (+0.35, p 0.13) | 0.456 (+0.01, p 0.4) | 49.5% / 5.5 (p 0.0001; length only 3.0% / 20.1) | 31.8% / 7.7 (p 0.0001; length only 5.6% / 16.1) |
| stagedft | 81.48 (+0.87, p 0.026) | 0.483 (-0.01, p 0.075) | 100.92 (+0.74, p 0.016) | 0.443 (-0.00, p 0.85) | 44.5% / 6.7 (p 0.0001; length only 4.5% / 19.6) | 32.3% / 8.5 (p 0.0001; length only 6.6% / 15.0) |
| stagedfrozen | 80.67 (+0.06, p 0.7) | 0.477 (-0.02, p 0.011) | 101.09 (+0.90, p 0.0034) | 0.443 (-0.00, p 0.58) | 47.0% / 6.3 (p 0.0001; length only 3.0% / 20.7) | 29.8% / 8.2 (p 0.0001; length only 7.1% / 16.0) |
| attempted | – | – | 81.55 | 0.606 | – | 65.2% / 2.6 (p 0.0001; length only 5.1% / 18.2) |
