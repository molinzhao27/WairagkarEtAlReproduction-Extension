# D-099: T15 inner-speech voice with the D-086 voice target (200 held-out trials, 2 seeds)

| cell | Whisper WER | WER sped up | PER | words read right | top-1 of 50 |
|---|---|---|---|---|---|
| base | 73.3% | 66.1% | 45.1% | 44.1% | 85.2% |
| voice | 74.3% | 70.5% | 45.4% | 44.2% | 85.8% |

`voice` vs `base`: WER +1.0 points (p 0.57); sped up +4.4 (p 0.063); PER +0.3 (p 0.52).

Primary (WER at least 3.0 points lower, p < 0.05): not met
