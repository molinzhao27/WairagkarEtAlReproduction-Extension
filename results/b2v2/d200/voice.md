# D-200: T15's inner-speech voice through the causal post-net (voice from the read-out) (200 held-out trials, 2 seeds)

| cell | Whisper WER | WER sped up | PER | words read right | top-1 of 50 |
|---|---|---|---|---|---|
| raw | 73.3% | 66.1% | 45.1% | 44.1% | 85.2% |
| postnet | 73.9% | 70.1% | 45.1% | 44.1% | 85.2% |

`postnet` vs `raw`: WER +0.6 points (p 0.36); sped up +4.0 (p 0.04); PER +0.0 (p 1).

Primary (WER at least 3.0 points lower, p < 0.05): not met
