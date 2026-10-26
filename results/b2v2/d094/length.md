# D-094: the inner-speech read-out vs training length (no adapter; 200 held-out March trials, 2 seeds)

Baselines: `gan_read` PER 51.9%.

| steps | PER s0 | PER s1 | PER mean | words read right | top-1 | validation's step (per fold) | PER vs gan_read (p) |
|---|---|---|---|---|---|---|---|
| len3k | 48.5% | 48.7% | 48.6% | 39.5% | 85.0% | 1600,2600,1100,1000; 1300,1700,800,2000 | -3.4 (0.00017) |
| len6k | 48.8% | 49.3% | 49.1% | 39.0% | 81.5% | 1200,3800,2900,2500; 4100,1500,2000,1400 | -2.7 (0.007) |
| len10k | 50.3% | 49.2% | 49.8% | 38.7% | 83.2% | 2600,3900,5300,1400; 2400,2300,1300,2400 | -2.1 (0.05) |

Contrasts per trial (seed-averaged; PER in points, negative = fewer errors; words read right in points):

- len6k-len3k (primary): PER +0.68 (p 0.13); words read right -0.47 (p 0.52)
- len10k-len3k (primary): PER +1.31 (p 0.039); words read right -0.76 (p 0.17)
- len10k-len6k: PER +0.62 (p 0.52); words read right -0.30 (p 0.78)

Primary (each counts if PER is lower with p < 0.025): len6k-len3k not met; len10k-len3k not met
