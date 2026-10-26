# D-097: T15's inner speech with the decoder's first layer trained too (PC; 200 held-out March trials, 2 seeds)

Baselines: `gan_read` PER 51.9%.

| steps | PER s0 | PER s1 | PER mean | words read right | top-1 | validation's step (per fold) | PER vs gan_read (p) |
|---|---|---|---|---|---|---|---|
| base | 48.5% | 48.7% | 48.6% | 39.5% | 85.0% | 1600,2600,1100,1000; 1300,1700,800,2000 | -3.4 (0.00017) |
| unfreeze | 45.3% | 45.0% | 45.1% | 44.1% | 85.2% | 2900,1500,1100,2400; 2000,900,2200,1800 | -6.6 (3.1e-09) |

Contrasts per trial (seed-averaged; PER in points, negative = fewer errors; words read right in points):

- unfreeze-base (primary): PER -3.25 (p 1e-05); words read right +4.65 (p 1.5e-05)

Primary (each counts if PER is at least 3 points lower with p < 0.05): unfreeze-base met
