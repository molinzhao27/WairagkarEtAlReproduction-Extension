# D-098: T15's inner speech, 1 vs 2 vs 4 decoder layers trained (PC; 200 held-out March trials, 2 seeds)

Baselines: `gan_read` PER 51.9%.

| steps | PER s0 | PER s1 | PER mean | words read right | top-1 | validation's step (per fold) | PER vs gan_read (p) |
|---|---|---|---|---|---|---|---|
| l1 | 45.3% | 45.0% | 45.1% | 44.1% | 85.2% | 2900,1500,1100,2400; 2000,900,2200,1800 | -6.6 (3.1e-09) |
| l2 | 45.3% | 46.0% | 45.7% | 42.7% | 85.0% | 1500,1900,600,1700; 1000,1100,900,800 | -6.3 (8.6e-10) |
| l4 | 44.9% | 44.4% | 44.6% | 44.2% | 87.2% | 800,1600,1100,1400; 1400,900,800,800 | -7.1 (2.5e-11) |

Contrasts per trial (seed-averaged; PER in points, negative = fewer errors; words read right in points):

- l2-l1 (primary): PER +0.36 (p 0.44); words read right -1.38 (p 0.23)
- l4-l1 (primary): PER -0.48 (p 0.55); words read right +0.07 (p 0.95)
- l4-l2: PER -0.83 (p 0.29); words read right +1.45 (p 0.2)

Primary (each counts if PER is at least 3 points lower with p < 0.025): l2-l1 not met; l4-l1 not met
