# D-093: the March-only shared adapter x training length (200 held-out March trials, 2 seeds per cell)

Baseline `gan_read` (d086_gan_read): PER 51.9%, words read right 37.4%, top-1 78.0%.
Baseline `ganreadv3` (d091_ganreadv3): PER 54.0%, words read right 34.0%, top-1 77.0%.

| cell | PER s0 | PER s1 | PER mean | words read right (mean) | top-1 (mean) | PER vs gan_read (p) | PER vs ganreadv3 (p) |
|---|---|---|---|---|---|---|---|
| ad1k | 50.9% | 50.4% | 50.7% | 36.6% | 81.2% | -1.2 (0.1) | -2.6 (0.0074) |
| ad3k | 49.3% | 48.8% | 49.1% | 40.4% | 85.0% | -2.7 (0.016) | -4.1 (5.3e-05) |
| noad1k | 53.5% | 52.4% | 53.0% | 34.4% | 79.5% | +0.9 (0.19) | -0.5 (0.49) |
| noad3k | 48.5% | 48.7% | 48.6% | 39.5% | 85.0% | -3.4 (0.00017) | -4.7 (2.6e-07) |

Effects per trial (seed-averaged; PER in points, negative = fewer errors):

- per: adapter -0.73 (p 0.26); length -2.89 (p 4.8e-06); interaction +2.79 (p 0.0027)
- words_read_right: adapter +1.56 (p 0.059); length +4.46 (p 1.4e-06); interaction -1.31 (p 0.27)

Primary (the adapter lowers PER, p < 0.05): not met
