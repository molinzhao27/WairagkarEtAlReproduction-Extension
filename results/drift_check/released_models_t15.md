# Drift check `released_models_t15` (D-063)

| model | session | gap (d) | z Frechet | z MMD² | raw Frechet | raw mean shift | feat r |
|---|---|---|---|---|---|---|---|
| brain2voice_t15_day165.h5 | t15_day165 (half B vs half A) | 0 | 4783.86 | 0.0004 | 0.5 | 0.005 | 0.690 |
| brain2voice_t15_day165.h5 | t15_day172 | 7 | 32557.71 | 0.0008 | 4.9 | 0.012 | 0.536 |
| brain2voice_t15_day165.h5 | t15_day174 | 9 | 88986.63 | 0.0058 | 6.9 | 0.011 | 0.543 |
| brain2voice_t15_day165.h5 | t15_day179 | 14 | 102333.66 | 0.0045 | 8.0 | 0.017 | 0.567 |
| brain2voice_t15_day165.h5 | t15_day181 | 16 | 79102.73 | 0.0038 | 8.0 | 0.018 | 0.524 |
| brain2voice_t15_day165.h5 | t15_day186 | 21 | 100364.42 | 0.0043 | 9.6 | 0.024 | 0.481 |
| brain2voice_t15_day165.h5 | t15_day188 | 23 | 94874.04 | 0.0050 | 10.1 | 0.029 | 0.459 |
| brain2voice_t15_day165.h5 | t15_day193 | 28 | 100858.96 | 0.0042 | 14.1 | 0.034 | 0.466 |
| brain2voice_t15_day165.h5 | t15_day195 | 30 | 108941.45 | 0.0049 | 10.9 | 0.021 | 0.448 |
| brain2voice_t15_day174.h5 | t15_day174 (half B vs half A) | 0 | 17372.75 | 0.0001 | 0.7 | 0.004 | 0.664 |
| brain2voice_t15_day174.h5 | t15_day179 | 5 | 88238.93 | 0.0010 | 5.9 | 0.014 | 0.601 |
| brain2voice_t15_day174.h5 | t15_day181 | 7 | 141741.65 | 0.0019 | 6.2 | 0.016 | 0.562 |
| brain2voice_t15_day174.h5 | t15_day186 | 12 | 163520.87 | 0.0017 | 9.5 | 0.024 | 0.538 |
| brain2voice_t15_day174.h5 | t15_day188 | 14 | 185791.18 | 0.0022 | 10.1 | 0.029 | 0.496 |
| brain2voice_t15_day174.h5 | t15_day193 | 19 | 272378.43 | 0.0033 | 14.9 | 0.034 | 0.495 |
| brain2voice_t15_day174.h5 | t15_day195 | 21 | 272372.84 | 0.0030 | 10.7 | 0.021 | 0.483 |
| brain2voice_t15_day181.h5 | t15_day181 (half B vs half A) | 0 | 32227.59 | 0.0001 | 0.7 | 0.003 | 0.676 |
| brain2voice_t15_day181.h5 | t15_day186 | 5 | 433510.77 | 0.0025 | 9.3 | 0.035 | 0.582 |
| brain2voice_t15_day181.h5 | t15_day188 | 7 | 431254.16 | 0.0028 | 9.4 | 0.040 | 0.530 |
| brain2voice_t15_day181.h5 | t15_day193 | 12 | 644269.32 | 0.0035 | 14.7 | 0.045 | 0.521 |
| brain2voice_t15_day181.h5 | t15_day195 | 14 | 505534.29 | 0.0025 | 9.3 | 0.031 | 0.528 |
| brain2voice_t15_day188.h5 | t15_day188 (half B vs half A) | 0 | 68096.39 | 0.0001 | 0.7 | 0.006 | 0.700 |
| brain2voice_t15_day188.h5 | t15_day193 | 5 | 837247.67 | 0.0031 | 12.1 | 0.018 | 0.590 |
| brain2voice_t15_day188.h5 | t15_day195 | 7 | 647112.26 | 0.0027 | 8.2 | 0.016 | 0.595 |

Spearman vs feature r over held-out pairs (n = 20):

- z_frechet: rho +0.20 (p 0.41); given the gap: rho +0.06 (p 0.81)
- z_mmd2: rho -0.48 (p 0.031); given the gap: rho +0.04 (p 0.87)
- raw_frechet: rho -0.62 (p 0.0039); given the gap: rho -0.08 (p 0.74)
- raw_mean_shift: rho -0.46 (p 0.039)
- gap_days: rho -0.90 (p 4.7e-08)
