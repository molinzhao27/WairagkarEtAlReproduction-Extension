# Alignment check `m165_adversarial` (D-063): feature r on held-out halves

| session | zero_shot | adv_id0 | adv_id1 |
|---|---|---|---|
| t15_day165 undrifted (identity check) | 0.674 | 0.660 (-1.4, p 3.5e-06) | 0.659 (-1.5, p 5.3e-07) |
| t15_day165 + synthetic drift | 0.627 | 0.640 (+1.3, p 0.0044) | 0.639 (+1.2, p 0.0086) |
| t15_day172 | 0.509 | 0.497 (-1.1, p 0.019) | 0.498 (-1.1, p 0.032) |
| t15_day179 | 0.586 | 0.532 (-5.4, p 6e-11) | 0.535 (-5.2, p 4.5e-11) |
| t15_day186 | 0.482 | 0.468 (-1.4, p 0.037) | 0.467 (-1.5, p 0.053) |
| t15_day193 | 0.460 | 0.457 (-0.2, p 0.71) | 0.464 (+0.4, p 0.65) |
