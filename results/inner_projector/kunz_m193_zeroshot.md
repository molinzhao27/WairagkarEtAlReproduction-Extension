# Inner -> attempted conversion `kunz_m193_zeroshot` (D-067): 200 held-out imagined trials

Attempted decodes' speech fraction: 0.431

| transform | mean own rank / 50 | Δ vs none (p) | top-1 | top-5 | speech fraction |
|---|---|---|---|---|---|
| none | 21.84 | – | 7.0% | 19.5% | 0.083 |
| gain | 15.95 | -5.89 (8e-08) | 14.0% | 32.5% | 0.215 |
| projector | 19.04 | -2.81 (0.004) | 11.5% | 24.5% | 0.161 |
| paired | 14.81 | -7.03 (1.6e-09) | 10.5% | 35.5% | 0.250 |
| paired_shuffled | 16.62 | -5.21 (1.9e-06) | 8.0% | 28.5% | 0.241 |

Post hoc (not pre-registered) length control: rank among the own + 10 attempted trials closest in length to the own attempted trial (chance 6.0); length-only baseline over all 50: mean rank 24.73, top-1 0.5%

| transform | length-matched mean rank | Δ vs none (p) | length-matched top-1 |
|---|---|---|---|
| none | 5.20 | – | 20.5% |
| gain | 4.07 | -1.13 (6.8e-06) | 30.0% |
| projector | 4.72 | -0.47 (0.039) | 23.5% |
| paired | 3.94 | -1.25 (3.5e-07) | 29.0% |
| paired_shuffled | 4.37 | -0.82 (0.00021) | 25.5% |

paired vs paired_shuffled, length-matched: -0.43 (p 0.004)

paired vs paired_shuffled: rank -1.81 (p 0.0053)

Per fold (projector scale; share of the attempted − imagined mean difference inside the subspace):

- A3/I4: scale 1.09, mean-difference share 0.97, paired canonical r [0.84, 0.69, 0.56, 0.51, 0.46], shuffled [0.81, 0.58, 0.46, 0.35, 0.29]
- A6/I7: scale 1.06, mean-difference share 0.98, paired canonical r [0.87, 0.72, 0.57, 0.55, 0.49], shuffled [0.84, 0.61, 0.49, 0.36, 0.36]
- A11/I9: scale 1.17, mean-difference share 0.98, paired canonical r [0.86, 0.71, 0.53, 0.52, 0.45], shuffled [0.83, 0.62, 0.49, 0.34, 0.32]
- A13/I12: scale 1.19, mean-difference share 0.98, paired canonical r [0.84, 0.68, 0.51, 0.49, 0.45], shuffled [0.82, 0.61, 0.46, 0.37, 0.29]
