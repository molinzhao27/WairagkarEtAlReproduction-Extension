# Same-day neural matching, imagined -> attempted (pre-registered, D-032)

Does an imagined trial's neural activity resemble the attempted trial of the same sentence (same day, paired blocks) more than attempted trials of other sentences? No decoder, no audio targets, no drift.

- Representation: 5-bin mean pooling, PCA to 32 components fitted on the other three pairs (standardized: True).
- Primary: DTW (cost cosine, steps [[1, 1], [0, 1], [1, 0]], band none), mean Pearson r over components; own attempted trial ranked among itself + 10 attempted trials closest in cue TTS speech duration (speech_span).
- One-sided p: 10000 seeded relabelings (seed 42); exact p by convolution as a check. Ranks: 1 = best, ties mid-ranked; top-1 = rank exactly 1.

**Deviations from the pre-registration:** none.

**Additions to the pre-registration (reported next to it, not replacing it):** block-normalized (added 2026-10-21 before any run): Length-only baseline on durations divided by the median go-window duration of the trial's own block: -|log((attempted / median of its block) / (imagined / median of its block))|; same rank statistics (length-matched and all). Imagined trials are ~1.5x longer than attempted ones, so the raw ratio favours long attempted trials rather than the proportionally matched one. (switch: additional_baselines.length_block_normalized)

Table cells: mean rank (chance), median rank, top-1 rate (chance), p_perm, p_exact.

## PRIMARY: DTW trajectory correlation (PCA, primary), candidates: length-matched

| Trials | n | Mean rank (chance) | Median | Top-1 (chance) | p (perm) | p (exact) |
|---|---|---|---|---|---|---|
| pooled | 200 | 2.48 (6.00) | 1.0 | 0.545 (0.091) | 0.0001 | 1.8e-65 |
| A03-I04 | 50 | 2.06 (6.00) | 1.0 | 0.560 (0.091) | 0.0001 | 6.4e-23 |
| A06-I07 | 50 | 3.32 (6.00) | 2.0 | 0.420 (0.091) | 0.0001 | 2.8e-10 |
| A11-I09 | 50 | 2.94 (6.00) | 1.0 | 0.540 (0.091) | 0.0001 | 3.2e-13 |
| A13-I12 | 50 | 1.60 (6.00) | 1.0 | 0.660 (0.091) | 0.0001 | 7.6e-31 |

Descriptive: own similarity minus the other candidates' mean = 0.0530 (own higher in 88% of 200 trials; Wilcoxon p 1.9e-29).

## DTW trajectory correlation (PCA, primary), candidates: all

| Trials | n | Mean rank (chance) | Median | Top-1 (chance) | p (perm) | p (exact) |
|---|---|---|---|---|---|---|
| pooled | 200 | 8.20 (25.50) | 2.0 | 0.405 (0.020) | 0.0001 | 4e-78 |
| A03-I04 | 50 | 6.76 (25.50) | 2.0 | 0.400 (0.020) | 0.0001 | 2.3e-25 |
| A06-I07 | 50 | 12.72 (25.50) | 7.5 | 0.260 (0.020) | 0.0001 | 3.4e-11 |
| A11-I09 | 50 | 9.52 (25.50) | 2.5 | 0.420 (0.020) | 0.0001 | 1.7e-17 |
| A13-I12 | 50 | 3.78 (25.50) | 1.0 | 0.540 (0.020) | 0.0001 | 2e-39 |

Descriptive: own similarity minus the other candidates' mean = 0.0562 (own higher in 88% of 200 trials; Wilcoxon p 3.4e-30).

## time-averaged pattern correlation (512-dim, secondary), candidates: length-matched

| Trials | n | Mean rank (chance) | Median | Top-1 (chance) | p (perm) | p (exact) |
|---|---|---|---|---|---|---|
| pooled | 200 | 4.08 (6.00) | 3.0 | 0.345 (0.091) | 0.0001 | 9.3e-19 |
| A03-I04 | 50 | 4.42 (6.00) | 3.5 | 0.280 (0.091) | 0.0004 | 0.0002 |
| A06-I07 | 50 | 5.12 (6.00) | 4.5 | 0.200 (0.091) | 0.025 | 0.026 |
| A11-I09 | 50 | 3.62 (6.00) | 2.0 | 0.400 (0.091) | 0.0001 | 2.5e-08 |
| A13-I12 | 50 | 3.14 (6.00) | 1.5 | 0.500 (0.091) | 0.0001 | 1.3e-11 |

Descriptive: own similarity minus the other candidates' mean = 0.0486 (own higher in 74% of 200 trials; Wilcoxon p 8.4e-11).

## time-averaged pattern correlation (512-dim, secondary), candidates: all

| Trials | n | Mean rank (chance) | Median | Top-1 (chance) | p (perm) | p (exact) |
|---|---|---|---|---|---|---|
| pooled | 200 | 16.36 (25.50) | 10.0 | 0.140 (0.020) | 0.0001 | 2.9e-20 |
| A03-I04 | 50 | 17.70 (25.50) | 13.0 | 0.100 (0.020) | 0.0003 | 5.6e-05 |
| A06-I07 | 50 | 20.92 (25.50) | 22.0 | 0.060 (0.020) | 0.013 | 0.012 |
| A11-I09 | 50 | 14.98 (25.50) | 6.5 | 0.160 (0.020) | 0.0001 | 6.3e-08 |
| A13-I12 | 50 | 11.82 (25.50) | 3.0 | 0.240 (0.020) | 0.0001 | 9.6e-13 |

Descriptive: own similarity minus the other candidates' mean = 0.0512 (own higher in 72% of 200 trials; Wilcoxon p 1.9e-10).

## length-only baseline, -|log duration ratio|, raw (pre-registered), candidates: length-matched

| Trials | n | Mean rank (chance) | Median | Top-1 (chance) | p (perm) | p (exact) |
|---|---|---|---|---|---|---|
| pooled | 200 | 5.91 (6.00) | 6.0 | 0.075 (0.090) | 0.34 | 0.35 |
| A03-I04 | 50 | 6.00 (6.00) | 6.0 | 0.060 (0.091) | 0.51 | 0.5 |
| A06-I07 | 50 | 5.86 (6.00) | 6.0 | 0.100 (0.089) | 0.38 | 0.38 |
| A11-I09 | 50 | 5.94 (6.00) | 6.0 | 0.060 (0.091) | 0.45 | 0.46 |
| A13-I12 | 50 | 5.84 (6.00) | 6.0 | 0.080 (0.091) | 0.37 | 0.37 |

Descriptive: own similarity minus the other candidates' mean = 0.0037 (own higher in 52% of 200 trials; Wilcoxon p 0.69).

## length-only baseline, -|log duration ratio|, raw (pre-registered), candidates: all

| Trials | n | Mean rank (chance) | Median | Top-1 (chance) | p (perm) | p (exact) |
|---|---|---|---|---|---|---|
| pooled | 200 | 24.89 (25.50) | 25.5 | 0.005 (0.020) | 0.28 | 0.28 |
| A03-I04 | 50 | 23.62 (25.50) | 21.5 | 0.020 (0.020) | 0.19 | 0.18 |
| A06-I07 | 50 | 24.80 (25.50) | 26.5 | 0.000 (0.020) | 0.36 | 0.37 |
| A11-I09 | 50 | 26.48 (25.50) | 26.0 | 0.000 (0.020) | 0.68 | 0.68 |
| A13-I12 | 50 | 24.68 (25.50) | 25.5 | 0.000 (0.020) | 0.34 | 0.35 |

Descriptive: own similarity minus the other candidates' mean = 0.0077 (own higher in 52% of 200 trials; Wilcoxon p 0.56).

## length-only baseline, -|log duration ratio|, block-normalized (added 2026-10-21 before any run), candidates: length-matched

| Trials | n | Mean rank (chance) | Median | Top-1 (chance) | p (perm) | p (exact) |
|---|---|---|---|---|---|---|
| pooled | 200 | 5.51 (6.00) | 5.0 | 0.115 (0.090) | 0.017 | 0.014 |
| A03-I04 | 50 | 5.92 (6.00) | 5.0 | 0.060 (0.091) | 0.44 | 0.43 |
| A06-I07 | 50 | 5.84 (6.00) | 6.0 | 0.140 (0.091) | 0.37 | 0.36 |
| A11-I09 | 50 | 5.58 (6.00) | 5.5 | 0.120 (0.091) | 0.17 | 0.18 |
| A13-I12 | 50 | 4.70 (6.00) | 4.5 | 0.140 (0.089) | 0.0015 | 0.0018 |

Descriptive: own similarity minus the other candidates' mean = 0.0183 (own higher in 64% of 200 trials; Wilcoxon p 0.00094).

## length-only baseline, -|log duration ratio|, block-normalized (added 2026-10-21 before any run), candidates: all

| Trials | n | Mean rank (chance) | Median | Top-1 (chance) | p (perm) | p (exact) |
|---|---|---|---|---|---|---|
| pooled | 200 | 20.58 (25.50) | 17.5 | 0.020 (0.020) | 0.0001 | 6.3e-07 |
| A03-I04 | 50 | 22.74 (25.50) | 19.5 | 0.000 (0.019) | 0.092 | 0.089 |
| A06-I07 | 50 | 24.94 (25.50) | 26.5 | 0.040 (0.020) | 0.39 | 0.39 |
| A11-I09 | 50 | 20.10 (25.50) | 15.5 | 0.040 (0.020) | 0.0037 | 0.004 |
| A13-I12 | 50 | 14.54 (25.50) | 10.5 | 0.000 (0.020) | 0.0001 | 1.7e-08 |

Descriptive: own similarity minus the other candidates' mean = 0.0539 (own higher in 70% of 200 trials; Wilcoxon p 2e-06).

## PCA (fitted on the other pairs)

| Pair | Train trials | Train frames | Explained variance |
|---|---|---|---|
| A03-I04 | 300 | 62246 | 0.493 |
| A06-I07 | 300 | 60144 | 0.492 |
| A11-I09 | 300 | 61837 | 0.481 |
| A13-I12 | 300 | 66237 | 0.478 |

Cue TTS speech duration (speech_span, voice Aman (English (India))): mean 1.08 s, range 0.52-1.98 s.

## Limitations

- No within-mode positive control: no sentence repeats within a condition on this day, so the analysis has no same-sentence ceiling (how well a trial matches another trial of the same sentence in the same mode is unknown).
- At chance is NOT evidence of no shared code: single-trial matching may simply be too noisy.
- Direction imagined -> attempted only; trials are matched within a block pair (same day, same arrays).

Interpretation rule: Fixed in advance (D-032): above chance = imagined and attempted share sentence-specific neural structure on this day; at chance = not evidence of no shared code (no positive control).

Code 34145ae770f31bbf86bffa65777bb61d9bdf85aa (dirty: False), config sha256 61e7833a417e.
