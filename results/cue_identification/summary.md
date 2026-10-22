# Own-cue identification on Wairagkar data (Kunz scoring)

Trials: `Training data (50-word vocab)`. Each decoded output vs the TTS of every distinct cue of the set; DTW mel_corr; rank of the own cue (1 = best); excess over the 10 length-matched cues. Same functions as the Kunz runs (scripts/cue_identification.py). The models were never trained on these sessions. Every trial here has its own sentence, so the last two columns rescale to 50 candidates, as in a Kunz block (expected over random draws of the other 49).

| cell | decoder age (days) | n | cues | top-1 (chance) | median rank (p) | excess length-matched (p) | top-1 among 50 (chance 0.02) | median rank / 50 |
|---|---|---|---|---|---|---|---|---|
| day195_m193 | 2 | 146 | 146 | 0.260 (0.007) | 4.0 (5.0e-91) | 0.106 (1.6e-25) | 0.429 | 2.0 |
| day195_m172 | 23 | 146 | 146 | 0.226 (0.007) | 7.0 (8.9e-79) | 0.088 (2.9e-25) | 0.340 | 3.0 |
| day195_m165 | 30 | 146 | 146 | 0.199 (0.007) | 9.0 (7.0e-59) | 0.075 (3.6e-23) | 0.285 | 3.7 |
