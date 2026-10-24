# Voice check `synth_transfer_day195` (D-074 A1): zero-shot output on 158 day195 training trials

Alex targets (their true syllables) vs their targets: r 0.514.

| decoder | r vs their targets | r vs Alex targets | Alex − theirs |
|---|---|---|---|
| voice_transfer | 0.412 | 0.412 | -0.000 |
| theirs_day165 | 0.440 | 0.353 | -0.087 |

| paired (voice_transfer − theirs_day165) | mean diff | p (Wilcoxon) |
|---|---|---|
| shift_new_vs_theirs | +0.086 | 3e-26 |
| r_vs_alex_new_vs_theirs | +0.059 | 1.1e-21 |
| r_vs_theirs_new_vs_theirs | -0.028 | 5.6e-12 |
