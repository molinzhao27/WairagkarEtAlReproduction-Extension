# Speed-normalized ASR `day195_dev_tests` (D-073): pooled WER, 147 day195 test trials

Fixed factor 3.25 (their target audio / Alex, median of 158 training trials).

| clips | original | fixed speed-up | per-trial speed-up |
|---|---|---|---|
| alex_tts/alex_tts | 0.7% | – | – |
| target_audio/target_audio | 0.3% | 0.2% | 0.2% |
| voice_cost_day195/zero_shot | 81.4% | 74.8% | 73.9% |
| voice_cost_day195/trained_theirs | 74.0% | 68.0% | 70.7% |
| voice_cost_day195/trained_tts_oracle | 95.7% | 89.7% | 87.1% |
| voice_cost_day195/trained_tts_auto | 90.0% | 89.1% | 89.0% |
| voice_conv_day195/trained_conv_oracle | 94.5% | 94.8% | 92.3% |
| voice_conv_day195/trained_conv_auto | 99.6% | 95.0% | 96.7% |
| splice_day195/trained_splice_oracle | 79.0% | 72.3% | 72.0% |
| splice_day195/trained_splice_auto | 80.7% | 79.0% | 78.4% |
| knnvc_day195/trained_tts_oracle | 88.5% | 85.6% | 84.6% |
| knnvc_day195/trained_tts_auto | 93.3% | 89.0% | 90.7% |
| f5clone_day195/trained_tts_oracle | 88.4% | 81.5% | 81.1% |
| f5clone_day195/trained_tts_auto | 88.5% | 84.8% | 84.5% |
| splice_manual_day195/trained_splice_manual | 83.8% | 76.0% | 76.5% |

| model vs its run's zero-shot | original Δ (p) | fixed Δ (p) | per-trial Δ (p) |
|---|---|---|---|
| voice_cost_day195/trained_theirs | -6.9 (0.039) | -5.8 (0.047) | -3.4 (0.24) |
| voice_cost_day195/trained_tts_oracle | +14.6 (4.9e-06) | +15.6 (2.2e-06) | +13.4 (1.2e-06) |
| voice_cost_day195/trained_tts_auto | +8.8 (0.0023) | +15.6 (2.2e-07) | +14.7 (1.2e-07) |
| voice_conv_day195/trained_conv_oracle | +12.7 (2e-06) | +21.1 (2.2e-09) | +17.0 (1e-08) |
| voice_conv_day195/trained_conv_auto | +18.1 (1.1e-06) | +21.0 (2.9e-11) | +21.7 (5.9e-11) |
| splice_day195/trained_splice_oracle | -2.2 (0.62) | -0.9 (0.91) | -2.1 (0.69) |
| splice_day195/trained_splice_auto | -0.9 (0.88) | +5.7 (0.04) | +3.5 (0.062) |
| knnvc_day195/trained_tts_oracle | +6.6 (0.033) | +11.7 (4.2e-05) | +10.6 (9.9e-05) |
| knnvc_day195/trained_tts_auto | +10.8 (0.00091) | +15.0 (1e-06) | +16.6 (4.3e-07) |
| f5clone_day195/trained_tts_oracle | +6.3 (0.092) | +7.4 (0.0095) | +6.4 (0.017) |
| f5clone_day195/trained_tts_auto | +5.6 (0.035) | +9.7 (0.00064) | +9.4 (0.00013) |
| splice_manual_day195/trained_splice_manual | +2.3 (0.45) | +1.8 (0.62) | +1.0 (0.93) |
