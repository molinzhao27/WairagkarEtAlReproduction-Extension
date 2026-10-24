# Speed-normalized ASR `day195_dev_tests_validate` (D-073): pooled WER, 147 day195 test trials

Fixed factor 3.25 (their target audio / Alex, median of 158 training trials).

| clips | original | fixed speed-up | per-trial speed-up |
|---|---|---|---|
| alex_tts/alex_tts | 0.7% | – | – |
| target_audio/target_audio | 0.3% | 0.2% | 0.2% |
| voice_cost_day195/zero_shot | 81.4% | 74.8% | 73.9% |
