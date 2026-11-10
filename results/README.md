# Results index

Small result files of every run: JSON with per-trial values and a Markdown summary beside most of
them. No audio, neural data or model weights are here. The [main README](../README.md) has the
summary tables and the limits.

Experiments carry the IDs of the project's lab notebook (`D-075`, `D-097`, ...). The notebook itself
is not published; this index says what each ID tested and how it came out. "Rule" means the success
rule written down before the run.

## First-generation decoder

| ID | Question | Folder | Outcome |
|---|---|---|---|
| D-022, D-026 | Does offline inference reproduce their live output? | `reproduce_inference_t15_day195/` | Yes: mel correlation 0.958 with their live audio |
| D-027, D-037 | Can syllable timing be recovered for target audio? | `alignment_validation/`, `syllable_benchmark/` | Partly: target correlation 0.32 against an upper bound of 0.46 |
| D-041, D-044 | Syllable timing marked by LLM annotators | `manual_alignment/` | Better: 0.40; adopted for the targets |
| D-029, D-030 | Their decoder on the inner-speech day, pilot and no adaptation | `kunz_pilot/`, `kunz_zeroshot/` | Attempted speech above inner speech in 4 of 4 block pairs; no inner-speech content detected |
| D-035 | Does an inner-speech trial match its attempted sentence in the raw neural data? | `neural_match/` | Yes: top-1 of 50, 40% |
| D-034, D-036, D-039 | Adaptation recipe on a day with real targets | `sanity_train_day195*/` | Frozen base, 15 epochs: WER 82% → 74.5% |
| D-046, D-048 | Main runs on the inner-speech day | `kunz_v3_manual/`, `kunz_v4_astra/`, `cue_identification/` | Attempted top-1 17.5% → 27%; inner speech marginal |
| D-050 | What does a text-to-speech target voice cost? | `voice_cost_day195/` | WER 95.7% vs 74.0% with real targets |
| D-057, D-060, D-062 | Other target voices: linear conversion, spliced words, kNN-VC, F5 clone | `voice_conv_day195/`, `splice_day195/`, `splice_manual_day195/`, `knnvc_day195/`, `f5clone_day195/`, `voice_search_day195_comparison.json` | None passes its rule |
| D-063 to D-066 | Aligning days without targets | `drift_check/`, `word_align_check/`, `align_check/` | All fail their checks |
| D-067 to D-070 | Converting inner speech toward attempted speech before this decoder | `inner_projector/`, `kunz_inner_*/`, `kunz_inner_compare.txt` | Helps matching to attempted trials; no gain against clean reference audio |
| D-071 | How far has each session drifted? | `drift_map/` | The inner-speech day is further off than any gap in the training data |
| D-073 | Does speeding the audio up help the recognizer? | `speed_asr/` | Up to 8 points of WER; changes no conclusion |
| D-074 | A decoder trained toward a synthetic voice from scratch | `synth_transfer_*/`, `voice_check/` | Removes the harm of synthetic targets; no gain |

## Second-generation decoder (`b2v2/`)

Per-run folders (`b2v2/b2v2_*`) hold `split.json`, `log.jsonl`, `summary.json` and score files. The
folders below hold the analyses.

| ID | Question | Folder | Outcome |
|---|---|---|---|
| D-075 | Does the rebuild work on their benchmark? Does text-only day adaptation work? | `b2v2/b2v2_all/`, `b2v2/b2v2_le165*/` | 800 ms windows: WER 74.7% (rule failed). Long context: 31.8% vs 48.7% for their live audio. Text-only adaptation: PER better, WER not |
| D-076 | Is T15's inner speech identifiable? | `b2v2/identify/` (`z_*`, `f_*`, `checks.md`) | Yes: top-1 of 50, 31.5%; attempted 100% |
| D-077 | Inner-to-attempted transforms in front of the decoder | `b2v2/project/`, `b2v2/identify/f_imagined_*` | Identification up to 55.5%, voice unchanged; a text-trained input does both (74.5%, WER 82.7%) |
| D-078 | Denoise, pitch, sharpen | `b2v2/postproc/` | No variant passes its rule |
| D-079, D-080 | Other behaviors: voice decoder and per-behavior classifiers | `b2v2/iso/` (`aall.md`, `classifiers.md`, `contamination.md`) | The decoder reads production only; classifiers read every behavior, listening at 80% |
| D-081, D-082 | Open-vocabulary inner speech and silent counting, 11–12 months later | `b2v2/b2v2_ov_*`, `b2v2/count/` | Negative: the decoder outputs almost nothing |
| D-083, D-085, D-087 | Adapting the decoder to heard words | `b2v2/iso/d083_*`, `d085_*`, `d087_*` | Phoneme output 35.7% to 47.1% of 7; the voice says none |
| D-086 | Six ways to make the inner-speech voice clearer; adversarial base | `b2v2/d086/`, `b2v2/b2v2_all_gan/` | Self-distillation passes (WER 82.7% → 76.2%); adversarial base: benchmark 31.8% → 22.0% |
| D-088 | Confidence-gated volume | `b2v2/d088/` | Rule not met |
| D-089 | Hidden-state distillation, averaging three decoders | `b2v2/d089/` | No gain; averaging is worse |
| D-091, D-092 | Pooling T15's inner speech across four days | `b2v2/d091*/`, `b2v2/d092*/` | No gain, or worse |
| D-093, D-094 | Adapter vs training length; how long to train | `b2v2/d093/`, `b2v2/d094/` | 3,000 steps is the gain (PER 51.9% → 48.6%); longer does not help |
| D-095 | T15's decoder for T12 and T16 | `b2v2/d095/` | Every rule met; a random decoder stays at chance |
| D-096 | Strengthening that adaptation | `b2v2/d096/` | Training the first layer: top-1 of 40, 19.7% → 40.6%; more data: nothing |
| D-097 | The same change for T15 | `b2v2/d097/` | PER 48.6% → 45.1% (final recipe) |
| D-098 | Training two or four layers | `b2v2/d098/`, `b2v2/d098_t15/` | Rule not met |
| D-099 | A voice target on the final recipe | `b2v2/d099/` | WER 73.3% → 74.3%, rule not met |
| D-200 | A causal post-network from phonemes to voice | `b2v2/d200/`, `b2v2/b2v2_postnet/` | WER 73.3% → 73.9%, rule not met |

## Other folders

- `parity/`, `kunz_v3_manual_gpu_parity/`: checks that a second machine reproduces the first.
- `perf/`: how much repeated runs differ on one machine and across machines.
- `data_inspection/`: shapes and value summaries of the released files.
- `time_training/`: timing of the first training runs.
