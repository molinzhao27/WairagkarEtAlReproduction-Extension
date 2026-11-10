# Brain-to-voice: reproducing Wairagkar et al. 2025 and extending it to inner speech

An offline reproduction of the brain-to-voice decoder of Wairagkar et al. 2025 (*Nature*) from their
public code and data, followed by an attempt to make the same kind of decoder speak **inner speech**
(silent, imagined speech), using the public data of Kunz et al. 2025 (*Cell*).

**Status: finished, November 2026.** No further work is planned. This is independent, exploratory work
on public datasets. It is not peer reviewed, most results come from one recording session per
participant, and nothing here was run live with a participant. Read [Limits](#limits) before quoting a
number.

## Short version

- **The reproduction works.** Their released model, run offline through our pipeline, gives the same
  audio as their decoder produced live.
- **Their first-generation decoder could not be moved to the inner-speech day.** It only adapts to a
  new day with real voice targets, and the inner-speech dataset has none. Five kinds of synthetic
  target were tried; none gave a reliable gain.
- **A rebuild of their second-generation decoder, which can adapt from text alone, works well on
  attempted speech.** On their benchmark trials a speech recognizer gets 22.0% of the words wrong on
  our decoder's audio, against 48.7% on the audio their first decoder produced live.
- **Inner speech: the sentence can be read, the voice cannot be understood.** From T15's inner speech
  the decoder picks the right sentence out of 50 in 85% of trials (chance 2%), on sentences it was
  never adapted on. The synthesized voice of the same trials is still mostly unintelligible (73% of
  words wrong).
- **The limit is the signal, not the voice model.** T15's inner speech carries the attempted-speech
  code at about a third of its strength. Fixes aimed at the voice itself gave small gains at best,
  and none on the final recipe.
- **The decoder carries over to two other people.** With text only and a small adapted part, T15's
  decoder identifies inner-speech sentences of T12 (52.5% of 40) and T16 (20.8% of 40; chance 2.5%).
  Their voices are not intelligible.
- **Heard and read words are in the same arrays, in a different code.** A simple classifier reads
  which of seven words T15 is hearing in 80% of trials. The voice decoder cannot say them.

Terms used below:

- **WER**: word error rate of an automatic speech recognizer (Whisper `small.en`) transcribing the
  decoder's synthesized audio, pooled over trials. Lower is better; it can exceed 100%.
- **PER**: phoneme error rate of the decoder's phoneme output against the cued sentence.
- **Top-1 of N**: how often the cued sentence ranks first among the N candidate sentences of its
  block, by the likelihood of each under the decoder's phoneme output.
- **T15, T12, T16**: BrainGate2 participants, as named in the two papers.

## Results

Every number below is in `results/`; [results/README.md](results/README.md) maps each experiment to
its folder. From the same-day matching test onward, each experiment's success rule was written down
before it ran, and results are reported against that rule, including the failures.

### 1. Reproduction (first-generation decoder)

Their released day-193 model, run offline on all 305 trials of session `t15_day195`.

| Check | Result |
|---|---|
| Our offline audio vs the audio their decoder produced live (130 evaluation trials) | mel correlation 0.958 |
| Against the target voice: ours / their live audio | 0.777 / 0.782 |
| Their electrode reordering on / off (feature correlation with the target) | 0.607 / −0.005 |

The paper reports 0.83 against the target. Their own live audio scores 0.78 under our metric, so the
gap comes from the metric or the reference audio, not from inference. It was not tuned away.

### 2. Moving the first-generation decoder to a new day

Test setting: a 30-day-old decoder on session `t15_day195` (147 test trials).

| Adaptation targets | WER |
|---|---|
| None (30-day-old decoder as is) | 81.4% |
| Their real voice targets | 74.0% (p 0.039) |
| Text-to-speech voice | 95.7% |
| Words spliced from their own recordings | 79.0% (p 0.62) |
| Voice clone (F5-TTS) | 88.4% |
| Voice conversion (kNN-VC) | 88.5% |
| Linear voice conversion | 94.5% |
| For scale: a 2-day-old decoder | 57.8% |

The synthetic rows use the true word timing, which the inner-speech data does not have; with guessed
timing none of them beats leaving the decoder alone either. Only real voice targets give a reliable
gain. Spliced words do no harm,
and every other voice makes the decoder worse than leaving it alone. Four ways of aligning the neural
data without any targets also failed. On the inner-speech day this decoder showed sentence content in
attempted speech (top-1 of 50: 17.5%, 27% after adaptation) and only a marginal trace in inner
speech.

One result from this phase needs no decoder: an inner-speech trial's raw neural activity identifies
the attempted trial of the same sentence (top-1 of 50: 40%). The information is in the recording.

### 3. Rebuild of the second-generation decoder, attempted speech

A PyTorch rebuild of the lab's follow-up decoder (causal Transformer with a voice head and a phoneme
head; preprint, Wairagkar et al. 2026). The phoneme head trains from text alone, which is what the
inner-speech data has. Benchmark: their 127 held-out trials, same recognizer for every row.

| Audio | WER | PER |
|---|---|---|
| Their first-generation decoder, recorded live | 48.7% | – |
| Rebuild, 800 ms windows (the planned live setting) | 74.7% | 59.2% |
| Rebuild, 3.2–6.4 s of past context | 31.8% | 10.0% |
| + adversarial training | 22.0% | 10.1% |

The rebuild failed its planned test with 800 ms windows and works with longer past context. It is
still causal (no future data), but that is not the setting the paper runs live. The rebuild leaves
out the paper's token and self-supervised heads; the preprint reports 8.04% WER (Whisper) for the
full model on its benchmark.

Across days, the rebuild trained on older sessions holds up far better than the first decoder
(30-day gap: 53.3% vs 81.4%). Adapting it from text alone improves the phonemes (PER 29.3% → 25.6%,
p 0.0005) but not reliably the voice (WER 53.3% → 50.9%, p 0.67).

### 4. Inner speech, T15

Kunz et al. session 2024.03.03, three to six weeks after the decoder's last training session. 200
inner-speech trials in four blocks. Each block is tested with a decoder adapted on other blocks, which
hold other sentences.

| Decoder | Top-1 of 50 | PER | WER |
|---|---|---|---|
| No adaptation | 9.5% | 87.9% | 94.6% |
| Input layer adapted on attempted speech (text only) | 31.5% | 77.1% | 90.1% |
| Input layer adapted on inner speech (text only) | 74.5% | 59.5% | 82.7% |
| + adversarial base, augmentation, more inner trials | 78.0% | 51.9% | 79.9% |
| + 3,000 training steps instead of 1,000 | 85.0% | 48.6% | – |
| + first decoder layer trained too | **85.2%** | **45.1%** | **73.3%** |
| For scale: attempted speech, same day, held-out blocks | 100% | 10.9% | 31.9% |

Chance for top-1 is 2%. Checks on the identification, run on the first three rows: it holds among
candidate sentences with the same number of words, falls to chance when the phoneme output is
reversed in time, and holds on sentences that appear nowhere in the decoder's training data.

Why the voice stays unclear:

- The inner-speech voice has one burst of sound per word at about the right time, but inside a word
  it holds a flat, averaged sound where attempted speech moves. Multi-syllable words collapse.
- The phoneme output gets 44% of inner-speech words fully right, against 83% for attempted speech.
  Where it has the word, the recognizer hears the right word about two times in three.
- Templates built from attempted speech recognize T15's inner-speech words exactly as well as
  templates built from inner speech itself (14.0% vs 14.0%; attempted on attempted 34.3%). Inner
  speech here is the attempted-speech code, weaker and noisier.
- Adversarial training makes the voice commit to a sound. Missing words halve (25% → 13%), but the
  words it has wrong now come out as clear wrong syllables, so WER barely moves.

### 5. Other participants

T12 (2023.11.28) and T16 (2024.04.29), one session each. T15's decoder stays frozen except a new
input layer and, in the last row, its first Transformer layer. Text only; neither has voice
recordings. Top-1 of 40 candidate sentences, chance 2.5%.

| | T12 | T16 |
|---|---|---|
| Attempted speech, T15's decoder | 11.5% | 55.0% |
| Attempted speech, same model with random weights | 2.5% | 2.5% |
| Inner speech, input layer adapted | 24.0% | 12.5% |
| Inner speech, first decoder layer trained too | 52.5% | 20.8% |

The random-weights control stays at chance, so the transfer comes from T15's training. PER stays
high (T12 66.6%, T16 74.9%) and the voice is unintelligible for both (WER 94–103%).

### 6. Other speech behaviors

Kunz et al. isolated-word session, T15, 2024.04.07. Seven words, chance 14.3%.

| Behavior | Voice decoder, adapted on attempted speech | Classifier trained on that behavior |
|---|---|---|
| Attempted | 98.6% | 92.1% |
| Mouthed | 87.1% | 86.4% |
| Imagined speaking | 17.1% | 31.9% |
| Imagined hearing own voice | 13.3% | 25.7% |
| Imagined listening | 18.1% | 40.5% |
| Passive listening | 13.6% | 80.0% |
| Silent reading | 12.9% | 30.0% |

The voice decoder reads speech production. Hearing a word is in the arrays in a different code: a
classifier trained on listening also reads imagined listening, where no audio plays, which sound
leaking into the recording could not produce. Adapting the decoder's input on listening trials lets
its phoneme output tell heard words apart (35.7%; 47.1% for a variant chosen on the test trials), but
the voice says none of them.

### 7. What did not work

| Tried | Outcome |
|---|---|
| Synthetic voice targets for the first decoder (five kinds) | No reliable gain; four of five do harm |
| Aligning days without targets (CORAL, MMD, same-word, adversarial) | Does not undo real drift |
| Rotating and scaling inner speech to look like attempted speech | Top-1 31.5% → 55.5%; voice unchanged (WER 90–96%) |
| Denoising, pitch shift, sharpening of the decoded audio | No WER gain |
| Lowering the volume where the decoder is unsure | No WER gain |
| Matching hidden states to attempted speech; averaging three decoders | No PER gain; averaging is worse |
| Pooling T15's inner speech from four days | No gain, or worse |
| Training longer than 3,000 steps; training two or four layers | No gain |
| Training toward the decoder's own attempted voice | WER 82.7% → 76.2% on an early recipe; 73.3% → 74.3% on the final one |
| A causal post-network from phonemes to voice | WER 73.3% → 73.9% |
| Open-vocabulary inner speech, 11–12 months after training | WER about 100%, PER 97% |
| Silent counting on the same late day | Not readable; spoken counts were not readable either |

## Limits

- **Offline only.** Nothing was run in closed loop. The second-generation rebuild needs seconds of
  past context and fails with the 800 ms windows planned for live use.
- **Closed sets.** Identification is among 40–50 known sentences from a 50-word vocabulary. It is not
  open decoding: the best inner-speech PER is 45%.
- **One session per result.** T15's inner speech is one day; T12 and T16 are one day each.
- **The same 200 test trials were used throughout.** The later T15 recipe choices (3,000 steps, the
  first layer) were made after seeing earlier results on those trials, so the final numbers are
  somewhat optimistic.
- **A recognizer, not listeners.** WER comes from Whisper `small.en`, which has a language prior. No
  blind human listening test was run.
- **Sentence overlap.** 155 of the 250 inner-speech sentences also occur in the attempted-speech
  training cues. The effects hold on the trials whose sentences do not.
- **Inner speech may include faint attempted movement.** T15 has ALS with severe dysarthria, and the
  release has no audio or muscle recordings to rule that out.
- **A partial rebuild.** The second-generation paper's code was not public. Settings the paper does
  not state are ours (tagged `[OURS]` in the configs), and two of its heads are not built.
- **Few seeds.** One or two per experiment.

## Citations

- Wairagkar M, Card NS, Singer-Clark T, Hou X, Iacobacci C, Miller LM, Hochberg LR,
  Brandman DM, Stavisky SD. **An instantaneous voice-synthesis neuroprosthesis.** *Nature*
  (2025). doi:[10.1038/s41586-025-09127-3](https://doi.org/10.1038/s41586-025-09127-3).
  Code: <https://github.com/Neuroprosthetics-Lab/brain-to-voice-2025>.
  Data: Dryad doi:[10.5061/dryad.2280gb64f](https://doi.org/10.5061/dryad.2280gb64f).
- Kunz EM, et al. **Inner speech in motor cortex and implications for speech
  neuroprostheses.** *Cell* (2025).
  doi:[10.1016/j.cell.2025.06.015](https://doi.org/10.1016/j.cell.2025.06.015).
  Data: Dryad doi:[10.5061/dryad.gf1vhhn1j](https://doi.org/10.5061/dryad.gf1vhhn1j).
- Wairagkar M, et al. Second-generation brain-to-voice decoder. bioRxiv preprint (2026).
  doi:[10.64898/2026.06.30.735633](https://doi.org/10.64898/2026.06.30.735633). Rebuilt here from
  the paper's description.
- Valin J-M, Skoglund J. **LPCNet: Improving neural speech synthesis through linear
  prediction.** ICASSP 2019. Code: <https://github.com/xiph/LPCNet>.

## Third-party code

The Wairagkar et al. code is **not committed** to this repo. Their license
(academic, non-commercial UC license) only allows copies to go to academic or nonprofit
researchers. Instead, `scripts/fetch_third_party.py` clones their pinned commit
(`configs/third_party.yaml`) into `third_party/brain_to_voice/` and verifies every file
against `third_party/brain_to_voice.sha256`. Full provenance and license details are in
`third_party/PROVENANCE.md`; the license text is in
`third_party/brain_to_voice/LICENCE.md` (after fetching).

**Required acknowledgment** in any publication using this work:

> The Software used in this research was created by Maitreyee Wairagkar of The University of California, Davis. © 2025.

LPCNet (xiph) is committed in `third_party/LPCNet/` under a BSD-3-Clause license.
It is built from source by `scripts/build_lpcnet.sh`, which produces the
`lpcnet_demo` binary used for feature extraction and speech synthesis.

## Data

No data is in this repo. `data/` is gitignored, and so are audio and model weights. Dataset configs
and SHA-256 checksums are in `configs/data/kunz.yaml` and `configs/data/wairagkar.yaml`.

**Wairagkar et al. 2025.** `t15_neural_data.zip` (26.16 GB) and `t15_control_experiment.zip`
(711 MB). Dryad requires an API token for downloads. The default run is a dry run that checks sizes
and free disk:

```bash
python scripts/fetch_wairagkar_data.py
```

**Kunz et al. 2025.** Download `sentenceData.zip` from the Dryad record, then extract the T15
2024.03.03 files:

```bash
python -I scripts/extract_kunz_subset.py --zip /path/to/sentenceData.zip
```

The later experiments also read T12 and T16 sentence sessions and T15's later sessions from the same
archive, and `isolatedVerbalBehaviors.zip` from the same record.

## Setup

```bash
git clone git@github.com:molinzhao27/WairagkarEtAlReproduction-Extension.git
cd WairagkarEtAlReproduction-Extension
git config core.hooksPath .githooks
python scripts/fetch_third_party.py
pytest
```

The first-generation pipeline uses TensorFlow 2.21 with tf-keras.

```bash
# Mac (Apple silicon, CPU)
mamba env create -f environment.yml           # creates env "wvx"
mamba run -n wvx bash scripts/build_lpcnet.sh # builds third_party/LPCNet/lpcnet_demo
mamba run -n wvx python scripts/check_env.py  # forward pass, fit step, LPCNet smoke test

# Linux (CUDA)
python -m venv .venv
.venv/bin/pip install -r requirements-linux-cuda.txt
.venv/bin/pip install -e . --no-deps
bash scripts/build_lpcnet.sh
.venv/bin/python scripts/check_env.py
```

LPCNet needs autoconf, automake, libtool, make, pkg-config and a C compiler.

The second-generation decoder (`b2v2_*`) uses PyTorch. No pinned environment file is included for
it; the runs here used PyTorch 2.5.1 on an RTX 4070 and a later release on an RTX 5090. A base
model trains in about 12–25 minutes on the 4070, and an adaptation in 1–3 minutes.

## Pipeline

Each script's docstring gives its exact command. All settings live in `configs/`.

**First-generation decoder (reproduction)**

- `scripts/run_inference.py`: offline inference on a session with a released model, scored.
- `scripts/compare_closed_loop.py`: offline audio against the audio their decoder produced live.
- `scripts/prepare_kunz.py`: Kunz `.mat` files to the decoder's input layout.
- `scripts/train.py`, `scripts/kunz_experiment.py`, `scripts/summarize_kunz.py`: training and the
  attempted-to-imagined experiment with this decoder.
- `scripts/asr_eval.py`: Whisper scoring of synthesized audio.

**Second-generation decoder**

- `scripts/b2v2_phonemes.py`, `scripts/build_b2v2_data.py`: phoneme labels and the 37-session
  training set.
- `scripts/train_b2v2.py`: train a base model, or adapt one to a new day (text only, voice targets,
  or both). `configs/experiments/b2v2_all_gan.yaml` is the base model used for the final results.
- `scripts/decode_b2v2.py`, `scripts/score_b2v2.py`: causal decoding, then WER and PER.
- `scripts/build_b2v2_kunz.py`, `scripts/build_b2v2_kunz_other.py`, `scripts/build_b2v2_iso.py`,
  `scripts/build_b2v2_openvocab.py`: the Kunz sessions in the same format.
- `scripts/train_b2v2_pool.py`: inner-speech adaptation (the final recipe:
  `configs/experiments/b2v2_kunz_d097_unfreeze_*.yaml`).
- `scripts/identify_b2v2.py`: sentence identification and PER from the phoneme output.
- `scripts/d0*.py`: the analysis of each numbered experiment.

## Layout

```
src/wvx/            project code (data adapters, models, training, metrics, synthesis)
scripts/            one entry point per pipeline step
configs/            all settings (default.yaml, data/, experiments/)
tests/              fast CPU tests
results/            small result files of every run (JSON and Markdown; no audio)
third_party/        brain_to_voice (fetched), LPCNet (BSD-3-Clause), PROVENANCE.md
data/               local only (gitignored)
```

## License

This repo's own code is released under the [MIT License](LICENSE). Third-party code
under `third_party/` retains its own licenses; see
[third_party/PROVENANCE.md](third_party/PROVENANCE.md) for details.
