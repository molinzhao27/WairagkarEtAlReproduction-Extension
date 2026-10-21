# Wairagkar et al. 2025 — reproduction and inner-speech extension

> 🚧 **Work in progress.** Setup and the offline inference reproduction are done. First inner-speech runs
> exist, but they are preliminary and not results to cite.

Reproduce the offline brain-to-voice decoder from Wairagkar et al. 2025 (*Nature*),
then extend it to inner speech using Kunz et al. 2025 (*Cell*) data.

**Core experiment:** participant T15, session 2024.03.03. Train on attempted 50-word
sentences, test on imagined 50-word sentences from the same day. The arrays and sentence
set are identical across conditions, so any drop in performance reflects speech mode rather
than day-to-day drift.

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
`lpcnet_demo` binary used for feature extraction and speech synthesis. Provenance
is in `third_party/PROVENANCE.md`.

## Where things run

Code is edited and unit-tested on a Mac. Data and all data-touching compute live on a
remote Linux GPU machine inside a Docker container. Code is device-agnostic: it runs on
Apple silicon (CPU) and on NVIDIA CUDA GPUs.

## Data

All data lives under `data/` on the GPU host only. `data/` is gitignored. Dataset configs
and SHA-256 checksums are in `configs/data/kunz.yaml` and `configs/data/wairagkar.yaml`.

**Kunz et al. 2025** — T15, session 2024.03.03. Download `sentenceData.zip` from the
Dryad record, then extract the subset this project needs:

```bash
python -I scripts/extract_kunz_subset.py --zip /path/to/sentenceData.zip
```

**Wairagkar et al. 2025** — `t15_neural_data.zip` (26.16 GB) and
`t15_control_experiment.zip` (711 MB). Dryad now requires an API token for downloads.
The default run is a dry run that checks sizes and free disk:

```bash
python scripts/fetch_wairagkar_data.py
```

## Setup

```bash
git clone git@github.com:molinzhao27/WairagkarEtAlReproduction-Extension.git
cd WairagkarEtAlReproduction-Extension
git config core.hooksPath .githooks
python scripts/fetch_third_party.py
pytest
```

### Mac (Apple silicon, CPU)

```bash
mamba env create -f environment.yml           # creates env "wvx" (TF 2.21 + tf-keras, CPU)
mamba run -n wvx bash scripts/build_lpcnet.sh # builds third_party/LPCNet/lpcnet_demo
mamba run -n wvx python scripts/check_env.py  # forward pass, fit step, LPCNet smoke test
```

TF 2.21 is the only TF/Keras-2 version reported to run on the Blackwell GPU. The same
version is used on both machines to avoid skew.

### Linux (CUDA)

Create a Python venv from `requirements-linux-cuda.txt` (or use `environment.linux-cuda.yml`
with conda/mamba). Build LPCNet and run the environment check:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-linux-cuda.txt
.venv/bin/pip install -e . --no-deps
bash scripts/build_lpcnet.sh
.venv/bin/python scripts/check_env.py
.venv/bin/python -m pytest
```

LPCNet requires autoconf, automake, libtool, make, pkg-config, and a C compiler (gcc).
Pending GPU verification on the RTX PRO 6000.

## Pipeline

Scripts that touch data run on the GPU host unless noted otherwise.

- `scripts/extract_kunz_subset.py` — extract the Kunz T15 2024.03.03 files from the Dryad archive.
- `scripts/fetch_wairagkar_data.py` — download and verify the Wairagkar Dryad dataset (dry run by default).
- `scripts/prepare_kunz.py` — convert Kunz `.mat` files into decoder-layout SessionData (`.npz`).
- `scripts/make_tts.py` — generate LPCNet target features from macOS TTS, rank voices, push to host. **Mac only.**
- `scripts/align_targets.py` — align TTS targets to decoder output for Kunz trials (bootstrap method).
- `scripts/validate_alignment.py` — validate alignment against Wairagkar sessions' syllable timings and targets.
- `scripts/run_inference.py` — reproduce offline inference on a held-out session and score the output.
- `scripts/train.py` — train the brain-to-voice decoder on prepared sessions.
- `scripts/kunz_experiment.py` — core experiment: train on attempted, decode attempted and imagined.
- `scripts/summarize_kunz.py` — cross-fold summary of a Kunz experiment run.
- `scripts/asr_eval.py` — ASR scoring (Whisper) of synthesized audio in a results directory.
- `scripts/compare_closed_loop.py` — compare offline inference with the audio their decoder produced live.
- `scripts/inspect_mat.py` — generic `.mat` dumper (shapes, dtypes, value summaries).

## License

This repo's own code is released under the [MIT License](LICENSE). Third-party code
under `third_party/` retains its own licenses; see
[third_party/PROVENANCE.md](third_party/PROVENANCE.md) for details.

## Layout

```
src/wvx/            project code (config, seeding, data adapters, inference, training, metrics, synthesis)
scripts/            pipeline entry points
configs/            all settings (default.yaml, data/, third_party.yaml)
tests/              fast CPU tests
results/            small per-trial outputs of our runs; preliminary
figures/            small outputs only
third_party/        brain_to_voice (fetched), LPCNet (BSD-3-Clause), PROVENANCE.md
data/               GPU host only (gitignored)
```
