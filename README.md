# Wairagkar et al. 2025 — reproduction and inner-speech extension

Reproduce the offline brain-to-voice decoder from Wairagkar et al. 2025 (*Nature*),
then extend it to inner speech using Kunz et al. 2025 (*Cell*) data.

**Core experiment:** participant T15, session 2024.03.03. Train on attempted 50-word
sentences, test on imagined 50-word sentences from the same day. The arrays and sentence
set are identical across conditions, so any drop in performance reflects speech mode rather
than day-to-day drift.

**Status:** setup only — no results yet. See [notes/PLAN.md](notes/PLAN.md) for phases.

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

## Where things run

Code is edited and unit-tested on a Mac. **Data and all data-touching compute live on a
remote GPU host** inside a Docker container. The host is never named in tracked files;
configure it in `scripts/remote/host.env` (gitignored; template:
`scripts/remote/host.env.example`).

- `scripts/remote/setup_container.sh` — one-time container setup on the host.
- `scripts/remote/sync.sh push` / `pull` — sync code up, results down.
- `scripts/remote/gpu_job.sh` — run GPU jobs under the host's shared lock.

## Data

All data lives under `data/` on the GPU host only. `data/` is gitignored; checksums are
in [data/MANIFEST.md](data/MANIFEST.md). Dataset configs are in `configs/data/kunz.yaml`
and `configs/data/wairagkar.yaml`.

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

<!-- ENVIRONMENT: to be filled in -->

## License

This repo's own code is released under the [MIT License](LICENSE). Third-party code
under `third_party/` retains its own licenses; see
[third_party/PROVENANCE.md](third_party/PROVENANCE.md) for details.

## Layout

```
src/wvx/           project code (config, seeding)
scripts/            pipeline entry points + remote host scripts
configs/            all settings (default.yaml, data/, third_party.yaml)
tests/              fast CPU tests
notes/              PLAN.md, DATA_FORMATS.md, WAIRAGKAR_CODE.md, DECISIONS.md
results/            small outputs only
figures/            small outputs only
third_party/        brain_to_voice (fetched), PROVENANCE.md
data/               GPU host only; MANIFEST.md is tracked
```
