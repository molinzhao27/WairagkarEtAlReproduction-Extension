# Wairagkar et al. 2025 — reproduction and inner-speech extension

1. **Reproduce** the offline brain-to-voice decoder from Wairagkar et al. 2025.
2. **Extend** it to inner speech with the Kunz et al. 2025 data. Train on *attempted* 50-word sentences
   and test on *imagined* 50-word sentences from the same session (T15, 2024.03.03). Because the day,
   arrays and sentence set are the same, any drop in performance reflects speech mode, not drift.

Agents and contributors: start with [AGENTS.md](AGENTS.md). The plan is in [notes/PLAN.md](notes/PLAN.md).

## Citations
- Wairagkar M, Card NS, Singer-Clark T, Hou X, Iacobacci C, Miller LM, Hochberg LR, Brandman DM,
  Stavisky SD. **An instantaneous voice-synthesis neuroprosthesis.** *Nature* (2025).
  doi:[10.1038/s41586-025-09127-3](https://doi.org/10.1038/s41586-025-09127-3).
  Code: <https://github.com/Neuroprosthetics-Lab/brain-to-voice-2025>.
  Data: Dryad doi:[10.5061/dryad.2280gb64f](https://doi.org/10.5061/dryad.2280gb64f).
- Kunz EM, et al. **Inner speech in motor cortex and implications for speech neuroprostheses.**
  *Cell* (2025). doi:[10.1016/j.cell.2025.06.015](https://doi.org/10.1016/j.cell.2025.06.015).
  Data: Dryad doi:[10.5061/dryad.gf1vhhn1j](https://doi.org/10.5061/dryad.gf1vhhn1j).
- Valin J-M, Skoglund J. **LPCNet: Improving neural speech synthesis through linear prediction.**
  ICASSP 2019. Code: <https://github.com/xiph/LPCNet>.

**Required acknowledgment** (license of `third_party/brain_to_voice`) in any publication using this work:
> The Software used in this research was created by Maitreyee Wairagkar of The University of California, Davis. © 2025.

The vendored brain-to-voice code is under an academic, non-commercial license
([LICENCE.md](third_party/brain_to_voice/LICENCE.md)). See [third_party/PROVENANCE.md](third_party/PROVENANCE.md).

## Setup
```bash
git clone <this repo> && cd WairagkarEtAlReproduction-Extension
git config core.hooksPath .githooks          # enables the pre-commit guard (not cloned automatically)
mamba env create -f environment.yml          # creates env "wvx" (Apple silicon)
mamba run -n wvx pip install -e . --no-deps
mamba run -n wvx bash scripts/build_lpcnet.sh
mamba run -n wvx python scripts/check_env.py
mamba run -n wvx pytest
```
For the Linux CUDA training machine, use `environment.linux-cuda.yml` instead (see its header).

## Where things run
Code is edited and unit-tested on a Mac. **Data and all data-touching compute live on a GPU host**
(RTX PRO 6000), inside this project's Docker container `wvx`. See [AGENTS.md → Where things run](AGENTS.md#where-things-run).
```bash
cp scripts/remote/host.env.example scripts/remote/host.env    # then set WVX_HOST=user@host
ssh "$WVX_HOST" 'bash -s' < scripts/remote/setup_container.sh # one time, on the host
scripts/remote/sync.sh push                                   # code -> container
scripts/remote/sync.sh pull                                   # results/, figures/ <- container
```
GPU jobs go through `scripts/remote/gpu_job.sh` (run on the host), which takes the host's shared GPU lock.

## Data
All data lives under `data/` on the GPU host, and `data/` is gitignored. Checksums are in [data/MANIFEST.md](data/MANIFEST.md).

**Kunz et al. 2025** (T15 2024.03.03 subset). Download `sentenceData.zip` from the Dryad record (SHA-256 is
in `configs/data/kunz.yaml` / MANIFEST). Then extract only the files this project needs, on the host:
```bash
python -I scripts/extract_kunz_subset.py --zip /path/to/sentenceData.zip
```

**Wairagkar et al. 2025.** `t15_neural_data.zip` is 26.16 GB and `t15_control_experiment.zip` is 711 MB.
Dryad now requires an API token for downloads (`DRYAD_API_TOKEN`). The default run is a dry run that checks
sizes and free disk space:
```bash
python scripts/fetch_wairagkar_data.py
```

## Pipeline
| Step | Command | Status |
|---|---|---|
| Inspect a `.mat` file | `python -I scripts/inspect_mat.py <file>` | ✅ |
| Check environment | `python scripts/check_env.py` | ✅ |
| Reproduce Wairagkar (inference, then training) | — | Phase 1 |
| Generate Kunz target speech | — | Phase 2 |
| Kunz → decoder input adapter | — | Phase 3 |
| Attempted → imagined experiment | — | Phase 4 |

## Layout
See [AGENTS.md → Layout](AGENTS.md#layout).
