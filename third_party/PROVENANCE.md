# Third-party provenance

This file records where each vendored component under `third_party/` came from, its license,
and every local change made to it. **Any future local edit to a vendored component must be
logged here before it is committed.** This includes adding, removing, renaming or patching a
file inside a vendored directory, and building or installing anything into it. Log the date,
the path(s), the reason, and a short description or diff summary under that component's
"Local modifications" list. Prefer wrapping from `src/wvx` over editing vendored code.

## brain_to_voice

- **What:** Training and offline-inference code for Wairagkar et al. 2025, *Nature*,
  "An instantaneous voice-synthesis neuroprosthesis" (doi:10.1038/s41586-025-09127-3).
- **Source:** https://github.com/Neuroprosthetics-Lab/brain-to-voice-2025
- **Commit:** `cff4875decd35ff0f98a944035377a25b43ba79c`
- **Commit date:** 2026-07-22T23:25:06-07:00
- **Clone date:** 2026-10-20
- **Path in this repo:** `third_party/brain_to_voice/`
- **Not vendored:** the Dryad dataset and pretrained models (doi:10.5061/dryad.2280gb64f), and
  the LPCNet vocoder (the upstream README says to install it into
  `third_party/brain_to_voice/dependencies/`. We will not do that; see the LPCNet section).

### License

The license is the academic, non-commercial UC license "TDG-Attribution-NonCommercial-ShareAlike",
© 2025 The Regents of the University of California. The full text is in
`third_party/brain_to_voice/LICENCE.md`. Summary (this summary does not replace the license text):

- Only academic or nonprofit researchers may use it. It is a free, nonexclusive license to use, copy and
  make derivative works, solely for educational or academic research purposes. Copies may be
  redistributed free of charge only to other academic or nonprofit researchers, for the same purposes.
- Any distributed copy, including a derivative work, must include a copy of the license and its
  copyright notice, and must be distributed under the same license terms. Breaking these conditions
  ends the license.
- There is no warranty. The software is provided "as is", and the Institution accepts no liability for damages.
- Commercial entities must contact copyright@ucdavis.edu to ask about licensing.

### Required publication acknowledgment (verbatim)

Any academic or scholarly publication arising from use of this software or its derivatives must
include:

> The Software used in this research was created by Maitreyee Wairagkar of The University of California, Davis. © 2025.

### Local modifications

- The upstream `.git` directory was removed at vendoring time (2026-10-20).
- No source edits. No other file has been added, changed or removed.

Note: the vendored `third_party/brain_to_voice/.gitignore` stays in place, so it also applies inside this
repository's git tree for that subtree. It ignores `*.mat`, `*.h5`, `*.tf`, `*.wav`, `*.pcm`, `*.f32`,
`ckpt/`, `aud/*/`, `data/*/` and `dependencies/*/`.

### Integrity reference (SHA-256 at audit time, 2026-10-20)

Use these hashes to detect unlogged edits. Paths are relative to `third_party/brain_to_voice/`.

| File | SHA-256 |
|---|---|
| `.gitignore` | `030ff0ab0f67c6859fa9ca767bf850db3aa4303a806bd0fb35fbe866cd1e3e81` |
| `LICENCE.md` | `578d584362e6c954819d376d57bfb632d17db2f9f737897285e956be49dda56c` |
| `README.md` | `76be869ee189a46ec8b71e5850316e7b81c548328b0f838187d1a467bdfe1b4f` |
| `requirements.yml` | `b731a4498dc12c9a09af49894f670f303ca6af63dc85231ae4dadce6916e0d29` |
| `brain-to-voice_pipeline.png` | `b969e91b56b58d5542f5a4794aae9d335633c13e8e6ecb8eca9ac8b02ae978e2` |
| `training.ipynb` | `989d69d40b2e63aa25badbfdaa6d5cc8e6b769e0c1ea3156d8fefdcd4fd16c67` |
| `inference.ipynb` | `5da9844fdf985250c81fec9bcd41464bb1ba919c70ceb38ff44eda6d112d8216` |
| `configs/config.yaml` | `00029dbd8352703b715677ab25d8936b6f56d4782d013e7ba5bf297cfeb526be` |
| `configs/training_data_args.yaml` | `77338b4b90f2c689ebde349b1d21e02dc78ca47901936fe45c7bb1e906d0b41b` |
| `dependencies/T15_electrode_mapping_256.json` | `bd9a932a3fbc7ce171e1e6c8d83fc674b62042156eb79db475836097e73e3e48` |
| `dependencies/T15_lpcnet_features_norm_macOS_TTS.json` | `9cfb71bc371ed5274b3d53690548ed239fcf946e62076c7d77282b9b206acd82` |
| `lib/data_utils.py` | `40fc34f11486f8e51b6a45d6af011bdab31d1c3670e9efa82511e5f21c7935b9` |
| `lib/dataloader.py` | `f41b6e16fdfada0228f06a1edb3270d50415c1682167a13dba8d36b30c5c27e2` |
| `lib/inference_utils.py` | `7b270bf7023c34d63b5d3ff8842e68b971a58ba6adf40760599de2b4b31d6b8f` |
| `lib/model_utils.py` | `263fecdbf1da6d7dcf259e86d6b89cf59047689be1264ecfe113d1c2d58a6613` |
| `lib/speech_processor.py` | `9d56ce227d79891d1378588c5c3184a16b7350eeb87832ad392b0f030aafdd39` |
| `lib/training_utils.py` | `95e6d1c1138d29d96079570a968b7f19e4f83fc5d33efb105bb7cafcc55c65f0` |
| `lib/viz_utils.py` | `f41c41a8a80d1d25fbd3cfdb27faf842e0b1d58ab9e6074a1df77c172a3b10b6` |
| `models/brain2voice.py` | `acbf8daaea998c473e573c796d14d48a606777c3ad76b55bb523d5584400f138` |

Other files present at audit time: `aud/README.md`, `data/README.md`, `dependencies/README.md`
(one-line placeholder READMEs).

## LPCNet

_To be filled in by the toolchain step._
