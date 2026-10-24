#!/usr/bin/env python
"""Phoneme labels of every Wairagkar cue for the brain2voice-2.0-style decoder's CTC head (D-075).

Host, the syllable env (it has the ``cmudict`` package), CPU:
    CUDA_VISIBLE_DEVICES= nice -n 19 /workspace/envs/wvx-syl/bin/python -I scripts/b2v2_phonemes.py --config configs/experiments/b2v2_base.yaml

Words come from the forced alignment of the Alex TTS of each cue (corpora in ``b2v2.phonemes.corpora``; MFA's
text normalization); each word's pronunciation is its first CMUdict pronunciation, else the aligner's phones for it
(``wvx.phonemes.word_pronunciations``). Writes ``<b2v2.data_dir>/phonemes.json``: per cue (``wvx.targets.text_key``)
its words, per-word phonemes and sources; counts of each source and the cues with an unknown word.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvx.config import resolve_path  # noqa: E402
from wvx.experiment_config import load_experiment_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/b2v2_base.yaml")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    import cmudict

    from wvx import phonemes as ph, syllables as sy, targets as tg
    from wvx.provenance import code_version

    p = cfg["b2v2"]["phonemes"]
    cfg_v = load_experiment_config(p["voice_config"])
    fa = cfg_v["syllables"]["forced_alignment"]
    voice = cfg_v["tts"]["voice"]
    first = {w: prons[0] for w, prons in cmudict.dict().items() if prons}
    table, sources, unknown_cues, missing = {}, {}, [], []
    for corpus in p["corpora"]:
        d = sy.tts_dir(cfg_v, voice, corpus)
        index = json.loads((d / "index.json").read_text())
        for u in index["utterances"]:
            key = tg.text_key(u["text"])
            js = d / f"mfa_out_{corpus}" / f"{u['name']}.json"
            if key in table:
                continue
            if not js.is_file():
                missing.append(key)
                continue
            words, phones = sy.read_aligner_json(js, cfg_v)
            prons = ph.word_pronunciations(words, phones, first, silence=fa["silence_labels"],
                                           unknown=str(fa["unknown_label"]))
            for w in prons:
                sources[w["source"]] = sources.get(w["source"], 0) + 1
            if any(w["phones"] is None for w in prons):
                unknown_cues.append(key)
            table[key] = {"words": [w["word"] for w in prons], "phones": [w["phones"] for w in prons],
                          "sources": [w["source"] for w in prons]}
    out_dir = resolve_path(cfg["b2v2"]["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "code_version": code_version(),
           "voice": voice, "corpora": p["corpora"], "vocabulary": list(ph.VOCAB), "n_cues": len(table),
           "word_sources": sources, "cues_with_unknown_word": unknown_cues, "cues_without_alignment": missing,
           "cues": table}
    (out_dir / "phonemes.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(table)} cues; word sources {sources}; {len(unknown_cues)} cues with an unknown word; "
          f"{len(missing)} without alignment -> {out_dir / 'phonemes.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
