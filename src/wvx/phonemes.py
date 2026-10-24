"""Phoneme labels for the brain2voice-2.0-style decoder's CTC head (D-075).

Vocabulary: CTC blank (id 0), ``SIL`` (id 1, a word terminator: after every word, as in brain-to-text decoders),
then the 39 CMUdict phonemes without stress (ids 2-40): 41 classes. The paper lists "39 phonemes, a silence token,
plus a blank token". A cue's label is the per-word pronunciations, each followed by ``SIL``.

Pronunciations (``word_pronunciations``): the first CMUdict pronunciation of each aligner word; a word not in
CMUdict takes the phones the forced aligner gave it for this cue (its G2P), stress removed.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

PHONEMES = ("AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY",
            "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y",
            "Z", "ZH")
BLANK, SIL = "<b>", "SIL"
VOCAB = (BLANK, SIL) + PHONEMES
ID = {p: i for i, p in enumerate(VOCAB)}
N_CLASSES = len(VOCAB)


def strip_stress(phone: str) -> str:
    """``AH0`` -> ``AH``; raises on a label outside the 39 phonemes."""
    p = phone.rstrip("012").upper()
    if p not in PHONEMES:
        raise ValueError(f"not a CMUdict phoneme: {phone!r}")
    return p


def word_pronunciations(words: Sequence[tuple], phones: Sequence[tuple], cmu: Mapping[str, Sequence[str]],
                        silence: Iterable[str] = ("", "sil", "sp", "<eps>"),
                        unknown: str = "spn") -> list[dict]:
    """Per aligner word (silences skipped): ``{"word", "phones", "source"}`` with source ``cmudict`` or ``aligner``,
    or ``unknown`` (phones None) when the aligner itself had no pronunciation (``spn``) and CMUdict has none.

    ``words`` / ``phones`` are (start, end, label) intervals; a phone belongs to the word holding its midpoint.
    ``cmu`` maps a lower-case word to its first pronunciation (stress digits allowed).
    """
    sil = {s.lower() for s in silence}
    ph = [p for p in phones if p[2].lower() not in sil]
    out = []
    for ws, we, w in words:
        if w.lower() in sil:
            continue
        mine = [p[2] for p in ph if ws <= (p[0] + p[1]) / 2 < we]
        key = w.lower()
        if key in cmu:
            out.append({"word": w, "phones": [strip_stress(p) for p in cmu[key]], "source": "cmudict"})
        elif mine and unknown not in {m.lower() for m in mine}:
            out.append({"word": w, "phones": [strip_stress(p) for p in mine], "source": "aligner"})
        else:
            out.append({"word": w, "phones": None, "source": "unknown"})
    return out


def label(word_phones: Sequence[Sequence[str]]) -> list[int]:
    """Ids of the words' phones, each word followed by SIL."""
    out = []
    for ph in word_phones:
        out += [ID[p] for p in ph] + [ID[SIL]]
    return out


def ctc_greedy(best: Iterable[int]) -> list[int]:
    """Best path per frame -> label ids: collapse runs, then drop blanks."""
    out, prev = [], None
    for b in best:
        b = int(b)
        if b != prev and b != ID[BLANK]:
            out.append(b)
        prev = b
    return out


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance (substitutions, insertions, deletions all cost 1)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def phoneme_error(ref: Sequence[int], hyp: Sequence[int], drop: Optional[Iterable[int]] = (ID[SIL],)) -> tuple[int, int]:
    """(edit errors, reference length) with ``drop`` ids (default SIL) removed from both sides first."""
    d = set(drop or ())
    r = [x for x in ref if x not in d]
    h = [x for x in hyp if x not in d]
    return edit_distance(r, h), len(r)


def ctc_log_likelihood(logp: np.ndarray, label: Sequence[int], blank: int = 0) -> float:
    """log p(label | x) under CTC (forward algorithm) from per-frame log-probabilities ``logp`` (T, C)."""
    T = logp.shape[0]
    ext = np.array([blank] + [v for lab in label for v in (int(lab), blank)])
    S = len(ext)
    if T == 0:
        return float("-inf")
    skip = np.zeros(S, bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    alpha = np.full(S, -np.inf)
    alpha[0] = logp[0, blank]
    if S > 1:
        alpha[1] = logp[0, ext[1]]
    a1, a2 = np.full(S, -np.inf), np.full(S, -np.inf)
    for t in range(1, T):
        a1[1:] = alpha[:-1]
        a2[2:] = alpha[:-2]
        alpha = np.logaddexp(np.logaddexp(alpha, a1), np.where(skip, a2, -np.inf)) + logp[t, ext]
    return float(np.logaddexp(alpha[-1], alpha[-2]) if S > 1 else alpha[-1])
