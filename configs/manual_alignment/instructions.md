# Syllable timing: instructions

## The task

Each item is one attempt by a participant with paralysis to say a short English sentence (the cue). He cannot
speak intelligibly. A decoder turned his brain activity into sound, and the item shows that decoded sound. It is
a smeared, unintelligible reconstruction. The participant speaks slowly, roughly one word at a time, with pauses
between words.

**For every syllable listed in the item, in order, give the start and end time (seconds) of the burst of decoded
sound where he attempted that syllable.**

The listed syllables are those of a standard pronunciation of the cue. For example, `thirsty 1/2` is the first
syllable of "thirsty", and `hope` is a one-syllable word. `items/<id>.json` lists them with their phones.

## What is in an item

- `items/<id>.png`: the overview image.
- `items/<id>.json`: the cue (`text`), the syllables to time (`syllables`), the item length (`duration_s`), and
  an answer template (`response_template`).
- `items/<id>.npz`: the arrays behind the image (time axis, spectrogram, loudness, neural activity). The tools read
  them, so you don't need to.

## Reading the image

All panels share one time axis in seconds. The overview has {{PX_PER_SECOND}} pixels per second, a major grid line
every 1 s and a minor one every 0.1 s. Zoomed views get a finer grid, down to 10 ms. Items longer than
{{SECONDS_PER_ROW}} s wrap into rows of {{SECONDS_PER_ROW}} s; each row repeats {{ROW_OVERLAP_S}} s of its
neighbours, so read the time off the axis. Grey shading is outside the item, and a black line marks its end.

1. **Spectrogram** of the decoded sound (0-8 kHz, dB relative to the item's loudest point). Attempts show up as
   bright vertical bands; pauses are dark.
2. **Loudness** of the decoded sound (dB relative to the item's loudest point, lightly smoothed). Syllables are
   humps, pauses are low stretches, and a dip between two humps often separates two syllables.
3. **Neural activity**: the average activity of the recorded brain channels, smoothed, z-scored within the item.
   It tends to rise around each attempt and can confirm a burst. It may lead the sound slightly, so take exact
   edges from panels 1 and 2.

Things to know:
- The first 0.6 s of every item is silent by construction: the decoder needs 0.6 s of brain activity before it
  produces anything.
- Don't try to recognise words; the sound is not intelligible. Use the timing structure instead: the number and
  order of bursts, the pauses, the loudness dips and the spectral changes.
- The syllables of one word usually form one burst, often with a dip between them. Words are usually separated by
  pauses.
- Not every burst is speech: decoder noise, breaths and false starts happen. Not every syllable has a clear burst.

## Workflow, per item

Run the tools from the package folder (setup: see README.md). Write only inside `answers/<worker_id>/`.

1. Look at the overview `items/<id>.png`.
2. Zoom in where it matters:
   `python tools/inspect_item.py render <id> --start S --end E --out answers/<worker_id>/views/<id>_S_E.png`
3. Locate bursts and dips with numbers: `python tools/inspect_item.py peaks <id> --start S --end E` lists loudness
   peaks and the dips between them, and `python tools/inspect_item.py values <id> --start S --end E --bands` prints
   loudness, neural activity and band energies every 10 ms.
4. Write a draft: copy `response_template` from `items/<id>.json` into `answers/<worker_id>/drafts/<id>.json` and
   fill in every `start_s` and `end_s`.
5. Check it on the image and refine:
   `python tools/inspect_item.py overlay <id> answers/<worker_id>/drafts/<id>.json --out answers/<worker_id>/views/<id>_draft.png`
   (add `--start S --end E` to zoom).
6. Validate the draft: `python tools/inspect_item.py check answers/<worker_id>/drafts/<id>.json`. When the whole
   batch is done, put the item answers into the batch file (format below) and run `check` on that file too.

## Rules for the spans

- One span per listed syllable, in the listed order. Spans must not overlap; touching is fine. Each span is at
  least {{MIN_SPAN_S}} s long and lies inside the item (0 to `duration_s`). Two decimals (10 ms) are enough.
- **Fewer visible bursts than syllables:** split the bursts. Usually two syllables of one word share a burst: cut
  at its deepest loudness dip, or in proportion when there is no dip. Mark the syllables you had to split with
  `"confidence": "low"`.
- **A word with no visible burst:** place its syllables where the order and pauses say they belong, keep them
  short, and mark them `"confidence": "low"`.
- **More bursts than syllables:** give each syllable the burst that best fits the order and the pauses. Leave the
  extra bursts unassigned; you may list them in `"unassigned_bursts"`.
- Never leave a syllable out, and never guess silently: use `confidence` (`high`, `medium`, `low`) and `notes`.

## Answer format (exact)

One file per batch: `answers/<worker_id>/<batch_id>.json` (for example `answers/w07/batch_012.json`). Answers for
items given to you outside your batches go to `answers/<worker_id>/extra.json` (then `extra_2.json`, ...) with
`"batch": "{{EXTRA_BATCH_ID}}"`. Practice answers go to `answers/<worker_id>/practice/round<r>.json` with your
practice batch id.

```json
{
  "schema": "{{RESPONSE_SCHEMA}}",
  "batch": "batch_012",
  "worker_id": "w07",
  "items": [
    {
      "item_id": "0a1b2c3d4e",
      "syllables": [
        {"index": 0, "label": "i",    "start_s": 1.62, "end_s": 1.98, "confidence": "high"},
        {"index": 1, "label": "hope", "start_s": 3.05, "end_s": 3.71}
      ],
      "confidence": "medium",
      "notes": "optional free text",
      "unassigned_bursts": [[5.20, 5.42]]
    }
  ],
  "notes": "optional free text"
}
```

- Required: `schema`, `batch`, `worker_id` (must equal your folder name), `items`; for each item `item_id` and
  `syllables`; for each syllable `index` (0, 1, 2, ... in order), `label` (exactly as in the item file),
  `start_s` and `end_s` (numbers of seconds).
- Optional: `confidence` (item or syllable; one of {{CONFIDENCE_VALUES}}), `notes` (strings), and
  `unassigned_bursts` (a list of `[start_s, end_s]`).
- Nothing else is allowed. `check` tells you exactly what is wrong. A rejected item answer is lost, not repaired.

## Worked examples

`examples/` holds items like the real ones, with reference spans: `<id>.png` is the item, `<id>_answer.json` is a
correct answer and `<id>_overlay.png` shows its spans (the green dashed lines and the reference lane). Study them
before you start: how bursts, pauses and dips line up with the listed syllables, and how far a reference syllable
can be from a clean hump.

{{EXAMPLE_LIST}}

## Practice

`practice/items/` holds {{N_PRACTICE}} practice items (ids starting with `{{PRACTICE_PREFIX}}`), each with reference
spans in `practice/truth/`. Do not open a truth file before you have answered that item. Your practice batch in
`manifest.json` has {{PRACTICE_ROUNDS}} rounds of {{ITEMS_PER_ROUND}} fresh items. For each round:

1. Answer its items exactly like real items; save them as `answers/<worker_id>/practice/round<r>.json` with
   `"batch": "<your practice batch>"`.
2. Score them: `python tools/score_practice.py answers/<worker_id>/practice/round<r>.json --overlay answers/<worker_id>/practice/overlays`
3. Study the per-syllable table and the overlays (your spans against the reference), then adjust your technique.

The score is the share of reference syllable starts within {{TOLERANCE_MS}} ms of one of your starts, the median
start error, whether the syllable counts agree, and the error of the first start and the last end. Sometimes the
reference has more or fewer syllables than the item lists, because the participant merged or split syllables. Your
answer still lists every requested syllable.
