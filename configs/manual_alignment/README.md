# Syllable timing package `{{NAME}}`

Mark, for every listed syllable of a cue, when a participant attempted it, using the image of a decoded,
unintelligible reconstruction of his speech attempt. Read `instructions.md` for the task, the workflow and the
answer format. `ORCHESTRATOR_PROMPT.md` is the prompt for the agent that coordinates the workers.

## Setup (once, macOS or Linux, Python >= 3.9)

```sh
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
.venv/bin/python tools/inspect_item.py --help
```

## Contents

| Path | What |
|---|---|
| `instructions.md` | the task, how to read the images, the workflow, the exact answer format |
| `ORCHESTRATOR_PROMPT.md` | the coordinator's prompt (team, phases, quality control) |
| `manifest.json` | workers, their batches and practice rounds, the counts |
| `settings.json` | settings of the tools (image scale, answer rules, thresholds) |
| `items/<id>.png`, `.json`, `.npz` | {{N_ITEMS}} items: overview image, cue + syllables + answer template, arrays |
| `practice/items/`, `practice/truth/` | {{N_PRACTICE}} practice items and their reference spans |
| `examples/` | {{N_EXAMPLES}} worked examples: image, correct answer, overlay of the answer |
| `tools/inspect_item.py` | `render`, `overlay`, `values`, `peaks`, `check` |
| `tools/score_practice.py` | scores practice answers against their reference spans (practice items only) |
| `tools/agreement.py` | compares the workers' answers per item and flags disagreements |
| `tools/requirements.txt` | pinned numpy and matplotlib |
| `answers/` | empty: workers write `answers/<worker_id>/...` |
| `files.sha256` | checksums of every file above |

Counts: {{N_ITEMS}} items in {{N_BATCHES}} batches of up to {{ITEMS_PER_BATCH}}, for {{N_WORKERS}} workers; every
item goes to {{K}} different workers. Practice: {{PRACTICE_ROUNDS}} rounds of {{ITEMS_PER_ROUND}} items per worker.

## Submitting

Each worker writes `answers/<worker_id>/<batch_id>.json` for each of its batches (format in `instructions.md`) and
checks it with `.venv/bin/python tools/inspect_item.py check answers/<worker_id>/<batch_id>.json`. When every item has
its answers, the whole `answers/` folder is returned; nothing else in the package changes.
