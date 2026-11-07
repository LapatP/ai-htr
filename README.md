# ai-htr — field extraction from hospital OPD cards

Locates and crops the **blood pressure** field on a scanned or photographed
hospital outpatient card, then attempts to read it. Two halves: *find the value
on the page*, then *read the handwriting*. Everything runs offline on one
machine — the eventual inputs are patient records.

Currently handles four form templates and three structural patterns for how a
BP is written: one blank with a hand-drawn slash (`143/90`), two blanks flanking
a printed slash, and two separately labelled blanks.

---

## Read this before you start

**This repository deliberately ships with no scans and no templates.**

The pipeline identifies a form by matching it against a *template*, and a
template is built from a real card. With one sample per form, the sample **is**
the template — so every template in this project is a patient page, and none of
them are distributable. `.gitignore` excludes `data/raw/`, `data/templates/*.png`
and `data/inbox/` for that reason.

The practical consequence: **you must supply your own OPD card before anything
runs.** Out of the box you will get

```
no templates in data\templates — run scripts/build_templates.py first
```

which is the expected state, not a broken install. Setup below walks through it.

---

## Status, stated honestly

| Stage | | Status |
|---|---|---|
| 0 | Normalize — dewarp, deskew, denoise | Built, measured |
| 1 | Identify form | Built, measured |
| 2 | Register into canonical space | Built, measured |
| 3 | Crop + ink refinement | Built, measured |
| 4 | **Read the handwriting** | **Wired, but not usable** |
| 5 | Clinical validation | Built |

**Cropping works.** Coverage of the true field is min 0.973 / mean 0.996, flat
from clean to heavy synthetic degradation. Registration error is sub-pixel.
Form identification refuses unknown forms rather than guessing — every failure
measured so far has been a refusal, never a misidentification.

**Reading does not work.** The current reader is a template-correlation baseline
over vector fonts with zero learned parameters. It turns a real `143/90` into
`111/10`. It is included because the pipeline needs *something* in that slot,
not because it is fit for use — which is why `--read` is opt-in and off by
default. There are no trained models anywhere in this project.

The blocker is data, not code: three genuinely filled BP fields exist in the
whole corpus, all on one card in one person's handwriting. No reader can be
honestly evaluated against that.

Every accuracy figure quoted anywhere in this repository is from **synthetic**
perturbation of real pages. Nothing has been tested against a second real scan.

---

## Requirements

- **Python 3.12.** Not 3.13/3.14 — ML wheels are patchy above 3.12, and that
  matters for the reader work still to come.
- No GPU, no network, no models to download. Dependencies are OpenCV, NumPy and
  two YAML parsers; the whole environment is about 200 MB.
- Developed on Windows with PowerShell. Nothing is Windows-specific except the
  drag-and-drop launcher `Process dropped scans.cmd`.

---

## Setup

```powershell
cd <wherever-you-put-this>
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pip install pytest
```

Installed editable, so `python -m aihtr` works from any directory with no
`PYTHONPATH` to remember, and edits to `src/` take effect immediately.

If this copy came with a **`wheels/`** folder, it carries every dependency and
installs with no network at all:

```powershell
.venv\Scripts\python.exe -m pip install --no-index --find-links wheels -e .
.venv\Scripts\python.exe -m pip install --no-index --find-links wheels pytest
```

`--no-index` forbids pip from contacting PyPI, so success means it came
entirely from the bundle. Those wheels are Windows x64 + CPython 3.12 only —
opencv, numpy, pillow and lxml are compiled binaries, so any other platform or
Python version needs the online install above. `wheels/` is not tracked in git;
it is built per export from `requirements-lock.txt`, so an offline install and
a networked one land on the same versions.

**Install the commit guard — one command per clone.** Git does not clone hooks,
so a fresh checkout has no protection until you run:

```powershell
git config core.hooksPath scripts/hooks
```

`scripts/hooks/pre-commit` then refuses any commit containing `data/raw/`,
`data/interim/`, `testdata/`, `wheels/`, an `out*/` folder, a template image,
any image or PDF at all, or any file over 2 MiB — **including one force-added
past `.gitignore` with `git add -f`**. `.gitignore` prevents an accident; the
hook also prevents a deliberate mistake, which matters once a remote is
involved. Override a commit you have genuinely thought about with
`git commit --no-verify`.

**Reproducing a measured number** takes the versions it was measured against,
not merely compatible ones:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

`requirements.txt` keeps loose `>=` ranges, which is right for development and
wrong for reproduction — cropping coverage comes out of OpenCV's feature
matching, so a newer OpenCV can shift the scores with no change to this code.
When quoting a figure, quote the commit and note that it was measured against
`requirements-lock.txt`.

Confirm the install. This needs no scans, so run it before you have any data:

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q
```

On a fresh checkout expect **65 passed, 62 skipped**, in a couple of seconds. The
skips are the tests that need a corpus or a built template — they start running
once you have data, at which point the full suite is **127 passed** in roughly
60 seconds. A *failure* at this stage means the install is wrong; skips do not.

> **PowerShell note.** There is no `&&` in PowerShell 5.1, so each command goes
> on its own line. To run an executable by path, prefix it with `&` —
> a bare leading `.` makes PowerShell try to dot-source it.

---

## Quick check: the bundled test data

If this copy came with a `testdata/` folder, you can watch the whole pipeline
work before you have a card of your own:

```powershell
.venv\Scripts\python.exe scripts\run_testdata.py
Invoke-Item out-testdata\report.html
```

It builds templates from `testdata/template-sources/`, crops all 24 synthetic
pages, and scores every crop against the values the generator rendered into
them. Expect 23–24 pages recognised and mean coverage ≥ 0.98. One heavily
degraded page is normally *refused* as ambiguous — that is the identifier
declining to guess, which is a pass.

It writes to `data/templates-testdata/` and `out-testdata/`, never to
`data/templates/` or `data/raw/`, so it cannot disturb a working setup.

Those pages are synthetic, but they are rendered on a form background cut from
a real card. Copy them between your own machines; do not push them anywhere
public. `.gitignore` excludes `testdata/` for that reason.

---

## First run: bring your own card

The one-time part. Only step 2 needs a human.

**1. Put a clean example in the corpus and build its template.**

```powershell
Copy-Item yourcard_001.jpg data\raw\
.venv\Scripts\python.exe scripts\build_templates.py --all
```

Add a matching entry to `configs/corpus.yaml`, and a form config in
`configs/forms/` — copy `nutrition-0190-v1.yaml` as the skeleton and leave
`bbox: null` on each field. Pages named in `configs/corpus.yaml` that you do
not have print as `MISSING`; that is expected, not an error.

`build_templates.py` also cuts every **insert** template — the triage sticker —
out of its parent form using the `source_region` in that form's config. Without
it the sticker's fields are never cropped, so a card carrying a second BP
quietly yields one reading instead of two. If one is missing the CLI now says
so at startup rather than running on in silence.

**2. Draw the boxes.** Opens a self-contained page in your browser; drag a
rectangle around each value.

```powershell
.venv\Scripts\python.exe scripts\make_box_tool.py
Invoke-Item docs\box-tool.html
.venv\Scripts\python.exe scripts\apply_boxes.py data\boxes\boxes.json
```

**3. Check every box actually landed.** Do not skip this — a misplaced box
produces a perfectly plausible crop of the wrong thing.

```powershell
.venv\Scripts\python.exe -m aihtr.schema
.venv\Scripts\python.exe scripts\preview_crops.py --refine
```

Then *look at* `data/interim/crops_contact_sheet.png`.

**4. Run it.**

```powershell
.venv\Scripts\python.exe -m aihtr "C:\path\to\scans" --out out
Invoke-Item out\report.html
```

`report.html` is the thing to open: raw input beside every crop, systolic and
diastolic shown separately, flagged rows highlighted. It is self-contained, and
that cuts both ways: the raw input it embeds is the **whole scanned card**, so
a report built from a real card is exactly as identifiable as the scan itself.
Keep `out/` on the machine that holds `data/raw/`, and never email or upload a
report made from a real card. `--no-report` gives a manifest-only run.

Full detail, including all the CLI flags and what every `manifest.csv` column
means, is in **[docs/running.md](docs/running.md)**.

> Absolute paths in `docs/running.md` are the original author's — substitute
> your own checkout location.

---

## Understanding what it did

To watch every stage operate on a single page — as received, straightened,
landmark matches, alignment overlay, each crop, each glyph with its score:

```powershell
.venv\Scripts\python.exe scripts\trace_page.py data\raw\yourcard_001.jpg --degrade
Invoke-Item out-trace\trace.html
```

`--degrade` knocks the page about first, so every stage has real work to do.

---

## Design decisions worth knowing

- **Coverage, not IoU.** A crop that clips a digit has failed however good its
  overlap score; extra whitespace is free. A clipped leading digit turns 143
  into 43.
- **Refinement is grow-only.** The nominal box is a human's statement of where
  the field is. Ink may extend it, never overrule it — measured, free
  refinement was *itself* clipping on degraded scans.
- **The reader never guesses.** Closed set of eleven symbols; it abstains rather
  than inventing. `field_empty` and `unreadable` stay distinct facts.
- **Readings are two integers, never a string.** `143/90` misread as `14/390` is
  a one-character slip and clinical nonsense.
- **A form is data, not code.** Adding a form means adding a YAML file in
  `configs/forms/`. Never edit Python to add one.
- **Every reading is graded before it is trusted** (`src/aihtr/validate.py`).
  The load-bearing rule is that systolic must exceed diastolic — definitional,
  free, and the only thing that catches a clipped leading digit. It grades; it
  never corrects. Silently repairing `43/90` to `143/90` would be exactly the
  confident fabrication that makes a language model dangerous here.

---

## Documentation

| File | |
|---|---|
| [docs/START-HERE.md](docs/START-HERE.md) | Running an export copy (with bundled test data) on another computer |
| [docs/running.md](docs/running.md) | Full runbook; how to onboard a new form |
| [docs/cropping-evaluation.md](docs/cropping-evaluation.md) | How cropping was measured, and what the numbers mean |
| [docs/synthetic-data.md](docs/synthetic-data.md) | Generating test pages with known ground truth |
| [docs/stage3-ink-refinement.md](docs/stage3-ink-refinement.md) | Refinement design, and where it deviates |
| [docs/status-report.html](docs/status-report.html) | Current status summary |
| `CLAUDE.md` | Full working notes, including every trap hit so far |

---

## Known issues

1. **Crop refinement does not follow ink downward.** A value written low on the
   field gets its bottom clipped; sideways displacement is absorbed fine.
2. ~~**A missed field is not treated as a disagreement.**~~ **Fixed.**
   `cross_check()` now takes how many BP sources the page actually carries, and
   a page that declares two but reads one is routed to review instead of
   passing as *plausible*.
3. **The reader is not usable**, and its confidence figure is uncalibrated —
   it has reported "confident" on every glyph of a completely wrong reading.
4. **`MARGIN_RATIO = 6.5` needs recalibrating**; it predates insert masking, so
   recall is being lost to false refusals.
5. **Refinement constants are tuned for digits on ruled lines only.** Checkboxes,
   circled Y/N tokens and grid cells are declared but unmeasured.
6. **Source resolution is ~103 dpi** (messaging-app compressed). The scanning
   standard is 300. This caps the reader; it does not affect cropping.
