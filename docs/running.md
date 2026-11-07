# Running the cropper on a set of scans

## The easy way: drop and go

**`Process dropped scans.cmd`** in the project root. Two ways to use it:

- **Drag** image files (or a folder of them) straight onto it, or
- **Double-click** it to process whatever is already in `data\inbox\`

Either way it copies the images into the inbox, runs the cropper, and opens
`out\report.html` for you. No terminal, no paths to type.

Clear `data\inbox\` out between batches — it processes everything in there.

> Editing that .cmd? Keep it **CRLF and plain ASCII**. cmd.exe seeks by byte
> offset assuming CRLF, so an LF-only batch file silently eats the first
> characters of lines and fails incomprehensibly. `tests/test_launcher.py`
> guards it.


## One-time setup

Install the package into the project venv, editable. After this there is no
`PYTHONPATH` to remember and it runs from any directory:

```powershell
cd C:\Users\USER\Projects\ai-htr
.venv\Scripts\python.exe -m pip install -e .
```

Editable means edits to `src/` take effect immediately — no reinstall.

## If the forms are already configured

One command. **PowerShell 5.1 has no `&&`**, so `cd` goes on its own line:

```powershell
cd C:\Users\USER\Projects\ai-htr
.venv\Scripts\python.exe -m aihtr "C:\path\to\scans" --out out
```

Currently configured: `base-opd-card-v1`, `nutrition-0190-v1`, `ms-form-v2_4`,
`neuro-v1`. Anything else comes back `UNKNOWN` and is listed, never cropped
against a wrong template.

Useful flags:

| flag | effect |
|---|---|
| `-r` | recurse into subfolders |
| `--limit N` | stop after N images — do this first on a big set |
| `--read` | also run the reader + clinical check (**the baseline reader is weak**) |
| `--no-crops` | manifest only, write no images |
| `--no-refine` | skip ink refinement |
| `--clean` | delete crops from previous runs first |

Roughly 0.4 s per page, so a few hundred scans is a couple of minutes.

## What comes out

```
out/
  report.html           <- OPEN THIS ONE
  manifest.csv          one row per field
  crops/
    <page-stem>/
      bp.tight.png        just the value  -> what a reader would get
      bp.context.png      value + printed label -> what a human checks against
```

### Each run overwrites the last

`report.html` and `manifest.csv` are rewritten wholesale every run, so they
always describe the run that just finished. `crops/` is different — it is
**additive**, so a folder from an earlier run stays on disk and no longer
corresponds to anything in the report. The run tells you when that has happened
and `--clean` removes them. Copy anything you want to keep before re-running.

**`report.html` is the one to open.** Every page shows the **raw input** on the
left and its crops on the right, so a reading can be checked against the actual
card without opening anything else. Readings appear as **systolic and diastolic
separately**, because that is what the pipeline produces — `143/90` is only a
rendering of two integers. Flagged rows are outlined in amber, unrecognised
pages listed at the bottom.
Self-contained — the raw page and crops are embedded, so it opens by
double-click. That also makes it **as identifiable as the scan**: a report built
from a real card carries the whole card, name and HN included. Treat `out/` like
`data/raw/` and never send a real-card report off the machine; reports from the
synthetic `testdata/` pages are the shareable kind.

```powershell
Invoke-Item out\report.html
```

The CSV is for filtering and joining; the report is for *looking*, which is the
part that matters. No automated check can tell you a box landed on the wrong
value — only the context crop can.

`manifest.csv` columns:

| column | meaning |
|---|---|
| `page`, `form`, `owner` | source file, identified form, and which form or insert the field belongs to |
| `field` | field id from the config |
| `projected` | 1 if the field came via an insert located at runtime |
| `x1 y1 x2 y2` | final crop box in canonical space (300 dpi A4) |
| `flags` | `field_empty`, `possibly_clipped`, `occluded_by:<insert>`, `box_adjusted` |
| `needs_review` | 1 if a human should look |
| `reg_inliers`, `reg_error_px` | how well the page aligned — low inliers is a warning sign |
| `page_ms` | wall-clock for the whole page: decode, identify, register, crop, write |
| `t_normalize_ms` … `t_write_ms` | the same time split by stage. Repeated on every field row of a page, like `reg_inliers` |
| `reading` | `143/90`, a rendering of the two numbers below |
| `systolic`, `diastolic` | the actual contract: two integers, carried separately |
| `read_confidence`, `verdict` | only with `--read` |

**Start by filtering `needs_review == 1`.** That is the review queue.

Reading is off by default deliberately: the baseline reader misreads `143/90` as
`111/10`, and emitting numbers by default invites someone to believe them.

## Seeing what each stage did to one page

```powershell
.venv\Scripts\python.exe scripts\trace_page.py data\raw\base-opd-card-v1_001.jpg --degrade
Invoke-Item out-trace\trace.html
```

One page, every stage, with the actual intermediate image from each: as
received, straightened, the landmark matches (green agreed, red discarded), the
alignment overlay, each crop with its boxes, each glyph with its score, and the
final verdict. `--degrade` knocks the page about first so every stage has real
work to do. Use it to understand the pipeline, or to debug a page that came out
wrong.

## Adding a form that is not configured yet

Five steps. Only step 3 needs a human.

**1. Put a clean example in the corpus** and register it:

```powershell
Copy-Item yourform_001.jpg data\raw\
```

Add an entry to `configs/corpus.yaml` (copy an existing block) with `v1: true`.

**2. Write the form config** — `configs/forms/<form-id>.yaml`. Copy
`nutrition-0190-v1.yaml` as the skeleton. Declare each field with its `type`,
and leave `bbox: null`; the boxes come in step 3.

**3. Build the template, draw the boxes, apply them:**

```powershell
.venv\Scripts\python.exe scripts\build_templates.py --all
.venv\Scripts\python.exe scripts\make_box_tool.py
Invoke-Item docs\box-tool.html       # drag a rectangle round each value
.venv\Scripts\python.exe scripts\apply_boxes.py data\boxes\boxes.json
```

If the form carries a pasted insert, **draw the insert's outer region first** —
its fields are stored relative to it, and page registration masks that area out.

**4. Check every box landed:**

```powershell
.venv\Scripts\python.exe -m aihtr.schema        # want N/N
.venv\Scripts\python.exe scripts\preview_crops.py --refine
```

Then **look at** `data/interim/crops_contact_sheet.png`. Do not skip this. A box
in the wrong place produces a perfectly plausible crop of the wrong thing — that
is how a box 80 px off once cropped the word "Survey" and reported success.

**5. Confirm the form is distinguishable** from the ones already configured:

```powershell
.venv\Scripts\python.exe scripts\check_formid.py
```

Every known form should be identified, every unknown one refused. If a new form
is confused with an existing one, they share too much printed layout — see the
margin calibration in `scripts/calibrate_margin.py`.

## When pages come back UNKNOWN

The reason is printed with each one:

- **`too few inliers`** — nothing matched. Usually a form with no template
  (expected), or a page so degraded that registration cannot work.
- **`ambiguous (margin 1.4x)`** — it looks like several known forms at once.
  Normal for a form sharing the base OPD card layout but having no template of
  its own. Add one.

Both are safe outcomes: an unrecognised page is never cropped against a guess.
