# ai-htr — hospital OPD form field extraction

Two-part pipeline: **crop** regions out of scanned/photographed hospital OPD
cards, then **read** them into structured data. Everything runs locally — the
eventual inputs are patient records, so nothing leaves the machine.

## v1 scope: "BP everywhere"

One clinical concept, five sites, four templates, **three structural patterns**.
Narrow enough to finish, broad enough to prove the multi-template design.

| Pattern | Encoding | Sites |
|---|---|---|
| **A** | one blank, writer supplies the slash `143/90` | base card, Nutrition |
| **B** | two blanks flanking a **printed** slash | triage sticker, MS form |
| **C** | two separately labelled blanks, no slash | Neuro `Home SBP`/`DBP` |

Readers emit **two integers**, never a string: `143/90` misread as `14/390` is a
one-character slip and clinical nonsense. Entity-level accuracy, not just CER.

Every reading passes through `validate.py` before it is trusted. The load-bearing
rule is **systolic must exceed diastolic** — definitional, free, and the only thing
in the pipeline that catches a clipped leading digit. It grades, it never corrects:
silently repairing `43/90` to `143/90` would be the confident fabrication that makes
a VLM dangerous here. Range checks cannot catch a plausible misread (`143`->`148`);
`cross_check()` can, because the base card and the triage sticker record the same
visit twice.

## Architecture

```
0. normalize    dewarp / deskew / denoise            form-agnostic
1. form id      ORB match vs every template,          ← keystone
                pick highest RANSAC inlier count
2. register     warp input into canonical space
3. crop         fixed boxes from per-form YAML
4. read         one reader per field archetype
```

Stage 1 by inlier count (not a trained classifier) because it hands stage 2 its
homography for free **and** reports "unknown form" instead of guessing.

**A form is data, not code.** Adding a form = a YAML in `configs/forms/`.
Never edit Python to add one. Seven field archetypes are declared in
`schema.py`; only `line_value` is implemented. `diagram` (stick figures, foot
monofilament points) is deliberately deferred — it is a research problem.

All `bbox` values are in **canonical template space**, not input-image space.

## Environment — three gotchas

1. **RTX 5050 is Blackwell (sm_120).** torch must come from the CUDA 12.8 index
   or it installs fine and fails at runtime with "no kernel image is available".
2. **Use Python 3.12**, not the `py` default of 3.14 (ML wheels are patchy).
3. **Print run is NOT a form ID.** nutrition-0190 and chest both carry
   `3591/30,000/Mat.10039712` with completely different layouts — it marks the
   paper stock, not the form. Form identification must be feature-based.
4. **8 GB VRAM** is the ceiling. Typhoon OCR 1.5 2B QAT (~2 GB) fits comfortably;
   bf16 (~6-8 GB) is borderline. LocateAnything-3B was dropped — Linux-only,
   no WSL installed, and would OOM.

PowerShell is `RemoteSigned`, so call `.venv\Scripts\python.exe` directly rather
than relying on `Activate.ps1`. Windows console is cp1252 — reconfigure stdout
to utf-8 before printing Thai.

## Resolution ceiling — read this before trusting any accuracy number

All 12 source pages are LINE exports, downscaled to **1200 px tall = ~103 DPI at
A4** (the base card is slightly better at 1537 px / ~131 DPI). Document scanning
standard is 300 DPI; OCR wants 200+ minimum.

At 103 DPI a handwritten `143/90` is roughly **60 x 25 px**, and 8pt printed Thai
is ~11 px tall — effectively unreadable.

Consequence: **stages 0-3 (cropping) are fine** — ORB finds ample keypoints on
ruled lines and printed blocks at 1 MP. **Stage 4 (reading) is capped**, and
digits in vitals are exactly the highest-stakes characters.

Decision: **canonical space is defined at 300 DPI (2480x3508)** even though
current sources are 103 DPI. Upsampling today is lossy-but-harmless; if we
defined canonical space at 103 DPI, future good scans would have to be
downsampled and that loss would be permanent. Configs never need rewriting.

## Running it on a set of scans

Easiest: drag scans onto **`Process dropped scans.cmd`**, or double-click it to
process `data\inbox\`. It opens the report itself. That .cmd must stay **CRLF +
ASCII** — cmd.exe seeks by byte offset assuming CRLF, and an LF-only batch file
eats the first characters of lines (`rem` -> `m`). Guarded by
`tests/test_launcher.py` after it bit us once.

Or by hand:

Installed editable (`pip install -e .`), so no PYTHONPATH is needed.
**The development shell is PowerShell 5.1.** Three rules for any command written for it:

1. No `&&` (parse error) and no `VAR=x cmd` (not valid syntax) - one command per line.
2. **Absolute paths.** The shell is often not sitting in the project folder, so
   relative paths like `.venv\Scripts\python.exe` fail.
3. **Prefix an executable path with `&`.** A bare leading `.` makes PowerShell try
   to dot-source or auto-load a module: `.venv\Scripts\python.exe` errors with
   "The module '.venv' could not be loaded".

Works from any directory (the package is installed editable, and its default
`--forms`/`--templates` paths resolve from the package location, not the cwd):

```powershell
& "C:\Users\USER\Projects\ai-htr\.venv\Scripts\python.exe" -m aihtr "C:\path\to\scans" --out "C:\path\to\results"
```

```powershell
cd C:\Users\USER\Projects\ai-htr
.venv\Scripts\python.exe -m aihtr "C:\path\to\scans" --out out
```

Writes `out/report.html` (**the thing to open** — raw input beside every crop,
systolic/diastolic shown separately, flagged rows highlighted), `out/manifest.csv`, and `out/crops/<page>/<field>.{tight,context}.png`.
~0.4 s per page. Unrecognised forms are listed, never cropped against a guess.
Reading is **off by default** — the baseline reader misreads `143/90` as `111/10`,
and emitting numbers by default invites someone to believe them; `--read` opts in.
Full runbook, including how to onboard a new form: `docs/running.md`.

## Commands

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
PYTHONPATH=src .venv/Scripts/python.exe -m aihtr.schema        # which boxes still need drawing
PYTHONPATH=src .venv/Scripts/python.exe scripts/make_box_tool.py    # regenerate docs/box-tool.html
PYTHONPATH=src .venv/Scripts/python.exe scripts/apply_boxes.py data/boxes/boxes.json
PYTHONPATH=src .venv/Scripts/python.exe scripts/preview_crops.py   # tight vs context sheet
PYTHONPATH=src .venv/Scripts/python.exe scripts/grid_preview.py FORM x0 y0 x1 y1 --step 25
```

Box tool: open `docs/box-tool.html` in a browser (self-contained, no server).
Draw the sticker region **first** — the two triage BP boxes are rebased into its
coordinates. Export the JSON, then run `apply_boxes.py` on it. Configs are edited
with ruamel.yaml so the explanatory comments survive; boxes are written flow-style
because block-style lists let ruamel relocate neighbouring comments into the value.

## Inserts must be masked out of the template

`mask_inserts()` blanks declared insert regions before form ID *and* page
registration. Without it the sticker carries 89-100% of the inliers and the page
gets aligned to the sticker: moving it 60 px dragged the card's own `bp` field
from 1.000 similarity to 0.018. A card with no sticker also failed form ID
outright (2.66x margin, under the 6.5x gate). Inserts record `source_region` in
their YAML so masking knows where to blank.

A field sitting *under* an insert is flagged `occluded_by:<insert>`: inserts are
located before base fields are cropped, so a sticker covering a field is caught
rather than silently cropping the sticker's content.

Third finding rooted in the same fact: **the base card template is a filled page
with a sticker on it.** A blank base OPD card would dissolve this, the template-
subtraction blocker, and the tuned constants in `refine.py` together.

## Deskew declines under perspective

`estimate_skew` is exact for pure rotation (0/36 errors over +/-6 deg) and
catastrophic under keystone: it reported -8.5 deg on a page tilted -2.5, because
non-parallel rules make the median meaningless. normalize_page then rotated by
the bogus angle and registration FAILED (19 inliers vs 63 with no normalisation).

Guarded by the spread of the line angles: parallel rules give an IQR of
0.06-0.47 deg, perspective gives 3.5-7.3. Above `MAX_SKEW_IQR` it returns 0.0
and does not rotate. Safe because a homography absorbs rotation anyway --
measured 10 deg registering fine with 2000+ inliers and no deskew.
Heavy-degradation form ID went 25/32 -> 27/32.

## Synthetic test data

`scripts/make_synthetic.py` renders known BP values into known boxes and writes
`truth.csv`; `scripts/score_synthetic.py` scores a run against it. See
`docs/synthetic-data.md`. Coverage 0.98-1.00, and values deliberately written
over their box score 1.000 - refinement earning its keep.

**Open issue it surfaced:** `MARGIN_RATIO = 6.5` was calibrated before
`mask_inserts()` existed. Masked margins under degradation reach 3.81x, which
overlaps the old wrong-form range. Top-1 stayed correct 16/16, so these are
false refusals (safe direction), but the gate wants recalibrating.

## Cropping is measured

`docs/cropping-evaluation.md`. Coverage is flat at min 0.973 / mean 0.996 across
clean, mild, moderate and heavy degradation. Form ID and registration are 32/32
up to moderate, 25/32 at heavy — and **all 6 heavy failures were refusals, 0 were
misidentifications**.

Refinement must stay **grow-only**. Free refinement measured min coverage 0.230
under moderate degradation against 0.973 with no refinement at all: the feature
built to stop clipping was itself clipping on degraded scans. The nominal box is
a human's statement of where the field is; ink may extend it, never overrule it.

## State

- [x] Schema, config loader, 4 form configs
- [x] 12 source pages identified and copied to `data/raw/` (`configs/corpus.yaml`)
- [x] Templates built for the four v1 forms
- [x] **Stages 0-3 done** — normalize, form ID, registration, crop. 123 tests passing.
- [x] **Clinical plausibility guardrail** (`validate.py`) — it catches
      the exact clipped-digit bug that slipped past every image-level diagnostic
- [x] Box tool + `apply_boxes.py`; all 9 boxes drawn 2026-08-20 (`data/boxes/boxes.json`)
- [x] Ink-aware crop refinement, **grow-only** + `possibly_clipped` / `field_empty` flags
      (`refine.py`; deviates from the original design — see `docs/stage3-ink-refinement.md`)
- [~] **Stage 4 wired end to end** (`digits.py`, `read.py`, `pipeline.py`) but the
      baseline reader is **not usable**: it reads `143/90` as `111/10`. Hershey vector
      templates do not resemble handwriting at 103 DPI. Every error was caught by
      validate.py + cross_check, so nothing wrong escaped — but 0/1 correct is 0/1.
- [ ] **Reader model decision** — real handwritten digit training data, or the
      Typhoon OCR VLM path. Blocked more by data than by model: with one filled BP
      field in the whole corpus, no reader can be evaluated.
- [~] **Stage 4 wired end to end** (`digits.py`, `read.py`, `pipeline.py`) but the
      baseline reader is **not usable**: it reads `143/90` as `111/10`. Hershey vector
      templates do not resemble handwriting at 103 DPI. Every error was caught by
      validate.py + cross_check, so nothing wrong escaped — but 0/1 correct is 0/1.
- [ ] **Reader model decision** — real handwritten digit training data, or the
      Typhoon OCR VLM path. Blocked more by data than by model: with one filled BP
      field in the whole corpus, no reader can be evaluated.

### Measured results (2026-08-20)

Registration, validated by **synthetic perturbation** — the only honest option
with one page per form. Warp a template by a homography we chose, ask the
pipeline to recover it, measure where a grid of points lands:

| | mean | p95 | max | failures |
|---|---|---|---|---|
| clean | 0.35 px | 1.21 px | 7.25 px | 0/20 |
| blur+noise+JPEG | 0.30 px | 0.95 px | 3.43 px | 0/20 |

Sub-pixel, against a BP field ~175 x 73 px in canonical space. (The two runs are
not the same perturbations — the degrade path consumes RNG draws — so
"degraded scored better" is noise, not a finding.)

Form identification: **4/4 known forms correct, 8/8 unknown forms refused.**

That second number took a correction. `MARGIN_RATIO` was guessed at 1.5x and let
five wrong forms through, because seven of the twelve pages share the identical
base OPD card background and score real inliers on shared furniture.
`scripts/calibrate_margin.py` measured the actual distributions under
perturbation — correct 14.3x-152x, wrong 1.0x-3.1x — and 6.5 sits in the gap.

**Do not trust the raw inlier counts in `check_formid.py`** (they saturate at
5000): each template IS the page under test. Same for the overlays — perfect
alignment there is a plumbing check, not evidence. The perturbation numbers are
the evidence.

## Related

- Dataset `sidataplus/annesi-opd-card-v1-500-v0.1.0` (gated, access requested):
  500 synthetic pages, 14 instances each, **one** template. Content and noise
  variety only — zero layout variety. Its `compiled-template/template.yaml` and
  `templates.jsonl` validate this config-driven approach.
- Domain gap is the main risk: synthetic renders lack skew, binder blobs, faded
  print, crooked stickers. A hand-labelled **real** eval set is essential.
