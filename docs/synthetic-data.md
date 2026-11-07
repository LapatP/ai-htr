# Synthetic test data

Generates OPD pages with **known values in known places**, so the cropper can be
scored rather than eyeballed.

```powershell
.venv\Scripts\python.exe scripts\make_synthetic.py --per-form 5 --move-sticker
.venv\Scripts\python.exe -m aihtr data\synthetic --out out-synth --clean --read
.venv\Scripts\python.exe scripts\score_synthetic.py
```

## What it produces

`data/synthetic/` with one JPEG per page plus `truth.csv` / `truth.json`
recording, for every field: the value written, the canonical box it was written
into, the degradation level, and whether it was deliberately made to overflow
its box.

Each page gets a plausible BP (systolic 92-185, diastolic at least 18 below it),
rendered in a Hershey script font with random size, thickness, ink darkness and
jitter, then warped and degraded at one of four levels: clean, mild, moderate,
heavy.

Two options worth using:

- `--overflow-rate 0.25` writes a quarter of values **larger than their box**,
  which is what ink refinement exists to catch. The real card does this.
- `--move-sticker` lifts inserts off and re-glues them elsewhere, exercising the
  runtime insert-location path.

## Measured, 16 pages / 32 fields

| level | declared | cropped | meanCov | minCov |
|---|---|---|---|---|
| clean | 8 | 8 | 1.000 | 1.000 |
| mild | 8 | 8 | 0.996 | 0.970 |
| moderate | 8 | 3 | 1.000 | 1.000 |
| heavy | 8 | 6 | 0.980 | 0.879 |

Values written **over** their box: coverage **1.000**. Refinement caught every
one, which is the clearest evidence yet that it does its job.

Reader: **2/8 exact (25%)** on rendered vector digits - far cleaner than pen on
a 103 dpi scan. That is a floor, not a ceiling.

## What it does NOT test

Pages are rendered from the very templates the pipeline matches against, so form
identification and registration are graded on their own answer sheet. Real scans
carry artefacts nobody thought to simulate. A good score here means the
machinery works; it says nothing about real cards.

The handwriting is vector fonts. It exercises segmentation and cropping. It is
not a substitute for real handwriting when evaluating a reader.

`clear_field()` blanks a field before writing into it, because the base card's
template already carries `143/90`. That erases the dotted rule underneath - a
visible artefact of synthesis that a real card would not show.

## The finding it produced immediately

Three of 16 pages came back `UNKNOWN` with margins of 3.81x-5.86x, under the
6.5x gate. **Top-1 was correct 16/16** - these were false refusals, not
misidentifications, so the failure was in the safe direction.

But the cause matters: `MARGIN_RATIO = 6.5` was calibrated *before*
`mask_inserts()` existed. Masking drops the base card's margin from 278x to 28x,
and under degradation it can reach 3.81x - which overlaps the old wrong-form
range (max 3.1x). **The gate needs recalibrating in the masked regime**, and the
separation may genuinely be narrower than it was. Lowering it trades false
refusals against misidentification risk, so it is a decision, not a tweak.
