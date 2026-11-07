# Start here — running this on another computer

This copy carries the code, the configs, and a **synthetic test dataset**, so
you can see the whole cropping pipeline work before you have a card on that
machine. It carries **no patient scans and no templates built from them**.

## 1. Install (about 2 minutes, no network needed)

Needs **Python 3.12** — not 3.13 or 3.14. Check what is on the machine first:

```powershell
py -0
```

If 3.12 is not in that list, install it from python.org before going further.
The `py -3.12` below then picks it whatever else is installed, so you do not
have to remove a newer Python.

Every wheel this needs is bundled in `wheels/`, so the install runs offline:

```powershell
cd <wherever-you-put-this>
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --no-index --find-links wheels -e .
.venv\Scripts\python.exe -m pip install --no-index --find-links wheels pytest
```

`--no-index` forbids pip from contacting PyPI, so if those commands succeed
they succeeded entirely from the bundle. Drop both flags to pull current
versions from the network instead.

`wheels/` is **Windows x64 + CPython 3.12 only**. opencv, numpy, pillow and
lxml ship as compiled binaries, so a Mac, a Linux box or any other Python
version will refuse them and will need the online install.

Installed editable, so edits to `src/` take effect with no reinstall.

The bundle also carries the optional `samples` extra (pillow, python-docx),
which only `scripts/opd_sample_sheet.py` uses — nothing in `src/` imports
either. Add it with
`.venv\Scripts\python.exe -m pip install --no-index --find-links wheels -e .[samples]`.

## 2. Check the install

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Expect **65 passed, 62 skipped** in a couple of seconds. The skips need a corpus of
real scans, which this copy deliberately does not have. A *failure* means the
install is wrong; skips do not.

## 3. Watch it actually run

```powershell
.venv\Scripts\python.exe scripts\run_testdata.py
```

It builds four templates from `testdata/template-sources/`, crops all 24
synthetic pages in `testdata/pages/`, and scores every crop against the values
the generator rendered into them.

What a good run looks like, measured on this machine:

| level | declared | cropped | mean coverage | min coverage |
|---|---|---|---|---|
| clean | 16 | 16 | 1.000 | 1.000 |
| mild | 16 | 16 | 0.992 | 0.879 |
| moderate | 8 | 8 | 0.987 | 0.894 |
| heavy | 8 | 6 | 0.992 | 0.955 |

23 of 24 pages recognised; one heavily degraded page **refused** as ambiguous.
A refusal is a pass — the identifier declines rather than guessing. Nothing was
misidentified.

Then open the report:

```powershell
Invoke-Item out-testdata\report.html
```

Raw page beside every crop, flagged rows highlighted, self-contained in one
file.

This step writes only to `data/templates-testdata/` and `out-testdata/`. It
never touches `data/templates/` or `data/raw/`, so it cannot disturb a working
setup if you later put real cards on this machine.

## 4. When you want to use a real card

Follow **First run: bring your own card** in `README.md`. Short version: drop
the scan in `data/raw/`, add it to `configs/corpus.yaml` and a form config in
`configs/forms/`, run `scripts/build_templates.py`, draw the field boxes with
`scripts/make_box_tool.py`, and check the contact sheet before trusting a crop.

---

## What is in the test data

24 pages across 4 form layouts, in 4 degradation levels (clean, mild, moderate,
heavy), with the triage sticker repositioned per page so the insert-location
path gets exercised. 10 of the 48 values are written deliberately **over** their
box, which is what ink refinement exists to catch. `testdata/truth.csv` holds
every value and its true pixel box.

**What this test data does and does not prove.** It exercises the machinery:
form identification, registration, cropping, refinement, insert location. It
does not test the domain gap — the pages are rendered from the same templates
the pipeline matches against, so form ID and registration are graded on their
own answer sheet, and the handwriting is a vector font, not a pen. Good scores
here mean the machinery works, never that it works on real cards.

**Handling.** Every page is generated — no patient wrote on any of them, and no
identifiers appear. But each is rendered on a form background cut from a real
card, so treat it as institutional imagery: copy it between your own machines,
and keep it off any public remote. `.gitignore` already excludes `testdata/`,
along with `data/raw/`, `data/templates/*.png` and `data/inbox/`.

## Reading, as opposed to cropping

Cropping works. **Reading does not** — the current reader is a
template-correlation baseline with zero learned parameters, and it turns a real
`143/90` into `111/10`. It is off by default (`--read` opts in). Known issues
are listed at the end of `README.md`.
