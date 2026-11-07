"""One command to prove a checkout works, using the bundled synthetic test data.

    python scripts/run_testdata.py

Builds templates from `testdata/template-sources/`, runs the pipeline over
`testdata/pages/`, and scores every crop against the values the generator
rendered into those pages.

Everything it writes is kept away from a working setup: templates go to
`data/templates-testdata/`, results to `out-testdata/`. It never reads or
writes `data/raw/` or `data/templates/`, so running it cannot disturb the
templates you built from your own cards.

Every page here is synthetic (scripts/make_synthetic.py), so this exercises the
machinery -- form ID, registration, cropping, refinement, insert location --
not the domain gap. Pages are rendered from the same templates the pipeline
matches against, which means form ID and registration are being graded on
their own answer sheet.

Expected on a good run: 23-24 of 24 pages recognised, mean coverage >= 0.98,
nothing misidentified. A refusal ("ambiguous") is a pass, not a failure: the
identifier declines rather than guessing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "testdata"


def run(step: str, *args: str) -> None:
    print(f"\n=== {step} " + "=" * max(4, 66 - len(step)))
    rc = subprocess.call([sys.executable, *args], cwd=ROOT)
    if rc != 0:
        sys.exit(f"\nFAILED at: {step}  (exit {rc})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out-testdata", help="where results go")
    ap.add_argument("--templates", default="data/templates-testdata",
                    help="where the test templates are built")
    args = ap.parse_args()

    if not TESTDATA.is_dir():
        sys.exit(f"no {TESTDATA.name}/ folder — this checkout has no bundled test data")

    pages = sorted((TESTDATA / "pages").glob("*.jpg"))
    sources = sorted((TESTDATA / "template-sources").glob("*.jpg"))
    print(f"{len(sources)} template source(s), {len(pages)} test page(s)")

    run("build templates (and cut the triage insert from its parent)",
        "scripts/build_templates.py", "--all",
        "--corpus", "configs/corpus-testdata.yaml",
        "--raw", str(TESTDATA / "template-sources"),
        "--out", args.templates)

    run("crop every test page",
        "-m", "aihtr", str(TESTDATA / "pages"),
        "--out", args.out, "--templates", args.templates, "--clean")

    run("score the crops against the rendered ground truth",
        "scripts/score_synthetic.py",
        "--truth", str(TESTDATA / "truth.csv"),
        "--manifest", f"{args.out}/manifest.csv")

    print(f"\nreport: {args.out}/report.html")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
