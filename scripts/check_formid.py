"""Stage 1 check: identify every page against the four v1 templates.

Two things are under test.

1. Do the four known forms identify correctly?
2. Do the eight forms with NO template get refused rather than guessed?

(2) is the real test. Seven of the eight share the identical base OPD card
background with neuro-v1, so a naive matcher will happily score them high on
shared printed furniture. Whether the margin rule catches that is the whole
question.

Also writes a red/cyan overlay per identified page to data/interim/ so
alignment can be judged by eye rather than taken on trust.

Usage:  PYTHONPATH=src python scripts/check_formid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr.formid import FormIdentifier          # noqa: E402
from aihtr.normalize import CANONICAL_SIZE, normalize_page  # noqa: E402
from aihtr.register import overlay, warp         # noqa: E402

RAW = ROOT / "data" / "raw"
TEMPLATES = ROOT / "data" / "templates"
INTERIM = ROOT / "data" / "interim"


def main() -> int:
    tmpls = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
             for p in sorted(TEMPLATES.glob("*.png"))}
    if not tmpls:
        print("no templates -- run scripts/build_templates.py first")
        return 1

    ident = FormIdentifier(tmpls)
    corpus = yaml.safe_load((ROOT / "configs" / "corpus.yaml").read_text(encoding="utf-8"))
    INTERIM.mkdir(parents=True, exist_ok=True)

    print(f"{len(tmpls)} templates: {', '.join(tmpls)}\n")
    print(f"{'page':<28} {'expected':<22} {'identified':<22} verdict")
    print("-" * 100)

    right = wrong = correctly_refused = falsely_claimed = 0

    for page in corpus["pages"]:
        src = RAW / page["file"]
        if not src.exists():
            continue

        norm = normalize_page(src)
        res = ident.identify(norm.image)

        has_template = page["form"] in tmpls
        expected = page["form"] if has_template else "(no template)"
        got = res.form_id or "UNKNOWN"

        if has_template:
            good = got == page["form"]
            right += good
            wrong += not good
            verdict = "OK" if good else "*** WRONG ***"
        else:
            good = res.form_id is None
            correctly_refused += good
            falsely_claimed += not good
            verdict = "OK (refused)" if good else f"*** FALSE CLAIM: {got} ***"

        scores = "  ".join(f"{s.form_id.split('-')[0]}={s.n_inliers}"
                           for s in res.ranked)
        print(f"{page['file'][:27]:<28} {expected:<22} {got:<22} {verdict}")
        print(f"{'':<28} {res.reason:<22} {scores}")

        if has_template and good and res.best.result.ok:
            w = warp(norm.image, res.best.result.homography, CANONICAL_SIZE)
            ov = overlay(w, tmpls[page["form"]])
            out = INTERIM / f"overlay_{page['form']}.png"
            cv2.imwrite(str(out), cv2.resize(ov, (744, 1052)))

    print("-" * 100)
    print(f"known forms:   {right} correct, {wrong} wrong")
    print(f"unknown forms: {correctly_refused} correctly refused, "
          f"{falsely_claimed} falsely claimed")
    print(f"\noverlays -> {INTERIM.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
