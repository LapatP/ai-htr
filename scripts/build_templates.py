"""Build canonical templates from the raw corpus.

A template is just a normalized page: deskewed, contrast-evened, letterboxed
into canonical space. With one sample per form, the sample *is* the template.

That circularity is real and is why registration is not validated here -- see
scripts/eval_registration.py, which perturbs a page by a known homography and
measures whether the pipeline recovers it.

Usage:  python scripts/build_templates.py [--all]
        python scripts/build_templates.py --corpus configs/corpus-testdata.yaml --raw testdata/template-sources
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr import schema  # noqa: E402
from aihtr.normalize import CANONICAL_SIZE, normalize_page  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "templates"
FORMS = ROOT / "configs" / "forms"


def build_inserts(out_dir: Path = OUT) -> list[str]:
    """Cut every declared insert template out of its parent form template.

    An insert -- the triage sticker -- is a separate sheet laid on the card,
    and `source_region` records where it sat on the page the form template was
    cut from. So the insert template is derivable, and must be: without it
    crop_fields() cannot locate the sticker and silently drops its fields,
    which on a base OPD card means losing the second BP reading and with it the
    whole cross-check.
    """
    built: list[str] = []
    for form_id, spec in sorted(schema.load_all(FORMS).items()):
        parent = out_dir / f"{form_id}.png"
        for ins in spec.inserts:
            if ins.source_region is None:
                print(f"{ins.id:<26} SKIPPED — no source_region in {form_id}.yaml")
                continue
            if not parent.exists():
                print(f"{ins.id:<26} SKIPPED — {parent.name} not built")
                continue
            img = cv2.imread(str(parent), cv2.IMREAD_GRAYSCALE)
            r = ins.source_region
            cut = img[r.y1:r.y2, r.x1:r.x2]
            if cut.size == 0:
                print(f"{ins.id:<26} FAILED  — source_region lies outside {parent.name}")
                continue
            dst = out_dir / f"{ins.id}.png"
            cv2.imwrite(str(dst), cut)
            built.append(ins.id)
            print(f"{ins.id:<26} {cut.shape[1]:>5}x{cut.shape[0]:<5} "
                  f"{'':>7}{'':>6}  {dst.name}  (cut from {parent.name})")
    return built


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="build every page, not just the v1 four")
    ap.add_argument("--corpus", type=Path, default=ROOT / "configs" / "corpus.yaml",
                    help="corpus file listing which raw page builds which form")
    ap.add_argument("--raw", type=Path, default=RAW,
                    help="folder holding the raw pages named in the corpus")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="where templates are written (default data/templates)")
    args = ap.parse_args()

    corpus = yaml.safe_load(args.corpus.read_text(encoding="utf-8"))
    pages = [p for p in corpus["pages"] if args.all or p.get("v1")]

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"canonical space: {CANONICAL_SIZE[0]}x{CANONICAL_SIZE[1]}\n")
    print(f"{'form':<26} {'src WxH':>11} {'skew':>7} {'scale':>6}  out")
    print("-" * 78)

    for page in pages:
        src = args.raw / page["file"]
        if not src.exists():
            print(f"{page['form']:<26} MISSING {src.name}")
            continue

        norm = normalize_page(src)
        dst = out_dir / f"{page['form']}.png"
        cv2.imwrite(str(dst), norm.image)

        raw = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        h, w = raw.shape[:2]
        print(f"{page['form']:<26} {w:>5}x{h:<5} {norm.skew_deg:>6.2f}° "
              f"{norm.scale:>6.2f}  {dst.name}")

    build_inserts(out_dir)

    print(f"\n{len(list(OUT.glob('*.png')))} templates in {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
