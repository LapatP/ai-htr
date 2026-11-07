"""Apply boxes.json from the box tool back into the form configs.

Three jobs:

1. **Crop the triage sticker template.** The box tool asks for the sticker's
   outer rectangle on the base card. That rectangle becomes
   data/templates/triage-card-v1.png.

2. **Rebase the sticker's fields.** The two triage BP boxes are drawn in base-card
   coordinates, but they belong to the *insert*, whose position varies from page
   to page. Subtracting the region origin converts them into sticker-relative
   coordinates, which is what makes them survive the sticker being glued on
   crooked.

3. **Derive a context box** for each field, and write both into the YAML.

Configs are edited with ruamel.yaml so the comments explaining each field
survive the round trip -- they are the most useful thing in those files.

Usage:  PYTHONPATH=src python scripts/apply_boxes.py path/to/boxes.json [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr.normalize import CANONICAL_SIZE  # noqa: E402

FORMS = ROOT / "configs" / "forms"
TEMPLATES = ROOT / "data" / "templates"
REGION_PREFIX = "__region__."

# Context boxes are grown from the tight box to take in the printed label
# ("BP", "mmHg") that sits alongside the value. Horizontal generously, because
# labels sit left and right; vertical modestly, because rows are close together.
PAD_X, PAD_Y = 0.55, 0.45


def seq(values) -> CommentedSeq:
    """A flow-style list: `[1900, 560, 2210, 700]`.

    Not cosmetic. A block-style list gives ruamel four new nodes to reattach
    surrounding comments to, and it will happily relocate the comment that
    belonged to the *next* key into the middle of the list. Flow style keeps
    the value a single node, so neighbouring comments stay put.
    """
    s = CommentedSeq(int(v) for v in values)
    s.fa.set_flow_style()
    return s


def clear_stale_comment(node, key: str) -> None:
    """Drop a `# TODO: draw on template` marker once the box exists."""
    tok = node.ca.items.get(key)
    if tok and any(t is not None and "TODO" in getattr(t, "value", "") for t in tok):
        node.ca.items.pop(key, None)


def grow(box, w, h, px=PAD_X, py=PAD_Y):
    x1, y1, x2, y2 = box
    dx, dy = int((x2 - x1) * px), int((y2 - y1) * py)
    return [max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy)]


def rebase(box, origin):
    ox, oy = origin
    return [box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("boxes", help="boxes.json exported from the box tool")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.boxes).read_text(encoding="utf-8"))
    W, H = CANONICAL_SIZE
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    written = skipped = 0

    for form_id, boxes in data.items():
        cfg = FORMS / f"{form_id}.yaml"
        if not cfg.exists():
            print(f"  ! no config for {form_id}, skipping")
            continue
        if not boxes:
            continue

        doc = yaml.load(cfg.read_text(encoding="utf-8"))

        # --- 1 & 2: sticker region, if this form has one -------------------
        origin = None
        for key, box in list(boxes.items()):
            if not key.startswith(REGION_PREFIX):
                continue
            region_id = key[len(REGION_PREFIX):]
            origin = (box[0], box[1])
            out = TEMPLATES / f"{region_id}.png"
            src = TEMPLATES / f"{form_id}.png"
            if src.exists() and not args.dry_run:
                page = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(str(out), page[box[1]:box[3], box[0]:box[2]])
            # Record it on the insert: page registration must mask this area
            # out, or the sticker's features dominate and the card's own fields
            # move whenever the sticker is glued on differently.
            for ins in doc.get("inserts") or []:
                if ins.get("id") == region_id:
                    ins["source_region"] = seq(box)
            print(f"  region {region_id}: {box[2]-box[0]}x{box[3]-box[1]} px "
                  f"-> {out.name}{' (dry run)' if args.dry_run else ''}")

        # --- 3: write field boxes -----------------------------------------
        def place(field_list, ids_in_insert: bool):
            nonlocal written, skipped
            for fld in field_list or []:
                box = boxes.get(fld["id"])
                if box is None:
                    skipped += 1
                    continue
                if ids_in_insert:
                    if origin is None:
                        print(f"  ! {fld['id']} is on an insert but no region drawn")
                        skipped += 1
                        continue
                    box = rebase(box, origin)
                    lim_w, lim_h = 10**6, 10**6      # insert space, no page clamp
                else:
                    lim_w, lim_h = W, H
                clear_stale_comment(fld, "bbox")
                fld["bbox"] = seq(box)
                fld["context_bbox"] = seq(grow(box, lim_w, lim_h))
                written += 1
                print(f"  {fld['id']:<28} {box}")

        place(doc.get("fields"), False)
        for ins in doc.get("inserts") or []:
            place(ins.get("fields"), True)

        if not args.dry_run:
            with cfg.open("w", encoding="utf-8") as fh:
                yaml.dump(doc, fh)
        print(f"  -> {cfg.name}{' (dry run)' if args.dry_run else ''}\n")

    print(f"{written} boxes written, {skipped} still missing")
    if not args.dry_run:
        print("\nverify with:  PYTHONPATH=src python -m aihtr.schema")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
