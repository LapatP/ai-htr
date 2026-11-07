# -*- coding: utf-8 -*-
"""Compile a Word-ready sample image set from the AnneSI synthetic OPD-card release.

    https://huggingface.co/datasets/sidataplus/annesi-opd-card-v1-500-v0.1.0

The repo is gated, so give it access one of two ways:

    # a folder holding (part of) the repo: manifest.jsonl + samples/<case>/...
    python scripts/opd_sample_sheet.py --local <folder> --out <folder>

    # or a read token for an account that has been granted access
    $env:HF_TOKEN = "hf_..."
    python scripts/opd_sample_sheet.py --count 6 --out <folder>

Selection is manifest-driven: manifest.jsonl names the real BP crops (two rows per
page — the top-of-card field and the triage sticker), with the written values, so
captions carry ground truth rather than a guess.

Outputs:
    fig01-page.png / -top-context.png / -top-tight.png / -triage-context.png
    contact-sheet.png            every page in one labelled grid
    opd-card-samples.docx        one table per figure, image over caption
    index.csv                    figure list with both BP values and the writer font

These pages are synthetic and non-clinical, but they are rendered from a private
hospital template, so keep the outputs out of git.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = "sidataplus/annesi-opd-card-v1-500-v0.1.0"
API = f"https://huggingface.co/api/datasets/{REPO}"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main"

PAGE_WIDTH_PX = 1600
CROP_WIDTH_PX = 1000
SHEET_COLS = 3


# ---------------------------------------------------------------- source

class Source:
    """The gated HF repo (needs a token) or a local copy of its layout."""

    def __init__(self, local: Path | None, token: str | None):
        self.local = local
        self.token = token

    def read(self, rel: str) -> bytes | None:
        if self.local:
            p = self.local / rel
            return p.read_bytes() if p.is_file() else None
        req = urllib.request.Request(f"{RAW}/{rel}")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (401, 403):
                sys.exit(
                    f"ABORT: {e.code} on {rel}. The dataset is gated — your account needs "
                    "access granted, and $env:HF_TOKEN must hold a read token. "
                    "Or download the repo and pass --local <folder>."
                )
            raise


def load_manifest(src: Source) -> list[dict]:
    for name in ("manifest.jsonl", "manifests/instances.jsonl"):
        blob = src.read(name)
        if blob:
            return [json.loads(l) for l in blob.decode("utf-8").splitlines() if l.strip()]
    sys.exit("ABORT: no manifest.jsonl found in the source")


# ---------------------------------------------------------------- images

def save_png(blob: bytes, width: int, dest: Path) -> tuple[int, int]:
    img = Image.open(io.BytesIO(blob))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "PNG", optimize=True)
    return img.size


def contact_sheet(paths: list[Path], labels: list[str], dest: Path, cols: int = SHEET_COLS) -> None:
    thumbs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im.thumbnail((700, 700), Image.LANCZOS)
        thumbs.append(im)
    cw = max(t.width for t in thumbs) + 20
    ch = max(t.height for t in thumbs) + 46
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * ch), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    for i, (t, label) in enumerate(zip(thumbs, labels)):
        x, y = (i % cols) * cw + 10, (i // cols) * ch + 10
        sheet.paste(t, (x, y))
        draw.rectangle([x, y, x + t.width, y + t.height], outline=(190, 190, 190))
        draw.text((x, y + t.height + 8), label, fill=(20, 20, 20), font=font)
    sheet.save(dest, "PNG", optimize=True)


# ---------------------------------------------------------------- word document

def crops_sheet(rows: list[dict], dest: Path, scale: int = 2) -> None:
    """All the little BP crops in one labelled image: a row per page, a column per crop."""
    cols = [("top_context", "Top of card, with context"),
            ("top_tight", "Tight crop"),
            ("triage_context", "Triage sticker, with context")]
    pad, label_w, header_h, gap = 18, 430 * scale // 2, 46 * scale // 2, 14 * scale // 2

    cells: list[list[Image.Image | None]] = []
    for row in rows:
        line = []
        for key, _ in cols:
            p = row.get(key)
            if not p:
                line.append(None)
                continue
            im = Image.open(p).convert("RGB")
            line.append(im.resize((im.width * scale, im.height * scale), Image.LANCZOS))
        cells.append(line)

    col_w = [max((c[i].width for c in cells if c[i]), default=10) + gap for i in range(len(cols))]
    row_h = [max((im.height for im in line if im), default=10) + gap + 22 * scale // 2 for line in cells]
    width = label_w + sum(col_w) + pad * 2
    height = header_h + sum(row_h) + pad * 2

    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    def font_at(size: int):
        for name in ("arialbd.ttf", "arial.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    f_head = font_at(11 * scale)
    f_label = font_at(10 * scale)
    f_small = font_at(9 * scale)

    x = pad + label_w
    for (key, title), w in zip(cols, col_w):
        draw.text((x, pad + 6), title, fill=(20, 20, 20), font=f_head)
        x += w
    draw.line([(pad, header_h + pad - 8), (width - pad, header_h + pad - 8)], fill=(150, 150, 150), width=2)

    y = header_h + pad
    for row, line, h in zip(rows, cells, row_h):
        draw.text((pad, y + 4), row["sample"], fill=(20, 20, 20), font=f_label)
        draw.text((pad, y + 4 + 15 * scale),
                  f"top {row['bp_top']}   triage {row['bp_triage']}",
                  fill=(90, 90, 90), font=f_small)
        draw.text((pad, y + 4 + 30 * scale), row["font"].replace("google-", ""),
                  fill=(140, 140, 140), font=f_small)
        x = pad + label_w
        for im, w in zip(line, col_w):
            if im:
                sheet.paste(im, (x, y + (h - gap - im.height) // 2))
                draw.rectangle([x - 2, y + (h - gap - im.height) // 2 - 2,
                                x + im.width + 1, y + (h - gap - im.height) // 2 + im.height + 1],
                               outline=(205, 205, 205))
            x += w
        y += h
        draw.line([(pad, y - gap // 2), (width - pad, y - gap // 2)], fill=(228, 228, 228), width=1)

    sheet.save(dest, "PNG", optimize=True)


def build_docx(rows: list[dict], sheet: Path, dest: Path, crops: Path | None = None) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1.0)
    page = doc.sections[0]
    usable = page.page_width - page.left_margin - page.right_margin

    doc.add_heading("Synthetic OPD-card sample pages", level=1)
    intro = doc.add_paragraph(
        "Synthetic, non-clinical pages rendered from the OPD-card template "
        f"({REPO}; 500 pages). Each page carries a handwritten blood pressure in two "
        "places — the top of the card and the triage sticker — with the written values "
        "recorded in the dataset manifest. No patient data appears in these images."
    )
    intro.runs[0].font.size = Pt(10)

    doc.add_picture(str(sheet), width=usable)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Figure 1  Sample synthetic OPD-card pages (n={len(rows)}).")
    r.font.size, r.italic = Pt(9), True

    fig = 2
    if crops and crops.is_file():
        doc.add_picture(str(crops), width=usable)
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(
            f"Figure {fig}  The blood pressure fields cropped from each page: at the top of "
            "the card, the tight crop shipped with the dataset, and on the triage sticker."
        )
        r.font.size, r.italic = Pt(9), True
        fig += 1
    for row in rows:
        doc.add_page_break()
        doc.add_heading(row["sample"], level=2)

        table = doc.add_table(rows=2, cols=1)
        cell = table.cell(0, 0).paragraphs[0]
        cell.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.add_run().add_picture(row["page"], width=usable * 0.72)
        cp = table.cell(1, 0).paragraphs[0]
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cp.add_run(
            f"Figure {fig}  Full synthetic page. Blood pressure written at the top of the "
            f"card: {row['bp_top']} mmHg; on the triage sticker: {row['bp_triage']} mmHg."
        )
        r.font.size, r.italic = Pt(9), True
        fig += 1

        crops = [(row.get("top_context"), f"BP field at the top of the card, with context — written value {row['bp_top']} mmHg"),
                 (row.get("top_tight"), "The same field, tight crop as shipped in the dataset. "
                                        "Note that this crop holds the systolic value and the "
                                        f"slash only; the diastolic value ({row['bp_top'].split('/')[-1]}) "
                                        "falls outside it"),
                 (row.get("triage_context"), f"BP field on the triage sticker, with context — written value {row['bp_triage']} mmHg")]
        for path, what in [(p, w) for p, w in crops if p]:
            t2 = doc.add_table(rows=2, cols=1)
            c = t2.cell(0, 0).paragraphs[0]
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            c.add_run().add_picture(path, width=usable * 0.62)
            c2 = t2.cell(1, 0).paragraphs[0]
            c2.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = c2.add_run(f"Figure {fig}  {what}.")
            r.font.size, r.italic = Pt(9), True
            fig += 1
            doc.add_paragraph()

    try:
        doc.save(dest)
    except PermissionError:                       # the file is open in Word
        alt = dest.with_name(dest.stem + "-new" + dest.suffix)
        doc.save(alt)
        print(f"  ({dest.name} is open in Word — wrote {alt.name} instead)")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--out", default="opd-samples")
    ap.add_argument("--local", help="folder holding a copy of the repo")
    ap.add_argument("--token", help="HF read token (else $HF_TOKEN)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop-scale", type=int, default=2, help="upscale factor for the combined crop image")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    local = Path(args.local) if args.local else None
    if local and not local.is_dir():
        sys.exit(f"ABORT: --local {local} is not a folder")
    if not local and not token:
        sys.exit("ABORT: no access — pass --local <folder> or set $env:HF_TOKEN")

    src = Source(local, token)
    manifest = load_manifest(src)

    cases: dict[str, dict] = {}
    for rec in manifest:
        c = cases.setdefault(rec["sample_id"], {"page": rec["page_path"]})
        c[rec["field_context"]] = rec            # "top" / "triage"

    names = sorted(cases)
    random.Random(args.seed).shuffle(names)
    chosen = sorted(names[: args.count])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, sample in enumerate(chosen, 1):
        case = cases[sample]
        top, triage = case.get("top"), case.get("triage")
        blob = src.read(case["page"])
        if not blob:
            print(f"  skip {sample}: page not available locally")
            continue
        tag = f"fig{i:02d}"
        page_png = out / f"{tag}-page.png"
        size = save_png(blob, PAGE_WIDTH_PX, page_png)

        row = {
            "sample": sample,
            "page": str(page_png),
            "page_px": f"{size[0]}x{size[1]}",
            "bp_top": top["normalized_transcription"] if top else "-",
            "bp_triage": triage["normalized_transcription"] if triage else "-",
            "font": (top or triage or {}).get("source_asset_id", ""),
            "writer": (top or triage or {}).get("writer_id", ""),
        }

        wanted = []
        if top:
            wanted += [("top_context", top["crop_path"], CROP_WIDTH_PX),
                       ("top_tight", top["crop_path"].replace("context-", "tight-"), 700)]
        if triage:
            wanted.append(("triage_context", triage["crop_path"], CROP_WIDTH_PX))
        for key, rel, width in wanted:
            cb = src.read(rel)
            if cb:
                dest = out / f"{tag}-{key.replace('_', '-')}.png"
                save_png(cb, width, dest)
                row[key] = str(dest)

        rows.append(row)
        print(f"  {tag}  {sample}  page {row['page_px']}  top {row['bp_top']}  "
              f"triage {row['bp_triage']}  [{row['font']}]")

    if not rows:
        sys.exit("ABORT: nothing compiled")

    sheet = out / "contact-sheet.png"
    contact_sheet([Path(r["page"]) for r in rows],
                  [f"fig{i:02d}  {r['sample']}  top {r['bp_top']} / triage {r['bp_triage']}"
                   for i, r in enumerate(rows, 1)], sheet)

    crops = out / "bp-crops.png"
    crops_sheet(rows, crops, scale=args.crop_scale)

    build_docx(rows, sheet, out / "opd-card-samples.docx", crops)

    with (out / "index.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["figure", "sample", "bp_top", "bp_triage", "writer_font",
                    "page_pixels", "page_png", "top_context", "top_tight", "triage_context"])
        for i, r in enumerate(rows, 1):
            w.writerow([f"fig{i:02d}", r["sample"], r["bp_top"], r["bp_triage"], r["font"],
                        r["page_px"], r["page"], r.get("top_context", ""),
                        r.get("top_tight", ""), r.get("triage_context", "")])

    print(f"\n{len(rows)} samples -> {out}")


if __name__ == "__main__":
    main()
