"""Batch entrypoint: point it at a folder of scans, get crops back.

    python -m aihtr data/raw --out out/

Processes every image through normalize -> identify -> register -> crop, writes
a tight and a context crop per field, and records one row per field in
`manifest.csv`. Pages whose form is not recognised are listed, not guessed at.

Reading (stage 4) is OFF by default. The baseline reader is not accurate enough
to trust, so producing numbers by default would invite someone to believe them.
`--read` turns it on and adds the reading and its clinical verdict to the
manifest, always alongside the flags.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import cv2

from . import schema
from .crop import crop_fields, mask_inserts
from .formid import FormIdentifier
from .normalize import CANONICAL_SIZE, normalize_page
from .read import read_pattern_a, read_pattern_split
from .register import register, warp
from .report import write_report
from .validate import validate_bp

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
ROOT = Path(__file__).resolve().parents[2]


def find_images(target: Path, recursive: bool) -> list[Path]:
    if target.is_file():
        return [target]
    it = target.rglob("*") if recursive else target.glob("*")
    return sorted(p for p in it if p.suffix.lower() in EXTS)


def load_env(forms_dir: Path, templates_dir: Path):
    """Load form specs and every built template.

    Returns the missing templates in two buckets. They fail differently and
    only one of them used to be reported: a missing FORM template means pages
    of that form are refused and counted as unrecognised, which is loud. A
    missing INSERT template is silent -- crop_fields() cannot locate the insert
    and simply drops its fields, so a card whose triage sticker carries a
    second BP yields one reading instead of two, with nothing in the manifest
    to say the other was never looked for.
    """
    forms = schema.load_all(forms_dir)
    templates = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                 for p in sorted(templates_dir.glob("*.png"))}
    missing = [f for f in forms if f not in templates]
    missing_inserts = sorted({
        f"{ins.id} (on {form_id})"
        for form_id, spec in forms.items()
        if form_id in templates
        for ins in spec.inserts
        if ins.id not in templates
    })
    return forms, templates, missing, missing_inserts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aihtr", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="image file, or folder of images")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--forms", type=Path, default=ROOT / "configs" / "forms")
    ap.add_argument("--templates", type=Path, default=ROOT / "data" / "templates")
    ap.add_argument("-r", "--recursive", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N images")
    ap.add_argument("--read", action="store_true",
                    help="also run the reader and the clinical check (baseline is weak)")
    ap.add_argument("--no-refine", action="store_true", help="skip ink refinement")
    ap.add_argument("--no-crops", action="store_true", help="manifest only, write no images")
    ap.add_argument("--no-report", action="store_true", help="skip the HTML report")
    ap.add_argument("--clean", action="store_true",
                    help="delete crops from previous runs before starting")
    args = ap.parse_args(argv)

    images = find_images(args.input, args.recursive)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"no images found in {args.input}")
        return 1

    forms, templates, missing, missing_inserts = load_env(args.forms, args.templates)
    if not templates:
        print(f"no templates in {args.templates} — run scripts/build_templates.py first")
        return 1
    if missing:
        print(f"warning: configured but no template built: {', '.join(missing)}")
    if missing_inserts:
        print(f"warning: insert template missing, its fields will NOT be cropped: "
              f"{', '.join(missing_inserts)}")
        print("         run scripts/build_templates.py to cut it from the form template")
    if missing or missing_inserts:
        print()

    # Identify and register against the CARD, never against a sticker on it.
    ident = FormIdentifier({k: mask_inserts(v, forms[k])
                            for k, v in templates.items() if k in forms})

    args.out.mkdir(parents=True, exist_ok=True)
    crop_root = args.out / "crops"
    # report.html and manifest.csv are rewritten wholesale each run, but crops/
    # is additive: a folder from a previous run survives and no longer
    # corresponds to anything in the report. Removing it is opt-in, because
    # --out may point somewhere the user keeps other things.
    if args.clean and crop_root.exists():
        shutil.rmtree(crop_root)
        print(f"cleaned previous crops from {crop_root}\n")

    rows, unknown, failed = [], [], []
    page_sources: dict[str, Path] = {}   # so the report can show the raw input
    # Wall-clock per page, kept for EVERY image including the ones that come
    # back unknown or fail — a page that was slow and then refused still cost
    # the time, and averaging only over successes would hide that.
    page_secs: dict[str, float] = {}
    stage_totals: dict[str, float] = {}
    t0 = time.time()

    print(f"{len(images)} image(s) -> {args.out}\n")
    print(f"{'page':<34} {'form':<22} {'fields':>7} {'review':>7} {'time':>8}")
    print("-" * 83)

    for img_path in images:
        # perf_counter, not time(): monotonic and higher resolution, so a clock
        # adjustment mid-run cannot produce a negative duration.
        t_page = time.perf_counter()
        stage: dict[str, float] = {}

        def _lap(name: str, since: float) -> float:
            now = time.perf_counter()
            stage[name] = now - since
            return now

        try:
            page = normalize_page(img_path).image
        except Exception as exc:                       # unreadable file
            page_secs[img_path.name] = time.perf_counter() - t_page
            failed.append((img_path.name, str(exc)))
            continue
        mark = _lap("normalize", t_page)

        res = ident.identify(page)
        mark = _lap("identify", mark)
        if res.form_id is None:
            page_secs[img_path.name] = time.perf_counter() - t_page
            unknown.append((img_path.name, res.reason))
            print(f"{img_path.name[:33]:<34} {'UNKNOWN':<22} {'-':>7} {'-':>7} "
                  f"{page_secs[img_path.name]:>7.2f}s")
            continue

        spec = forms[res.form_id]
        reg = register(page, mask_inserts(templates[res.form_id], spec))
        if not reg.ok:
            page_secs[img_path.name] = time.perf_counter() - t_page
            failed.append((img_path.name, f"registration failed ({reg})"))
            print(f"{img_path.name[:33]:<34} {res.form_id[:21]:<22} {'REG FAIL':>7} "
                  f"{'':>7} {page_secs[img_path.name]:>7.2f}s")
            continue
        page = warp(page, reg.homography, CANONICAL_SIZE)
        mark = _lap("register", mark)

        crops = crop_fields(page, spec, templates, do_refine=not args.no_refine)
        mark = _lap("crop", mark)
        stem = img_path.stem
        page_sources[img_path.name] = img_path
        if not args.no_crops and crops:
            (crop_root / stem).mkdir(parents=True, exist_ok=True)

        by_id = {c.field_id: c for c in crops}
        n_review = 0
        for c in crops:
            if not args.no_crops:
                d = crop_root / stem
                cv2.imwrite(str(d / f"{c.field_id}.tight.png"), c.tight)
                cv2.imwrite(str(d / f"{c.field_id}.context.png"), c.context)

            row = {
                "page": img_path.name, "form": res.form_id, "owner": c.owner,
                "field": c.field_id, "projected": int(c.projected),
                "x1": c.tight_box[0], "y1": c.tight_box[1],
                "x2": c.tight_box[2], "y2": c.tight_box[3],
                "flags": "|".join(c.flags), "needs_review": int(c.needs_review),
                "reg_inliers": reg.n_inliers, "reg_error_px": round(reg.reproj_error, 2),
            }
            n_review += c.needs_review
            rows.append(row)

        _lap("write", mark)
        if args.read:
            _add_readings(rows, by_id, img_path.name)

        # Page total covers everything above: decode, identify, register, crop,
        # and writing the crop PNGs. It is the wall-clock the operator waits.
        secs = time.perf_counter() - t_page
        page_secs[img_path.name] = secs
        for k, v in stage.items():
            stage_totals[k] = stage_totals.get(k, 0.0) + v
        for row in rows[-len(crops):] if crops else []:
            row["page_ms"] = round(secs * 1000)
            for k, v in stage.items():
                row[f"t_{k}_ms"] = round(v * 1000)

        print(f"{img_path.name[:33]:<34} {res.form_id[:21]:<22} {len(crops):>7} "
              f"{n_review:>7} {secs:>7.2f}s")

    # ---- manifest ----
    if rows:
        cols = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    dt = time.time() - t0
    done = len({r["page"] for r in rows})
    fresh = {Path(r["page"]).stem for r in rows}
    stale = sorted(d.name for d in crop_root.iterdir()
                   if d.is_dir() and d.name not in fresh) if crop_root.exists() else []
    print("-" * 83)
    print(f"{done} page(s) cropped, {len(rows)} field(s), "
          f"{sum(r['needs_review'] for r in rows)} flagged for review")
    print(f"{len(unknown)} unrecognised, {len(failed)} failed   ({dt:.1f}s total)")

    # Median and slowest, not just the mean. A mean hides the one page that
    # took six seconds, and that page is the interesting one.
    if page_secs:
        vals = sorted(page_secs.values())
        mid = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals)//2 - 1] + vals[len(vals)//2]) / 2
        slow_name, slow = max(page_secs.items(), key=lambda kv: kv[1])
        print(f"per page: {mid:.2f}s median · {vals[0]:.2f}s fastest · "
              f"{slow:.2f}s slowest ({slow_name[:40]})")
        if stage_totals and done:
            order = ("normalize", "identify", "register", "crop", "write")
            bits = " · ".join(f"{k} {stage_totals[k]/done*1000:.0f}ms"
                              for k in order if k in stage_totals)
            print(f"mean split: {bits}")
    if unknown:
        print("\nunrecognised — these need a template and boxes before they can be cropped:")
        for name, why in unknown[:20]:
            print(f"  {name:<40} {why}")
        if len(unknown) > 20:
            print(f"  ... and {len(unknown)-20} more")
    if failed:
        print("\nfailed:")
        for name, why in failed[:20]:
            print(f"  {name:<40} {why}")
    if stale:
        shown = ", ".join(stale[:6]) + (" ..." if len(stale) > 6 else "")
        print(f"\n{len(stale)} crop folder(s) left over from an earlier run, "
              f"not part of this report:")
        print(f"  {shown}")
        print("  re-run with --clean to remove them")
    if rows:
        print(f"\nmanifest: {args.out / 'manifest.csv'}")
        if not args.no_crops:
            print(f"crops:    {crop_root}")

    # The report is the thing a person actually opens. A CSV cannot show you
    # whether a box landed on the right value; the context crops can.
    if not args.no_report and not args.no_crops and (rows or unknown or failed):
        rp = write_report(rows, unknown, failed, args.out, crop_root,
                          str(args.input), dt, stale, page_sources)
        print(f"REPORT:   {rp}   <- open this one")
    return 0


def _add_readings(rows, by_id, page_name) -> None:
    """Attach stage 4/5 output to the rows for this page, pairing split fields."""
    from .pipeline import _role
    groups: dict[str, dict] = {}
    for fid, c in by_id.items():
        g, role = _role(fid)
        key = g if not c.projected else f"{c.owner}/{g}"
        groups.setdefault(key, {})[role] = c

    for key, parts in groups.items():
        used = [parts[k] for k in ("single", "systolic", "diastolic") if k in parts]
        sys_v = dia_v = ""
        if any("field_empty" in c.flags for c in used):
            text, conf, verdict = "", 0.0, "field_empty"
        elif "single" in parts:
            rd = read_pattern_a(parts["single"].tight)
            text, conf = rd.text, rd.confidence
            verdict = validate_bp(rd.systolic, rd.diastolic).verdict.value if rd.readable else "unreadable"
            sys_v, dia_v = (rd.systolic, rd.diastolic) if rd.readable else ("", "")
        elif "systolic" in parts and "diastolic" in parts:
            rd = read_pattern_split(parts["systolic"].tight, parts["diastolic"].tight)
            text, conf = rd.text, rd.confidence
            verdict = validate_bp(rd.systolic, rd.diastolic).verdict.value if rd.readable else "unreadable"
            sys_v, dia_v = (rd.systolic, rd.diastolic) if rd.readable else ("", "")
        else:
            continue
        for c in used:
            for r in rows:
                if r["page"] == page_name and r["field"] == c.field_id:
                    r["group"] = key
                    r["reading"] = text
                    # Carried separately as well as combined: the pipeline's
                    # contract is two integers, and "143/90" is only a rendering
                    # of them. A consumer joining on this should never have to
                    # split a string back apart.
                    r["systolic"] = sys_v
                    r["diastolic"] = dia_v
                    r["read_confidence"] = round(conf, 3)
                    r["verdict"] = verdict


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
