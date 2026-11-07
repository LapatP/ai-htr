"""Run ONE page through every stage and show what each one produced.

Writes a self-contained `trace.html` with the actual intermediate image from
each step - the page as received, straightened, the landmark matches, the
alignment overlay, each crop, each glyph, the verdict. Useful for understanding
the pipeline, and for debugging a page that came out wrong.

Usage:
  python scripts/trace_page.py data/raw/base-opd-card-v1_001.jpg
  python scripts/trace_page.py somepage.jpg --out out-trace --degrade
"""
from __future__ import annotations

import argparse
import base64
import html as H
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr import schema                                                    # noqa: E402
from aihtr.crop import crop_fields, mask_inserts                            # noqa: E402
from aihtr.digits import read_glyphs                                        # noqa: E402
from aihtr.formid import MARGIN_RATIO, FormIdentifier                       # noqa: E402
from aihtr.normalize import (CANONICAL_SIZE, estimate_skew, load_gray,      # noqa: E402
                             normalize_image, normalize_page)
from aihtr.read import read_pattern_a, read_pattern_split                   # noqa: E402
from aihtr.register import detect, match, overlay, register, warp           # noqa: E402
from aihtr.synth import random_homography                                   # noqa: E402
from aihtr.validate import validate_bp                                      # noqa: E402

W, HGT = CANONICAL_SIZE
TEMPLATES = ROOT / "data" / "templates"


def embed(img, max_w=900) -> str:
    """Inline an image as a data URI.

    JPEG for anything page-sized, PNG for small crops. A scanned page is
    photographic — PNG stores it almost losslessly and the trace ballooned to
    11 MB. Crops stay PNG because at 2x zoom the digit edges matter.
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, (max_w, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)

    if img.shape[0] * img.shape[1] > 250_000:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        mime = "jpeg"
    else:
        ok, enc = cv2.imencode(".png", img)
        mime = "png"
    if not ok:
        return ""
    return f"data:image/{mime};base64," + base64.b64encode(enc.tobytes()).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "out-trace")
    ap.add_argument("--degrade", action="store_true",
                    help="knock the page about first, so each stage has real work to do")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    forms = schema.load_all(ROOT / "configs" / "forms")
    templates = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                 for p in TEMPLATES.glob("*.png")}
    args.out.mkdir(parents=True, exist_ok=True)
    S: list[str] = []

    def stage(n, title, sub):
        S.append(f'<section><div class="sh"><span class="n">{n}</span>'
                 f'<h2>{title}</h2><p>{sub}</p></div>')

    def shot(img, cap, wide=900):
        S.append(f'<figure><img src="{embed(img, wide)}" alt="">'
                 f'<figcaption>{cap}</figcaption></figure>')

    def facts(pairs):
        S.append('<dl class="facts">' + "".join(
            f"<dt>{k}</dt><dd>{v}</dd>" for k, v in pairs) + "</dl>")

    # ---- input -------------------------------------------------------------
    raw = load_gray(args.image)
    if args.degrade:
        rng = np.random.default_rng(args.seed)
        base = normalize_page(args.image).image
        Ht = random_homography(CANONICAL_SIZE, rng, rot=3.0, scale=0.05,
                               trans=50, persp=6e-5)
        raw = cv2.GaussianBlur(warp(base, Ht, CANONICAL_SIZE), (3, 3), 0)
        raw = np.clip(raw.astype(np.int16) + rng.normal(0, 6, raw.shape),
                      0, 255).astype(np.uint8)

    stage(0, "Normalize", "any photo or scan &rarr; upright grayscale page, 2480 &times; 3508")
    skew_in = estimate_skew(raw)
    norm = normalize_image(raw, source=args.image)
    shot(raw, f"as received &mdash; {raw.shape[1]} &times; {raw.shape[0]} px")
    shot(norm.image, "straightened, contrast evened, letterboxed into canonical space")
    facts([("tilt measured", f"{skew_in:+.2f}&deg;"
            + (" &mdash; <b>declined</b>, the printed rules disagreed too much to trust"
               if skew_in == 0.0 else "")),
           ("residual tilt", f"{estimate_skew(norm.image):+.2f}&deg;"),
           ("scale applied", f"&times;{norm.scale:.3f}")])
    S.append("</section>")

    # ---- form id -----------------------------------------------------------
    stage(1, "Identify the form", "page &rarr; which template this is, or UNKNOWN")
    ident = FormIdentifier({k: mask_inserts(v, forms[k])
                            for k, v in templates.items() if k in forms})
    res = ident.identify(norm.image)
    top = max((s.n_inliers for s in res.ranked), default=1)
    S.append('<div class="bars">')
    for s in res.ranked:
        win = s.form_id == res.form_id
        S.append(f'<div class="bar"><span class="lbl">{H.escape(s.form_id)}</span>'
                 f'<span class="track"><i class="{"win" if win else ""}" '
                 f'style="width:{max(2, s.n_inliers / top * 100):.1f}%"></i></span>'
                 f'<span class="val">{s.n_inliers}</span></div>')
    S.append("</div>")
    facts([("verdict", f"<b>{res.form_id or 'UNKNOWN'}</b>"),
           ("reason", H.escape(res.reason)),
           ("gate", f"winner must beat runner-up by {MARGIN_RATIO}&times;")])
    S.append("</section>")
    if res.form_id is None:
        S.append('<p class="note">Stopped here. An unrecognised page is never '
                 'cropped against a guess.</p>')
        return _write(S, args, str(args.image))

    spec = forms[res.form_id]
    tmpl = mask_inserts(templates[res.form_id], spec)

    # ---- registration ------------------------------------------------------
    stage(2, "Align to the template",
          "page + template &rarr; the page moved into the template's exact position")
    kp_p, des_p = detect(norm.image)
    kp_t, des_t = detect(tmpl)
    knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(des_p, des_t, k=2)
    good = match(des_p, des_t)
    reg = register(norm.image, tmpl)
    pp = np.float32([kp_p[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pt = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    Hm, mask = cv2.findHomography(pp, pt, cv2.RANSAC, 4.0)
    inl = mask.ravel().astype(bool)

    rng2 = np.random.default_rng(3)
    sel = rng2.choice(len(good), size=min(55, len(good)), replace=False)
    vis = cv2.cvtColor(np.hstack([norm.image, tmpl]), cv2.COLOR_GRAY2BGR)
    for i in sel:
        m = good[i]
        p1 = tuple(int(v) for v in kp_p[m.queryIdx].pt)
        p2 = (int(kp_t[m.trainIdx].pt[0]) + W, int(kp_t[m.trainIdx].pt[1]))
        c = (70, 160, 60) if inl[i] else (60, 60, 210)
        cv2.line(vis, p1, p2, c, 5, cv2.LINE_AA)
        cv2.circle(vis, p1, 13, c, -1)
        cv2.circle(vis, p2, 13, c, -1)
    shot(vis, "landmark matches &mdash; green agree on one transform, red discarded", 1000)

    aligned = warp(norm.image, reg.homography, CANONICAL_SIZE)
    shot(overlay(aligned, tmpl),
         "aligned page over the template &mdash; sharp = agreement, colour fringing = drift")
    a = reg.homography[:2, :2]
    facts([("landmarks", f"{len(kp_p)} page / {len(kp_t)} template"),
           ("pairings tried", f"{len(knn)}"),
           ("unambiguous", f"{len(good)} ({len(good)/max(len(knn),1)*100:.0f}% kept)"),
           ("self-consistent", f"<b>{reg.n_inliers}</b> ({reg.inlier_ratio:.0%} of those)"),
           ("residual", f"<b>{reg.reproj_error:.2f} px</b>"),
           ("recovered", f"rotation {np.degrees(np.arctan2(a[1,0], a[0,0])):+.2f}&deg;, "
                         f"scale {np.hypot(a[0,0], a[1,0]):.3f}, "
                         f"shift {reg.homography[0,2]:+.0f}, {reg.homography[1,2]:+.0f} px")])
    S.append("</section>")

    # ---- crop + refine -----------------------------------------------------
    stage(3, "Cut the fields out",
          "aligned page + box coordinates &rarr; a tight and a context crop per field")
    crops = crop_fields(aligned, spec, templates, do_refine=True)
    S.append('<div class="grid">')
    for fc in crops:
        r = fc.refinement
        ctx = cv2.cvtColor(fc.context, cv2.COLOR_GRAY2BGR)
        cx, cy = fc.context_box[0], fc.context_box[1]
        if r and r.grew:
            n = r.original
            cv2.rectangle(ctx, (n[0]-cx, n[1]-cy), (n[2]-cx, n[3]-cy), (190, 190, 190), 2)
        b = fc.tight_box
        cv2.rectangle(ctx, (b[0]-cx, b[1]-cy), (b[2]-cx, b[3]-cy), (190, 95, 45), 3)
        chips = "".join(f'<span class="chip">{H.escape(f)}</span>' for f in fc.flags) \
            or '<span class="chip ok">clean</span>'
        S.append(f'<div class="card"><h3>{H.escape(fc.field_id)}'
                 + ('<span class="tag">via insert</span>' if fc.projected else "")
                 + f'</h3><img src="{embed(ctx, 420)}" alt="">'
                 f'<img class="tight" src="{embed(fc.tight, 300)}" alt="">'
                 f'<div class="meta">{b}</div><div>{chips}</div></div>')
    S.append("</div></section>")

    # ---- read + validate ---------------------------------------------------
    stage(4, "Read the digits", "tight crop &rarr; two integers, never a string")
    S.append('<div class="grid">')
    for fc in crops:
        if "field_empty" in fc.flags:
            S.append(f'<div class="card"><h3>{H.escape(fc.field_id)}</h3>'
                     f'<p class="empty">field is empty &mdash; nothing to read</p></div>')
            continue
        g = read_glyphs(fc.tight)
        big = cv2.cvtColor(cv2.resize(fc.tight, None, fx=2.2, fy=2.2,
                                      interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        for gl in g:
            p = [int(v * 2.2) for v in gl.box]
            col = (70, 140, 60) if gl.confident else (60, 60, 200)
            cv2.rectangle(big, (p[0], p[1]), (p[2], p[3]), col, 2)
            cv2.putText(big, gl.symbol, (p[0] + 2, max(14, p[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        rows = "".join(f"<tr><td>{H.escape(x.symbol)}</td><td>{x.score:.2f}</td>"
                       f"<td>{x.margin:.2f}</td></tr>" for x in g)
        S.append(f'<div class="card"><h3>{H.escape(fc.field_id)}</h3>'
                 f'<img src="{embed(big, 420)}" alt="">'
                 f'<table><tr><th>sym</th><th>score</th><th>margin</th></tr>{rows}</table></div>')
    S.append("</div></section>")

    stage(5, "Check it makes sense",
          "two integers &rarr; accept / review / reject, with reasons")
    groups: dict = {}
    for fc in crops:
        gid = fc.field_id.rsplit(".", 1)[0] if fc.field_id.endswith((".systolic", ".diastolic")) \
            else fc.field_id.replace("_sbp", "").replace("_dbp", "")
        # Qualify by owner: the card's `bp` and the sticker's `bp.systolic` both
        # reduce to "bp", and unqualified they render as two rows with the same
        # name — the same collision that silently lost a reading in pipeline.py.
        label = gid if fc.owner == spec.form_id else f"{fc.owner} / {gid}"
        groups.setdefault((fc.owner, gid, label), []).append(fc)
    for (owner, gid, label), members in groups.items():
        if any("field_empty" in m.flags for m in members):
            S.append(f'<div class="verdict empty"><b>{H.escape(label)}</b> '
                     f'&mdash; field_empty (a fact, not a failure)</div>')
            continue
        if len(members) == 1:
            rd = read_pattern_a(members[0].tight)
        else:
            s = next((m for m in members if m.field_id.endswith((".systolic", "_sbp"))), None)
            d = next((m for m in members if m.field_id.endswith((".diastolic", "_dbp"))), None)
            if not s or not d:
                continue
            rd = read_pattern_split(s.tight, d.tight)
        v = validate_bp(rd.systolic, rd.diastolic) if rd.readable else None
        cls = v.verdict.value if v else "unreadable"
        why = "; ".join(v.reasons) if v and v.reasons else ("could not read the value" if not v else "")
        hint = ("<div class='hint'>" + H.escape(v.hints[0]) + "</div>") if v and v.hints else ""
        S.append(f'<div class="verdict {cls}"><b>{H.escape(label)}</b> &mdash; '
                 f'<span class="big">{H.escape(rd.text)}</span> '
                 f'<span class="tag">{cls}</span>'
                 f'<div class="why">{H.escape(why)}</div>{hint}</div>')
    S.append("</section>")
    return _write(S, args, str(args.image))


def _write(S, args, source) -> int:
    doc = _HEAD.replace("__TITLE__", H.escape(Path(source).name)) + "\n".join(S) + "</div></body></html>"
    dst = args.out / "trace.html"
    dst.write_text(doc, encoding="utf-8")
    print(f"trace -> {dst}")
    return 0


_HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>pipeline trace</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{--g:#EEF1F2;--s:#FBFCFD;--s2:#E6EBEC;--ink:#172126;--i2:#5C6C73;--i3:#8798A0;
--r:#C7D1D5;--rs:#DBE2E5;--ac:#2E5FA3;--acs:#DCE6F3;--ok:#2C7A52;--oks:#D6E9DF;
--wn:#99691A;--wns:#F0E3C9;--bd:#A03A3A;--bds:#F2DBDB}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--g:#10161A;--s:#182126;
--s2:#1F2A30;--ink:#E2EAED;--i2:#93A4AC;--i3:#6B7C85;--r:#2C383E;--rs:#232E34;--ac:#79ACEC;
--acs:#1D2E44;--ok:#62B98D;--oks:#17301F;--wn:#D6A64B;--wns:#332714;--bd:#E08B8B;--bds:#361D1D}}
:root[data-theme="dark"]{--g:#10161A;--s:#182126;--s2:#1F2A30;--ink:#E2EAED;--i2:#93A4AC;
--i3:#6B7C85;--r:#2C383E;--rs:#232E34;--ac:#79ACEC;--acs:#1D2E44;--ok:#62B98D;--oks:#17301F;
--wn:#D6A64B;--wns:#332714;--bd:#E08B8B;--bds:#361D1D}
*{box-sizing:border-box}
body{margin:0;background:var(--g);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.55}
.wrap{max-width:1120px;margin:0 auto;padding:44px 26px 90px}
h1{font-size:30px;font-weight:600;margin:0 0 6px;letter-spacing:-.02em}
.lead{font-family:"IBM Plex Mono",monospace;font-size:12.5px;color:var(--i3);margin:0 0 34px}
section{background:var(--s);border:1px solid var(--r);margin-bottom:20px;padding:20px 24px 24px}
.sh{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;border-bottom:1px solid var(--rs);padding-bottom:12px;margin-bottom:18px}
.sh .n{font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:600;color:#fff;background:var(--ac);width:26px;height:26px;display:grid;place-items:center;border-radius:3px}
.sh h2{font-size:19px;font-weight:600;margin:0;letter-spacing:-.01em}
.sh p{margin:0;font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--i3);flex-basis:100%}
figure{margin:0 0 16px}
figure img{display:block;max-width:100%;height:auto;border:1px solid var(--r);background:#fff}
figcaption{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--i3);margin-top:6px}
.facts{display:grid;grid-template-columns:auto 1fr;gap:4px 18px;margin:14px 0 0;font-size:13.5px}
.facts dt{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--i3);padding-top:3px}
.facts dd{margin:0;font-family:"IBM Plex Mono",monospace;font-size:12.5px}
.bars{display:flex;flex-direction:column;gap:7px}
.bar{display:flex;align-items:center;gap:12px}
.bar .lbl{font-family:"IBM Plex Mono",monospace;font-size:12px;width:190px;flex:none;color:var(--i2)}
.bar .track{flex:1;background:var(--s2);height:19px;position:relative}
.bar i{display:block;height:100%;background:var(--i3)}
.bar i.win{background:var(--ok)}
.bar .val{font-family:"IBM Plex Mono",monospace;font-size:12px;width:56px;text-align:right;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.card{background:var(--g);border:1px solid var(--r);padding:13px 15px}
.card h3{font-family:"IBM Plex Mono",monospace;font-size:12.5px;font-weight:600;margin:0 0 9px}
.card img{display:block;max-width:100%;height:auto;border:1px solid var(--r);background:#fff;margin-bottom:7px}
.card img.tight{max-width:70%}
.card .meta{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--i3);margin:5px 0 6px}
.card table{border-collapse:collapse;font-family:"IBM Plex Mono",monospace;font-size:11px;width:100%}
.card th{text-align:left;color:var(--i3);font-weight:500;padding:2px 6px 2px 0}
.card td{padding:1px 6px 1px 0;font-variant-numeric:tabular-nums}
.tag{font-family:"IBM Plex Mono",monospace;font-size:9.5px;letter-spacing:.09em;text-transform:uppercase;border:1px solid var(--r);padding:1px 6px;border-radius:2px;color:var(--i3);margin-left:7px}
.chip{font-family:"IBM Plex Mono",monospace;font-size:9.5px;background:var(--wns);color:var(--wn);padding:2px 7px;border-radius:3px;margin-right:4px}
.chip.ok{background:var(--oks);color:var(--ok)}
.empty{font-size:13px;color:var(--i3);margin:4px 0}
.verdict{border-left:4px solid var(--i3);background:var(--g);padding:12px 16px;margin-bottom:10px;font-family:"IBM Plex Mono",monospace;font-size:13px}
.verdict .big{font-size:19px;font-weight:600}
.verdict.plausible{border-left-color:var(--ok)} .verdict.implausible{border-left-color:var(--wn)}
.verdict.impossible{border-left-color:var(--bd)} .verdict.impossible .big{color:var(--bd)}
.verdict .why{font-size:11.5px;color:var(--i2);margin-top:4px}
.verdict .hint{font-size:11px;color:var(--i3);margin-top:5px}
.note{background:var(--wns);color:var(--wn);padding:11px 15px;font-size:13px;border-left:3px solid var(--wn)}
</style></head><body><div class="wrap">
<h1>Pipeline trace</h1><p class="lead">__TITLE__</p>
"""

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
