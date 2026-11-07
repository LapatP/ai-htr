"""Write a single self-contained HTML page showing what a run produced.

Digging through `out/crops/<page>/<field>.tight.png` to see whether a run worked
is miserable, and the thing that most needs human eyes — is this box on the
right value? — is exactly the thing a CSV cannot show you.

Each page shows the **raw input** alongside its crops, so a reading can be
checked against the actual card without opening another window, and systolic and
diastolic are shown as two separate numbers because that is what the pipeline
actually produces. `143/90` is a rendering convenience; the contract is two
integers.

Crops and the raw page are embedded as data URIs so the file opens by
double-click. The same property makes it as identifiable as the input: a report
built from a real card contains the whole card, so it stays on the machine that
holds the scans. Only reports from synthetic pages are safe to send.
"""
from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

import cv2

MAX_EMBED_PX = 420        # crops are tiny; this is a guard, not a resize policy
PAGE_THUMB_W = 460        # the raw input shown beside each page's crops


def _data_uri(img, jpeg_quality: int | None = None) -> str:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if jpeg_quality is not None:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        mime = "jpeg"
    else:
        ok, enc = cv2.imencode(".png", img)
        mime = "png"
    if not ok:
        return ""
    return f"data:image/{mime};base64," + base64.b64encode(enc.tobytes()).decode("ascii")


def _img_tag(path: Path, cls: str) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return '<span class="missing">missing</span>'
    h, w = img.shape[:2]
    if max(h, w) > MAX_EMBED_PX:
        s = MAX_EMBED_PX / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    uri = _data_uri(img)
    return f'<img class="{cls}" src="{uri}" alt="">' if uri else \
        '<span class="missing">unreadable</span>'


def _page_tag(path) -> str:
    """The raw input page, so a reading can be checked against the real card.

    JPEG, not PNG — a scanned page is photographic, and PNG stores it almost
    losslessly. On a 20-page run that is the difference between a couple of MB
    and tens of them.
    """
    if path is None:
        return ""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return '<span class="missing">source image not found</span>'
    h, w = img.shape[:2]
    s = PAGE_THUMB_W / w
    img = cv2.resize(img, (PAGE_THUMB_W, max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    uri = _data_uri(img, jpeg_quality=78)
    return f'<img class="src" src="{uri}" alt="">' if uri else ""


def _reading_block(r: dict) -> str:
    """Systolic and diastolic as two numbers, not one string."""
    sys_v, dia_v = r.get("systolic", ""), r.get("diastolic", "")
    verdict = r.get("verdict", "")
    conf = r.get("read_confidence", "")

    if sys_v in ("", None) or dia_v in ("", None):
        text = html.escape(str(r.get("reading") or "unreadable"))
        return (f'<div class="reading {html.escape(verdict)}">'
                f'<span class="unread">{text}</span>'
                f'<span class="verdict">{html.escape(verdict)}</span></div>')

    return (f'<div class="reading {html.escape(verdict)}">'
            f'<div class="pair">'
            f'<span class="v"><b>{html.escape(str(sys_v))}</b><small>systolic</small></span>'
            f'<span class="slash">/</span>'
            f'<span class="v"><b>{html.escape(str(dia_v))}</b><small>diastolic</small></span>'
            f'<span class="unit">mmHg</span></div>'
            f'<span class="verdict">{html.escape(verdict)}'
            f'{f" · conf {conf}" if conf != "" else ""}</span></div>')


def write_report(rows, unknown, failed, out_dir: Path, crop_root: Path,
                 source: str, seconds: float, stale: list[str] | None = None,
                 sources: dict | None = None) -> Path:
    sources = sources or {}
    pages: dict[str, list[dict]] = {}
    for r in rows:
        pages.setdefault(r["page"], []).append(r)

    n_fields = len(rows)
    n_review = sum(int(r["needs_review"]) for r in rows)
    has_reading = any("reading" in r for r in rows)

    parts = [_HEAD, f"""
<header>
  <p class="eyebrow">ai-htr · crop report</p>
  <h1>{html.escape(Path(source).name or source)}</h1>
  <p class="sub">{datetime.now():%d %B %Y, %H:%M} · {seconds:.1f}s</p>
</header>

<dl class="strip">
  <div><dt>Pages cropped</dt><dd>{len(pages)}</dd></div>
  <div><dt>Fields</dt><dd>{n_fields}</dd></div>
  <div><dt>Need review</dt><dd class="{'warn' if n_review else ''}">{n_review}</dd></div>
  <div><dt>Unrecognised</dt><dd class="{'warn' if unknown else ''}">{len(unknown)}</dd></div>
  <div><dt>Failed</dt><dd class="{'bad' if failed else ''}">{len(failed)}</dd></div>
</dl>
"""]

    if stale:
        parts.append(f'<p class="note">{len(stale)} crop folder(s) in '
                     f'<code>crops/</code> are left over from a previous run and are '
                     f'<b>not</b> shown here: {html.escape(", ".join(stale[:8]))}'
                     f'{" ..." if len(stale) > 8 else ""}. Re-run with '
                     f'<code>--clean</code> to remove them.</p>')
    if n_review:
        parts.append('<p class="note">Rows outlined in amber need a human to look. '
                     'Check the <b>context</b> crop against the raw page on the left: '
                     'is the box on the right value?</p>')

    for page, frows in pages.items():
        stem = Path(page).stem
        form = frows[0]["form"]
        reg = f'{frows[0]["reg_inliers"]} inliers · {frows[0]["reg_error_px"]} px'
        ms = frows[0].get("page_ms")
        if ms not in (None, ""):
            reg += f' · {int(ms)/1000:.2f}s'
        parts.append(f'''
<section class="page">
  <div class="page-head">
    <h2>{html.escape(page)}</h2>
    <span class="form">{html.escape(form)}</span>
    <span class="reg">{reg}</span>
  </div>
  <div class="body">
    <div class="source">{_page_tag(sources.get(page))}
      <figcaption>raw input</figcaption></div>
    <div class="fields">''')
        for r in frows:
            review = int(r["needs_review"])
            flags = r["flags"] or ""
            chips = "".join(
                f'<span class="chip {"chip-warn" if f.split(":")[0] in ("possibly_clipped","field_empty","occluded_by") else ""}">{html.escape(f)}</span>'
                for f in flags.split("|") if f)
            reading = _reading_block(r) if has_reading and r.get("reading") else ""
            d = crop_root / stem
            parts.append(f'''
      <div class="field{' review' if review else ''}">
        <div class="fid">{html.escape(r["field"])}
          {'<span class="proj">via insert</span>' if int(r["projected"]) else ''}</div>
        <div class="crops">
          <figure>{_img_tag(d / f'{r["field"]}.context.png', "ctx")}<figcaption>context</figcaption></figure>
          <figure>{_img_tag(d / f'{r["field"]}.tight.png', "tight")}<figcaption>tight</figcaption></figure>
        </div>
        {reading}
        <div class="meta">[{r["x1"]}, {r["y1"]}, {r["x2"]}, {r["y2"]}]</div>
        <div class="chips">{chips or '<span class="chip ok">clean</span>'}</div>
      </div>''')
        parts.append("    </div>\n  </div>\n</section>")

    if unknown:
        parts.append('<section class="list"><h2>Unrecognised</h2>'
                     '<p class="note">No template for these. They were listed, never cropped '
                     'against a guess. See docs/running.md to add a form.</p><ul>')
        for name, why in unknown:
            parts.append(f"<li><b>{html.escape(name)}</b> — {html.escape(why)}</li>")
        parts.append("</ul></section>")

    if failed:
        parts.append('<section class="list"><h2>Failed</h2><ul>')
        for name, why in failed:
            parts.append(f"<li><b>{html.escape(name)}</b> — {html.escape(why)}</li>")
        parts.append("</ul></section>")

    parts.append("</div></body></html>")
    dst = out_dir / "report.html"
    dst.write_text("\n".join(parts), encoding="utf-8")
    return dst


_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ai-htr crop report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{--ground:#EEF1F2;--surface:#FBFCFD;--surface-2:#E6EBEC;--ink:#172126;--ink-2:#5C6C73;
--ink-3:#8798A0;--rule:#C7D1D5;--rule-soft:#DBE2E5;--accent:#2E5FA3;--accent-soft:#DCE6F3;
--done:#2C7A52;--done-soft:#D6E9DF;--warn:#99691A;--warn-soft:#F0E3C9;--bad:#A03A3A;--bad-soft:#F2DBDB}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#10161A;--surface:#182126;
--surface-2:#1F2A30;--ink:#E2EAED;--ink-2:#93A4AC;--ink-3:#6B7C85;--rule:#2C383E;--rule-soft:#232E34;
--accent:#79ACEC;--accent-soft:#1D2E44;--done:#62B98D;--done-soft:#17301F;--warn:#D6A64B;
--warn-soft:#332714;--bad:#E08B8B;--bad-soft:#361D1D}}
:root[data-theme="dark"]{--ground:#10161A;--surface:#182126;--surface-2:#1F2A30;--ink:#E2EAED;
--ink-2:#93A4AC;--ink-3:#6B7C85;--rule:#2C383E;--rule-soft:#232E34;--accent:#79ACEC;
--accent-soft:#1D2E44;--done:#62B98D;--done-soft:#17301F;--warn:#D6A64B;--warn-soft:#332714;
--bad:#E08B8B;--bad-soft:#361D1D}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.5}
.wrap{max-width:1420px;margin:0 auto;padding:0 26px 90px}
header{border-bottom:2px solid var(--ink);padding:48px 0 18px;margin-bottom:28px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--ink-2);margin:0 0 12px}
h1{font-size:clamp(24px,4vw,36px);font-weight:600;margin:0 0 8px;letter-spacing:-.02em;word-break:break-all}
.sub{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink-3);margin:0}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);margin:0 0 30px}
.strip>div{background:var(--surface);padding:13px 15px}
.strip dt{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-2);margin:0 0 5px}
.strip dd{margin:0;font-size:23px;font-weight:600;font-variant-numeric:tabular-nums}
.strip dd.warn{color:var(--warn)} .strip dd.bad{color:var(--bad)}
.note{font-size:13.5px;color:var(--ink-2);background:var(--surface);border-left:3px solid var(--warn);padding:10px 14px;margin:0 0 26px}
.page{background:var(--surface);border:1px solid var(--rule);margin-bottom:20px}
.page-head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;padding:13px 18px;border-bottom:1px solid var(--rule-soft)}
.page-head h2{font-size:15px;font-weight:600;margin:0;font-family:"IBM Plex Mono",monospace;word-break:break-all}
.form{font-family:"IBM Plex Mono",monospace;font-size:11px;background:var(--accent-soft);color:var(--accent);padding:2px 8px;border-radius:3px}
.reg{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--ink-3);margin-left:auto}
.body{display:flex;gap:1px;background:var(--rule-soft);align-items:stretch}
.source{background:var(--surface);padding:14px 16px;flex:none;text-align:center}
.source img.src{display:block;max-width:100%;width:340px;height:auto;border:1px solid var(--rule);background:#fff}
.source figcaption{font-family:"IBM Plex Mono",monospace;font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3);margin-top:7px}
.fields{flex:1;display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1px;background:var(--rule-soft);align-content:start}
.field{background:var(--surface);padding:14px 16px;border-left:3px solid transparent}
.field.review{border-left-color:var(--warn)}
.fid{font-family:"IBM Plex Mono",monospace;font-size:12.5px;font-weight:600;margin-bottom:9px}
.proj{font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);border:1px solid var(--rule);padding:1px 5px;border-radius:2px;margin-left:6px}
.crops{display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap}
.crops figure{margin:0}
.crops img{display:block;background:#fff;border:1px solid var(--rule);max-width:100%;height:auto;image-rendering:crisp-edges;max-height:76px}
.crops figcaption{font-family:"IBM Plex Mono",monospace;font-size:9.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);margin-top:4px}
.reading{margin-top:11px}
.reading .pair{display:flex;align-items:baseline;gap:8px;font-family:"IBM Plex Mono",monospace}
.reading .v{display:flex;flex-direction:column;line-height:1.1}
.reading .v b{font-size:23px;font-weight:600;font-variant-numeric:tabular-nums}
.reading .v small{font-size:8.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3);margin-top:3px}
.reading .slash{font-size:20px;color:var(--ink-3)}
.reading .unit{font-size:10px;color:var(--ink-3);align-self:flex-end;margin-left:2px}
.reading .unread{font-family:"IBM Plex Mono",monospace;font-size:16px;font-weight:600;color:var(--ink-3)}
.reading .verdict{display:block;font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);margin-top:6px}
.reading.impossible .v b{color:var(--bad)} .reading.impossible .verdict{color:var(--bad)}
.reading.implausible .v b{color:var(--warn)} .reading.implausible .verdict{color:var(--warn)}
.reading.plausible .v b{color:var(--done)} .reading.plausible .verdict{color:var(--done)}
.meta{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--ink-3);margin-top:9px;font-variant-numeric:tabular-nums}
.chips{margin-top:7px;display:flex;flex-wrap:wrap;gap:5px}
.chip{font-family:"IBM Plex Mono",monospace;font-size:9.5px;padding:2px 7px;border-radius:3px;background:var(--surface-2);color:var(--ink-2)}
.chip-warn{background:var(--warn-soft);color:var(--warn)}
.chip.ok{background:var(--done-soft);color:var(--done)}
.missing{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--bad)}
.list{background:var(--surface);border:1px solid var(--rule);padding:16px 20px;margin-bottom:20px}
.list h2{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.14em;text-transform:uppercase;margin:0 0 10px;color:var(--ink)}
.list ul{margin:0;padding-left:18px}
.list li{font-size:13px;color:var(--ink-2);margin-bottom:5px}
.list li b{font-family:"IBM Plex Mono",monospace;color:var(--ink);font-weight:500}
@media(max-width:900px){.body{flex-direction:column}.source img.src{width:100%;max-width:420px}}
</style></head><body><div class="wrap">"""
