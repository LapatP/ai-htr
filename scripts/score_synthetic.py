"""Score a run against synthetic ground truth.

Pairs `data/synthetic/truth.csv` (what was rendered, and where) with a run's
`manifest.csv` (what the pipeline found) and reports, per degradation level:

* how many fields were cropped at all
* coverage of the true value box - the metric that matters, since a crop
  clipping a digit has failed however good its overlap
* whether deliberately overflowing values were still captured
* reader accuracy, if the run used --read

Usage:
  python scripts/score_synthetic.py                                  # defaults
  python scripts/score_synthetic.py --truth data/synthetic/truth.csv \
                                    --manifest out-synth/manifest.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("clean", "mild", "moderate", "heavy")


def coverage(truth_box, got_box) -> float:
    iw = max(0, min(truth_box[2], got_box[2]) - max(truth_box[0], got_box[0]))
    ih = max(0, min(truth_box[3], got_box[3]) - max(truth_box[1], got_box[1]))
    area = max(1, (truth_box[2] - truth_box[0]) * (truth_box[3] - truth_box[1]))
    return iw * ih / area


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=Path, default=ROOT / "data" / "synthetic" / "truth.csv")
    ap.add_argument("--manifest", type=Path, default=ROOT / "out-synth" / "manifest.csv")
    args = ap.parse_args()

    for p in (args.truth, args.manifest):
        if not p.exists():
            print(f"missing: {p}")
            return 1

    truth = {(r["file"], r["field"]): r
             for r in csv.DictReader(args.truth.open(encoding="utf-8"))}
    got = {(r["page"], r["field"]): r
           for r in csv.DictReader(args.manifest.open(encoding="utf-8"))}

    per_level = collections.defaultdict(list)
    declared = collections.Counter()
    overflow, normal, misses, reads = [], [], [], []
    group_truth: dict = {}
    group_read: dict = {}

    for key, t in truth.items():
        lvl = t["level"]
        declared[lvl] += 1
        g = got.get(key)
        if g is None:
            misses.append((key[0], key[1], lvl, "not cropped"))
            continue
        tb = [int(t[k]) for k in ("x1", "y1", "x2", "y2")]
        gb = [int(g[k]) for k in ("x1", "y1", "x2", "y2")]
        c = coverage(tb, gb)
        per_level[lvl].append(c)
        (overflow if int(t["overflowed"]) else normal).append(c)
        if c < 0.90:
            misses.append((key[0], key[1], lvl, f"coverage {c:.3f}"))
        # Reader output is per GROUP ("143/84"), truth is per FIELD ("143" for a
        # split systolic). Comparing them directly can never match, so rebuild
        # the group's true "sys/dia" and compare like with like.
        if g.get("reading") and g.get("group"):
            group_truth.setdefault((key[0], g["group"]), {})[key[1]] = t["value"]
            group_read[(key[0], g["group"])] = g["reading"]

    print(f"{'level':<10} {'declared':>9} {'cropped':>8} {'meanCov':>8} "
          f"{'minCov':>7} {'>=0.95':>7}")
    print("-" * 54)
    for lvl in LEVELS:
        cs = per_level[lvl]
        if not cs:
            print(f"{lvl:<10} {declared[lvl]:>9} {0:>8}   (none cropped)")
            continue
        print(f"{lvl:<10} {declared[lvl]:>9} {len(cs):>8} {sum(cs)/len(cs):>8.3f} "
              f"{min(cs):>7.3f} {sum(c >= 0.95 for c in cs):>7}")

    print("-" * 54)
    if overflow:
        print(f"values written OVER their box: n={len(overflow):<3} "
              f"meanCov {sum(overflow)/len(overflow):.3f}  minCov {min(overflow):.3f}")
        print("  (these are what ink refinement exists to catch)")
    if normal:
        print(f"values inside their box:       n={len(normal):<3} "
              f"meanCov {sum(normal)/len(normal):.3f}  minCov {min(normal):.3f}")

    if misses:
        print(f"\n{len(misses)} field(s) missed or poorly covered:")
        for f, fid, lvl, why in misses[:12]:
            print(f"  {f[:44]:<46} {fid:<24} {lvl:<9} {why}")
        if len(misses) > 12:
            print(f"  ... and {len(misses)-12} more")

    for gkey, parts in group_truth.items():
        vals = list(parts.values())
        if len(vals) == 1 and "/" in vals[0]:
            expected = vals[0]                       # pattern A: one field holds both
        elif len(vals) == 2:
            sys_v = next((v for k, v in parts.items()
                          if k.endswith((".systolic", "_sbp"))), None)
            dia_v = next((v for k, v in parts.items()
                          if k.endswith((".diastolic", "_dbp"))), None)
            if sys_v is None or dia_v is None:
                continue
            expected = f"{sys_v}/{dia_v}"
        else:
            continue
        reads.append((expected, group_read[gkey]))

    if reads:
        hits = sum(1 for a, b in reads if a == b)
        print(f"\nreader: {hits}/{len(reads)} exact ({hits/len(reads)*100:.0f}%)")
        wrong = [(a, b) for a, b in reads if a != b][:6]
        for a, b in wrong:
            print(f"  wrote {a:<10} read {b}")
        print("\n  Note these are rendered vector digits, far cleaner than pen on a")
        print("  103 dpi scan. A poor score here is a floor, not a ceiling.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
