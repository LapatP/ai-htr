"""Clinical plausibility checks for blood-pressure readings.

A guardrail, not a corrector. Nothing here ever rewrites a value — it grades it
and says why. Silently "fixing" `43/90` into `143/90` would be exactly the
confident fabrication that makes a document VLM dangerous in a clinical
pipeline; the point is to make a bad read *loud*, not to make it look good.

Why this earns its place: the pipeline's worst failure mode is a crop that
clips a digit while every upstream diagnostic reports success. Registration
reports sub-pixel error, the reader confidently transcribes what it was given,
and `143/90` silently becomes `43/90`. No amount of image processing catches
that. One domain invariant does — **systolic must exceed diastolic** — and it
costs nothing.

Thresholds are documented rather than tuned. They should be reviewed by someone
with prescribing authority before this is used on real records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# --- hard bounds: outside these, the reading cannot be a real measurement ----
# Deliberately wide. The job is to catch OCR damage, not to second-guess
# clinicians. Extremes do get recorded, so only the genuinely impossible is
# rejected outright.
SBP_POSSIBLE = (50, 300)
DBP_POSSIBLE = (20, 200)

# --- soft bounds: possible, but rare enough in an outpatient clinic to review -
SBP_USUAL = (80, 220)
DBP_USUAL = (40, 130)

# Pulse pressure = SBP - DBP. Normally ~40.
# Narrow (<25) suggests aortic stenosis, tamponade, hypovolaemia.
# Wide (>60) is common in elderly stiff arteries; >100 suggests aortic
# regurgitation. Both are real, so these are flags rather than rejections.
PULSE_PRESSURE_USUAL = (20, 100)

# Decision thresholds where a single misread digit changes management.
# A reading near one of these deserves review even when it looks perfectly
# plausible — this is the "clinical severity weighting" the metric needs, and
# it is the reason plain CER is the wrong headline number.
DECISION_THRESHOLDS = [
    (180, 120, "hypertensive crisis"),
    (140, 90, "hypertension diagnosis"),
    (130, 80, "stage 1 hypertension"),
    (90, 60, "hypotension"),
]
NEAR_THRESHOLD_MMHG = 5


class Verdict(str, Enum):
    PLAUSIBLE = "plausible"        # accept
    IMPLAUSIBLE = "implausible"    # possible but unusual — review
    IMPOSSIBLE = "impossible"      # cannot be a real reading — reject


@dataclass
class Validation:
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    near_thresholds: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return self.verdict is not Verdict.PLAUSIBLE or bool(self.near_thresholds)

    def __str__(self) -> str:
        s = self.verdict.value.upper()
        if self.reasons:
            s += ": " + "; ".join(self.reasons)
        if self.near_thresholds:
            s += "  [near " + ", ".join(self.near_thresholds) + "]"
        if self.hints:
            s += "\n    hint: " + "\n    hint: ".join(self.hints)
        return s


def _digits(n: int) -> int:
    return len(str(abs(int(n))))


def clipping_hints(systolic: int, diastolic: int) -> list[str]:
    """If a leading digit had been clipped, would the reading make sense?

    This does NOT correct anything. A hint that prepending `1` turns an
    impossible reading into a plausible one is evidence about the *crop* — it
    tells stage 3 to widen the box and re-read, not stage 4 to guess a digit.
    """
    out = []
    for lead in (1, 2):
        for label, s, d in (
            ("systolic", int(f"{lead}{systolic}"), diastolic),
            ("diastolic", systolic, int(f"{lead}{diastolic}")),
        ):
            # _hints=False breaks the recursion: hint generation must not
            # itself request hints.
            if validate_bp(s, d, _hints=False).verdict is Verdict.PLAUSIBLE:
                out.append(
                    f"prepending '{lead}' to the {label} gives {s}/{d}, which is "
                    f"plausible — suspect a clipped leading digit, re-crop wider"
                )
    return out


def validate_bp(systolic: int, diastolic: int, _hints: bool = True) -> Validation:
    """Grade one BP reading. Never modifies it.

    `_hints` is internal: clipping_hints() re-validates candidate repairs and
    must not recurse into hint generation.
    """
    reasons: list[str] = []
    hard = False

    # 1. The definitional invariant. Pulse pressure must be positive: the heart
    #    cannot eject against a pressure it never exceeds. This single rule
    #    catches the clipped-leading-digit failure outright.
    if systolic <= diastolic:
        hard = True
        reasons.append(
            f"systolic ({systolic}) must exceed diastolic ({diastolic})"
        )

    # 2. Digit counts. BP is written as 2-3 digits per component; anything else
    #    is a segmentation fault rather than a measurement.
    for name, v in (("systolic", systolic), ("diastolic", diastolic)):
        n = _digits(v)
        if n < 2:
            hard = True
            reasons.append(f"{name} has {n} digit — a digit was probably clipped")
        elif n > 3:
            hard = True
            reasons.append(f"{name} has {n} digits — digits were probably merged")

    # 3. Survivable range.
    if not SBP_POSSIBLE[0] <= systolic <= SBP_POSSIBLE[1]:
        hard = True
        reasons.append(f"systolic {systolic} outside {SBP_POSSIBLE} — not a real reading")
    if not DBP_POSSIBLE[0] <= diastolic <= DBP_POSSIBLE[1]:
        hard = True
        reasons.append(f"diastolic {diastolic} outside {DBP_POSSIBLE} — not a real reading")

    if hard:
        hints = clipping_hints(systolic, diastolic) if _hints else []
        return Validation(Verdict.IMPOSSIBLE, reasons, hints)

    # 4. Unusual but real.
    soft: list[str] = []
    if not SBP_USUAL[0] <= systolic <= SBP_USUAL[1]:
        soft.append(f"systolic {systolic} outside the usual outpatient range {SBP_USUAL}")
    if not DBP_USUAL[0] <= diastolic <= DBP_USUAL[1]:
        soft.append(f"diastolic {diastolic} outside the usual outpatient range {DBP_USUAL}")

    pp = systolic - diastolic
    if pp < PULSE_PRESSURE_USUAL[0]:
        soft.append(f"pulse pressure {pp} is narrow (<{PULSE_PRESSURE_USUAL[0]})")
    elif pp > PULSE_PRESSURE_USUAL[1]:
        soft.append(f"pulse pressure {pp} is wide (>{PULSE_PRESSURE_USUAL[1]})")

    near = [
        name
        for s, d, name in DECISION_THRESHOLDS
        if abs(systolic - s) <= NEAR_THRESHOLD_MMHG or abs(diastolic - d) <= NEAR_THRESHOLD_MMHG
    ]

    verdict = Verdict.IMPLAUSIBLE if soft else Verdict.PLAUSIBLE
    return Validation(verdict, soft, [], near)


def cross_check(readings: dict[str, tuple[int, int]],
                expected: int | None = None) -> Validation:
    """Compare BP readings taken from the same page.

    Free redundancy specific to this form family: on a base OPD card carrying a
    triage sticker, the top vitals block and the sticker record the *same*
    measurement for the same visit. Two independent crops, two independent
    reads, one ground truth. Disagreement means at least one is wrong, and it
    is detectable without any reference data at all.

    ``expected`` is how many BP sources the page actually carries. Pass it
    whenever the page declares more than one: a source that yielded nothing is
    not agreement, it is the redundancy going missing, and reporting that as
    *plausible* is how a page with one unread BP ends up looking as safe as a
    page where both agree. A wrong number is caught; a missing one has to be
    too.
    """
    expected = len(readings) if expected is None else expected

    if expected >= 2 and len(readings) < expected:
        got = ", ".join(sorted(readings)) or "none"
        return Validation(
            Verdict.IMPLAUSIBLE,
            [f"{len(readings)} of {expected} BP sources on the page produced a "
             f"reading (got: {got}); cross-check unavailable"],
            ["the missing source was empty or unreadable — it is not evidence "
             "of agreement; route the page to human review"],
        )

    if len(readings) < 2:
        return Validation(Verdict.PLAUSIBLE, ["only one reading; nothing to cross-check"])

    distinct = set(readings.values())
    if len(distinct) == 1:
        return Validation(Verdict.PLAUSIBLE, ["all readings on the page agree"])

    shown = ", ".join(f"{k}={s}/{d}" for k, (s, d) in sorted(readings.items()))
    return Validation(
        Verdict.IMPOSSIBLE,
        [f"readings disagree on the same page: {shown}"],
        ["at least one crop or read is wrong; route the page to human review"],
    )
