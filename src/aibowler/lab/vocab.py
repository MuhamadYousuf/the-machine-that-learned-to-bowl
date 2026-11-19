"""Coach vocabulary → simulator zones.

Cards (and the LLM that drafts them) never write raw numbers for where a weakness is.
They pick words a coach would use ("good length", "fourth stump", "slower ball") and this
module turns those words into the Gaussian zones the batting model understands. That keeps
LLM output checkable and stops it inventing zones the simulator cannot express.

Coordinates (see docs/SPEC.md): length is metres in front of the batter's stumps,
line is metres across the pitch with off side positive for a right-hander,
height is ball height at the popping crease, speed delta is relative to what the
batter expects, late movement is sideways movement after pitching.
"""

from __future__ import annotations

# name -> (centre, width)
LENGTHS: dict[str, tuple[float, float]] = {
    "full_toss": (0.0, 0.6),
    "yorker": (1.0, 0.9),
    "full": (4.0, 1.4),
    "good": (7.0, 1.0),
    "short": (9.3, 1.0),
    "bouncer": (12.0, 1.6),
}

LINES: dict[str, tuple[float, float]] = {
    "down_leg": (-0.45, 0.18),
    "at_body": (-0.25, 0.15),
    "leg_stump": (-0.11, 0.10),
    "stumps": (0.0, 0.15),
    "fourth_stump": (0.25, 0.14),
    "outside_off": (0.45, 0.18),
    "wide_outside_off": (0.75, 0.22),
}

PACE: dict[str, tuple[float, float]] = {
    "slower_ball": (-18.0, 7.0),
    "pace_on": (6.0, 5.0),
}

MOVEMENT: dict[str, tuple[float, float]] = {
    "none": (0.0, 0.05),
    "away": (0.10, 0.07),  # late movement towards the off side (away from a right-hander)
    "in": (-0.10, 0.07),
}

HEIGHT: dict[str, tuple[float, float]] = {
    "low": (0.25, 0.2),
    "waist": (0.9, 0.3),
    "chest": (1.3, 0.25),
    "head": (1.65, 0.3),
}

DISMISSALS = ("bowled_lbw", "caught_behind", "caught_field")

SEVERITY_STRENGTH = {"low": 0.12, "medium": 0.24, "high": 0.38}
SEVERITY_BONUS = {"low": 0.45, "medium": 0.85, "high": 1.3}

SKILL = {"club": 0.45, "domestic": 0.65, "international": 0.82, "elite": 0.92}
STYLE_AGGRESSION = {"defensive": 0.25, "balanced": 0.55, "aggressive": 0.8, "ultra_aggressive": 0.95}
STARTER_CONFIDENCE = {"nervous_starter": 0.06, "normal": 0.15, "fast_starter": 0.3}

VOCAB = {
    "length": LENGTHS,
    "line": LINES,
    "pace": PACE,
    "movement": MOVEMENT,
    "height": HEIGHT,
}
FEATURE_FOR = {
    "length": "length_m",
    "line": "line_m",
    "pace": "speed_delta_kmh",
    "movement": "late_movement_m",
    "height": "height_at_crease_m",
}


def zone_for(terms: dict[str, str]) -> tuple[dict[str, float], dict[str, float]]:
    """Turn {"length": "good", "line": "fourth_stump"} into (centres, widths)."""
    if not terms:
        raise ValueError("a zone needs at least one of: " + ", ".join(VOCAB))
    centres, widths = {}, {}
    for kind, word in terms.items():
        if kind not in VOCAB:
            raise ValueError(f"unknown zone dimension {kind!r}; use one of {sorted(VOCAB)}")
        table = VOCAB[kind]
        if word not in table:
            raise ValueError(f"unknown {kind} {word!r}; use one of {sorted(table)}")
        centre, width = table[word]
        centres[FEATURE_FOR[kind]] = centre
        widths[FEATURE_FOR[kind]] = width
    return centres, widths


def describe(terms: dict[str, str]) -> str:
    """Human words for a zone, e.g. 'good length, fourth stump, moving away'."""
    order = ("pace", "length", "line", "movement", "height")
    parts = []
    for kind in order:
        if kind not in terms:
            continue
        word = terms[kind].replace("_", " ")
        parts.append({
            "length": f"{word} length" if word in ("good", "full", "short") else word,
            "line": f"on {word}" if word in ("stumps", "leg stump", "fourth stump") else word,
            "pace": word,
            "movement": f"moving {word}" if word != "none" else "no movement",
            "height": f"at {word} height",
        }[kind])
    return ", ".join(parts)
