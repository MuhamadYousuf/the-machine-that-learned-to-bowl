"""Real-player profiles from Cricsheet ball-by-ball data (https://cricsheet.org, ODC-BY licence).

Cricsheet has every ball's runs, extras and dismissals, but no ball tracking (line, length,
speed). So it constrains a batter's *outcomes* (strike rate, dot and boundary rates, how often
and how he gets out) and calibration fits the card's free parameters to those outcomes. Where
the ball was pitched still comes from the card's evidence and stays labelled as such.

Usage: download a JSON zip (e.g. ``t20s_json.zip`` or ``odis_json.zip``) from cricsheet.org/downloads
and point :func:`load_matches` at the zip or an unzipped folder.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

BOWLER_DISMISSALS = {"bowled", "lbw", "caught", "caught and bowled", "stumped", "hit wicket"}


@dataclass
class Profile:
    """Batting outcomes per ball faced (wides are not balls faced)."""

    player: str
    matches: int = 0
    balls: int = 0
    runs: int = 0
    dots: int = 0
    fours: int = 0
    sixes: int = 0
    dismissals: dict[str, int] = field(default_factory=dict)
    first_date: str | None = None
    last_date: str | None = None

    @property
    def outs(self) -> int:
        return sum(self.dismissals.values())

    def rates(self) -> dict[str, float]:
        b = max(1, self.balls)
        outs = self.outs
        caught = self.dismissals.get("caught", 0) + self.dismissals.get("caught and bowled", 0)
        bowled_lbw = self.dismissals.get("bowled", 0) + self.dismissals.get("lbw", 0)
        return {
            "strike_rate": 100.0 * self.runs / b,
            "dot_pct": 100.0 * self.dots / b,
            "boundary_pct": 100.0 * (self.fours + self.sixes) / b,
            "balls_per_dismissal": b / outs if outs else float("inf"),
            "caught_share": caught / outs if outs else float("nan"),
            "bowled_lbw_share": bowled_lbw / outs if outs else float("nan"),
        }

    def to_dict(self) -> dict:
        return {**asdict(self), "rates": self.rates()}


def _iter_json(path: Path) -> Iterator[dict]:
    if path.is_file() and path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith(".json"):
                    yield json.loads(z.read(name))
    elif path.is_dir():
        for p in sorted(path.glob("*.json")):
            yield json.loads(p.read_text(encoding="utf-8"))
    elif path.suffix == ".json":
        yield json.loads(path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"expected a Cricsheet .zip, folder of .json files, or one .json: {path}")


def load_matches(path: str | Path, player: str | None = None) -> list[dict]:
    """Load matches (optionally only those the player batted in), sorted by first date."""
    matches = []
    for m in _iter_json(Path(path)):
        if player is not None and not _batted(m, player):
            continue
        matches.append(m)
    matches.sort(key=lambda m: m.get("info", {}).get("dates", ["0000"])[0])
    return matches


def _deliveries(match: dict) -> Iterable[dict]:
    for inning in match.get("innings", []):
        for over in inning.get("overs", []):
            for d in over.get("deliveries", []):
                yield {**d, "_over": over.get("over", 0)}


def _batted(match: dict, player: str) -> bool:
    return any(d.get("batter") == player for d in _deliveries(match))


def find_players(path: str | Path, query: str, limit: int = 20) -> list[tuple[str, int]]:
    """Cricsheet names are like 'V Kohli' or 'Babar Azam'. Search by substring; returns (name, balls)."""
    q = query.lower()
    counts: dict[str, int] = {}
    for m in _iter_json(Path(path)):
        for d in _deliveries(m):
            name = d.get("batter", "")
            if q in name.lower():
                counts[name] = counts.get(name, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:limit]


def profile(matches: list[dict], player: str) -> Profile:
    prof = Profile(player=player)
    for m in matches:
        faced_here = False
        for d in _deliveries(m):
            extras = d.get("extras", {})
            if d.get("batter") == player and "wides" not in extras:
                faced_here = True
                prof.balls += 1
                r = int(d.get("runs", {}).get("batter", 0))
                prof.runs += r
                boundary = not d.get("runs", {}).get("non_boundary", False)
                prof.dots += int(r == 0)
                prof.fours += int(r == 4 and boundary)
                prof.sixes += int(r == 6 and boundary)
            for w in d.get("wickets", []):
                if w.get("player_out") == player and w.get("kind") in BOWLER_DISMISSALS:
                    prof.dismissals[w["kind"]] = prof.dismissals.get(w["kind"], 0) + 1
        if faced_here:
            prof.matches += 1
            date = m.get("info", {}).get("dates", [None])[0]
            prof.first_date = prof.first_date or date
            prof.last_date = date
    return prof


def split_profiles(matches: list[dict], player: str, holdout: float = 0.25) -> tuple[Profile, Profile]:
    """Fit on the earlier matches, hold out the most recent ones (time split, no leakage)."""
    n_hold = max(1, int(round(len(matches) * holdout))) if len(matches) >= 4 else 0
    fit = matches[: len(matches) - n_hold] if n_hold else matches
    hold = matches[len(matches) - n_hold:] if n_hold else []
    return profile(fit, player), profile(hold, player)
