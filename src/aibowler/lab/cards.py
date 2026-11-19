"""Batter cards, bowler cards and reward specs.

These are the contracts between people, the LLM and the simulator:

* A **batter card** describes a player in coach words (level, style, weaknesses, strengths)
  with evidence for every claim. It compiles to a :class:`BatterArchetype`.
* A **bowler card** describes what *your* bowler can actually do (pace band, accuracy,
  swing, variations), so plans are executable.
* A **reward spec** picks the match format and weights from vetted building blocks.
  Nobody (human or LLM) writes reward code.

The same pydantic models are used as the LLM's structured-output schema, so a drafted
card is valid by construction and still readable as YAML.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field

from ..batter import BatterArchetype, ScoringZone, WeaknessZone, sample_batter
from ..constants import SPEED_RANGE_KMH
from . import vocab

Length = Literal["full_toss", "yorker", "full", "good", "short", "bouncer"]
Line = Literal["down_leg", "at_body", "leg_stump", "stumps", "fourth_stump", "outside_off", "wide_outside_off"]
Pace = Literal["slower_ball", "pace_on"]
Movement = Literal["none", "away", "in"]
Height = Literal["low", "waist", "chest", "head"]
Severity = Literal["low", "medium", "high"]


class Evidence(BaseModel):
    """One piece of support for a claim. `kind` says how much to trust it."""

    claim: str
    kind: Literal["data", "source", "assumption"] = Field(
        description="data = computed from ball-by-ball records; source = a cited article/video; assumption = unsupported guess"
    )
    source: str | None = Field(default=None, description="URL, 'cricsheet', or who said it. None for assumptions.")


class Zone(BaseModel):
    """Where the ball is, in coach words. At least one field must be set."""

    length: Length | None = None
    line: Line | None = None
    pace: Pace | None = None
    movement: Movement | None = None
    height: Height | None = None

    def terms(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def describe(self) -> str:
        return vocab.describe(self.terms())


class Weakness(BaseModel):
    zone: Zone
    dismissal: Literal["bowled_lbw", "caught_behind", "caught_field"]
    severity: Severity
    evidence: list[Evidence] = Field(default_factory=list)


class Strength(BaseModel):
    zone: Zone
    severity: Severity
    evidence: list[Evidence] = Field(default_factory=list)


class BatterCard(BaseModel):
    schema_version: int = 1
    name: str
    description: str = ""
    hand: Literal["right", "left"] = "right"
    level: Literal["club", "domestic", "international", "elite"] = "international"
    style: Literal["defensive", "balanced", "aggressive", "ultra_aggressive"] = "balanced"
    starter: Literal["nervous_starter", "normal", "fast_starter"] = "normal"
    defends_straight: Literal["poor", "solid", "excellent"] = "solid"
    leaves_outside_off: Literal["rarely", "sometimes", "well"] = "sometimes"
    rotates_strike: Literal["rarely", "normal", "busy"] = Field(default="normal", description="How much they work singles off good balls")
    aerial_risk: Literal["low", "medium", "high"] = Field(default="medium", description="How often they hit in the air and risk a catch")
    power_shot: Literal["drive", "slog", "pull"] = "drive"
    caught_in_field_playing: Literal["drive", "slog", "pull", "fend"] = "drive"
    weaknesses: list[Weakness] = Field(default_factory=list)
    strengths: list[Strength] = Field(default_factory=list)
    uncertainty: Literal["low", "medium", "high"] = Field(
        default="medium", description="How unsure the evidence is overall; widens the batter population."
    )
    evidence: list[Evidence] = Field(default_factory=list, description="General claims (level, style, starter...)")

    # ---------------------------------------------------------------- compile

    def base_params(self) -> dict[str, float]:
        return {
            "skill": vocab.SKILL[self.level],
            "aggression": vocab.STYLE_AGGRESSION[self.style],
            "starting_confidence": vocab.STARTER_CONFIDENCE[self.starter],
            "strike_rotation": {"rarely": 0.25, "normal": 0.55, "busy": 0.8}[self.rotates_strike],
            "aerial_risk": {"low": 0.15, "medium": 0.4, "high": 0.7}[self.aerial_risk],
        }

    def to_archetype(self, params: dict[str, float] | None = None) -> BatterArchetype:
        """Compile to the simulator's batting model.

        `params` overrides the card's defaults (used by calibration): skill, aggression,
        starting_confidence, and `weakness_scale` / `strength_scale` multipliers.
        """
        p = {**self.base_params(), "weakness_scale": 1.0, "strength_scale": 1.0, **(params or {})}
        flip = -1.0 if self.hand == "left" else 1.0

        def compile_zone(zone: Zone) -> tuple[dict, dict]:
            centres, widths = vocab.zone_for(zone.terms())
            for key in ("line_m", "late_movement_m"):
                if key in centres:
                    centres[key] *= flip
            return centres, widths

        weak = []
        for w in self.weaknesses:
            c, wd = compile_zone(w.zone)
            weak.append(WeaknessZone(c, wd, vocab.SEVERITY_STRENGTH[w.severity] * p["weakness_scale"], w.dismissal))
        score = [
            # Universal cricket: yorkers are hard to score off; full tosses and half-volleys are gifts.
            ScoringZone({"length_m": 0.8, "line_m": 0.0, "height_at_crease_m": 0.18},
                        {"length_m": 1.1, "line_m": 0.32, "height_at_crease_m": 0.18}, -0.95),
            ScoringZone({"length_m": 0.0}, {"length_m": 0.9}, 0.6 + 1.0 * p["aggression"]),
            ScoringZone({"length_m": 3.6}, {"length_m": 1.2}, 0.3 + 0.9 * p["skill"]),
        ]
        for s in self.strengths:
            c, wd = compile_zone(s.zone)
            score.append(ScoringZone(c, wd, vocab.SEVERITY_BONUS[s.severity] * p["strength_scale"]))
        return BatterArchetype(
            name=self.slug(),
            starting_confidence=float(np.clip(p["starting_confidence"], 0.0, 1.0)),
            skill=float(np.clip(p["skill"], 0.0, 1.0)),
            aggression=float(np.clip(p["aggression"], 0.0, 1.0)),
            weakness_zones=tuple(weak),
            scoring_zones=tuple(score),
            straight_defense={"poor": 0.15, "solid": 0.5, "excellent": 0.9}[self.defends_straight],
            leave_outside_off_line_m={"rarely": 1.4, "sometimes": 0.9, "well": 0.5}[self.leaves_outside_off],
            power_shot=self.power_shot,
            caught_field_shot=self.caught_in_field_playing,
            strike_rotation=float(np.clip(p["strike_rotation"], 0.0, 1.0)),
            aerial_risk=float(np.clip(p["aerial_risk"], 0.0, 1.0)),
        )

    def spread(self) -> float:
        return {"low": 0.08, "medium": 0.15, "high": 0.25}[self.uncertainty]

    def slug(self) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in self.name.lower()).strip("_") or "batter"

    def assumption_share(self) -> float:
        """Share of all evidence items that are unsupported assumptions (or claims with none)."""
        items = list(self.evidence)
        unsupported_claims = 0
        for claim in [*self.weaknesses, *self.strengths]:
            if not claim.evidence:
                unsupported_claims += 1
            items += claim.evidence
        total = len(items) + unsupported_claims
        if total == 0:
            return 1.0
        return (sum(1 for e in items if e.kind == "assumption") + unsupported_claims) / total


class BowlerCard(BaseModel):
    """What your bowler can actually execute. Only right-arm pace/seam is simulated today."""

    schema_version: int = 1
    name: str = "Bowler"
    min_kmh: float = Field(default=118.0, description="Slowest delivery they bowl on purpose (km/h)")
    max_kmh: float = Field(default=140.0, description="Quickest delivery (km/h)")
    accuracy: Literal["club", "domestic", "international", "elite"] = "international"
    swing: Literal["none", "some", "big"] = "some"
    bouncer: bool = True

    def speed_band(self) -> tuple[float, float]:
        lo, hi = SPEED_RANGE_KMH
        a = float(np.clip(min(self.min_kmh, self.max_kmh), lo, hi))
        b = float(np.clip(max(self.min_kmh, self.max_kmh), lo, hi))
        return a, max(b, a + 1.0)

    def noise_scale(self) -> float:
        return {"club": 0.12, "domestic": 0.08, "international": 0.05, "elite": 0.035}[self.accuracy]

    def swing_scale(self) -> float:
        return {"none": 0.0, "some": 0.55, "big": 1.0}[self.swing]


class RewardSpec(BaseModel):
    """Reward from vetted blocks only: per-ball expected wickets, runs and extras, plus the laws."""

    schema_version: int = 1
    format: Literal["test", "odi", "t20"] = "odi"
    wicket_value: float | None = Field(default=None, description="Runs a wicket is worth; default depends on format")
    run_cost: float = 1.0
    extra_cost: float = 3.0
    max_bouncers_per_over: int | None = Field(default=None, description="Law: Tests/ODIs 2, T20 1")

    def resolved(self) -> dict[str, float]:
        defaults = {"test": (30.0, 2), "odi": (25.0, 2), "t20": (18.0, 1)}[self.format]
        wicket = self.wicket_value if self.wicket_value is not None else defaults[0]
        bouncers = self.max_bouncers_per_over if self.max_bouncers_per_over is not None else defaults[1]
        if not 1.0 <= wicket <= 60.0:
            raise ValueError("wicket_value must be between 1 and 60 runs")
        if not 0.1 <= self.run_cost <= 5.0 or not 0.0 <= self.extra_cost <= 10.0:
            raise ValueError("run_cost must be in [0.1, 5] and extra_cost in [0, 10]")
        if not 0 <= bouncers <= 2:
            raise ValueError("max_bouncers_per_over must be 0, 1 or 2")
        return {"wicket": wicket, "run": self.run_cost, "extra": self.extra_cost, "max_bouncers": int(bouncers)}


# ---------------------------------------------------------------- populations

class BatterPopulation:
    """Many plausible versions of one batter: calibrated parameter sets plus small per-over variation."""

    def __init__(self, card: BatterCard, params: list[dict[str, float]] | None = None):
        self.card = card
        self.params = params or [{}]

    def sample(self, rng: np.random.Generator) -> BatterArchetype:
        chosen = self.params[int(rng.integers(0, len(self.params)))]
        return sample_batter(self.card.to_archetype(chosen), rng, spread=min(0.9, self.card.spread()))

    def member(self, index: int) -> BatterArchetype:
        return self.card.to_archetype(self.params[index % len(self.params)])

    def __len__(self) -> int:
        return len(self.params)


# ---------------------------------------------------------------- YAML IO

def _dump(model: BaseModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(model.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True), encoding="utf-8")


def save_card(card: BaseModel, path: str | Path) -> None:
    _dump(card, path)


def load_batter(path: str | Path) -> BatterCard:
    return BatterCard.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_bowler(path: str | Path) -> BowlerCard:
    return BowlerCard.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_reward(path: str | Path | None) -> RewardSpec:
    if path is None:
        return RewardSpec()
    return RewardSpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def with_params(archetype: BatterArchetype, **changes) -> BatterArchetype:
    return replace(archetype, **changes)
