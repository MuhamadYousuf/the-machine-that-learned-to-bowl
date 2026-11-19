"""Data-driven right-handed batter archetypes and delivery features."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, hypot, sqrt
from typing import Mapping

import numpy as np

from .constants import BALL_RADIUS_M, BATTER_POPPING_CREASE_X_M, STUMPS_WIDTH_M
from .physics import Trajectory


ZONE_FEATURES = frozenset(
    {
        "length_m",
        "line_m",
        "late_movement_m",
        "height_at_crease_m",
        "speed_delta_kmh",
    }
)
VARIATION_SHIFT_LIMITS = {
    "length_m": 0.50,
    "line_m": 0.06,
    "late_movement_m": 0.06,
    "height_at_crease_m": 0.15,
    "speed_delta_kmh": 3.0,
}


@dataclass(frozen=True)
class DeliveryFeatures:
    """Trajectory measurements used by the batter and outcome models.

    Line is measured at the popping crease relative to off stump: zero is off
    stump and positive values are wider outside off. Late movement measures
    the final six metres before the bounce (or before the popping crease for a
    full toss) against an extrapolation from the preceding path. This retains
    the outswing signal when the ball has already stopped swinging after its
    bounce. ``speed_delta_kmh`` is state-relative, so use :func:`feature_value`
    to compute it for a batter.
    """

    length_m: float | None
    line_m: float
    height_at_crease_m: float
    speed_at_crease_kmh: float
    late_movement_m: float
    hits_stumps: bool
    pitched_outside_leg: bool
    is_wide: bool
    is_beamer: bool
    length_category: str

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory) -> DeliveryFeatures:
        crease = trajectory.state_at_popping_crease
        if crease is None:
            raise ValueError("trajectory must reach the batter's popping crease")

        x = trajectory.pos[:, 0]
        y = trajectory.pos[:, 1]
        eval_x = (
            float(trajectory.bounce_point[0])
            if trajectory.bounce_point is not None
            else BATTER_POPPING_CREASE_X_M
        )
        x_late = eval_x - 6.0
        x_early = max(float(x[0]), x_late - 6.0)
        x_late = max(float(x[0]), x_late)
        y_early = float(np.interp(x_early, x, y))
        y_late = float(np.interp(x_late, x, y))
        y_end = float(np.interp(eval_x, x, y))
        prior_slope = (y_late - y_early) / max(1e-6, x_late - x_early)
        straight_extrapolation = y_late + prior_slope * (
            eval_x - x_late
        )
        late_movement = y_end - straight_extrapolation

        bounce = trajectory.bounce_point
        pitched_outside_leg = (
            bounce is not None
            and float(bounce[1]) + BALL_RADIUS_M < -STUMPS_WIDTH_M / 2.0
        )
        return cls(
            length_m=trajectory.length_from_stumps,
            line_m=float(crease.pos[1]) - STUMPS_WIDTH_M / 2.0,
            height_at_crease_m=float(crease.pos[2]),
            speed_at_crease_kmh=float(hypot(*crease.vel) * 3.6),
            late_movement_m=float(late_movement),
            hits_stumps=trajectory.hits_stumps,
            pitched_outside_leg=pitched_outside_leg,
            is_wide=trajectory.is_wide,
            is_beamer=trajectory.is_beamer,
            length_category=trajectory.length_category,
        )


def delivery_features(trajectory: Trajectory) -> DeliveryFeatures:
    """Build DeliveryFeatures from a simulated trajectory."""
    return DeliveryFeatures.from_trajectory(trajectory)


def effective_length_m(features: DeliveryFeatures) -> float:
    """Map full tosses to zero for zone matching and expectations."""
    return 0.0 if features.length_m is None else float(features.length_m)


def feature_value(
    name: str, features: DeliveryFeatures, state: BatterState
) -> float:
    """Read a zone feature, including speed relative to batter expectation."""
    if name not in ZONE_FEATURES:
        raise ValueError(f"unsupported zone feature {name!r}")
    if name == "length_m":
        return effective_length_m(features)
    if name == "speed_delta_kmh":
        return float(features.speed_at_crease_kmh - state.expected_speed_kmh)
    return float(getattr(features, name))


def _zone_match(
    centres: Mapping[str, float],
    widths: Mapping[str, float],
    features: DeliveryFeatures,
    state: BatterState,
) -> float:
    match = 1.0
    for name, centre in centres.items():
        value = feature_value(name, features, state)
        width = widths[name]
        match *= exp(-0.5 * ((value - centre) / width) ** 2)
    return float(np.clip(match, 0.0, 1.0))


def _validate_zone(centres: Mapping[str, float], widths: Mapping[str, float]) -> None:
    if not centres or set(centres) != set(widths):
        raise ValueError("zone centres and widths must have the same non-empty keys")
    if not set(centres).issubset(ZONE_FEATURES):
        raise ValueError(f"zone features must be drawn from {sorted(ZONE_FEATURES)}")
    if any(width <= 0.0 for width in widths.values()):
        raise ValueError("zone widths must be positive")


@dataclass(frozen=True)
class WeaknessZone:
    """A smooth weakness with per-feature Gaussian centres and widths."""

    centres: Mapping[str, float]
    widths: Mapping[str, float]
    strength: float
    dismissal_mode: str

    def __post_init__(self) -> None:
        _validate_zone(self.centres, self.widths)
        if self.strength < 0.0:
            raise ValueError("weakness strength must be non-negative")
        if self.dismissal_mode not in {"bowled_lbw", "caught_behind", "caught_field"}:
            raise ValueError(f"unknown dismissal mode {self.dismissal_mode!r}")

    def match(self, features: DeliveryFeatures, state: BatterState) -> float:
        return _zone_match(self.centres, self.widths, features, state)


@dataclass(frozen=True)
class ScoringZone:
    """A smooth scoring opportunity or scoring penalty."""

    centres: Mapping[str, float]
    widths: Mapping[str, float]
    run_quality_bonus: float

    def __post_init__(self) -> None:
        _validate_zone(self.centres, self.widths)

    def match(self, features: DeliveryFeatures, state: BatterState) -> float:
        return _zone_match(self.centres, self.widths, features, state)


@dataclass(frozen=True)
class BatterArchetype:
    """Base batting parameters and data-defined weakness/scoring zones."""

    name: str
    starting_confidence: float
    skill: float
    aggression: float
    weakness_zones: tuple[WeaknessZone, ...] = ()
    scoring_zones: tuple[ScoringZone, ...] = ()
    straight_defense: float = 0.25
    leave_outside_off_line_m: float = 1.5
    power_shot: str = "drive"
    caught_field_shot: str = "slog"
    # How much the batter works singles off balls that would otherwise be dots (0 = the original
    # project's batting model, unchanged; the tactics lab calibrates it against real data).
    strike_rotation: float = 0.0
    # Extra chance of hitting a catch on any connected shot (T20 risk-taking). 0 = original model.
    aerial_risk: float = 0.0

    def __post_init__(self) -> None:
        for name in ("starting_confidence", "skill", "aggression", "straight_defense", "strike_rotation", "aerial_risk"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.leave_outside_off_line_m < 0.0:
            raise ValueError("leave threshold must be non-negative")


def _weakness(
    centres: dict[str, float],
    widths: dict[str, float],
    strength: float,
    mode: str,
) -> WeaknessZone:
    return WeaknessZone(centres, widths, strength, mode)


def _scoring(
    centres: dict[str, float], widths: dict[str, float], bonus: float
) -> ScoringZone:
    return ScoringZone(centres, widths, bonus)


_YORKER_PENALTY = _scoring(
    {"length_m": 0.8, "line_m": 0.0, "height_at_crease_m": 0.18},
    {"length_m": 1.1, "line_m": 0.32, "height_at_crease_m": 0.18},
    -0.95,
)

SLOGGER = BatterArchetype(
    name="slogger",
    starting_confidence=0.14,
    skill=0.60,
    aggression=0.94,
    straight_defense=0.18,
    weakness_zones=(
        _weakness(
            {"length_m": 1.0, "line_m": -0.04},
            {"length_m": 1.0, "line_m": 0.25},
            0.25,
            "bowled_lbw",
        ),
        _weakness(
            {"length_m": 5.4, "speed_delta_kmh": -20.0},
            {"length_m": 2.5, "speed_delta_kmh": 7.0},
            0.19,
            "caught_field",
        ),
        _weakness(
            {"length_m": 12.0, "line_m": -0.08},
            {"length_m": 1.6, "line_m": 0.28},
            0.06,
            "caught_field",
        ),
    ),
    scoring_zones=(
        _YORKER_PENALTY,
        _scoring({"length_m": 0.0}, {"length_m": 1.2}, 1.85),
        _scoring({"length_m": 3.5}, {"length_m": 1.15}, 0.75),
        _scoring(
            {"length_m": 9.2, "line_m": -0.12},
            {"length_m": 1.05, "line_m": 0.32},
            0.55,
        ),
        _scoring(
            {"length_m": 9.2, "line_m": 0.66},
            {"length_m": 1.6, "line_m": 0.22},
            0.55,
        ),
        _scoring(
            {"length_m": 12.0, "line_m": -0.08},
            {"length_m": 1.7, "line_m": 0.32},
            0.55,
        ),
    ),
    leave_outside_off_line_m=1.4,
    power_shot="slog",
    caught_field_shot="slog",
)

TECHNICIAN = BatterArchetype(
    name="technician",
    starting_confidence=0.24,
    skill=0.91,
    aggression=0.60,
    straight_defense=0.92,
    leave_outside_off_line_m=0.50,
    power_shot="drive",
    caught_field_shot="drive",
    weakness_zones=(
        _weakness(
            {"length_m": 7.0, "line_m": 0.25, "late_movement_m": 0.08},
            {"length_m": 1.0, "line_m": 0.14, "late_movement_m": 0.07},
            0.40,
            "caught_behind",
        ),
    ),
    scoring_zones=(
        _YORKER_PENALTY,
        _scoring({"length_m": 0.0}, {"length_m": 0.65}, 1.45),
        _scoring({"length_m": 3.6}, {"length_m": 1.4}, 1.35),
        _scoring(
            {"length_m": 9.1, "line_m": 0.65},
            {"length_m": 1.7, "line_m": 0.23},
            0.95,
        ),
        _scoring(
            {"length_m": 7.0, "line_m": -0.45},
            {"length_m": 2.0, "line_m": 0.25},
            0.55,
        ),
        _scoring(
            {"length_m": 12.0, "line_m": -0.05},
            {"length_m": 1.6, "line_m": 0.40},
            -0.45,
        ),
        _scoring(
            {"length_m": 7.0, "line_m": 0.25, "late_movement_m": 0.20},
            {"length_m": 1.1, "line_m": 0.16, "late_movement_m": 0.12},
            -1.00,
        ),
    ),
)

TAILENDER = BatterArchetype(
    name="tailender",
    starting_confidence=0.08,
    skill=0.30,
    aggression=0.20,
    straight_defense=0.10,
    leave_outside_off_line_m=1.4,
    power_shot="slog",
    caught_field_shot="fend",
    weakness_zones=(
        _weakness(
            {"length_m": 12.0, "line_m": -0.10, "height_at_crease_m": 1.55},
            {"length_m": 1.7, "line_m": 0.22, "height_at_crease_m": 0.38},
            0.31,
            "caught_field",
        ),
        _weakness(
            {"length_m": 4.0, "line_m": 0.0},
            {"length_m": 0.9, "line_m": 0.22},
            0.28,
            "bowled_lbw",
        ),
    ),
    scoring_zones=(
        _YORKER_PENALTY,
        _scoring({"length_m": 0.0}, {"length_m": 1.2}, 0.52),
        _scoring({"length_m": 3.7}, {"length_m": 0.95}, 0.30),
    ),
)

ARCHETYPES = {SLOGGER.name: SLOGGER, TECHNICIAN.name: TECHNICIAN, TAILENDER.name: TAILENDER}


def get_archetype(archetype: BatterArchetype | str) -> BatterArchetype:
    if isinstance(archetype, BatterArchetype):
        return archetype
    try:
        return ARCHETYPES[archetype.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"unknown batter archetype: {archetype!r}") from error


@dataclass
class BatterState:
    """Mutable expectations, confidence, and repetition memory for one batter."""

    confidence: float = 0.15
    expected_length_m: float = 6.5
    expected_line_m: float = 0.0
    expected_late_movement_m: float = 0.0
    expected_speed_kmh: float = 135.0
    variance_length_m2: float = 0.0
    variance_line_m2: float = 0.0
    variance_late_movement_m2: float = 0.0
    variance_speed_kmh2: float = 0.0
    balls_faced: int = 0
    last_length_m: float | None = None
    last_line_m: float | None = None
    last_late_movement_m: float | None = None
    last_speed_kmh: float | None = None
    two_back_length_m: float | None = None
    two_back_line_m: float | None = None
    two_back_late_movement_m: float | None = None
    two_back_speed_kmh: float | None = None
    pattern_confidence: float = 0.0

    def __post_init__(self) -> None:
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        self.pattern_confidence = float(np.clip(self.pattern_confidence, 0.0, 1.0))

    @classmethod
    def for_archetype(
        cls,
        archetype: BatterArchetype | str,
        *,
        expected_length_m: float = 6.5,
        expected_line_m: float = 0.0,
        expected_late_movement_m: float = 0.0,
        expected_speed_kmh: float = 135.0,
        confidence: float | None = None,
    ) -> BatterState:
        params = get_archetype(archetype)
        return cls(
            confidence=params.starting_confidence if confidence is None else confidence,
            expected_length_m=expected_length_m,
            expected_line_m=expected_line_m,
            expected_late_movement_m=expected_late_movement_m,
            expected_speed_kmh=expected_speed_kmh,
        )


def surprise_score(features: DeliveryFeatures, state: BatterState) -> float:
    """Variance-aware novelty, discounted when a two-ball pattern is learned.

    Each standardized absolute deviation is smoothly squashed with
    ``1 - exp(-z**2 / 2)``.  A learned lag-2 sequence blends that novelty with
    surprise against the ball two deliveries ago, then takes the smaller
    score so predictable alternation stops earning repeat value.
    """
    speed_delta = speed_surprise_delta(features, state)
    values = (
        (effective_length_m(features), state.expected_length_m,
         state.variance_length_m2, 1.5),
        (features.line_m, state.expected_line_m, state.variance_line_m2, 0.15),
        (features.late_movement_m, state.expected_late_movement_m,
         state.variance_late_movement_m2, 0.10),
        (features.speed_at_crease_kmh, state.expected_speed_kmh,
         state.variance_speed_kmh2, 6.0),
    )
    z_scores = [
        abs(value - mean) / sqrt(max(0.0, variance) + floor * floor)
        for value, mean, variance, floor in values
    ]
    base = float(np.mean([1.0 - exp(-0.5 * z * z) for z in z_scores]))
    # Preserve the existing pace-change signal: an isolated slow ball after
    # fast deliveries differs from the same speed in a settled slow spell.
    pace_z = abs(speed_delta) / 10.0
    base = float(np.clip(base + 0.12 * (1.0 - exp(-0.5 * pace_z * pace_z)), 0.0, 1.0))
    lag2 = lag2_similarity(features, state)
    pattern_surprise = 1.0 - lag2
    weighted = (1.0 - state.pattern_confidence) * base + state.pattern_confidence * pattern_surprise
    return float(np.clip(min(base, weighted), 0.0, 1.0))


def speed_surprise_delta(features: DeliveryFeatures, state: BatterState) -> float:
    """State-relative speed surprise, discounted for a repeated pace.

    The initial 135 km/h expectation is a prior, not evidence that a first
    delivery was preceded by a fast ball. A repeated speed becomes familiar
    before the EMA has fully caught up; a real fast-to-slow change remains
    surprising.
    """
    if state.balls_faced == 0:
        return 0.0
    delta = features.speed_at_crease_kmh - state.expected_speed_kmh
    if state.last_speed_kmh is not None:
        repeat = exp(-0.5 * ((features.speed_at_crease_kmh - state.last_speed_kmh) / 6.0) ** 2)
        delta *= 1.0 - 0.75 * repeat
    return float(delta)


def repetition_similarity(features: DeliveryFeatures, state: BatterState) -> float:
    """Similarity to the previous delivery, including pace."""
    if state.last_length_m is None:
        return 0.0
    deltas = (
        (effective_length_m(features) - state.last_length_m) / 3.0,
        (features.line_m - float(state.last_line_m)) / 0.45,
        (features.late_movement_m - float(state.last_late_movement_m)) / 0.25,
        (features.speed_at_crease_kmh - float(state.last_speed_kmh)) / 12.0,
    )
    distance = float(np.sqrt(sum(delta * delta for delta in deltas) / 4.0))
    return float(exp(-2.0 * distance))


def lag2_similarity(features: DeliveryFeatures, state: BatterState) -> float:
    """Similarity to the delivery two balls back, on repetition's metric."""
    if state.two_back_length_m is None:
        return 0.0
    deltas = (
        (effective_length_m(features) - state.two_back_length_m) / 3.0,
        (features.line_m - float(state.two_back_line_m)) / 0.45,
        (features.late_movement_m - float(state.two_back_late_movement_m)) / 0.25,
        (features.speed_at_crease_kmh - float(state.two_back_speed_kmh)) / 12.0,
    )
    distance = float(np.sqrt(sum(delta * delta for delta in deltas) / 4.0))
    return float(exp(-2.0 * distance))


def weakness_profile(
    features: DeliveryFeatures,
    archetype: BatterArchetype | str,
    state: BatterState,
) -> tuple[float, str | None]:
    """Return combined Gaussian weakness strength and its dominant dismissal."""
    params = get_archetype(archetype)
    contributions = [
        (zone.strength * zone.match(features, state), zone.dismissal_mode)
        for zone in params.weakness_zones
    ]
    if not contributions:
        return 0.0, None
    strength = float(min(1.0, sum(value for value, _ in contributions)))
    mode = max(contributions, key=lambda item: item[0])[1]
    return strength, mode


def scoring_quality(
    features: DeliveryFeatures,
    archetype: BatterArchetype | str,
    state: BatterState,
) -> float:
    """Sum the archetype's smooth scoring-zone run-quality bonuses."""
    params = get_archetype(archetype)
    quality = sum(
        zone.run_quality_bonus * zone.match(features, state)
        for zone in params.scoring_zones
    )
    return float(np.clip(quality, -1.5, 1.8))


def sample_batter(
    archetype: BatterArchetype | str,
    rng: np.random.Generator,
    spread: float = 0.15,
) -> BatterArchetype:
    """Sample a varied copy while preserving the archetype's matchup shape."""
    if not 0.0 <= spread < 1.0:
        raise ValueError("spread must be in [0, 1)")
    base = get_archetype(archetype)

    def scaled(value: float) -> float:
        return value * (1.0 + float(rng.uniform(-spread, spread)))

    def varied_zone(zone):
        centres = {
            name: float(value)
            + float(
                rng.uniform(
                    -VARIATION_SHIFT_LIMITS[name] * spread / 0.15,
                    VARIATION_SHIFT_LIMITS[name] * spread / 0.15,
                )
            )
            for name, value in zone.centres.items()
        }
        width_factor = 1.0 + float(rng.uniform(-spread, spread))
        widths = {name: width * width_factor for name, width in zone.widths.items()}
        if isinstance(zone, WeaknessZone):
            return WeaknessZone(centres, widths, scaled(zone.strength), zone.dismissal_mode)
        return ScoringZone(centres, widths, scaled(zone.run_quality_bonus))

    return replace(
        base,
        starting_confidence=float(np.clip(scaled(base.starting_confidence), 0.0, 1.0)),
        skill=float(np.clip(scaled(base.skill), 0.0, 1.0)),
        aggression=float(np.clip(scaled(base.aggression), 0.0, 1.0)),
        straight_defense=float(np.clip(scaled(base.straight_defense), 0.0, 1.0)),
        leave_outside_off_line_m=max(0.0, scaled(base.leave_outside_off_line_m)),
        weakness_zones=tuple(varied_zone(zone) for zone in base.weakness_zones),
        scoring_zones=tuple(varied_zone(zone) for zone in base.scoring_zones),
    )
