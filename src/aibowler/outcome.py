"""Transparent Gaussian-zone batting probabilities and outcome sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .batter import (
    BatterArchetype,
    BatterState,
    DeliveryFeatures,
    get_archetype,
    lag2_similarity,
    repetition_similarity,
    scoring_quality,
    speed_surprise_delta,
    surprise_score,
    weakness_profile,
)


RUN_VALUES = (1, 2, 3, 4, 6)
RESULT_KEYS = (
    "dot",
    "run_1",
    "run_2",
    "run_3",
    "run_4",
    "run_6",
    "wide",
    "no_ball",
    "bowled",
    "lbw",
    "caught_behind",
    "caught_field",
)
WICKET_KEYS = ("bowled", "lbw", "caught_behind", "caught_field")


@dataclass(frozen=True)
class OutcomeProbabilities:
    """Probability vector and explainable latent batting probabilities."""

    probabilities: dict[str, float]
    expected_runs: float
    wicket_prob: float
    surprise: float
    miss_probability: float
    edge_probability: float
    connect_probability: float
    batting_run_probabilities: dict[str, float]


@dataclass(frozen=True)
class BallOutcome:
    """One sampled outcome and the probability distribution behind it."""

    result: str
    runs_conceded: int
    runs_scored: int
    is_wicket: bool
    is_legal_ball: bool
    shot: str
    shot_angle_deg: float | None
    shot_distance_m: float | None
    probabilities: dict[str, float]
    expected_runs: float
    wicket_prob: float
    surprise: float
    miss_probability: float
    edge_probability: float
    connect_probability: float
    batting_run_probabilities: dict[str, float]


def _empty_probabilities() -> dict[str, float]:
    return {key: 0.0 for key in RESULT_KEYS}


def _run_distribution(
    quality: float,
    aggression: float,
    confidence: float,
    surprise: float,
    rotation: float = 0.0,
) -> tuple[dict[str, float], float]:
    """Generate batting-run probabilities from matchup quality and state.

    ``rotation`` (0..1) turns some dot balls into singles and twos; 0 keeps the original model.
    """
    scoring_probability = float(
        np.clip(
            0.14
            + 0.13 * aggression
            + 0.40 * quality
            + 0.04 * surprise
            - 0.08 * confidence,
            0.01,
            0.97,
        )
    )
    if rotation:
        scoring_probability = float(np.clip(scoring_probability + 0.62 * rotation * (1.0 - scoring_probability), 0.01, 0.97))
    boundary_shift = float(np.clip(quality, 0.0, 1.0))
    conditional = {
        "run_1": 0.35 - 0.16 * boundary_shift,
        "run_2": 0.17,
        "run_3": 0.04,
        "run_4": 0.27 + 0.10 * boundary_shift,
        "run_6": 0.17 + 0.06 * boundary_shift,
    }
    if rotation:
        # Lab batters: blend towards a real-cricket scoring mix (mostly singles; sixes scale with intent).
        realistic = {"run_1": 0.62, "run_2": 0.12, "run_3": 0.01,
                     "run_4": 0.17 + 0.08 * boundary_shift, "run_6": 0.02 + 0.10 * aggression * boundary_shift}
        base_total = sum(conditional.values())
        real_total = sum(realistic.values())
        conditional = {k: (1.0 - rotation) * conditional[k] / base_total + rotation * realistic[k] / real_total
                       for k in conditional}
    total = sum(conditional.values())
    conditional = {key: value / total for key, value in conditional.items()}
    result = {"dot": 1.0 - scoring_probability}
    result.update({key: scoring_probability * value for key, value in conditional.items()})
    return result, scoring_probability


def _latent_probabilities(
    features: DeliveryFeatures,
    archetype: BatterArchetype,
    surprise: float,
    weakness: float,
    mode: str | None,
    *,
    fastness: float,
    pace_change: float,
    slow_expected: float,
) -> tuple[float, float, float, float]:
    """Estimate miss, edge, contact, and caught-field mis-hit chances."""
    # Fast deliveries are harder to control; a slow delivery only receives
    # the easy-scoring treatment once the batter expects that pace. A sudden
    # pace change remains disruptive.
    miss = float(
        np.clip(
            0.018
            + (1.0 - archetype.skill) * 0.07
            + 0.08 * surprise
            + 0.06 * fastness
            + 0.025 * pace_change
            - 0.05 * slow_expected
            + (0.035 * weakness if mode == "bowled_lbw" else 0.0),
            0.01,
            0.60,
        )
    )
    edge = float(
        np.clip(
            0.018
            + min(0.20, abs(features.late_movement_m) * 0.20)
            + 0.025 * fastness
            + 0.010 * pace_change
            + (0.17 * weakness if mode == "caught_behind" else 0.0),
            0.008,
            0.55,
        )
    )
    if miss + edge > 0.90:
        factor = 0.90 / (miss + edge)
        miss *= factor
        edge *= factor
    connect = 1.0 - miss - edge
    mishit = 0.003 + (1.0 - archetype.skill) * 0.010 + 0.035 * archetype.aerial_risk
    if mode == "caught_field":
        mishit += 0.035 * weakness
    return miss, edge, connect, float(np.clip(mishit, 0.0, 0.35))


def _wicket_weights(
    features: DeliveryFeatures, archetype: BatterArchetype, mode: str | None
) -> dict[str, float]:
    if mode == "caught_behind":
        return {"caught_behind": 0.96, "caught_field": 0.04}
    if mode == "caught_field":
        return {"caught_field": 0.94, "caught_behind": 0.06}
    if mode == "bowled_lbw":
        return {"bowled": 1.0} if features.pitched_outside_leg else {"bowled": 0.78, "lbw": 0.22}
    if features.hits_stumps:
        defense = archetype.straight_defense
        return {
            "bowled": 0.54 * (1.0 - defense),
            "lbw": 0.35 * (1.0 - defense),
            "caught_field": 0.08 + 0.65 * defense,
            "caught_behind": 0.03 + 0.26 * defense,
        }
    return {"caught_field": 0.70, "caught_behind": 0.30}


def calculate_probabilities(
    features: DeliveryFeatures,
    archetype: BatterArchetype | str,
    state: BatterState,
) -> OutcomeProbabilities:
    """Calculate outcomes without sampling or mutating the batter state."""
    params = get_archetype(archetype)
    surprise = surprise_score(features, state)
    weakness, mode = weakness_profile(features, params, state)
    # Fast pace taxes scoring control. Expected slow pace is easier to score;
    # an unexpected slower ball keeps its value through elevated surprise.
    speed_delta = speed_surprise_delta(features, state)
    fastness = float(np.clip((features.speed_at_crease_kmh - 100.0) / 45.0, 0.0, 1.0))
    pace_change = float(np.clip(abs(speed_delta) / 25.0, 0.0, 1.0))
    slow_pace_band = float(np.clip((135.0 - features.speed_at_crease_kmh) / 20.0, 0.0, 1.0))
    slow_expected = slow_pace_band * float(np.exp(-0.5 * (speed_delta / 10.0) ** 2))
    pace_repeat = (
        0.0
        if state.last_speed_kmh is None
        else float(np.exp(-0.5 * ((features.speed_at_crease_kmh - state.last_speed_kmh) / 2.0) ** 2))
    )
    miss, edge, connect, mishit = _latent_probabilities(
        features, params, surprise, weakness, mode,
        fastness=fastness, pace_change=pace_change, slow_expected=slow_expected,
    )

    if features.is_beamer:
        batting_runs, _ = _run_distribution(
            quality=1.8,
            aggression=1.0,
            confidence=0.0,
            surprise=max(0.5, surprise),
        )
        expected_runs = 1.0 + sum(
            probability * int(key[4:])
            for key, probability in batting_runs.items()
            if key.startswith("run_")
        )
        probabilities = _empty_probabilities()
        probabilities["no_ball"] = 1.0
        return OutcomeProbabilities(
            probabilities, float(expected_runs), 0.0, surprise,
            miss, edge, connect, batting_runs,
        )

    if features.is_wide:
        probabilities = _empty_probabilities()
        probabilities["wide"] = 1.0
        return OutcomeProbabilities(
            probabilities, 1.0, 0.0, surprise, miss, edge, connect, {"dot": 1.0}
        )

    base_wicket = (
        (0.018 if features.hits_stumps else 0.003)
        + miss * (0.17 if features.hits_stumps else 0.018)
        + edge * (0.075 if features.late_movement_m > 0.05 else 0.032)
        + connect * mishit
    )
    weakness_wicket = weakness * (0.012 + 0.75 * surprise ** 2.6)
    wicket_prob = (base_wicket * (0.82 + 0.60 * surprise) + weakness_wicket) * (
        1.0 - 0.42 * state.confidence
    )
    wicket_prob = float(np.clip(wicket_prob, 0.0, 0.45))

    weights = _wicket_weights(features, params, mode)
    weight_total = sum(weights.values())
    weights = {key: value / weight_total for key, value in weights.items()}
    # A settled slow pace is easiest to score when the preceding delivery
    # confirms it. This distinguishes constant slow bowling from a random
    # isolated speed match and leaves a fast-to-slow change-up disruptive.
    pace_quality = 1.50 * slow_expected * pace_repeat - 0.60 * fastness
    batting_runs, _ = _run_distribution(
        quality=float(np.clip(scoring_quality(features, params, state) + pace_quality, -1.5, 1.8)),
        aggression=params.aggression,
        confidence=state.confidence,
        surprise=surprise,
        rotation=params.strike_rotation,
    )

    probabilities = _empty_probabilities()
    for key in ("dot", "run_1", "run_2", "run_3", "run_4", "run_6"):
        probabilities[key] = (1.0 - wicket_prob) * batting_runs[key]
    for key, weight in weights.items():
        probabilities[key] = wicket_prob * weight
    total = sum(probabilities.values())
    probabilities = {key: value / total for key, value in probabilities.items()}
    expected_runs = sum(probabilities[f"run_{runs}"] * runs for runs in RUN_VALUES)
    actual_wicket_prob = sum(probabilities[key] for key in WICKET_KEYS)
    return OutcomeProbabilities(
        probabilities=probabilities,
        expected_runs=float(expected_runs),
        wicket_prob=float(actual_wicket_prob),
        surprise=surprise,
        miss_probability=miss,
        edge_probability=edge,
        connect_probability=connect,
        batting_run_probabilities=batting_runs,
    )


def _sample_key(probabilities: Mapping[str, float], rng: np.random.Generator) -> str:
    keys = tuple(probabilities)
    values = np.asarray([probabilities[key] for key in keys], dtype=float)
    values /= values.sum()
    return str(rng.choice(keys, p=values))


def _shot_for_scoring(
    features: DeliveryFeatures,
    archetype: BatterArchetype,
    runs: int,
    rng: np.random.Generator,
) -> tuple[str, float, float]:
    length = 0.0 if features.length_m is None else features.length_m
    if features.line_m > archetype.leave_outside_off_line_m:
        shot = "cut"
    elif length > 10.5:
        shot = "pull" if archetype.aggression >= 0.45 else "fend"
    elif features.line_m < -0.25:
        shot = "pull" if archetype.aggression >= 0.45 else archetype.power_shot
    elif length <= 5.0:
        shot = archetype.power_shot if runs >= 4 else "drive"
    elif features.speed_at_crease_kmh < 105.0:
        shot = archetype.power_shot
    else:
        shot = "defend" if runs <= 2 else archetype.power_shot

    if shot == "cut":
        mean_angle = 55.0
    elif shot in ("pull", "slog", "fend"):
        mean_angle = -25.0 if features.line_m < -0.1 else 20.0
    else:
        mean_angle = 5.0 if features.line_m >= 0.0 else -5.0
    angle = float(np.clip(rng.normal(mean_angle, 16.0), -170.0, 170.0))
    distance_by_runs = {1: 12.0, 2: 22.0, 3: 32.0, 4: 50.0, 6: 76.0}
    distance = float(
        max(4.0, rng.normal(distance_by_runs[runs], 5.0 if runs < 4 else 9.0))
    )
    return shot, angle, distance


def _shot_for_non_scoring(
    features: DeliveryFeatures,
    archetype: BatterArchetype,
    miss_probability: float,
    edge_probability: float,
    rng: np.random.Generator,
) -> str:
    if features.line_m > archetype.leave_outside_off_line_m:
        return "leave"
    if features.length_m is not None and features.length_m > 10.5:
        return archetype.caught_field_shot if rng.random() < 0.35 else "leave"
    latent = {
        "miss": miss_probability,
        "edge": edge_probability,
        "defend": max(0.05, 1.0 - miss_probability - edge_probability),
    }
    return _sample_key(latent, rng)


def _update_state(
    features: DeliveryFeatures,
    state: BatterState,
    *,
    is_wicket: bool,
    runs_scored: int,
    shot: str,
    similarity: float,
    lag2: float,
    alpha: float = 0.5,
) -> None:
    if is_wicket:
        state.confidence -= 0.10
    else:
        state.confidence += 0.015 + 0.035 * similarity
        if runs_scored:
            state.confidence += 0.035
        if shot in ("miss", "edge"):
            state.confidence -= 0.060
    state.confidence = float(np.clip(state.confidence, 0.0, 1.0))

    length = 0.0 if features.length_m is None else float(features.length_m)
    measurements = (
        (length, "expected_length_m", "variance_length_m2"),
        (features.line_m, "expected_line_m", "variance_line_m2"),
        (features.late_movement_m, "expected_late_movement_m", "variance_late_movement_m2"),
        (features.speed_at_crease_kmh, "expected_speed_kmh", "variance_speed_kmh2"),
    )
    for value, mean_name, variance_name in measurements:
        old_mean = float(getattr(state, mean_name))
        old_variance = float(getattr(state, variance_name))
        # Stable EMA variance around the prior mean; alternating extremes
        # quickly broaden the model and stop looking novel every delivery.
        new_variance = (1.0 - alpha) * (
            old_variance + alpha * (float(value) - old_mean) ** 2
        )
        setattr(state, variance_name, new_variance)
        setattr(state, mean_name, alpha * float(value) + (1.0 - alpha) * old_mean)
    state.pattern_confidence = float(
        np.clip(0.65 * lag2 + 0.35 * state.pattern_confidence, 0.0, 1.0)
    )
    state.two_back_length_m = state.last_length_m
    state.two_back_line_m = state.last_line_m
    state.two_back_late_movement_m = state.last_late_movement_m
    state.two_back_speed_kmh = state.last_speed_kmh
    state.last_length_m = length
    state.last_line_m = features.line_m
    state.last_late_movement_m = features.late_movement_m
    state.last_speed_kmh = features.speed_at_crease_kmh
    state.balls_faced += 1


def resolve(
    features: DeliveryFeatures,
    archetype: BatterArchetype | str,
    state: BatterState,
    rng: np.random.Generator,
    *,
    force_no_ball: bool = False,
) -> BallOutcome:
    """Sample one outcome, then update this batter's expectations and state."""
    params = get_archetype(archetype)
    similarity = repetition_similarity(features, state)
    lag2 = lag2_similarity(features, state)
    distribution = calculate_probabilities(features, params, state)
    if force_no_ball:
        probabilities = _empty_probabilities()
        probabilities["no_ball"] = 1.0
        _update_state(
            features, state, is_wicket=False, runs_scored=0, shot="leave",
            similarity=similarity, lag2=lag2,
        )
        return BallOutcome(
            result="no_ball", runs_conceded=1, runs_scored=0,
            is_wicket=False, is_legal_ball=False, shot="leave",
            shot_angle_deg=None, shot_distance_m=None,
            probabilities=probabilities, expected_runs=1.0, wicket_prob=0.0,
            surprise=distribution.surprise, miss_probability=distribution.miss_probability,
            edge_probability=distribution.edge_probability,
            connect_probability=distribution.connect_probability,
            batting_run_probabilities={"dot": 1.0},
        )
    probabilities = distribution.probabilities
    sampled_key = _sample_key(probabilities, rng)

    if sampled_key == "wide":
        result, runs_scored, runs_conceded = "wide", 0, 1
        is_wicket, is_legal_ball, shot = False, False, "leave"
        angle = distance = None
    elif sampled_key == "no_ball":
        result = "no_ball"
        run_key = _sample_key(distribution.batting_run_probabilities, rng)
        runs_scored = int(run_key[4:]) if run_key.startswith("run_") else 0
        runs_conceded = 1 + runs_scored
        is_wicket, is_legal_ball = False, False
        if runs_scored:
            shot, angle, distance = _shot_for_scoring(features, params, runs_scored, rng)
        else:
            shot = _shot_for_non_scoring(
                features, params, distribution.miss_probability,
                distribution.edge_probability, rng,
            )
            angle = distance = None
    elif sampled_key.startswith("run_"):
        result = "runs"
        runs_scored = int(sampled_key[4:])
        runs_conceded = runs_scored
        is_wicket, is_legal_ball = False, True
        shot, angle, distance = _shot_for_scoring(features, params, runs_scored, rng)
    else:
        result, runs_scored, runs_conceded = sampled_key, 0, 0
        is_wicket, is_legal_ball = sampled_key in WICKET_KEYS, True
        if sampled_key == "caught_behind":
            shot = "edge"
        elif sampled_key == "caught_field":
            shot = params.caught_field_shot
        elif sampled_key in ("bowled", "lbw"):
            shot = "miss"
        else:
            shot = _shot_for_non_scoring(
                features, params, distribution.miss_probability,
                distribution.edge_probability, rng,
            )
        angle = distance = None

    _update_state(
        features, state, is_wicket=is_wicket, runs_scored=runs_scored,
        shot=shot, similarity=similarity, lag2=lag2,
    )
    return BallOutcome(
        result=result,
        runs_conceded=runs_conceded,
        runs_scored=runs_scored,
        is_wicket=is_wicket,
        is_legal_ball=is_legal_ball,
        shot=shot,
        shot_angle_deg=angle,
        shot_distance_m=distance,
        probabilities=dict(probabilities),
        expected_runs=distribution.expected_runs,
        wicket_prob=distribution.wicket_prob,
        surprise=distribution.surprise,
        miss_probability=distribution.miss_probability,
        edge_probability=distribution.edge_probability,
        connect_probability=distribution.connect_probability,
        batting_run_probabilities=dict(distribution.batting_run_probabilities),
    )
