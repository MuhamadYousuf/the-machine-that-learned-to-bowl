import numpy as np

from aibowler.batter import (
    BatterState,
    DeliveryFeatures,
    SLOGGER,
    TAILENDER,
    TECHNICIAN,
    delivery_features,
    sample_batter,
)
from aibowler.constants import BATTER_STUMPS_X_M, RELEASE_POSITION_M
from aibowler.outcome import calculate_probabilities, resolve
from aibowler.physics import PitchConditions, Release, simulate_delivery


def make_features(
    *,
    length=6.5,
    line=0.0,
    height=0.65,
    speed=118.0,
    late=0.0,
    hits=False,
    outside_leg=False,
    wide=False,
    beamer=False,
    category="good",
):
    return DeliveryFeatures(
        length_m=length,
        line_m=line,
        height_at_crease_m=height,
        speed_at_crease_kmh=speed,
        late_movement_m=late,
        hits_stumps=hits,
        pitched_outside_leg=outside_leg,
        is_wide=wide,
        is_beamer=beamer,
        length_category=category,
    )


def high_surprise_state(archetype):
    return BatterState.for_archetype(
        archetype,
        confidence=0.12,
        expected_length_m=12.5,
        expected_line_m=-0.55,
        expected_late_movement_m=-0.25,
    )


def test_delivery_features_are_extracted_relative_to_off_stump():
    aimed = np.degrees(
        np.arctan2(-RELEASE_POSITION_M[1], BATTER_STUMPS_X_M - RELEASE_POSITION_M[0])
    )
    release = Release(140.0, -5.4, float(aimed), -0.8)
    trajectory = simulate_delivery(
        release, PitchConditions(seam_sd_deg=0.0), np.random.default_rng(1)
    )
    features = delivery_features(trajectory)
    assert features.length_m is not None
    assert features.speed_at_crease_kmh > 0.0
    assert features.line_m > 0.2
    assert features.late_movement_m > 0.0
    assert features.hits_stumps == trajectory.hits_stumps


def test_archetype_weaknesses_are_distinct_and_learnable():
    slogger_yorker = make_features(length=1.2, line=-0.05, height=0.2, hits=True)
    slogger_state = high_surprise_state(SLOGGER)
    slogger_wicket = calculate_probabilities(
        slogger_yorker, SLOGGER, slogger_state
    ).wicket_prob
    technician_same_ball = calculate_probabilities(
        slogger_yorker, TECHNICIAN, high_surprise_state(TECHNICIAN)
    ).wicket_prob
    assert slogger_wicket > technician_same_ball

    corridor_outswing = make_features(
        length=7.0, line=0.24, height=0.75, late=0.23, category="good"
    )
    straight = make_features(
        length=7.0, line=-0.02, height=0.75, late=0.0, category="good"
    )
    tech_state = high_surprise_state(TECHNICIAN)
    corridor_wicket = calculate_probabilities(
        corridor_outswing, TECHNICIAN, tech_state
    ).wicket_prob
    straight_wicket = calculate_probabilities(
        straight, TECHNICIAN, tech_state
    ).wicket_prob
    assert corridor_wicket > straight_wicket

    body_bouncer = make_features(
        length=11.5, line=0.05, height=1.65, late=0.0, category="bouncer"
    )
    tailender_wicket = calculate_probabilities(
        body_bouncer, TAILENDER, high_surprise_state(TAILENDER)
    ).wicket_prob
    slogger_bouncer_wicket = calculate_probabilities(
        body_bouncer, SLOGGER, high_surprise_state(SLOGGER)
    ).wicket_prob
    assert tailender_wicket > slogger_bouncer_wicket


def test_surprise_increases_dismissal_probability():
    feature = make_features(
        length=11.5, line=0.05, height=1.65, late=0.0, category="bouncer"
    )
    expected_state = BatterState.for_archetype(
        TAILENDER,
        confidence=0.15,
        expected_length_m=11.5,
        expected_line_m=0.05,
        expected_late_movement_m=0.0,
    )
    surprised_state = BatterState.for_archetype(
        TAILENDER,
        confidence=0.15,
        expected_length_m=3.0,
        expected_line_m=0.65,
        expected_late_movement_m=-0.25,
    )
    expected = calculate_probabilities(feature, TAILENDER, expected_state)
    surprised = calculate_probabilities(feature, TAILENDER, surprised_state)
    assert surprised.surprise > expected.surprise
    assert surprised.wicket_prob > expected.wicket_prob


def test_repeated_exploit_delivery_sets_batter_and_lowers_wicket_probability():
    feature = make_features(
        length=1.2, line=-0.05, height=0.2, hits=True, category="yorker"
    )
    state = high_surprise_state(SLOGGER)
    rng = np.random.default_rng(913)
    wicket_probabilities = []
    for _ in range(6):
        outcome = resolve(feature, SLOGGER, state, rng)
        wicket_probabilities.append(outcome.wicket_prob)
    assert state.balls_faced == 6
    assert wicket_probabilities[5] < wicket_probabilities[0]


def test_outcome_probabilities_sum_to_one_and_are_seed_deterministic():
    feature = make_features(length=7.0, line=0.2, late=0.15, hits=True)
    first_state = high_surprise_state(TECHNICIAN)
    second_state = high_surprise_state(TECHNICIAN)
    first_distribution = calculate_probabilities(feature, TECHNICIAN, first_state)
    assert abs(sum(first_distribution.probabilities.values()) - 1.0) < 1e-12

    first = resolve(feature, TECHNICIAN, first_state, np.random.default_rng(42))
    second = resolve(feature, TECHNICIAN, second_state, np.random.default_rng(42))
    assert first == second


def test_wide_and_beamer_no_ball_are_not_legal_balls():
    wide = make_features(wide=True)
    wide_outcome = resolve(
        wide, SLOGGER, BatterState.for_archetype(SLOGGER), np.random.default_rng(10)
    )
    assert wide_outcome.result == "wide"
    assert not wide_outcome.is_legal_ball
    assert wide_outcome.runs_conceded == 1

    beamer = make_features(length=None, height=1.2, beamer=True, category="full_toss")
    no_ball = resolve(
        beamer, SLOGGER, BatterState.for_archetype(SLOGGER), np.random.default_rng(11)
    )
    assert no_ball.result == "no_ball"
    assert not no_ball.is_legal_ball
    assert no_ball.runs_conceded >= 1
    assert no_ball.expected_runs >= 3.0
    assert abs(sum(no_ball.probabilities.values()) - 1.0) < 1e-12


def test_batter_starts_at_archetype_confidence_and_updates_expectations_by_ema():
    state = BatterState.for_archetype(SLOGGER)
    assert state.confidence == SLOGGER.starting_confidence
    feature = make_features(length=9.0, line=0.2, late=0.1, wide=True)
    outcome = resolve(feature, SLOGGER, state, np.random.default_rng(12))
    assert outcome.result == "wide"
    assert state.expected_length_m == 7.75
    assert abs(state.expected_line_m - 0.1) < 1e-12
    assert abs(state.expected_late_movement_m - 0.05) < 1e-12
    assert abs(state.expected_speed_kmh - 126.5) < 1e-12
    assert state.confidence > SLOGGER.starting_confidence
    assert state.balls_faced == 1


def test_sampled_batter_varies_numeric_params_and_keeps_slogger_yorker_weakness():
    sampled = sample_batter(SLOGGER, np.random.default_rng(91))
    assert sampled is not SLOGGER
    assert sampled != SLOGGER
    strongest = max(sampled.weakness_zones, key=lambda zone: zone.strength)
    assert strongest.dismissal_mode == "bowled_lbw"
    assert strongest.centres["length_m"] < 2.0
    assert strongest.widths["length_m"] > 0.0


def test_weakness_zones_match_smoothly_instead_of_rectangularly():
    zone = SLOGGER.weakness_zones[0]
    center = make_features(length=1.0, line=-0.04, height=0.18)
    nearby = make_features(length=2.0, line=-0.04, height=0.18)
    outside = make_features(length=5.0, line=-0.04, height=0.18)
    state = BatterState.for_archetype(SLOGGER)
    assert zone.match(center, state) > zone.match(nearby, state)
    assert zone.match(nearby, state) > zone.match(outside, state) > 0.0


def test_slower_ball_speed_delta_is_part_of_surprise_and_expectations():
    fast = make_features(length=6.5, speed=140.0)
    slow = make_features(length=6.5, speed=118.0)
    state = BatterState.for_archetype(
        SLOGGER,
        expected_length_m=6.5,
        expected_line_m=0.0,
        expected_late_movement_m=0.0,
        expected_speed_kmh=140.0,
    )
    state.balls_faced = 1
    assert calculate_probabilities(slow, SLOGGER, state).surprise > calculate_probabilities(
        fast, SLOGGER, state
    ).surprise
    outcome = resolve(slow, SLOGGER, state, np.random.default_rng(87))
    assert state.expected_speed_kmh == 129.0
    assert outcome.surprise > 0.0


def test_slogger_expects_at_least_three_runs_from_a_non_beamer_full_toss():
    full_toss = make_features(
        length=None, line=-0.1, height=0.55, speed=115.0, hits=False, category="full_toss"
    )
    outcome = calculate_probabilities(
        full_toss, SLOGGER, BatterState.for_archetype(SLOGGER)
    )
    assert outcome.expected_runs >= 3.0


def test_constant_expected_slow_pace_is_easier_to_score_for_every_archetype():
    slow = make_features(length=7.0, speed=102.0)
    fast = make_features(length=7.0, speed=122.0)
    for archetype in (SLOGGER, TECHNICIAN, TAILENDER):
        settled = BatterState.for_archetype(archetype, expected_speed_kmh=102.0)
        fast_state = BatterState.for_archetype(archetype, expected_speed_kmh=122.0)
        settled.balls_faced = fast_state.balls_faced = 2
        settled.last_speed_kmh = 102.0
        fast_state.last_speed_kmh = 122.0
        slow_outcome = calculate_probabilities(slow, archetype, settled)
        fast_outcome = calculate_probabilities(fast, archetype, fast_state)
        assert slow_outcome.expected_runs > fast_outcome.expected_runs
        assert slow_outcome.miss_probability < fast_outcome.miss_probability


def test_sudden_slow_delivery_remains_a_change_of_pace_weapon():
    slow = make_features(length=6.5, speed=102.0)
    expected_slow = BatterState.for_archetype(SLOGGER, expected_speed_kmh=102.0)
    after_fast_ball = BatterState.for_archetype(SLOGGER, expected_speed_kmh=124.0)
    expected_slow.balls_faced = after_fast_ball.balls_faced = 1
    expected_slow.last_speed_kmh = 102.0
    after_fast_ball.last_speed_kmh = 124.0
    settled_result = calculate_probabilities(slow, SLOGGER, expected_slow)
    changeup_result = calculate_probabilities(slow, SLOGGER, after_fast_ball)
    assert changeup_result.surprise > settled_result.surprise
    assert changeup_result.wicket_prob > settled_result.wicket_prob


def test_all_archetypes_score_from_full_tosses_and_half_volleys():
    for archetype, full_toss_runs, half_volley_runs in (
        (SLOGGER, 3.0, 1.8),
        (TECHNICIAN, 2.0, 2.0),
        (TAILENDER, 2.0, 1.5),
    ):
        state = BatterState.for_archetype(archetype, expected_speed_kmh=105.0)
        state.balls_faced = 2
        state.last_speed_kmh = 105.0
        full_toss = make_features(length=None, speed=105.0, height=0.5, category="full_toss")
        half_volley = make_features(length=3.6, speed=105.0, height=0.4, category="full")
        assert calculate_probabilities(full_toss, archetype, state).expected_runs >= full_toss_runs
        assert calculate_probabilities(half_volley, archetype, state).expected_runs >= half_volley_runs


def test_variance_aware_surprise_drops_after_mixed_extremes_raise_variance():
    from aibowler.batter import surprise_score

    outlier = make_features(length=12.0, line=0.0, height=0.7, speed=135.0, late=0.0)
    narrow = BatterState.for_archetype(
        SLOGGER, expected_length_m=6.5, expected_line_m=0.0,
        expected_late_movement_m=0.0, expected_speed_kmh=135.0,
    )
    broad = BatterState.for_archetype(
        SLOGGER, expected_length_m=6.5, expected_line_m=0.0,
        expected_late_movement_m=0.0, expected_speed_kmh=135.0,
    )
    broad.variance_length_m2 = 16.0
    assert surprise_score(outlier, broad) < surprise_score(outlier, narrow)


def test_lag_two_pattern_learns_bouncer_yorker_alternation_within_over():
    from aibowler.batter import surprise_score

    bouncer = make_features(length=12.0, line=0.0, height=1.6, speed=145.0, category="bouncer")
    yorker = make_features(length=1.0, line=0.0, height=0.18, speed=115.0, category="yorker")
    state = BatterState.for_archetype(SLOGGER)
    scores = []
    for index in range(6):
        feature = bouncer if index % 2 == 0 else yorker
        scores.append(surprise_score(feature, state))
        resolve(feature, SLOGGER, state, np.random.default_rng(1000 + index))
    assert state.pattern_confidence > 0.9
    assert scores[3] < scores[2]
    assert scores[5] < 0.05


def test_p9_champion_clone_stays_below_each_specialist_plan():
    import csv
    from pathlib import Path

    matrix_path = Path(__file__).resolve().parents[1] / "outputs" / "batter" / "plan_matrix.csv"
    with matrix_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for archetype, specialist in (
        ("slogger", "P3"), ("technician", "P4"), ("tailender", "P5")
    ):
        selected = {row["plan"]: row for row in rows if row["archetype"] == archetype}
        assert float(selected["P9"]["expected_fitness"]) < float(
            selected[specialist]["expected_fitness"]
        )
        assert float(selected["P8"]["expected_fitness"]) < float(
            selected["P7"]["expected_fitness"]
        )
