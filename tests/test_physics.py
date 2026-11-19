import numpy as np

from aibowler.constants import BALL_RADIUS_M, BATTER_STUMPS_X_M, RELEASE_POSITION_M
from aibowler.physics import PitchConditions, Release, release_from_action, simulate_delivery


def _release_for_elevation(elevation_deg: float, *, swing: float = 0.0) -> Release:
    aimed_azimuth = np.degrees(
        np.arctan2(-RELEASE_POSITION_M[1], BATTER_STUMPS_X_M - RELEASE_POSITION_M[0])
    )
    return Release(140.0, elevation_deg, float(aimed_azimuth), swing)


def _pitch_length_for_target(target_length_m: float) -> float:
    # Bisection selects an elevation whose simulated bounce is close to the
    # requested distance from the batter's stumps. It tests the model output
    # without depending on a hand-waved angle-to-length formula.
    lo, hi = -14.0, 1.0
    rng = np.random.default_rng(0)
    for _ in range(28):
        mid = (lo + hi) / 2.0
        trajectory = simulate_delivery(_release_for_elevation(mid), PitchConditions(seam_sd_deg=0.0), rng)
        length = trajectory.length_from_stumps
        if length is None or length < target_length_m:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def _trajectory_for_length(target_length_m: float):
    elevation = _pitch_length_for_target(target_length_m)
    return simulate_delivery(
        _release_for_elevation(elevation),
        PitchConditions(seam_sd_deg=0.0),
        np.random.default_rng(20),
    )


def test_140_kmh_delivery_reaches_stumps_in_realistic_time():
    trajectory = _trajectory_for_length(6.5)
    assert trajectory.flight_time_to_stumps is not None
    assert 0.45 <= trajectory.flight_time_to_stumps <= 0.60


def test_release_is_off_side_but_zero_azimuth_targets_middle_stumps():
    release = release_from_action(np.array([0.0, 0.0, 0.0, 0.0]), np.random.default_rng(2))
    assert release.position[1] == 0.30
    trajectory = simulate_delivery(
        _release_for_elevation(-2.5), PitchConditions(seam_sd_deg=0.0), np.random.default_rng(3)
    )
    assert trajectory.state_at_stumps is not None
    assert abs(trajectory.state_at_stumps.pos[1]) < 0.02


def test_zero_swing_and_zero_action_azimuth_stay_on_the_aim_line():
    release = release_from_action(np.array([0.0, -0.05, 0.0, 0.0]), np.random.default_rng(2))
    trajectory = simulate_delivery(release, PitchConditions(seam_sd_deg=0.0), np.random.default_rng(3))
    x0, y0, _ = release.position
    dx = trajectory.pos[:, 0] - x0
    expected_y = y0 + dx * np.tan(np.radians(release.azimuth_deg))
    deviation = np.max(np.abs(trajectory.pos[:, 1] - expected_y))
    assert deviation < 0.02


def test_positive_and_negative_swing_curve_in_opposite_directions():
    conditions = PitchConditions(seam_sd_deg=0.0)
    straight = simulate_delivery(_release_for_elevation(-6.0), conditions, np.random.default_rng(4))
    inswing = simulate_delivery(_release_for_elevation(-6.0, swing=1.0), conditions, np.random.default_rng(4))
    outswing = simulate_delivery(_release_for_elevation(-6.0, swing=-1.0), conditions, np.random.default_rng(4))
    assert straight.bounce_point is not None
    assert inswing.bounce_point is not None
    assert outswing.bounce_point is not None
    assert inswing.bounce_point[1] < straight.bounce_point[1]
    assert outswing.bounce_point[1] > straight.bounce_point[1]


def test_bounce_length_categories_cover_yorker_good_and_short():
    cases = [
        (1.0, "yorker"),
        (4.0, "full"),
        (6.5, "good"),
        (9.0, "short"),
        (11.5, "bouncer"),
    ]
    for target_length, expected_category in cases:
        elevation = _pitch_length_for_target(target_length)
        trajectory = simulate_delivery(
            _release_for_elevation(elevation),
            PitchConditions(seam_sd_deg=0.0),
            np.random.default_rng(5),
        )
        assert trajectory.length_from_stumps is not None
        assert abs(trajectory.length_from_stumps - target_length) < 0.15
        assert trajectory.length_category == expected_category


def test_bouncer_arrives_above_stumps_and_does_not_hit_them():
    trajectory = _trajectory_for_length(11.5)
    assert trajectory.state_at_stumps is not None
    assert trajectory.state_at_stumps.pos[2] > 0.711
    assert not trajectory.hits_stumps


def test_deliberately_aimed_straight_yorker_hits_stumps():
    trajectory = _trajectory_for_length(1.0)
    assert trajectory.length_category == "yorker"
    assert trajectory.hits_stumps


def test_bounce_calibration_targets_at_140_kmh():
    conditions = PitchConditions(seam_sd_deg=0.0)
    good = _trajectory_for_length(6.5)
    short = _trajectory_for_length(9.0)
    bouncer = _trajectory_for_length(11.5)

    assert 0.55 <= good.state_at_stumps.pos[2] <= 0.80
    assert 1.0 <= short.state_at_popping_crease.pos[2] <= 1.35
    assert 1.5 <= bouncer.state_at_popping_crease.pos[2] <= 1.95
    assert 0.84 <= np.linalg.norm(good.state_at_popping_crease.vel) * 3.6 / 140.0 <= 0.90
    assert 0.47 <= good.flight_time_to_stumps <= 0.58
    assert conditions.drag_coefficient == 0.40


def test_off_side_and_leg_side_wides_use_asymmetric_limits():
    aimed = np.degrees(
        np.arctan2(-RELEASE_POSITION_M[1], BATTER_STUMPS_X_M - RELEASE_POSITION_M[0])
    )
    offside = Release(140.0, -4.0, float(aimed + 3.5), 0.0)
    legside = Release(140.0, -4.0, float(aimed - 2.0), 0.0)
    assert simulate_delivery(
        offside, PitchConditions(seam_sd_deg=0.0), np.random.default_rng(30)
    ).is_wide
    assert simulate_delivery(
        legside, PitchConditions(seam_sd_deg=0.0), np.random.default_rng(31)
    ).is_wide


def test_execution_noise_is_deterministic_for_the_same_seed():
    action = np.array([0.2, -0.15, 0.1, -0.3])
    first = release_from_action(action, np.random.default_rng(42), noise_scale=0.08)
    second = release_from_action(action, np.random.default_rng(42), noise_scale=0.08)
    assert first == second


def test_bounce_uses_ball_centre_height():
    trajectory = simulate_delivery(
        _release_for_elevation(-6.0), PitchConditions(seam_sd_deg=0.0), np.random.default_rng(8)
    )
    assert trajectory.bounce_point is not None
    assert trajectory.bounce_point[2] == BALL_RADIUS_M
