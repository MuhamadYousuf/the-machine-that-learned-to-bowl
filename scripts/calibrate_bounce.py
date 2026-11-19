"""Search bounce constants against the architect's 140 km/h calibration points."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aibowler.constants import BATTER_STUMPS_X_M, RELEASE_POSITION_M
from aibowler.physics import PitchConditions, Release, simulate_delivery


TARGET_LENGTHS_M = (6.5, 9.0, 11.5)
RNG_SEED = 20261106


def aimed_azimuth_deg() -> float:
    x0, y0, _ = RELEASE_POSITION_M
    return float(np.degrees(np.arctan2(-y0, BATTER_STUMPS_X_M - x0)))


def release_for_length(target_m: float, baseline: PitchConditions) -> Release:
    """Find the elevation that places the first bounce at the requested length."""
    low, high = -14.0, 1.5
    azimuth = aimed_azimuth_deg()
    for _ in range(40):
        elevation = (low + high) / 2.0
        trajectory = simulate_delivery(
            Release(140.0, elevation, azimuth, 0.0),
            baseline,
            np.random.default_rng(RNG_SEED),
        )
        length = trajectory.length_from_stumps
        if length is None or length < target_m:
            high = elevation
        else:
            low = elevation
    return Release(140.0, (low + high) / 2.0, azimuth, 0.0)


def outside_range(value: float, low: float, high: float, scale: float) -> float:
    if value < low:
        return (low - value) / scale
    if value > high:
        return (value - high) / scale
    return 0.0


def inspect_candidate(
    restitution: float,
    slope: float,
    horizontal_retention: float,
    releases: tuple[Release, ...],
) -> tuple[float, list[dict[str, float]]]:
    pitch = PitchConditions(
        restitution=restitution,
        restitution_impact_slope_per_rad=slope,
        horizontal_retention=horizontal_retention,
        drag_coefficient=0.40,
        seam_sd_deg=0.0,
    )
    rows: list[dict[str, float]] = []
    for release in releases:
        trajectory = simulate_delivery(release, pitch, np.random.default_rng(RNG_SEED))
        if trajectory.bounce_point is None:
            return float("inf"), []
        crease = trajectory.state_at_popping_crease
        stumps = trajectory.state_at_stumps
        if crease is None or stumps is None:
            return float("inf"), []
        rows.append(
            {
                "length_m": float(trajectory.length_from_stumps),
                "height_crease_m": float(crease.pos[2]),
                "height_stumps_m": float(stumps.pos[2]),
                "speed_ratio": float(np.linalg.norm(crease.vel) * 3.6 / release.speed_kmh),
                "cross_pitch_time_s": float(trajectory.flight_time_to_stumps),
                "release_to_stumps_s": float(trajectory.release_to_stumps_time),
            }
        )

    good, short, bouncer = rows
    penalty = (
        outside_range(good["height_stumps_m"], 0.55, 0.80, 0.25) ** 2
        + outside_range(short["height_crease_m"], 1.00, 1.35, 0.35) ** 2
        + outside_range(bouncer["height_crease_m"], 1.50, 1.95, 0.45) ** 2
        + outside_range(good["speed_ratio"], 0.84, 0.90, 0.06) ** 2
        + outside_range(good["cross_pitch_time_s"], 0.47, 0.58, 0.11) ** 2
    )
    # Prefer the simple constants already selected when several candidates
    # satisfy all windows; this keeps the calibration stable across machines.
    penalty += 0.002 * (
        abs(restitution - 0.67) / 0.10
        + abs(slope - 0.55) / 0.40
        + abs(horizontal_retention - 0.96) / 0.08
    )
    return penalty, rows


def main() -> None:
    baseline = PitchConditions(drag_coefficient=0.40, seam_sd_deg=0.0)
    releases = tuple(release_for_length(length, baseline) for length in TARGET_LENGTHS_M)

    best: tuple[float, float, float, float, list[dict[str, float]]] | None = None
    for restitution in (0.63, 0.65, 0.67, 0.69, 0.71, 0.73):
        for slope in (0.35, 0.45, 0.55, 0.65, 0.75):
            for retention in (0.92, 0.94, 0.96, 0.98, 1.00):
                score, rows = inspect_candidate(restitution, slope, retention, releases)
                candidate = (score, restitution, slope, retention, rows)
                if best is None or candidate[0] < best[0]:
                    best = candidate

    assert best is not None
    score, restitution, slope, retention, rows = best
    print("Bounce calibration search (Cd fixed at 0.40; 140 km/h, zero swing/seam)")
    print(
        f"Selected restitution={restitution:.2f}, impact-angle slope={slope:.2f}/rad, "
        f"horizontal retention={retention:.2f} (score {score:.4f})"
    )
    print(
        f"{'Target':>8} {'Bounce m':>9} {'Category':>10} {'Height crease':>15} "
        f"{'Height stumps':>15} {'Crease speed':>14} {'Cross pitch':>12} {'From release':>13}"
    )
    names = ("good", "short", "bouncer")
    for target, name, release, row in zip(TARGET_LENGTHS_M, names, releases, rows):
        trajectory = simulate_delivery(
            release,
            PitchConditions(
                restitution=restitution,
                restitution_impact_slope_per_rad=slope,
                horizontal_retention=retention,
                drag_coefficient=0.40,
                seam_sd_deg=0.0,
            ),
            np.random.default_rng(RNG_SEED),
        )
        crease_speed = np.linalg.norm(trajectory.state_at_popping_crease.vel) * 3.6
        print(
            f"{target:>7.1f}m {row['length_m']:>8.2f} {trajectory.length_category:>10} "
            f"{row['height_crease_m']:>13.3f}m {row['height_stumps_m']:>13.3f}m "
            f"{crease_speed:>11.1f}km/h {row['cross_pitch_time_s']:>10.3f}s "
            f"{row['release_to_stumps_s']:>11.3f}s"
        )
    print("Target windows: good stumps height 0.55-0.80 m; short crease height")
    print("1.00-1.35 m; bouncer crease height 1.50-1.95 m; good crease speed")
    print("84-90% of release; cross-pitch time 0.47-0.58 s.")


if __name__ == "__main__":
    main()
