"""Ball-flight physics for the AI Bowler project.

Coordinates use x along the pitch from the bowler's stumps (x=0) to the
batter's stumps (x=20.12 m), y laterally with positive toward the off side of
a right-handed batter, and z vertically above the ground. The right-arm,
over-the-wicket release is at (-1.22, +0.30, 2.20) m. Action azimuth is an
offset from the line aimed at the middle stumps; therefore zero azimuth aims
straight at the stumps despite the lateral release offset.

For a right-handed batter, swing=-1 is outswing toward positive y (off side)
and swing=+1 is inswing toward negative y (leg side). The first-bounce
category is measured from the batter's stumps: yorker <=2 m, full >2 through
6 m, good >6 through 8 m, short >8 through 10.5 m, and bouncer >10.5 m. A ball
with no bounce in front of the stumps is a full toss.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin, sqrt, tan

import numpy as np

from .constants import (
    AIR_DENSITY_KG_M3,
    BALL_CROSS_SECTION_M2,
    BALL_MASS_KG,
    BALL_RADIUS_M,
    BATTER_POPPING_CREASE_X_M,
    BATTER_STUMPS_X_M,
    DEFAULT_HORIZONTAL_RETENTION,
    DEFAULT_RESTITUTION,
    DEFAULT_RESTITUTION_IMPACT_SLOPE_PER_RAD,
    DEFAULT_RESTITUTION_REFERENCE_ANGLE_RAD,
    DRAG_COEFFICIENT,
    ELEVATION_RANGE_DEG,
    GRAVITY_M_S2,
    INTEGRATION_DT_S,
    LEG_SIDE_WIDE_GUIDELINE_M,
    RELEASE_POSITION_M,
    SPEED_RANGE_KMH,
    STUMPS_HEIGHT_M,
    STUMPS_WIDTH_M,
    SWING_ACCEL_COEFFICIENT_PER_M,
    WIDE_GUIDELINE_M,
    AZIMUTH_OFFSET_RANGE_DEG,
)


@dataclass(frozen=True)
class Release:
    """Initial ball state; speed is km/h, angles are degrees, position is m."""

    speed_kmh: float
    elevation_deg: float
    azimuth_deg: float
    swing: float
    position: tuple[float, float, float] = RELEASE_POSITION_M


@dataclass(frozen=True)
class PitchConditions:
    """Tunable pitch and integration settings for a delivery."""

    restitution: float = DEFAULT_RESTITUTION
    restitution_impact_slope_per_rad: float = DEFAULT_RESTITUTION_IMPACT_SLOPE_PER_RAD
    restitution_reference_angle_rad: float = DEFAULT_RESTITUTION_REFERENCE_ANGLE_RAD
    horizontal_retention: float = DEFAULT_HORIZONTAL_RETENTION
    seam_sd_deg: float = 0.25
    dt_s: float = INTEGRATION_DT_S
    air_density_kg_m3: float = AIR_DENSITY_KG_M3
    drag_coefficient: float = DRAG_COEFFICIENT
    gravity_m_s2: float = GRAVITY_M_S2
    swing_accel_coefficient_per_m: float = SWING_ACCEL_COEFFICIENT_PER_M


@dataclass(frozen=True)
class FlightState:
    """Interpolated ball state at a named pitch plane."""

    t: float
    pos: np.ndarray
    vel: np.ndarray


@dataclass(frozen=True)
class Trajectory:
    """Sampled trajectory and delivery outcomes."""

    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    bounce_point: np.ndarray | None
    state_at_popping_crease: FlightState | None
    state_at_stumps: FlightState | None
    hits_stumps: bool
    is_wide: bool
    is_beamer: bool
    length_from_stumps: float | None
    length_category: str

    @property
    def flight_time_to_stumps(self) -> float | None:
        """Time between the bowler's and batter's stump planes.

        The sampled trajectory clock starts at release behind the bowler's
        stumps. This pitch-length duration excludes the short run-up from the
        release point to x=0; use release_to_stumps_time for elapsed time from
        release to the batter's stumps.
        """
        if self.state_at_stumps is None:
            return None
        x_values = self.pos[:, 0]
        crossing_index = int(np.searchsorted(x_values, 0.0, side="left"))
        if crossing_index == 0 or crossing_index >= len(self.t):
            return None
        x0 = float(x_values[crossing_index - 1])
        x1 = float(x_values[crossing_index])
        fraction = (0.0 - x0) / (x1 - x0)
        time_at_bowler_stumps = float(
            self.t[crossing_index - 1]
            + fraction * (self.t[crossing_index] - self.t[crossing_index - 1])
        )
        return self.state_at_stumps.t - time_at_bowler_stumps

    @property
    def release_to_stumps_time(self) -> float | None:
        """Elapsed time from release to the batter's stumps plane."""
        if self.state_at_stumps is None:
            return None
        return self.state_at_stumps.t


def _map_unit(value: float, bounds: tuple[float, float]) -> float:
    return bounds[0] + (value + 1.0) * 0.5 * (bounds[1] - bounds[0])


def release_from_action(
    action: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float = 0.0,
) -> Release:
    """Map four normalized controls to a physical release.

    The action order is speed, elevation, azimuth offset, and swing. Each
    component is clipped to [-1, 1]. Execution noise is Gaussian in normalized
    action space and is applied before clipping, making repeated calls with
    the same Generator seed deterministic.
    """
    command = np.asarray(action, dtype=float)
    if command.shape != (4,):
        raise ValueError(f"action must have shape (4,), received {command.shape}")
    if not np.all(np.isfinite(command)):
        raise ValueError("action values must be finite")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")
    command = np.clip(command, -1.0, 1.0)
    if noise_scale:
        command = np.clip(command + rng.normal(0.0, noise_scale, size=4), -1.0, 1.0)

    x0, y0, _ = RELEASE_POSITION_M
    straight_to_middle_deg = degrees(
        np.arctan2(-y0, BATTER_STUMPS_X_M - x0)
    )
    azimuth_offset = _map_unit(command[2], AZIMUTH_OFFSET_RANGE_DEG)
    return Release(
        speed_kmh=_map_unit(command[0], SPEED_RANGE_KMH),
        elevation_deg=_map_unit(command[1], ELEVATION_RANGE_DEG),
        azimuth_deg=straight_to_middle_deg + azimuth_offset,
        swing=float(command[3]),
    )


def _length_category(length_m: float | None) -> str:
    if length_m is None or length_m < 0.0:
        return "full_toss"
    if length_m <= 2.0:
        return "yorker"
    if length_m <= 6.0:
        return "full"
    if length_m <= 8.0:
        return "good"
    if length_m <= 10.5:
        return "short"
    return "bouncer"


def _state_at_x(
    plane_x: float,
    t0: float,
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    t1: float,
    p1: tuple[float, float, float],
    v1: tuple[float, float, float],
) -> FlightState | None:
    if p0[0] <= plane_x <= p1[0] and p1[0] > p0[0]:
        fraction = (plane_x - p0[0]) / (p1[0] - p0[0])
        pos = np.array(
            [
                plane_x,
                p0[1] + fraction * (p1[1] - p0[1]),
                p0[2] + fraction * (p1[2] - p0[2]),
            ],
            dtype=float,
        )
        vel = np.array(
            [
                v0[0] + fraction * (v1[0] - v0[0]),
                v0[1] + fraction * (v1[1] - v0[1]),
                v0[2] + fraction * (v1[2] - v0[2]),
            ],
            dtype=float,
        )
        return FlightState(t0 + fraction * (t1 - t0), pos, vel)
    return None


def simulate_delivery(
    release: Release,
    pitch: PitchConditions,
    rng: np.random.Generator,
) -> Trajectory:
    """Integrate one delivery with semi-implicit Euler at dt <= 2 ms.

    Gravity and quadratic aerodynamic drag act throughout flight. Swing is a
    lateral acceleration proportional to speed squared and is switched off at
    the first bounce. The first bounce applies vertical restitution and
    horizontal speed retention, then a small random seam angle. Later ground
    contact ends the simulation; rolling and additional hops are omitted.
    """
    if not 0.0 < pitch.dt_s <= 0.002:
        raise ValueError("pitch.dt_s must be in (0, 0.002] seconds")
    if not 0.0 <= pitch.restitution <= 1.0:
        raise ValueError("pitch.restitution must be between 0 and 1")
    if pitch.restitution_impact_slope_per_rad < 0.0:
        raise ValueError("pitch.restitution_impact_slope_per_rad must be non-negative")
    if not 0.0 < pitch.horizontal_retention <= 1.0:
        raise ValueError("pitch.horizontal_retention must be in (0, 1]")

    px, py, pz = (float(v) for v in release.position)
    if pz < BALL_RADIUS_M:
        raise ValueError("release height must be at least the ball radius")
    speed = release.speed_kmh / 3.6
    elevation = radians(release.elevation_deg)
    azimuth = radians(release.azimuth_deg)
    vx = speed * cos(elevation) * cos(azimuth)
    vy = speed * cos(elevation) * sin(azimuth)
    vz = speed * sin(elevation)

    t = 0.0
    drag_k = (
        0.5
        * pitch.air_density_kg_m3
        * pitch.drag_coefficient
        * BALL_CROSS_SECTION_M2
        / BALL_MASS_KG
    )
    dt = pitch.dt_s
    x_stop = BATTER_STUMPS_X_M + 1.5
    times = [t]
    positions = [(px, py, pz)]
    velocities = [(vx, vy, vz)]
    bounce_point: np.ndarray | None = None
    crease_state: FlightState | None = None
    stumps_state: FlightState | None = None

    def save_segment(
        t0: float,
        p0: tuple[float, float, float],
        v0: tuple[float, float, float],
        t1: float,
        p1: tuple[float, float, float],
        v1: tuple[float, float, float],
    ) -> None:
        nonlocal crease_state, stumps_state
        if crease_state is None:
            crease_state = _state_at_x(
                BATTER_POPPING_CREASE_X_M, t0, p0, v0, t1, p1, v1
            )
        if stumps_state is None:
            stumps_state = _state_at_x(
                BATTER_STUMPS_X_M, t0, p0, v0, t1, p1, v1
            )

    for _ in range(2500):
        old_t = t
        old_p = (px, py, pz)
        old_v = (vx, vy, vz)
        current_speed = sqrt(vx * vx + vy * vy + vz * vz)
        if current_speed < 0.2:
            break
        if old_t >= 3.0 or px >= x_stop:
            break

        h = min(dt, 3.0 - t)
        ax = -drag_k * current_speed * vx
        ay = -drag_k * current_speed * vy
        if bounce_point is None:
            ay -= (
                release.swing
                * pitch.swing_accel_coefficient_per_m
                * current_speed
                * current_speed
            )
        az = -pitch.gravity_m_s2 - drag_k * current_speed * vz
        next_v = (vx + ax * h, vy + ay * h, vz + az * h)
        next_p = (
            px + next_v[0] * h,
            py + next_v[1] * h,
            pz + next_v[2] * h,
        )
        next_t = t + h

        ground_fraction: float | None = None
        if pz > BALL_RADIUS_M and next_p[2] <= BALL_RADIUS_M and next_v[2] < 0.0:
            ground_fraction = (pz - BALL_RADIUS_M) / (pz - next_p[2])

        if ground_fraction is None:
            save_segment(old_t, old_p, old_v, next_t, next_p, next_v)
            px, py, pz = next_p
            vx, vy, vz = next_v
            t = next_t
            times.append(t)
            positions.append((px, py, pz))
            velocities.append((vx, vy, vz))
            continue

        fraction = max(0.0, min(1.0, ground_fraction))
        event_t = old_t + fraction * h
        event_p = (
            old_p[0] + fraction * (next_p[0] - old_p[0]),
            old_p[1] + fraction * (next_p[1] - old_p[1]),
            BALL_RADIUS_M,
        )
        event_v = (
            old_v[0] + fraction * (next_v[0] - old_v[0]),
            old_v[1] + fraction * (next_v[1] - old_v[1]),
            old_v[2] + fraction * (next_v[2] - old_v[2]),
        )
        save_segment(old_t, old_p, old_v, event_t, event_p, event_v)

        # A bounce behind the stumps is a full toss for batter-side outcomes.
        first_bounce_in_play = bounce_point is None and event_p[0] <= BATTER_STUMPS_X_M
        if not first_bounce_in_play:
            px, py, pz = event_p
            vx, vy, vz = event_v
            t = event_t
            times.append(t)
            positions.append((px, py, pz))
            velocities.append((vx, vy, vz))
            break

        bounce_point = np.array(event_p, dtype=float)
        vx = event_v[0] * pitch.horizontal_retention
        vy = event_v[1] * pitch.horizontal_retention
        if pitch.seam_sd_deg:
            seam_angle = radians(float(rng.normal(0.0, pitch.seam_sd_deg)))
            vy += vx * tan(seam_angle)
        impact_angle = atan2(abs(event_v[2]), hypot(event_v[0], event_v[1]))
        effective_restitution = pitch.restitution + pitch.restitution_impact_slope_per_rad * (
            impact_angle - pitch.restitution_reference_angle_rad
        )
        effective_restitution = max(0.0, min(0.95, effective_restitution))
        vz = abs(event_v[2]) * effective_restitution
        px, py, pz = event_p
        t = event_t
        times.append(t)
        positions.append((px, py, pz))
        velocities.append((vx, vy, vz))

        # Integrate the unused part of this step after the velocity change.
        remainder = h * (1.0 - fraction)
        if remainder > 1e-12:
            old_t = t
            old_p = (px, py, pz)
            old_v = (vx, vy, vz)
            current_speed = sqrt(vx * vx + vy * vy + vz * vz)
            ax = -drag_k * current_speed * vx
            ay = -drag_k * current_speed * vy
            az = -pitch.gravity_m_s2 - drag_k * current_speed * vz
            next_v = (vx + ax * remainder, vy + ay * remainder, vz + az * remainder)
            next_p = (
                px + next_v[0] * remainder,
                py + next_v[1] * remainder,
                pz + next_v[2] * remainder,
            )
            next_t = t + remainder
            save_segment(old_t, old_p, old_v, next_t, next_p, next_v)
            px, py, pz = next_p
            vx, vy, vz = next_v
            t = next_t
            times.append(t)
            positions.append((px, py, pz))
            velocities.append((vx, vy, vz))

        if px >= x_stop or t >= 3.0:
            break

    if crease_state is None and len(positions) > 1:
        for index in range(1, len(positions)):
            crease_state = _state_at_x(
                BATTER_POPPING_CREASE_X_M,
                times[index - 1],
                positions[index - 1],
                velocities[index - 1],
                times[index],
                positions[index],
                velocities[index],
            )
            if crease_state is not None:
                break
    if stumps_state is None and len(positions) > 1:
        for index in range(1, len(positions)):
            stumps_state = _state_at_x(
                BATTER_STUMPS_X_M,
                times[index - 1],
                positions[index - 1],
                velocities[index - 1],
                times[index],
                positions[index],
                velocities[index],
            )
            if stumps_state is not None:
                break

    length_from_stumps = (
        BATTER_STUMPS_X_M - float(bounce_point[0])
        if bounce_point is not None
        else None
    )
    length_category = _length_category(length_from_stumps)
    is_wide = crease_state is not None and (
        float(crease_state.pos[1]) > WIDE_GUIDELINE_M
        or float(crease_state.pos[1]) < -LEG_SIDE_WIDE_GUIDELINE_M
    )
    is_beamer = (
        bounce_point is None
        and crease_state is not None
        and float(crease_state.pos[2]) > 1.0
    )
    hits_stumps = False
    if stumps_state is not None:
        y_at_stumps = float(stumps_state.pos[1])
        z_at_stumps = float(stumps_state.pos[2])
        hits_stumps = (
            abs(y_at_stumps) <= STUMPS_WIDTH_M / 2.0 + BALL_RADIUS_M
            and BALL_RADIUS_M - 1e-9 <= z_at_stumps <= STUMPS_HEIGHT_M + BALL_RADIUS_M
        )

    return Trajectory(
        t=np.asarray(times, dtype=float),
        pos=np.asarray(positions, dtype=float),
        vel=np.asarray(velocities, dtype=float),
        bounce_point=bounce_point,
        state_at_popping_crease=crease_state,
        state_at_stumps=stumps_state,
        hits_stumps=hits_stumps,
        is_wide=is_wide,
        is_beamer=is_beamer,
        length_from_stumps=length_from_stumps,
        length_category=length_category,
    )
