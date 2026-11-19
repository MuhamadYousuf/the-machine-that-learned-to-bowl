"""Six-legal-ball training environment for a single over."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any

import numpy as np

from .batter import (
    ARCHETYPES,
    BatterArchetype,
    BatterState,
    DeliveryFeatures,
    effective_length_m,
    get_archetype,
    sample_batter,
)
from .constants import SPEED_RANGE_KMH
from .outcome import BallOutcome, resolve
from .physics import PitchConditions, Release, Trajectory, release_from_action, simulate_delivery


LEGACY_OBSERVATION_SIZE = 17
OBSERVATION_SIZE = 18
ENVIRONMENT_VERSION = 2
BOUNCER_SHOULDER_HEIGHT_M = 1.25
ACTION_SIZE = 4
MAX_LEGAL_BALLS = 6
MAX_ILLEGAL_BALLS = 6
ARCHETYPE_NAMES = tuple(ARCHETYPES)


def _serialize_points(trajectory: Trajectory, every: int = 2) -> list[dict[str, float]]:
    indices = list(range(0, len(trajectory.t), every))
    if indices[-1] != len(trajectory.t) - 1:
        indices.append(len(trajectory.t) - 1)
    return [
        {
            "t_s": float(trajectory.t[index]),
            "x_m": float(trajectory.pos[index, 0]),
            "y_m": float(trajectory.pos[index, 1]),
            "z_m": float(trajectory.pos[index, 2]),
        }
        for index in indices
    ]


class OverEnv:
    """One over, with sampled batters, outcomes, and re-bowled extras."""

    def __init__(
        self,
        *,
        noise_scale: float = 0.05,
        illegal_ball_cap: int = MAX_ILLEGAL_BALLS,
        batter_spread: float = 0.15,
        pitch_conditions: PitchConditions | None = None,
        environment_version: int = ENVIRONMENT_VERSION,
    ) -> None:
        if noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative")
        if illegal_ball_cap < 0:
            raise ValueError("illegal_ball_cap must be non-negative")
        if environment_version not in (1, 2):
            raise ValueError("environment_version must be 1 or 2")
        self.noise_scale = float(noise_scale)
        self.illegal_ball_cap = int(illegal_ball_cap)
        self.batter_spread = float(batter_spread)
        self.pitch_conditions = pitch_conditions or PitchConditions()
        self.environment_version = int(environment_version)
        self.rng: np.random.Generator | None = None
        self.base_seed: int | None = None
        self.initial_archetype: BatterArchetype | None = None
        self.current_archetype: BatterArchetype | None = None
        self.state: BatterState | None = None
        self.history: deque[DeliveryFeatures] = deque(maxlen=2)
        self.legal_balls = 0
        self.illegal_balls = 0
        self.runs_conceded = 0
        self.wickets = 0
        self.extras = 0
        self.bouncers_this_over = 0
        self.bouncer_no_balls = 0
        self.done = False

    def reset(
        self, archetype: BatterArchetype | str, rng: np.random.Generator
    ) -> np.ndarray:
        """Start a fresh over against a sampled batter of the chosen type."""
        self.rng = rng
        self.base_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
        self.initial_archetype = get_archetype(archetype)
        self.current_archetype = sample_batter(
            self.initial_archetype, rng, spread=self.batter_spread
        )
        self.state = BatterState.for_archetype(self.current_archetype)
        self.history.clear()
        self.legal_balls = 0
        self.illegal_balls = 0
        self.runs_conceded = 0
        self.wickets = 0
        self.extras = 0
        self.bouncers_this_over = 0
        self.bouncer_no_balls = 0
        self.done = False
        return self._observation()

    def _stream_rng(self, delivery_index: int, stream_id: int) -> np.random.Generator:
        if self.base_seed is None:
            raise RuntimeError("environment has not been reset")
        seed = np.random.SeedSequence([self.base_seed, delivery_index, stream_id])
        return np.random.default_rng(seed)

    def _require_active(self) -> tuple[np.random.Generator, BatterArchetype, BatterState]:
        if self.rng is None or self.current_archetype is None or self.state is None:
            raise RuntimeError("call reset(archetype, rng) before step(action)")
        if self.done:
            raise RuntimeError("the over is complete; reset before taking another step")
        return self.rng, self.current_archetype, self.state

    @staticmethod
    def _normalize_history_feature(features: DeliveryFeatures) -> tuple[float, ...]:
        length = effective_length_m(features) / 14.0
        line = float(np.clip(features.line_m / 1.0, -1.0, 1.0))
        late = float(np.clip(features.late_movement_m / 0.5, -1.0, 1.0))
        low, high = SPEED_RANGE_KMH
        speed = float(
            np.clip(2.0 * (features.speed_at_crease_kmh / 3.6 - low / 3.6) / ((high - low) / 3.6) - 1.0, -1.0, 1.0)
        )
        return (float(np.clip(length, 0.0, 1.0)), line, late, speed, 1.0)

    def _observation(self) -> np.ndarray:
        if self.current_archetype is None or self.state is None:
            raise RuntimeError("environment has not been reset")
        one_hot = [float(self.current_archetype.name == name) for name in ARCHETYPE_NAMES]
        vector = [
            *one_hot,
            self.state.confidence,
            float(np.clip(self.state.balls_faced / 12.0, 0.0, 1.0)),
            float(np.clip(self.legal_balls / 6.0, 0.0, 1.0)),
            float(np.clip(self.runs_conceded / 12.0, 0.0, 1.0)),
        ]
        records = list(self.history)
        if len(records) < 2:
            records = [None] * (2 - len(records)) + records
        for features in records:
            vector.extend(
                (0.0, 0.0, 0.0, 0.0, 0.0)
                if features is None
                else self._normalize_history_feature(features)
            )
        if self.environment_version >= 2:
            vector.append(float(np.clip(self.bouncers_this_over / 2.0, 0.0, 1.0)))
        observation = np.asarray(vector, dtype=np.float32)
        expected_size = OBSERVATION_SIZE if self.environment_version >= 2 else LEGACY_OBSERVATION_SIZE
        if observation.shape != (expected_size,):
            raise RuntimeError(f"internal observation shape error: {observation.shape}")
        return observation

    @staticmethod
    def _key_points(trajectory: Trajectory) -> dict[str, Any]:
        def state_record(state):
            if state is None:
                return None
            return {
                "t_s": float(state.t),
                "position_m": [float(value) for value in state.pos],
                "velocity_m_s": [float(value) for value in state.vel],
            }

        return {
            "bounce_m": None
            if trajectory.bounce_point is None
            else [float(value) for value in trajectory.bounce_point],
            "popping_crease": state_record(trajectory.state_at_popping_crease),
            "stumps": state_record(trajectory.state_at_stumps),
        }

    @staticmethod
    def _event_times(trajectory: Trajectory) -> dict[str, Any]:
        bounce_time = None
        if trajectory.bounce_point is not None:
            distances = np.linalg.norm(trajectory.pos - trajectory.bounce_point, axis=1)
            bounce_time = float(trajectory.t[int(np.argmin(distances))])
        crease_time = (
            None if trajectory.state_at_popping_crease is None
            else float(trajectory.state_at_popping_crease.t)
        )
        stumps_time = (
            None if trajectory.state_at_stumps is None
            else float(trajectory.state_at_stumps.t)
        )
        return {
            "release_s": float(trajectory.t[0]),
            "bounce_s": bounce_time,
            "crease_s": crease_time,
            "stumps_s": stumps_time,
            # The batting model resolves the shot at the popping crease; this
            # is a timing proxy, not a simulated bat-ball collision.
            "bat_contact_s": crease_time,
            "bat_contact_is_crease_proxy": True,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Deliver one ball and return observation, reward, done, and record."""
        _, batter, state = self._require_active()
        delivery_index = self.legal_balls + self.illegal_balls
        noise_rng = self._stream_rng(delivery_index, 0)
        physics_rng = self._stream_rng(delivery_index, 1)
        outcome_rng = self._stream_rng(delivery_index, 2)
        batter_rng = self._stream_rng(delivery_index, 3)
        command = np.asarray(action, dtype=float)
        if command.shape != (ACTION_SIZE,):
            raise ValueError(f"action must have shape ({ACTION_SIZE},)")
        if not np.all(np.isfinite(command)):
            raise ValueError("action values must be finite")
        command = np.clip(command, -1.0, 1.0)
        if self.noise_scale:
            executed_action = np.clip(
                command + noise_rng.normal(0.0, self.noise_scale, size=ACTION_SIZE),
                -1.0,
                1.0,
            )
        else:
            executed_action = command.copy()
        release = release_from_action(executed_action, noise_rng, noise_scale=0.0)
        trajectory = simulate_delivery(release, self.pitch_conditions, physics_rng)
        features = DeliveryFeatures.from_trajectory(trajectory)
        is_counted_bouncer = (
            self.environment_version >= 2
            and features.length_category == "bouncer"
            and features.height_at_crease_m >= BOUNCER_SHOULDER_HEIGHT_M
        )
        if is_counted_bouncer:
            self.bouncers_this_over += 1
        bouncer_no_ball = is_counted_bouncer and self.bouncers_this_over > 2
        if bouncer_no_ball:
            self.bouncer_no_balls += 1
        outcome = (
            resolve(features, batter, state, outcome_rng, force_no_ball=True)
            if bouncer_no_ball
            else resolve(features, batter, state, outcome_rng)
        )

        extra = not outcome.is_legal_ball
        reward = 25.0 * outcome.wicket_prob - outcome.expected_runs - (3.0 if extra else 0.0)
        if outcome.is_legal_ball:
            self.legal_balls += 1
        else:
            self.illegal_balls += 1
            self.extras += 1
        self.runs_conceded += outcome.runs_conceded
        if outcome.is_wicket:
            self.wickets += 1
            self.history.clear()
            next_base = ARCHETYPES[
                ARCHETYPE_NAMES[int(batter_rng.integers(0, len(ARCHETYPE_NAMES)))]
            ]
            self.current_archetype = sample_batter(
                next_base, batter_rng, spread=self.batter_spread
            )
            self.state = BatterState.for_archetype(self.current_archetype)
        else:
            self.history.append(features)

        cap_penalty = self.illegal_balls >= self.illegal_ball_cap and self.legal_balls < MAX_LEGAL_BALLS
        if cap_penalty:
            reward -= 10.0
        self.done = self.legal_balls >= MAX_LEGAL_BALLS or cap_penalty

        info: dict[str, Any] = {
            "action": [float(value) for value in executed_action],
            "requested_action": [float(value) for value in command],
            "batter_archetype": batter.name,
            "legal_ball_number": self.legal_balls,
            "illegal_balls": self.illegal_balls,
            "bouncers_this_over": self.bouncers_this_over,
            "bouncer_no_ball": bool(bouncer_no_ball),
            "environment_version": self.environment_version,
            "extra_penalty_applied": bool(cap_penalty),
            "release": release,
            "trajectory_points": _serialize_points(trajectory),
            "key_points": self._key_points(trajectory),
            "event_times": self._event_times(trajectory),
            "features": features,
            "outcome": outcome,
            "sampled_runs": outcome.runs_conceded,
            "sampled_wicket": outcome.is_wicket,
            "sampled_extra": extra,
            "reward": float(reward),
            "expected_runs": outcome.expected_runs,
            "wicket_prob": outcome.wicket_prob,
        }
        return self._observation(), float(reward), self.done, info

    def summary(self) -> dict[str, float | int]:
        """Sampled statistics for the current over."""
        return {
            "legal_balls": self.legal_balls,
            "illegal_balls": self.illegal_balls,
            "runs_conceded": self.runs_conceded,
            "wickets": self.wickets,
            "extras": self.extras,
            "bouncers_this_over": self.bouncers_this_over,
            "bouncer_no_balls": self.bouncer_no_balls,
        }

