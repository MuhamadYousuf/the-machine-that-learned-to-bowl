"""One over against a batter population, bowled within a bowler card's limits, scored by a reward spec.

Same physics and batting model as :class:`aibowler.env.OverEnv`; differences:

* the batter comes from a :class:`BatterPopulation` (and after a wicket, the *same* player's
  population again: the analyst cares about one batter, not a random lineup);
* the policy's speed command is mapped into the bowler's own pace band, swing is scaled by
  the bowler's swing ability, and execution noise is the bowler's accuracy;
* rewards and the bouncer limit come from a :class:`RewardSpec`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..batter import BatterState, DeliveryFeatures
from ..constants import SPEED_RANGE_KMH
from ..env import ACTION_SIZE, BOUNCER_SHOULDER_HEIGHT_M, MAX_LEGAL_BALLS, OverEnv
from ..outcome import resolve
from ..physics import release_from_action, simulate_delivery
from .cards import BatterPopulation, BowlerCard, RewardSpec


def map_command(command: np.ndarray, bowler: BowlerCard) -> np.ndarray:
    """Policy output in [-1, 1]^4 → the simulator's action, inside this bowler's abilities."""
    lo, hi = SPEED_RANGE_KMH
    b_lo, b_hi = bowler.speed_band()
    kmh = b_lo + (float(command[0]) + 1.0) / 2.0 * (b_hi - b_lo)
    mapped = np.asarray(command, dtype=float).copy()
    mapped[0] = 2.0 * (kmh - lo) / (hi - lo) - 1.0
    mapped[3] = float(command[3]) * bowler.swing_scale()
    return mapped


class LabOverEnv(OverEnv):
    def __init__(self, population: BatterPopulation, bowler: BowlerCard, reward: RewardSpec, **kwargs):
        super().__init__(noise_scale=bowler.noise_scale(), batter_spread=0.0, **kwargs)
        self.population = population
        self.bowler = bowler
        self.weights = reward.resolved()

    def reset(self, archetype=None, rng: np.random.Generator | None = None) -> np.ndarray:  # type: ignore[override]
        if rng is None:
            raise ValueError("reset needs an rng")
        self.rng = rng
        self.base_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
        self.initial_archetype = self.current_archetype = self.population.sample(rng)
        self.state = BatterState.for_archetype(self.current_archetype)
        self.history.clear()
        self.legal_balls = self.illegal_balls = self.runs_conceded = self.wickets = self.extras = 0
        self.bouncers_this_over = self.bouncer_no_balls = 0
        self.done = False
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        _, batter, state = self._require_active()
        index = self.legal_balls + self.illegal_balls
        noise_rng, physics_rng, outcome_rng, batter_rng = (self._stream_rng(index, k) for k in range(4))
        command = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if command.shape != (ACTION_SIZE,):
            raise ValueError(f"action must have shape ({ACTION_SIZE},)")
        executed = np.clip(command + noise_rng.normal(0.0, self.noise_scale, size=ACTION_SIZE), -1.0, 1.0)
        release = release_from_action(map_command(executed, self.bowler), noise_rng, noise_scale=0.0)
        trajectory = simulate_delivery(release, self.pitch_conditions, physics_rng)
        features = DeliveryFeatures.from_trajectory(trajectory)
        counted_bouncer = features.length_category == "bouncer" and features.height_at_crease_m >= BOUNCER_SHOULDER_HEIGHT_M
        if counted_bouncer:
            self.bouncers_this_over += 1
        limit = self.weights["max_bouncers"] if self.bowler.bouncer else 0
        illegal_bouncer = counted_bouncer and self.bouncers_this_over > limit
        self.bouncer_no_balls += int(illegal_bouncer)
        outcome = resolve(features, batter, state, outcome_rng, force_no_ball=illegal_bouncer)

        extra = not outcome.is_legal_ball
        w = self.weights
        reward = w["wicket"] * outcome.wicket_prob - w["run"] * outcome.expected_runs - (w["extra"] if extra else 0.0)
        if outcome.is_legal_ball:
            self.legal_balls += 1
        else:
            self.illegal_balls += 1
            self.extras += 1
        self.runs_conceded += outcome.runs_conceded
        if outcome.is_wicket:
            self.wickets += 1
            self.history.clear()
            self.current_archetype = self.population.sample(batter_rng)
            self.state = BatterState.for_archetype(self.current_archetype)
        else:
            self.history.append(features)
        cap = self.illegal_balls >= self.illegal_ball_cap and self.legal_balls < MAX_LEGAL_BALLS
        if cap:
            reward -= 10.0
        self.done = self.legal_balls >= MAX_LEGAL_BALLS or cap
        info = {
            "action": executed.tolist(),
            "release": release,
            "features": features,
            "outcome": outcome,
            "batter_confidence": float(state.confidence),
            "legal_ball_number": self.legal_balls,
            "bouncer_no_ball": bool(illegal_bouncer),
            "reward": float(reward),
            "trajectory": trajectory,
        }
        return self._observation(), float(reward), self.done, info


def bowl_overs(policy, population: BatterPopulation, bowler: BowlerCard, reward: RewardSpec,
               seeds: list[int], record: bool = False) -> dict:
    """Bowl one over per seed with `policy(observation) -> action`. Returns totals and (optionally) every ball."""
    env = LabOverEnv(population, bowler, reward)
    total, wickets, runs, extras, bouncers, balls = 0.0, 0, 0, 0, 0, []
    actions = []
    for seed in seeds:
        obs = env.reset(rng=np.random.default_rng(seed))
        over_ball = 0
        while True:
            act = np.asarray(policy(obs), dtype=float)
            actions.append(act)
            obs, r, done, info = env.step(act)
            total += r
            if record:
                f, o = info["features"], info["outcome"]
                balls.append({
                    "seed": seed, "ball": over_ball, "legal": o.is_legal_ball,
                    "length_m": None if f.length_m is None else float(f.length_m),
                    "line_m": float(f.line_m), "height_m": float(f.height_at_crease_m),
                    "category": f.length_category, "speed_kmh": float(info["release"].speed_kmh),
                    "swing": float(info["release"].swing), "result": o.result, "runs": int(o.runs_conceded),
                    "wicket": bool(o.is_wicket), "shot": o.shot, "wicket_prob": float(o.wicket_prob),
                    "expected_runs": float(o.expected_runs), "confidence": info["batter_confidence"],
                })
            over_ball += 1
            if done:
                break
        s = env.summary()
        wickets += s["wickets"]
        runs += s["runs_conceded"]
        extras += s["extras"]
        bouncers += s["bouncers_this_over"]
    n = max(1, len(seeds))
    acts = np.asarray(actions) if actions else np.zeros((0, 4))
    return {
        "reward_per_over": total / n,
        "wickets_per_over": wickets / n,
        "economy": runs / n,
        "extras_per_over": extras / n,
        "bouncers_per_over": bouncers / n,
        "saturation_share": float(np.mean(np.abs(acts) > 0.95)) if len(acts) else 0.0,
        "balls": balls,
    }
