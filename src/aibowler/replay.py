"""Compressed replay generation and loading for later visualization stages."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .env import ARCHETYPE_NAMES, OverEnv
from .policy import MLPPolicy


REPLAY_SCHEMA_VERSION = 2


def _dump_gzip(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as file:
        json.dump(payload, file, separators=(",", ":"), allow_nan=False)


def load_replay(path: str | Path) -> dict[str, Any]:
    """Load current gzip JSON replays and legacy plain JSON replay files."""
    replay_path = Path(path)
    opener = gzip.open if replay_path.suffix.lower() == ".gz" else open
    with opener(replay_path, "rt", encoding="utf-8") as file:
        payload = json.load(file)
    if int(payload.get("schema_version", 1)) not in {1, REPLAY_SCHEMA_VERSION}:
        raise ValueError(f"unsupported replay schema version: {payload.get('schema_version')}")
    return payload


def write_replay_bundle(
    run_dir: str | Path,
    *,
    run_name: str,
    label: str,
    genome: np.ndarray,
    generation: int | None,
    fitness: float | None,
    noise_scale: float,
    seed: int,
    overs: int,
    combine: bool = False,
) -> list[Path]:
    """Write sampled, fine-step replay JSON for each batter archetype."""
    replay_dir = Path(run_dir) / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    policy = MLPPolicy(genome)
    documents = {}
    for archetype_index, archetype_name in enumerate(ARCHETYPE_NAMES):
        overs_data = []
        total_wickets = total_runs = total_extras = 0
        speed_total = swing_total = 0.0
        length_counts: dict[str, int] = {}
        delivery_count = 0
        for over_index in range(overs):
            over_seed = seed + archetype_index * 10_000_000 + over_index * 101
            rng = np.random.default_rng(over_seed)
            env = OverEnv(noise_scale=noise_scale)
            observation = env.reset(archetype_name, rng)
            deliveries = []
            reward_total = 0.0
            while True:
                action = policy.act(observation)
                observation, reward, done, info = env.step(action)
                release = info["release"]
                features = info["features"]
                outcome = info["outcome"]
                reward_total += reward
                delivery_count += 1
                speed_total += release.speed_kmh
                swing_total += release.swing
                length_counts[features.length_category] = length_counts.get(features.length_category, 0) + 1
                deliveries.append(
                    {
                        "action": info["action"],
                        "requested_action": info["requested_action"],
                        "batter_archetype": info["batter_archetype"],
                        "release": asdict(release),
                        "event_times_s": info["event_times"],
                        "trajectory_points": info["trajectory_points"],
                        "key_points": info["key_points"],
                        "features": asdict(features),
                        "outcome": asdict(outcome),
                        "reward": float(reward),
                    }
                )
                if done:
                    break
            summary = env.summary()
            total_wickets += int(summary["wickets"])
            total_runs += int(summary["runs_conceded"])
            total_extras += int(summary["extras"])
            overs_data.append(
                {
                    "over_index": over_index,
                    "seed": over_seed,
                    "summary": summary,
                    "reward": reward_total,
                    "deliveries": deliveries,
                }
            )

        documents[archetype_name] = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "run_name": run_name,
            "label": label,
            "generation": generation,
            "checkpoint_fitness": fitness,
            "initial_archetype": archetype_name,
            "noise_scale": noise_scale,
            "action_order": ["speed", "elevation", "azimuth", "swing"],
            "trajectory_point_interval_ms": 4,
            "units": {"position": "m", "time": "s", "speed": "km/h", "angles": "degrees"},
            "events": ["release", "bounce", "crease", "stumps", "bat_contact_proxy"],
            "summary": {
                "overs": overs,
                "wickets_per_over": total_wickets / overs,
                "runs_per_over": total_runs / overs,
                "extras_per_over": total_extras / overs,
                "mean_release_speed_kmh": speed_total / max(1, delivery_count),
                "mean_swing": swing_total / max(1, delivery_count),
                "length_distribution": {
                    category: count / max(1, delivery_count)
                    for category, count in length_counts.items()
                },
            },
            "overs_data": overs_data,
        }

    output_paths = []
    if combine:
        output_path = replay_dir / f"{label}.json.gz"
        _dump_gzip(output_path, {"schema_version": REPLAY_SCHEMA_VERSION, "archetypes": documents})
        output_paths.append(output_path)
    else:
        for archetype_name, document in documents.items():
            output_path = replay_dir / f"{label}_{archetype_name}.json.gz"
            _dump_gzip(output_path, document)
            output_paths.append(output_path)
    return output_paths


def write_swarm_replay(
    run_dir: str | Path,
    *,
    run_name: str,
    generation: int,
    population: np.ndarray,
    noise_scale: float,
    seed: int,
) -> Path:
    """Replay the first ball from every genome against common sampled types."""
    deliveries = []
    for archetype_index, archetype_name in enumerate(ARCHETYPE_NAMES):
        scenario_seed = seed + archetype_index * 1_000_000
        for genome_index, genome in enumerate(population):
            rng = np.random.default_rng(scenario_seed)
            env = OverEnv(noise_scale=noise_scale)
            observation = env.reset(archetype_name, rng)
            policy = MLPPolicy(genome)
            action = policy.act(observation)
            _, reward, _, info = env.step(action)
            deliveries.append(
                {
                    "genome_index": genome_index,
                    "initial_archetype": archetype_name,
                    "sampled_batter": info["batter_archetype"],
                    "action": info["action"],
                    "release": asdict(info["release"]),
                    "event_times_s": info["event_times"],
                    "trajectory_points": info["trajectory_points"],
                    "key_points": info["key_points"],
                    "features": asdict(info["features"]),
                    "outcome": asdict(info["outcome"]),
                    "reward": reward,
                }
            )
    path = Path(run_dir) / "replays" / f"swarm_gen_{generation:04d}.json.gz"
    _dump_gzip(
        path,
        {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "run_name": run_name,
            "generation": generation,
            "shared_seed_by_archetype": {
                name: seed + index * 1_000_000
                for index, name in enumerate(ARCHETYPE_NAMES)
            },
            "trajectory_point_interval_ms": 4,
            "deliveries": deliveries,
        },
    )
    return path
