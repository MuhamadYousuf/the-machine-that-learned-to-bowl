"""Multiprocess neuroevolution with balanced common-random-number evaluation."""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import pickle
import random
import re
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .batter import ARCHETYPES
from .env import ARCHETYPE_NAMES, ENVIRONMENT_VERSION, OverEnv
from .policy import GENOME_SIZE, MLPPolicy
from .replay import write_replay_bundle, write_swarm_replay


ELITE_COUNT = 8
TOURNAMENT_K = 4
MUTATION_START = 0.08
MUTATION_END = 0.015
CHECKPOINT_GENERATIONS = {1, 5, 10, 25, 50, 100, 250, 500, 750, 1000, 1250, 1500}
POPULATION_SNAPSHOT_GENERATIONS = {1, 10, 50, 250, 500, 1000, 1500}
POPULATION_SNAPSHOT_INTERVAL = 25
REPLAY_GENERATIONS = {1, 10, 50, 100, 250, 500, 1000, 1500}
VALIDATION_INTERVAL = 25
VALIDATION_OVERS = 200
VALIDATION_SEED_BASE = 1 << 40
LENGTH_CATEGORIES = ("yorker", "full", "good", "short", "bouncer", "full_toss")
RESUME_INTERVAL = 10
LINEAGE_FIELDS = (
    "generation", "child_id", "parent_a", "parent_b", "was_elite",
    "crossover_used", "sigma", "fitness",
)


def mutation_sigma(generation: int, generations: int) -> float:
    if generations <= 1:
        return MUTATION_START
    progress = (generation - 1) / (generations - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MUTATION_END + (MUTATION_START - MUTATION_END) * cosine


def breed_population(
    population: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    *,
    generation: int,
    generations: int,
    elite_count: int = ELITE_COUNT,
    tournament_k: int = TOURNAMENT_K,
    mutation_rate: float = 0.10,
) -> np.ndarray:
    """Create a population with elitism, tournament selection, and mutation."""
    population = np.asarray(population, dtype=float)
    fitness = np.asarray(fitness, dtype=float)
    if population.ndim != 2 or fitness.shape != (len(population),):
        raise ValueError("population must be 2D and fitness must match its rows")
    if len(population) == 0:
        raise ValueError("population cannot be empty")
    elite_count = min(max(1, int(elite_count)), len(population))
    order = np.argsort(fitness)[::-1]
    next_population = [population[index].copy() for index in order[:elite_count]]
    sigma = mutation_sigma(generation, generations)

    def select_parent() -> np.ndarray:
        competitors = rng.integers(0, len(population), size=max(1, tournament_k))
        selected = competitors[np.argmax(fitness[competitors])]
        return population[selected]

    while len(next_population) < len(population):
        first = select_parent()
        second = select_parent()
        mask = rng.random(population.shape[1]) < 0.5
        child = np.where(mask, first, second).copy()
        mutations = rng.random(population.shape[1]) < mutation_rate
        child[mutations] += rng.normal(0.0, sigma, size=int(mutations.sum()))
        next_population.append(child)
    return np.asarray(next_population, dtype=float)


def run_toy_ga(
    objective: Callable[[np.ndarray], float],
    *,
    genome_size: int = 8,
    population_size: int = 32,
    generations: int = 25,
    seed: int = 123,
) -> list[float]:
    """Small deterministic GA helper used by the evolution regression test."""
    rng = np.random.default_rng(seed)
    population = rng.normal(0.0, 1.0, size=(population_size, genome_size))
    best_history = []
    for generation in range(1, generations + 1):
        fitness = np.asarray([objective(genome) for genome in population])
        best_history.append(float(np.max(fitness)))
        if generation < generations:
            population = breed_population(
                population, fitness, rng, generation=generation,
                generations=generations, elite_count=min(4, population_size),
            )
    return best_history


def _evaluate_episode(
    genome: np.ndarray,
    archetype_name: str,
    seed: int,
    noise_scale: float,
    record_deliveries: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    environment = OverEnv(noise_scale=noise_scale)
    observation = environment.reset(archetype_name, rng)
    policy = MLPPolicy(genome)
    total_reward = 0.0
    speeds = []
    length_counts = {category: 0 for category in LENGTH_CATEGORIES}
    deliveries = []
    actions = []
    bouncer_no_balls = 0
    while True:
        action = policy.act(observation)
        actions.append(action.copy())
        observation, reward, done, info = environment.step(action)
        total_reward += reward
        speeds.append(info["release"].speed_kmh)
        length_counts[info["features"].length_category] += 1
        bouncer_no_balls += int(info.get("bouncer_no_ball", False))
        if record_deliveries:
            features = info["features"]
            outcome = info["outcome"]
            deliveries.append(
                {
                    "legal_ball_number": int(info["legal_ball_number"]),
                    "length_m": 0.0 if features.length_m is None else float(features.length_m),
                    "line_m": float(features.line_m),
                    "speed_kmh": float(info["release"].speed_kmh),
                    "swing": float(info["release"].swing),
                    "category": features.length_category,
                    "result": outcome.result,
                    "wicket": bool(outcome.is_wicket),
                    "runs": int(outcome.runs_conceded),
                    "legal": bool(outcome.is_legal_ball),
                }
            )
        if done:
            break
    summary = environment.summary()
    result = {
        "initial_archetype": archetype_name,
        "reward": total_reward,
        "wickets": summary["wickets"],
        "runs": summary["runs_conceded"],
        "extras": summary["extras"],
        "mean_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
        "length_counts": length_counts,
        "deliveries": sum(length_counts.values()),
        "legal_balls": summary["legal_balls"],
        "bouncers_per_over": summary["bouncers_this_over"],
        "bouncer_no_balls": bouncer_no_balls,
        "bouncer_no_ball_share": bouncer_no_balls / max(1, sum(length_counts.values())),
        "action_saturation_share": float(np.mean(np.abs(np.asarray(actions)) > 0.95)),
    }
    if record_deliveries:
        result["delivery_points"] = deliveries
        result["length_pattern"] = [row["category"] for row in deliveries if row["legal"]]
    return result


def normalize_reward(reward: float, random_reward: float, specialist_reward: float) -> float:
    """Map random bowling to 0 and the hand-designed specialist to 1."""
    span = float(specialist_reward) - float(random_reward)
    if span <= 0.0:
        raise ValueError("specialist reward must exceed random reward")
    return (float(reward) - float(random_reward)) / span


def _evaluate_genome_payload(payload: tuple[np.ndarray, list[tuple[str, int]], float, dict]) -> dict:
    genome, scenarios, noise_scale, baselines = payload
    episode_results = [
        _evaluate_episode(genome, archetype, seed, noise_scale)
        for archetype, seed in scenarios
    ]
    by_archetype = {}
    for name in ARCHETYPE_NAMES:
        selected = [row for row in episode_results if row["initial_archetype"] == name]
        denominator = max(1, len(selected))
        mean_reward = sum(row["reward"] for row in selected) / denominator
        random_reward = float(baselines[name]["random"])
        specialist_reward = float(baselines[name]["specialist"])
        normalized = normalize_reward(mean_reward, random_reward, specialist_reward)
        by_archetype[name] = {
            "reward_per_over": mean_reward,
            "normalized_score": normalized,
            "wickets_per_over": sum(row["wickets"] for row in selected) / denominator,
            "economy": sum(row["runs"] for row in selected) / denominator,
            "extras_per_over": sum(row["extras"] for row in selected) / denominator,
            "bouncers_per_over": sum(row["bouncers_per_over"] for row in selected) / denominator,
            "bouncer_no_ball_share": sum(row["bouncer_no_ball_share"] for row in selected) / denominator,
            "action_saturation_share": sum(row["action_saturation_share"] for row in selected) / denominator,
            "length_distribution": {
                category: sum(row["length_counts"][category] for row in selected)
                / max(1, sum(row["deliveries"] for row in selected))
                for category in LENGTH_CATEGORIES
            },
        }
    total_deliveries = max(1, sum(row["deliveries"] for row in episode_results))
    all_lengths = {
        category: sum(row["length_counts"][category] for row in episode_results)
        / total_deliveries
        for category in LENGTH_CATEGORIES
    }
    return {
        "fitness": float(np.mean([by_archetype[name]["normalized_score"] for name in ARCHETYPE_NAMES])),
        "mean_reward": float(np.mean([row["reward"] for row in episode_results])),
        "wickets_per_over": float(np.mean([row["wickets"] for row in episode_results])),
        "economy": float(np.mean([row["runs"] for row in episode_results])),
        "extras_per_over": float(np.mean([row["extras"] for row in episode_results])),
        "mean_speed_kmh": float(np.mean([row["mean_speed_kmh"] for row in episode_results])),
        "bouncers_per_over": float(np.mean([row["bouncers_per_over"] for row in episode_results])),
        "bouncer_no_ball_share": float(np.mean([row["bouncer_no_ball_share"] for row in episode_results])),
        "action_saturation_share": float(np.mean([row["action_saturation_share"] for row in episode_results])),
        "by_archetype": by_archetype,
        "length_distribution": all_lengths,
    }


def load_normalization_baselines(path: str | Path) -> dict[str, dict[str, float]]:
    """Read P6 random and ORACLE specialist expected rewards from the matrix."""
    by_key: dict[tuple[str, str], float] = {}
    with Path(path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            metric = row.get("expected_fitness") or row["fitness"]
            by_key[(row["archetype"], row["plan"])] = float(metric)
    baselines = {}
    for name in ARCHETYPE_NAMES:
        if (name, "P6") not in by_key or (name, "ORACLE") not in by_key:
            raise ValueError(f"plan matrix is missing P6 or ORACLE for {name}")
        random_reward = by_key[(name, "P6")]
        specialist_reward = by_key[(name, "ORACLE")]
        if specialist_reward <= random_reward:
            raise ValueError(f"specialist baseline must beat random for {name}")
        baselines[name] = {
            "random": random_reward,
            "specialist": specialist_reward,
            "span": specialist_reward - random_reward,
        }
    return baselines


def validation_seed_sets(seed: int, overs_per_archetype: int = VALIDATION_OVERS) -> dict[str, list[int]]:
    """Make fixed held-out seeds in a namespace disjoint from training seeds."""
    base = VALIDATION_SEED_BASE + int(seed) * 1_000_000
    return {
        name: [base + index * 100_000 + over * 101 for over in range(overs_per_archetype)]
        for index, name in enumerate(ARCHETYPE_NAMES)
    }


def _make_scenarios(
    rng: np.random.Generator, overs_per_archetype: int
) -> list[tuple[str, int]]:
    scenarios = []
    for name in ARCHETYPE_NAMES:
        seeds = rng.integers(0, np.iinfo(np.uint32).max, size=overs_per_archetype, dtype=np.uint32)
        scenarios.extend((name, int(seed)) for seed in seeds)
    return scenarios


def _log_columns() -> list[str]:
    columns = [
        "gen", "best_fitness", "mean_fitness", "median_fitness",
        "wickets_per_over", "economy", "extras_per_over", "mean_speed_kmh",
    ]
    for name in ARCHETYPE_NAMES:
        columns.extend(
            (
                f"{name}_normalized_score",
                f"{name}_reward_per_over",
                f"{name}_wickets_per_over",
                f"{name}_economy",
                f"{name}_extras_per_over",
            )
        )
    columns.extend(f"length_{category}_share" for category in LENGTH_CATEGORIES)
    for name in ARCHETYPE_NAMES:
        columns.extend(f"{name}_length_{category}_share" for category in LENGTH_CATEGORIES)
    columns.extend(
        (
            "bouncer_no_ball_share",
            "bouncers_per_over",
            "action_saturation_share",
            "seconds_per_generation",
            "validation_score",
            "champion_validation_score",
            "champion_generation",
        )
    )
    return columns


def _log_row(generation: int, fitness: np.ndarray, best_stats: dict, seconds: float) -> dict:
    row = {
        "gen": generation,
        "best_fitness": float(np.max(fitness)),
        "mean_fitness": float(np.mean(fitness)),
        "median_fitness": float(np.median(fitness)),
        "wickets_per_over": best_stats["wickets_per_over"],
        "economy": best_stats["economy"],
        "extras_per_over": best_stats["extras_per_over"],
        "mean_speed_kmh": best_stats["mean_speed_kmh"],
        "bouncer_no_ball_share": best_stats["bouncer_no_ball_share"],
        "bouncers_per_over": best_stats["bouncers_per_over"],
        "action_saturation_share": best_stats["action_saturation_share"],
        "seconds_per_generation": seconds,
    }
    for name in ARCHETYPE_NAMES:
        row[f"{name}_normalized_score"] = best_stats["by_archetype"][name]["normalized_score"]
        row[f"{name}_reward_per_over"] = best_stats["by_archetype"][name]["reward_per_over"]
        row[f"{name}_wickets_per_over"] = best_stats["by_archetype"][name]["wickets_per_over"]
        row[f"{name}_economy"] = best_stats["by_archetype"][name]["economy"]
        row[f"{name}_extras_per_over"] = best_stats["by_archetype"][name]["extras_per_over"]
    for category in LENGTH_CATEGORIES:
        row[f"length_{category}_share"] = best_stats["length_distribution"][category]
    for name in ARCHETYPE_NAMES:
        for category in LENGTH_CATEGORIES:
            row[f"{name}_length_{category}_share"] = best_stats["by_archetype"][name]["length_distribution"][category]
    return row


def _save_checkpoint(
    path: Path,
    genome: np.ndarray,
    *,
    generation: int,
    seed: int,
    noise_scale: float,
    fitness: float,
    stats: dict,
    extra_metadata: dict | None = None,
) -> None:
    metadata = {
        "schema_version": 3,
        "generation": generation,
        "seed": seed,
        "noise_scale": noise_scale,
        "fitness": fitness,
        "observation_size": MLPPolicy(np.asarray(genome)).input_size,
        "environment_version": ENVIRONMENT_VERSION,
        "action_size": 4,
        "hidden_sizes": [24, 24],
        "stats": stats,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    _atomic_save_npz(
        path,
        genome=np.asarray(genome),
        metadata_json=json.dumps(metadata),
    )


def _evaluate_episode_payload(payload: tuple) -> dict:
    genome, archetype_name, seed, noise_scale, record_deliveries = payload
    return _evaluate_episode(
        genome, archetype_name, seed, noise_scale,
        record_deliveries=record_deliveries,
    )


def _summarize_validation(
    episode_results: list[dict], baselines: dict[str, dict[str, float]]
) -> dict:
    per_archetype = {}
    for name in ARCHETYPE_NAMES:
        selected = [row for row in episode_results if row["initial_archetype"] == name]
        denominator = max(1, len(selected))
        mean_reward = sum(row["reward"] for row in selected) / denominator
        per_archetype[name] = {
            "reward_per_over": mean_reward,
            "normalized_score": normalize_reward(
                mean_reward,
                baselines[name]["random"],
                baselines[name]["specialist"],
            ),
            "wickets_per_over": sum(row["wickets"] for row in selected) / denominator,
            "economy": sum(row["runs"] for row in selected) / denominator,
            "extras_per_over": sum(row["extras"] for row in selected) / denominator,
            "bouncers_per_over": sum(row["bouncers_per_over"] for row in selected) / denominator,
            "bouncer_no_ball_share": sum(row["bouncer_no_ball_share"] for row in selected) / denominator,
            "action_saturation_share": sum(row["action_saturation_share"] for row in selected) / denominator,
            "length_distribution": {
                category: sum(row["length_counts"][category] for row in selected)
                / max(1, sum(row["deliveries"] for row in selected))
                for category in LENGTH_CATEGORIES
            },
            "pitch_points": [
                {
                    **point,
                    "over_index": over_index,
                    "ball_pattern_index": point["legal_ball_number"] - 1,
                }
                for over_index, row in enumerate(selected)
                for point in row.get("delivery_points", [])
            ],
            "ball_patterns": [row.get("length_pattern", []) for row in selected],
        }
    return {
        "score": float(np.mean([per_archetype[name]["normalized_score"] for name in ARCHETYPE_NAMES])),
        "per_archetype": per_archetype,
    }


def _run_validation(
    genome: np.ndarray,
    *,
    seed_sets: dict[str, list[int]],
    noise_scale: float,
    baselines: dict[str, dict[str, float]],
    executor: ProcessPoolExecutor | None,
) -> dict:
    payloads = [
        (genome, name, seed, noise_scale, True)
        for name in ARCHETYPE_NAMES
        for seed in seed_sets[name]
    ]
    if executor is None:
        episodes = [_evaluate_episode_payload(payload) for payload in payloads]
    else:
        episodes = list(executor.map(_evaluate_episode_payload, payloads, chunksize=2))
    return _summarize_validation(episodes, baselines)


def _save_population(
    path: Path,
    population: np.ndarray,
    *,
    generation: int,
    seed: int,
    fitness: np.ndarray | None = None,
    genome_ids: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": 2, "generation": generation, "seed": seed}
    arrays = {"population": population, "metadata_json": json.dumps(metadata)}
    if fitness is not None:
        arrays["fitness"] = np.asarray(fitness, dtype=float)
    if genome_ids is not None:
        arrays["genome_ids"] = np.asarray(genome_ids, dtype=np.int64)
    _atomic_save_npz(path, **arrays)


def _population_snapshot_schedule(generations: int) -> list[int]:
    scheduled = set(POPULATION_SNAPSHOT_GENERATIONS)
    scheduled.update(range(POPULATION_SNAPSHOT_INTERVAL, generations + 1, POPULATION_SNAPSHOT_INTERVAL))
    return sorted(generation for generation in scheduled if generation <= generations)


def _save_best_genome_snapshot(
    run_path: Path,
    *,
    generation: int,
    genome: np.ndarray,
    genome_id: int,
    fitness: float,
    stats: dict,
    snapshot_kind: str = "generation_best",
) -> None:
    """Atomically save a generation-best policy without consuming randomness."""
    per_archetype = {
        name: {
            key: float(stats.get("by_archetype", {}).get(name, {}).get(key, 0.0))
            for key in ("normalized_score", "reward_per_over", "wickets_per_over", "economy", "extras_per_over")
        }
        for name in ARCHETYPE_NAMES
    }
    metadata = {
        "schema_version": 1,
        "generation": int(generation),
        "genome_id": int(genome_id),
        "fitness": float(fitness),
        "snapshot_kind": snapshot_kind,
        "per_archetype": per_archetype,
    }
    target = run_path / "best_genomes" / f"gen_{generation:04d}.npz"
    _atomic_save_npz(
        target,
        genome=np.asarray(genome, dtype=float),
        generation=np.asarray(generation, dtype=np.int64),
        genome_id=np.asarray(genome_id, dtype=np.int64),
        fitness=np.asarray(fitness, dtype=float),
        archetype_names=np.asarray(ARCHETYPE_NAMES),
        per_archetype_scores=np.asarray(
            [per_archetype[name]["normalized_score"] for name in ARCHETYPE_NAMES], dtype=float
        ),
        metadata_json=json.dumps(metadata, separators=(",", ":")),
    )


def _backfill_best_genomes_from_checkpoints(run_path: Path, generation: int) -> None:
    """Seed sparse pre-feature history with exact saved best-policy checkpoints."""
    best_dir = run_path / "best_genomes"
    best_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted((run_path / "checkpoints").glob("gen_*.npz"))
    champion = run_path / "checkpoints" / "champion.npz"
    if champion.exists():
        sources.append(champion)
    for source in sources:
        match = re.search(r"(?:gen_)?(\d+)\.npz$", source.name)
        if source.name == "champion.npz":
            match = None
        if source.name == "champion.npz":
            with np.load(source, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                checkpoint_generation = int(metadata.get("generation", 0))
                genome = data["genome"].astype(float)
        else:
            if match is None:
                continue
            checkpoint_generation = int(match.group(1))
            with np.load(source, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                genome = data["genome"].astype(float)
        if checkpoint_generation < 1 or checkpoint_generation > generation:
            continue
        target = best_dir / f"gen_{checkpoint_generation:04d}.npz"
        if target.exists():
            continue
        _save_best_genome_snapshot(
            run_path,
            generation=checkpoint_generation,
            genome=genome,
            genome_id=int(metadata.get("genome_id", -1)),
            fitness=float(metadata.get("fitness", 0.0)),
            stats=metadata.get("stats", {}),
            snapshot_kind="validation_champion" if source.name == "champion.npz" else "checkpoint",
        )


def _breed_population_detailed(
    population: np.ndarray,
    fitness: np.ndarray,
    population_ids: np.ndarray,
    next_id: int,
    rng: np.random.Generator,
    *,
    generation: int,
    generations: int,
    elite_count: int = ELITE_COUNT,
    tournament_k: int = TOURNAMENT_K,
    mutation_rate: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, list[dict], int]:
    """Breed with stable ids and parent metadata for the lineage explorer."""
    population = np.asarray(population, dtype=float)
    fitness = np.asarray(fitness, dtype=float)
    population_ids = np.asarray(population_ids, dtype=np.int64)
    if population.ndim != 2 or fitness.shape != (len(population),):
        raise ValueError("population must be 2D and fitness must match its rows")
    if population_ids.shape != (len(population),):
        raise ValueError("population ids must match population rows")
    elite_count = min(max(1, int(elite_count)), len(population))
    order = np.argsort(fitness)[::-1]
    sigma = mutation_sigma(generation, generations)
    children: list[np.ndarray] = []
    child_ids: list[int] = []
    metadata: list[dict] = []
    for index in order[:elite_count]:
        children.append(population[index].copy())
        child_ids.append(next_id)
        metadata.append({
            "id": next_id, "parent_a": int(population_ids[index]), "parent_b": None,
            "was_elite": True, "crossover_used": False, "sigma": 0.0,
        })
        next_id += 1

    def select_parent_index() -> int:
        competitors = rng.integers(0, len(population), size=max(1, tournament_k))
        return int(competitors[np.argmax(fitness[competitors])])

    while len(children) < len(population):
        first_index = select_parent_index()
        second_index = select_parent_index()
        first, second = population[first_index], population[second_index]
        mask = rng.random(population.shape[1]) < 0.5
        child = np.where(mask, first, second).copy()
        mutations = rng.random(population.shape[1]) < mutation_rate
        child[mutations] += rng.normal(0.0, sigma, size=int(mutations.sum()))
        children.append(child)
        child_ids.append(next_id)
        metadata.append({
            "id": next_id,
            "parent_a": int(population_ids[first_index]),
            "parent_b": int(population_ids[second_index]),
            "was_elite": False,
            "crossover_used": first_index != second_index,
            "sigma": sigma,
        })
        next_id += 1
    return (
        np.asarray(children, dtype=float), np.asarray(child_ids, dtype=np.int64),
        metadata, next_id,
    )


def _append_lineage(path: Path, generation: int, metadata: list[dict], fitness: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with gzip.open(path, "at", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LINEAGE_FIELDS)
        if write_header:
            writer.writeheader()
        for item, score in zip(metadata, fitness):
            writer.writerow({
                "generation": generation,
                "child_id": item["id"],
                "parent_a": "" if item["parent_a"] is None else item["parent_a"],
                "parent_b": "" if item["parent_b"] is None else item["parent_b"],
                "was_elite": int(bool(item["was_elite"])),
                "crossover_used": int(bool(item["crossover_used"])),
                "sigma": item["sigma"],
                "fitness": float(score),
            })


def _replace_with_retry(source: Path, target: Path, *, attempts: int = 24) -> None:
    """Retry brief Windows sharing violations when a live reader has a snapshot open."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError as error:
            # Windows reports a sharing violation as ERROR_SHARING_VIOLATION
            # (32); antivirus scanners may briefly report ERROR_ACCESS_DENIED (5).
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt + 1 == attempts:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.25))


def _atomic_save_npz(path: Path, **arrays) -> None:
    """Publish a complete compressed snapshot, retrying brief reader locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.npz")
    with temp.open("wb") as file:
        np.savez_compressed(file, **arrays)
        file.flush()
        os.fsync(file.fileno())
    _replace_with_retry(temp, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    _replace_with_retry(temp, path)


def _atomic_save_resume_state(path: Path, state: dict) -> None:
    """Publish a complete state file and retain the previous valid one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name("resume_state.tmp.npz")
    with temp_path.open("wb") as file:
        np.savez_compressed(file, **state)
        file.flush()
        os.fsync(file.fileno())
    if path.exists():
        previous = path.with_name("resume_state_prev.npz")
        previous_temp = previous.with_name("resume_state_prev.tmp.npz")
        shutil.copyfile(path, previous_temp)
        with previous_temp.open("rb+") as file:
            file.flush()
            os.fsync(file.fileno())
        _replace_with_retry(previous_temp, previous)
    _replace_with_retry(temp_path, path)


def _load_resume_state(run_path: Path) -> tuple[dict, Path]:
    errors = []
    for path in (run_path / "resume_state.npz", run_path / "resume_state_prev.npz"):
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                state = {key: archive[key].copy() for key in archive.files}
            if int(state["schema_version"].item()) != 1:
                raise ValueError("unsupported resume-state schema")
            return state, path
        except Exception as error:  # try the previous atomic snapshot if newest is torn
            errors.append(f"{path.name}: {error}")
    raise FileNotFoundError("no readable resume state; " + "; ".join(errors))


def _rewrite_csv_through(path: Path, generation: int) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        rows = []
        for row in reader:
            try:
                if int(row.get("gen", "")) <= generation:
                    rows.append(row)
            except (TypeError, ValueError):
                continue
    if not fields:
        path.write_text("", encoding="utf-8")
        return
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    _replace_with_retry(temp, path)


def _rewrite_validation_through(path: Path, generation: int) -> None:
    if not path.exists():
        return
    lines = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                if int(record.get("generation", 0)) <= generation:
                    lines.append(json.dumps(record, separators=(",", ":"), allow_nan=False))
            except (json.JSONDecodeError, ValueError):
                continue
    _atomic_text(path, "".join(line + "\n" for line in lines))


def _rewrite_lineage_through(path: Path, generation: int) -> None:
    if not path.exists():
        return
    rows = []
    try:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = [row for row in reader if int(row.get("generation", 0)) <= generation]
    except (OSError, EOFError, gzip.BadGzipFile, ValueError):
        # Retain every complete gzip member/row that was readable before a cut.
        try:
            with gzip.open(path, "rt", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if int(row.get("generation", 0)) <= generation:
                        rows.append(row)
        except (OSError, EOFError, gzip.BadGzipFile, ValueError):
            pass
    temp = path.with_name(path.name + ".tmp.gz")
    with gzip.open(temp, "wt", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LINEAGE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _replace_with_retry(temp, path)


def _truncate_outputs(run_path: Path, generation: int) -> None:
    _rewrite_csv_through(run_path / "log.csv", generation)
    _rewrite_validation_through(run_path / "validation.jsonl", generation)
    _rewrite_lineage_through(run_path / "lineage.csv.gz", generation)
    numbered_patterns = (
        (run_path / "checkpoints", re.compile(r"gen_(\d+)\.npz$")),
        (run_path / "populations", re.compile(r"population_(\d+)\.npz$")),
        (run_path / "best_genomes", re.compile(r"gen_(\d+)\.npz$")),
        (run_path / "replays", re.compile(r"(?:gen_|swarm_gen_)(\d+)")),
    )
    for folder, pattern in numbered_patterns:
        if not folder.exists():
            continue
        for path in folder.iterdir():
            match = pattern.search(path.name)
            if match and int(match.group(1)) > generation:
                path.unlink(missing_ok=True)


def _save_resume_state(
    run_path: Path,
    *,
    generation: int,
    population: np.ndarray,
    fitness: np.ndarray,
    population_ids: np.ndarray,
    next_population: np.ndarray,
    next_population_ids: np.ndarray,
    next_lineage: list[dict],
    next_genome_id: int,
    rng: np.random.Generator,
    champion_genome: np.ndarray | None,
    champion_score: float,
    champion_generation: int | None,
    champion_record: dict | None,
    elite_ids: np.ndarray,
    sigma: float,
    worker_seeds: list[tuple[str, int]],
) -> None:
    state = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "generation": np.asarray(generation, dtype=np.int64),
        "population": np.asarray(population, dtype=float),
        "fitness": np.asarray(fitness, dtype=float),
        "population_ids": np.asarray(population_ids, dtype=np.int64),
        "next_population": np.asarray(next_population, dtype=float),
        "next_population_ids": np.asarray(next_population_ids, dtype=np.int64),
        "next_lineage_json": np.asarray(json.dumps(next_lineage, separators=(",", ":"))),
        "next_genome_id": np.asarray(next_genome_id, dtype=np.int64),
        "mutation_schedule_position": np.asarray(generation + 1, dtype=np.int64),
        "mutation_sigma": np.asarray(sigma, dtype=float),
        "champion_genome": np.asarray(
            champion_genome if champion_genome is not None else np.zeros(GENOME_SIZE), dtype=float
        ),
        "champion_has_genome": np.asarray(champion_genome is not None, dtype=np.bool_),
        "champion_score": np.asarray(champion_score, dtype=float),
        "champion_generation": np.asarray(-1 if champion_generation is None else champion_generation, dtype=np.int64),
        "champion_record_json": np.asarray(json.dumps(champion_record, separators=(",", ":")) if champion_record else ""),
        "elite_ids": np.asarray(elite_ids, dtype=np.int64),
        "numpy_rng_state_json": np.asarray(json.dumps(rng.bit_generator.state, separators=(",", ":"))),
        "python_rng_state": np.frombuffer(pickle.dumps(random.getstate(), protocol=4), dtype=np.uint8),
        "worker_seeds_json": np.asarray(json.dumps(worker_seeds, separators=(",", ":"))),
    }
    _atomic_save_resume_state(run_path / "resume_state.npz", state)


def reconstruct_population(
    champion_genome: np.ndarray,
    population_size: int,
    rng: np.random.Generator,
    *,
    generation: int,
    generations: int,
    local_fraction: float = 0.75,
) -> np.ndarray:
    """Warm-restart a GA from one saved champion when no population snapshot exists."""
    champion = np.asarray(champion_genome, dtype=float)
    if champion.shape != (GENOME_SIZE,):
        raise ValueError(f"champion genome must have shape ({GENOME_SIZE},)")
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if not 0.0 <= local_fraction <= 1.0:
        raise ValueError("local_fraction must be between 0 and 1")

    population = np.empty((population_size, GENOME_SIZE), dtype=float)
    population[0] = champion
    local_count = int((population_size - 1) * local_fraction)
    sigma = mutation_sigma(generation, generations)
    for index in range(1, population_size):
        if index <= local_count:
            candidate = champion.copy()
            mutations = rng.random(GENOME_SIZE) < 0.10
            candidate[mutations] += rng.normal(0.0, sigma, size=int(mutations.sum()))
            population[index] = candidate
        else:
            population[index] = MLPPolicy.random(rng).get_genome()
    return population


def _run_evolution_warm(
    run_dir: str | Path,
    *,
    generations: int = 200,
    population_size: int = 192,
    seed: int = 1,
    workers: int | None = None,
    noise_scale: float = 0.05,
    overs_per_archetype: int = 16,
    validation_overs: int = VALIDATION_OVERS,
    plan_matrix_path: str | Path | None = None,
    save_artifacts: bool = True,
    resume_from: str | Path | None = None,
) -> dict:
    """Run normalized-fitness neuroevolution and write validation/artifact data."""
    if generations < 1 or population_size < 2:
        raise ValueError("generations must be positive and population_size at least 2")
    if overs_per_archetype < 1:
        raise ValueError("overs_per_archetype must be positive")
    if validation_overs < 1:
        raise ValueError("validation_overs must be positive")
    run_path = Path(run_dir)
    if plan_matrix_path is None:
        plan_matrix_path = Path(__file__).resolve().parents[2] / "outputs" / "batter" / "plan_matrix.csv"
    plan_matrix_path = Path(plan_matrix_path)
    baselines = load_normalization_baselines(plan_matrix_path)
    log_path = run_path / "log.csv"
    has_existing_log = log_path.exists() and log_path.stat().st_size > 0
    if has_existing_log and resume_from is None:
        raise FileExistsError(f"refusing to append a new run to existing log: {log_path}")
    if resume_from is not None and not has_existing_log:
        raise FileNotFoundError(f"cannot resume without an existing training log: {log_path}")
    if has_existing_log:
        with log_path.open(newline="", encoding="utf-8") as log_file:
            prior_rows = list(csv.DictReader(log_file))
        if not prior_rows:
            raise ValueError(f"cannot resume from an empty training log: {log_path}")
        last_generation = int(prior_rows[-1]["gen"])
        start_generation = last_generation + 1
        if start_generation > generations:
            raise ValueError(
                f"training log already reached generation {last_generation}; requested total is {generations}"
            )
    else:
        last_generation = 0
        start_generation = 1
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_path / "replays").mkdir(parents=True, exist_ok=True)
    columns = _log_columns()
    write_header = not log_path.exists() or log_path.stat().st_size == 0

    rng_seed = seed if resume_from is None else seed + start_generation * 1_000_003
    rng = np.random.default_rng(rng_seed)
    resume_checkpoint_path = None
    resume_metadata: dict = {}
    if resume_from is None:
        # Seed genomes independently so the initial population is not a clone set.
        population = np.stack(
            [MLPPolicy.random(rng).get_genome() for _ in range(population_size)],
            axis=0,
        )
    else:
        resume_checkpoint_path = Path(resume_from)
        if not resume_checkpoint_path.is_absolute():
            resume_checkpoint_path = checkpoint_dir / resume_checkpoint_path
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_checkpoint_path}")
        with np.load(resume_checkpoint_path, allow_pickle=False) as data:
            resume_genome = data["genome"].astype(float)
            resume_metadata = json.loads(str(data["metadata_json"].item()))
        population = reconstruct_population(
            resume_genome,
            population_size,
            rng,
            generation=start_generation,
            generations=generations,
        )

    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    workers = max(1, int(workers))
    heldout_seeds = validation_seed_sets(seed, validation_overs)
    run_config = {
        "schema_version": 2,
        "name": run_path.name,
        "seed": seed,
        "generations": generations,
        "population_size": population_size,
        "elitism": min(ELITE_COUNT, population_size),
        "tournament_k": TOURNAMENT_K,
        "uniform_crossover_probability": 0.5,
        "mutation_rate": 0.10,
        "mutation_sigma_start": MUTATION_START,
        "mutation_sigma_end": MUTATION_END,
        "mutation_schedule": "cosine",
        "overs_per_archetype_per_generation": overs_per_archetype,
        "workers": workers,
        "noise_scale": noise_scale,
        "fitness_baselines": baselines,
        "validation_interval": VALIDATION_INTERVAL,
        "validation_overs_per_archetype": validation_overs,
        "validation_seeds": heldout_seeds,
        "training_seed_range": [0, np.iinfo(np.uint32).max],
        "checkpoint_generations": sorted(
            generation for generation in CHECKPOINT_GENERATIONS | {generations}
            if generation <= generations
        ),
        "population_snapshot_generations": sorted(
            generation for generation in POPULATION_SNAPSHOT_GENERATIONS
            if generation <= generations
        ),
        "replay_generations": sorted(
            generation for generation in REPLAY_GENERATIONS | {generations}
            if generation <= generations
        ),
        "plan_matrix_path": str(plan_matrix_path),
        "save_artifacts": save_artifacts,
    }
    run_path.mkdir(parents=True, exist_ok=True)
    config_path = run_path / "run_config.json"
    if resume_from is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"cannot resume without run configuration: {config_path}")
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        resume_expectations = {
            "seed": seed,
            "generations": generations,
            "population_size": population_size,
            "noise_scale": noise_scale,
            "overs_per_archetype_per_generation": overs_per_archetype,
            "validation_overs_per_archetype": validation_overs,
            "save_artifacts": save_artifacts,
        }
        for key, expected in resume_expectations.items():
            if previous_config.get(key) != expected:
                raise ValueError(
                    f"resume configuration mismatch for {key}: "
                    f"saved {previous_config.get(key)!r}, requested {expected!r}"
                )
        if int(resume_metadata.get("generation", 0)) > last_generation:
            raise ValueError("resume checkpoint is newer than the last logged generation")
        if resume_checkpoint_path is None:
            raise RuntimeError("resume checkpoint was not resolved")
        recovery_history = list(previous_config.get("recovery_history", []))
        recovery_history.append(
            {
                "interrupted_after_generation": last_generation,
                "resumed_at_generation": start_generation,
                "checkpoint": resume_checkpoint_path.name,
                "checkpoint_generation": int(resume_metadata.get("generation", 0)),
                "checkpoint_validation_score": resume_metadata.get("validation_score"),
                "population_reconstruction": "checkpoint genome + 75% local mutations + 25% fresh genomes",
                "rng_seed": rng_seed,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        run_config = previous_config
        run_config["workers"] = workers
        run_config["recovery_history"] = recovery_history
    config_path.write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    overall_started = time.perf_counter()
    champion_score = float("-inf")
    champion_genome: np.ndarray | None = None
    champion_generation: int | None = None
    if resume_from is not None:
        champion_path = checkpoint_dir / "champion.npz"
        if champion_path.exists():
            with np.load(champion_path, allow_pickle=False) as data:
                champion_genome = data["genome"].astype(float)
                champion_metadata = json.loads(str(data["metadata_json"].item()))
            champion_score = float(champion_metadata["validation_score"])
            champion_generation = int(champion_metadata["validation_generation"])
        elif "validation_score" in resume_metadata:
            champion_genome = resume_genome.copy()
            champion_score = float(resume_metadata["validation_score"])
            champion_generation = int(
                resume_metadata.get("validation_generation", resume_metadata["generation"])
            )
    history: list[dict] = []

    def evaluate_population(current_population: np.ndarray, scenarios: list[tuple[str, int]]):
        payloads = [(genome, scenarios, noise_scale, baselines) for genome in current_population]
        if executor is None:
            return [_evaluate_genome_payload(payload) for payload in payloads]
        return list(executor.map(_evaluate_genome_payload, payloads, chunksize=1))

    def save_typical(current_population: np.ndarray, evaluations: list[dict], generation: int):
        if not save_artifacts:
            return
        fitness_values = np.asarray([row["fitness"] for row in evaluations])
        order = np.argsort(fitness_values, kind="stable")
        middle = len(order) // 2
        chosen = order[np.clip(np.arange(middle - 2, middle + 3), 0, len(order) - 1)]
        for number, index in enumerate(chosen, start=1):
            write_replay_bundle(
                run_path,
                run_name=run_path.name,
                label=f"gen_{generation:04d}_typical_{number:02d}",
                genome=current_population[index],
                generation=generation,
                fitness=float(fitness_values[index]),
                noise_scale=noise_scale,
                seed=seed + generation * 100_000,
                overs=10,
                combine=True,
            )

    try:
        if resume_from is None:
            # Generation 0 is the untrained random population. Its median five
            # genomes are saved before natural selection begins.
            generation_zero_scenarios = _make_scenarios(rng, overs_per_archetype)
            generation_zero_evaluations = evaluate_population(population, generation_zero_scenarios)
            save_typical(population, generation_zero_evaluations, 0)
            generation_zero_fitness = np.asarray([row["fitness"] for row in generation_zero_evaluations])
            population = breed_population(
                population,
                generation_zero_fitness,
                rng,
                generation=1,
                generations=generations,
                elite_count=min(ELITE_COUNT, population_size),
            )

        with log_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            if write_header:
                writer.writeheader()
            for generation in range(start_generation, generations + 1):
                generation_started = time.perf_counter()
                scenarios = _make_scenarios(rng, overs_per_archetype)
                # Every genome, including copied elites, is evaluated on this
                # generation's common seeds. No stale fitness is retained.
                evaluations = evaluate_population(population, scenarios)
                fitness = np.asarray([entry["fitness"] for entry in evaluations])
                best_index = int(np.argmax(fitness))
                best_stats = evaluations[best_index]
                best_genome = population[best_index].copy()

                if generation == 1:
                    save_typical(population, evaluations, 1)

                should_checkpoint = generation in CHECKPOINT_GENERATIONS or generation == generations
                if should_checkpoint:
                    _save_checkpoint(
                        checkpoint_dir / f"gen_{generation:04d}.npz",
                        best_genome,
                        generation=generation,
                        seed=seed,
                        noise_scale=noise_scale,
                        fitness=float(fitness[best_index]),
                        stats=best_stats,
                        extra_metadata={"fitness_baselines": baselines},
                    )

                if save_artifacts and generation in POPULATION_SNAPSHOT_GENERATIONS:
                    _save_population(
                        run_path / "populations" / f"population_{generation:04d}.npz",
                        population,
                        generation=generation,
                        seed=seed,
                    )
                    write_swarm_replay(
                        run_path,
                        run_name=run_path.name,
                        generation=generation,
                        population=population,
                        noise_scale=noise_scale,
                        seed=seed + 8_000_000_000,
                    )

                if save_artifacts and generation in REPLAY_GENERATIONS:
                    write_replay_bundle(
                        run_path,
                        run_name=run_path.name,
                        label=f"gen_{generation:04d}",
                        genome=best_genome,
                        generation=generation,
                        fitness=float(fitness[best_index]),
                        noise_scale=noise_scale,
                        seed=seed + generation * 10_000_000,
                        overs=30,
                    )

                validation_record = None
                if generation % VALIDATION_INTERVAL == 0:
                    validation = _run_validation(
                        best_genome,
                        seed_sets=heldout_seeds,
                        noise_scale=noise_scale,
                        baselines=baselines,
                        executor=executor,
                    )
                    validation_record = {
                        "generation": generation,
                        "validation_score": validation["score"],
                        "per_archetype": validation["per_archetype"],
                    }
                    with (run_path / "validation.jsonl").open("a", encoding="utf-8") as validation_file:
                        validation_file.write(json.dumps(validation_record, separators=(",", ":"), allow_nan=False) + "\n")
                        validation_file.flush()
                    if validation["score"] > champion_score:
                        champion_score = validation["score"]
                        champion_genome = best_genome.copy()
                        champion_generation = generation
                        champion_validation_path = run_path / "champion_validation.json"
                        temporary_validation_path = champion_validation_path.with_suffix(".json.tmp")
                        temporary_validation_path.write_text(
                            json.dumps(validation_record, separators=(",", ":"), allow_nan=False),
                            encoding="utf-8",
                        )
                        _replace_with_retry(temporary_validation_path, champion_validation_path)
                        _save_checkpoint(
                            checkpoint_dir / "champion.npz",
                            champion_genome,
                            generation=generation,
                            seed=seed,
                            noise_scale=noise_scale,
                            fitness=float(fitness[best_index]),
                            stats=best_stats,
                            extra_metadata={
                                "checkpoint_kind": "validation_champion",
                                "validation_score": champion_score,
                                "validation_generation": generation,
                                "validation_normalized_scores": {
                                    name: validation["per_archetype"][name]["normalized_score"]
                                    for name in ARCHETYPE_NAMES
                                },
                                "fitness_baselines": baselines,
                            },
                        )

                seconds = time.perf_counter() - generation_started
                row = _log_row(generation, fitness, best_stats, seconds)
                if validation_record is not None:
                    row["validation_score"] = validation_record["validation_score"]
                    row["champion_validation_score"] = champion_score
                    row["champion_generation"] = champion_generation
                writer.writerow(row)
                csv_file.flush()
                history.append(row)

                score_text = ",".join(
                    f"{best_stats['by_archetype'][name]['normalized_score']:.2f}"
                    for name in ARCHETYPE_NAMES
                )
                print(
                    f"gen {generation:>4}/{generations} "
                    f"best={fitness[best_index]:8.3f} mean={np.mean(fitness):8.3f} "
                    f"median={np.median(fitness):8.3f} "
                    f"scores={score_text} "
                    f"speed={best_stats['mean_speed_kmh']:.1f} {seconds:6.1f}s",
                    flush=True,
                )
                if generation < generations:
                    population = breed_population(
                        population,
                        fitness,
                        rng,
                        generation=generation,
                        generations=generations,
                        elite_count=min(ELITE_COUNT, population_size),
                    )
        if champion_genome is None:
            final_checkpoint = checkpoint_dir / f"gen_{generations:04d}.npz"
            with np.load(final_checkpoint, allow_pickle=False) as data:
                champion_genome = data["genome"].astype(float)
                metadata = json.loads(str(data["metadata_json"].item()))
            final_validation = _run_validation(
                champion_genome,
                seed_sets=heldout_seeds,
                noise_scale=noise_scale,
                baselines=baselines,
                executor=executor,
            )
            champion_score = final_validation["score"]
            champion_generation = generations
            _save_checkpoint(
                checkpoint_dir / "champion.npz",
                champion_genome,
                generation=generations,
                seed=seed,
                noise_scale=noise_scale,
                fitness=float(metadata.get("fitness", 0.0)),
                stats=metadata.get("stats", {}),
                extra_metadata={
                    "checkpoint_kind": "validation_champion",
                    "validation_score": champion_score,
                    "validation_generation": generations,
                    "fitness_baselines": baselines,
                },
            )
        if save_artifacts and champion_genome is not None:
            write_replay_bundle(
                run_path,
                run_name=run_path.name,
                label="champion",
                genome=champion_genome,
                generation=champion_generation,
                fitness=champion_score,
                noise_scale=noise_scale,
                seed=seed + 99_000_000_000,
                overs=30,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return {
        "run_dir": str(run_path),
        "generations": generations,
        "start_generation": start_generation,
        "generations_this_session": generations - start_generation + 1,
        "population_size": population_size,
        "workers": workers,
        "seconds": time.perf_counter() - overall_started,
        "champion_generation": champion_generation,
        "champion_validation_score": champion_score,
        "history": history,
    }


def run_evolution(
    run_dir: str | Path,
    *,
    generations: int = 1500,
    population_size: int = 192,
    seed: int = 1,
    workers: int | None = None,
    noise_scale: float = 0.05,
    overs_per_archetype: int = 16,
    validation_overs: int = VALIDATION_OVERS,
    plan_matrix_path: str | Path | None = None,
    save_artifacts: bool = True,
    save_evolution_snapshots: bool = True,
    resume_from: str | Path | None = None,
    exact_resume: bool = False,
    stop_after_generation: int | None = None,
) -> dict:
    """Train with exact, crash-safe resume snapshots and per-genome lineage."""
    if resume_from is not None:
        if exact_resume:
            raise ValueError("choose exact resume or legacy warm resume, not both")
        return _run_evolution_warm(
            run_dir,
            generations=generations,
            population_size=population_size,
            seed=seed,
            workers=workers,
            noise_scale=noise_scale,
            overs_per_archetype=overs_per_archetype,
            validation_overs=validation_overs,
            plan_matrix_path=plan_matrix_path,
            save_artifacts=save_artifacts,
            resume_from=resume_from,
        )
    if generations < 1 or population_size < 2:
        raise ValueError("generations must be positive and population_size at least 2")
    if overs_per_archetype < 1 or validation_overs < 1:
        raise ValueError("training and validation overs must be positive")
    if stop_after_generation is not None and not 1 <= stop_after_generation <= generations:
        raise ValueError("stop_after_generation must be between 1 and generations")

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    config_path = run_path / "run_config.json"
    log_path = run_path / "log.csv"
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_path / "replays").mkdir(parents=True, exist_ok=True)
    if plan_matrix_path is None:
        plan_matrix_path = Path(__file__).resolve().parents[2] / "outputs" / "batter" / "plan_matrix.csv"
    plan_matrix_path = Path(plan_matrix_path)

    resume_state = None
    resume_source = None
    if exact_resume:
        if not config_path.exists():
            raise FileNotFoundError(f"cannot resume without run configuration: {config_path}")
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        resume_state, resume_source = _load_resume_state(run_path)
        saved_generation = int(resume_state["generation"].item())
        if run_config.get("generations") != generations:
            raise ValueError("resume generation total differs from the saved run configuration")
        if run_config.get("population_size") != population_size or run_config.get("seed") != seed:
            raise ValueError("resume population or seed differs from the saved run configuration")
        resume_checks = {
            "noise_scale": noise_scale,
            "overs_per_archetype_per_generation": overs_per_archetype,
            "validation_overs_per_archetype": validation_overs,
            "save_artifacts": save_artifacts,
        }
        for key, value in resume_checks.items():
            if run_config.get(key) != value:
                raise ValueError(f"resume configuration mismatch for {key}")
        if bool(run_config.get("save_evolution_snapshots", True)) != bool(save_evolution_snapshots):
            raise ValueError("resume configuration mismatch for save_evolution_snapshots")
        _truncate_outputs(run_path, saved_generation)
        baselines = run_config["fitness_baselines"]
        heldout_seeds = run_config["validation_seeds"]
        start_generation = saved_generation + 1
        if start_generation > generations:
            raise ValueError(f"resume state already reached generation {saved_generation}")
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(resume_state["numpy_rng_state_json"].item()))
        random.setstate(pickle.loads(resume_state["python_rng_state"].tobytes()))
        population = resume_state["next_population"].astype(float)
        population_ids = resume_state["next_population_ids"].astype(np.int64)
        lineage_metadata = json.loads(str(resume_state["next_lineage_json"].item()))
        next_genome_id = int(resume_state["next_genome_id"].item())
        has_champion = bool(resume_state["champion_has_genome"].item())
        champion_genome = resume_state["champion_genome"].astype(float) if has_champion else None
        champion_score = float(resume_state["champion_score"].item())
        champion_gen = int(resume_state["champion_generation"].item())
        champion_generation = None if champion_gen < 0 else champion_gen
        champion_json = str(resume_state["champion_record_json"].item())
        champion_record = json.loads(champion_json) if champion_json else None
        if save_evolution_snapshots:
            _backfill_best_genomes_from_checkpoints(run_path, saved_generation)
            if saved_generation % POPULATION_SNAPSHOT_INTERVAL == 0:
                population_path = run_path / "populations" / f"population_{saved_generation:04d}.npz"
                if not population_path.exists():
                    _save_population(
                        population_path,
                        resume_state["population"].astype(float),
                        generation=saved_generation,
                        seed=seed,
                        fitness=resume_state["fitness"].astype(float),
                        genome_ids=resume_state["population_ids"].astype(np.int64),
                    )
        champion_path = checkpoint_dir / "champion.npz"
        if champion_genome is None:
            champion_path.unlink(missing_ok=True)
            (run_path / "champion_validation.json").unlink(missing_ok=True)
        else:
            _save_checkpoint(
                champion_path, champion_genome, generation=int(champion_generation),
                seed=seed, noise_scale=noise_scale, fitness=champion_score,
                stats=champion_record or {},
                extra_metadata={
                    "checkpoint_kind": "validation_champion",
                    "validation_score": champion_score,
                    "validation_generation": champion_generation,
                    "fitness_baselines": baselines,
                    "policy_schema_version": 2,
                    "environment_version": ENVIRONMENT_VERSION,
                },
            )
            _atomic_text(
                run_path / "champion_validation.json",
                json.dumps(champion_record, separators=(",", ":"), allow_nan=False),
            )
        workers = int(workers or run_config.get("workers", 1))
        events = list(run_config.get("events", []))
        events.append({
            "type": "resume",
            "generation": saved_generation,
            "wall_time": datetime.now(timezone.utc).isoformat(),
            "note": f"exact continuation from {resume_source.name}; output files truncated through the saved generation",
        })
        run_config["events"] = events
        run_config["save_evolution_snapshots"] = bool(save_evolution_snapshots)
        run_config["best_genome_snapshot_interval"] = 1
        run_config["population_snapshot_interval"] = POPULATION_SNAPSHOT_INTERVAL
        run_config["population_snapshot_generations"] = _population_snapshot_schedule(generations)
        run_config["status"] = "training"
        _atomic_text(config_path, json.dumps(run_config, indent=2))
        print(
            f"Exact resume from generation {saved_generation} ({resume_source.name}); "
            f"continuing at {start_generation}.", flush=True,
        )
    else:
        if log_path.exists() and log_path.stat().st_size:
            raise FileExistsError(f"refusing to append a new run to existing log: {log_path}")
        baselines = load_normalization_baselines(plan_matrix_path)
        heldout_seeds = validation_seed_sets(seed, validation_overs)
        start_generation = 1
        rng = np.random.default_rng(seed)
        random.seed(seed)
        population = np.stack(
            [MLPPolicy.random(rng).get_genome() for _ in range(population_size)], axis=0
        )
        population_ids = np.arange(population_size, dtype=np.int64)
        next_genome_id = population_size
        lineage_metadata = [
            {
                "id": int(index), "parent_a": None, "parent_b": None,
                "was_elite": False, "crossover_used": False, "sigma": 0.0,
            }
            for index in population_ids
        ]
        champion_score = float("-inf")
        champion_genome = None
        champion_generation = None
        champion_record = None
        workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
        workers = max(1, int(workers))
        run_config = {
            "schema_version": 3,
            "name": run_path.name,
            "seed": seed,
            "generations": generations,
            "population_size": population_size,
            "elitism": min(ELITE_COUNT, population_size),
            "tournament_k": TOURNAMENT_K,
            "uniform_crossover_probability": 0.5,
            "mutation_rate": 0.10,
            "mutation_sigma_start": MUTATION_START,
            "mutation_sigma_end": MUTATION_END,
            "mutation_schedule": "cosine",
            "overs_per_archetype_per_generation": overs_per_archetype,
            "workers": workers,
            "noise_scale": noise_scale,
            "fitness_baselines": baselines,
            "environment_version": ENVIRONMENT_VERSION,
            "policy_schema_version": 2,
            "observation_size": 18,
            "validation_interval": VALIDATION_INTERVAL,
            "validation_overs_per_archetype": validation_overs,
            "validation_seeds": heldout_seeds,
            "training_seed_range": [0, np.iinfo(np.uint32).max],
            "resume_interval": RESUME_INTERVAL,
            "lineage": True,
            "checkpoint_generations": sorted(
                generation for generation in CHECKPOINT_GENERATIONS | {generations}
                if generation <= generations
            ),
            "population_snapshot_generations": sorted(
                _population_snapshot_schedule(generations)
            ),
            "replay_generations": sorted(
                generation for generation in REPLAY_GENERATIONS | {generations}
                if generation <= generations
            ),
            "plan_matrix_path": str(plan_matrix_path),
            "save_artifacts": save_artifacts,
            "save_evolution_snapshots": bool(save_evolution_snapshots),
            "best_genome_snapshot_interval": 1,
            "population_snapshot_interval": POPULATION_SNAPSHOT_INTERVAL,
            "population_snapshot_generations": _population_snapshot_schedule(generations),
            "events": [],
            "status": "training",
        }
        _atomic_text(config_path, json.dumps(run_config, indent=2))

    columns = _log_columns()
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    history: list[dict] = []
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    overall_started = time.perf_counter()
    last_scenarios: list[tuple[str, int]] = []

    def evaluate_population(current_population: np.ndarray, scenarios: list[tuple[str, int]]):
        payloads = [(genome, scenarios, noise_scale, baselines) for genome in current_population]
        if executor is None:
            return [_evaluate_genome_payload(payload) for payload in payloads]
        return list(executor.map(_evaluate_genome_payload, payloads, chunksize=1))

    def save_typical(current_population: np.ndarray, evaluations: list[dict], generation: int):
        if not save_artifacts:
            return
        scores = np.asarray([row["fitness"] for row in evaluations])
        order = np.argsort(scores, kind="stable")
        middle = len(order) // 2
        chosen = order[np.clip(np.arange(middle - 2, middle + 3), 0, len(order) - 1)]
        for number, index in enumerate(chosen, start=1):
            write_replay_bundle(
                run_path, run_name=run_path.name,
                label=f"gen_{generation:04d}_typical_{number:02d}",
                genome=current_population[index], generation=generation,
                fitness=float(scores[index]), noise_scale=noise_scale,
                seed=seed + generation * 100_000, overs=10, combine=True,
            )

    try:
        if not exact_resume:
            # Generation zero is the untrained random population; save its
            # median genomes and establish the generation-one lineage.
            generation_zero_scenarios = _make_scenarios(rng, overs_per_archetype)
            generation_zero_evaluations = evaluate_population(population, generation_zero_scenarios)
            generation_zero_fitness = np.asarray([item["fitness"] for item in generation_zero_evaluations])
            save_typical(population, generation_zero_evaluations, 0)
            _append_lineage(run_path / "lineage.csv.gz", 0, lineage_metadata, generation_zero_fitness)
            population, population_ids, lineage_metadata, next_genome_id = _breed_population_detailed(
                population, generation_zero_fitness, population_ids, next_genome_id,
                rng, generation=1, generations=generations,
                elite_count=min(ELITE_COUNT, population_size),
            )

        end_generation = min(
            generations,
            stop_after_generation if stop_after_generation is not None else generations,
        )
        with log_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            if write_header:
                writer.writeheader()
            for generation in range(start_generation, end_generation + 1):
                generation_started = time.perf_counter()
                scenarios = _make_scenarios(rng, overs_per_archetype)
                last_scenarios = scenarios
                evaluations = evaluate_population(population, scenarios)
                fitness = np.asarray([entry["fitness"] for entry in evaluations], dtype=float)
                best_index = int(np.argmax(fitness))
                best_stats = evaluations[best_index]
                best_genome = population[best_index].copy()

                if save_evolution_snapshots:
                    _save_best_genome_snapshot(
                        run_path,
                        generation=generation,
                        genome=best_genome,
                        genome_id=int(population_ids[best_index]),
                        fitness=float(fitness[best_index]),
                        stats=best_stats,
                    )

                if generation == 1:
                    save_typical(population, evaluations, 1)
                if generation in CHECKPOINT_GENERATIONS or generation == generations:
                    _save_checkpoint(
                        checkpoint_dir / f"gen_{generation:04d}.npz", best_genome,
                        generation=generation, seed=seed, noise_scale=noise_scale,
                        fitness=float(fitness[best_index]), stats=best_stats,
                        extra_metadata={
                            "fitness_baselines": baselines,
                            "policy_schema_version": 2,
                            "environment_version": ENVIRONMENT_VERSION,
                            "genome_id": int(population_ids[best_index]),
                        },
                    )
                population_due = (
                    (save_evolution_snapshots and generation % POPULATION_SNAPSHOT_INTERVAL == 0)
                    or (save_artifacts and generation in POPULATION_SNAPSHOT_GENERATIONS)
                )
                if population_due:
                    _save_population(
                        run_path / "populations" / f"population_{generation:04d}.npz",
                        population, generation=generation, seed=seed,
                        fitness=fitness, genome_ids=population_ids,
                    )
                if save_artifacts and generation in POPULATION_SNAPSHOT_GENERATIONS:
                    write_swarm_replay(
                        run_path, run_name=run_path.name, generation=generation,
                        population=population, noise_scale=noise_scale,
                        seed=seed + 8_000_000_000,
                    )
                if save_artifacts and generation in REPLAY_GENERATIONS:
                    write_replay_bundle(
                        run_path, run_name=run_path.name,
                        label=f"gen_{generation:04d}", genome=best_genome,
                        generation=generation, fitness=float(fitness[best_index]),
                        noise_scale=noise_scale, seed=seed + generation * 10_000_000,
                        overs=30,
                    )

                validation_record = None
                if generation % VALIDATION_INTERVAL == 0:
                    validation = _run_validation(
                        best_genome, seed_sets=heldout_seeds,
                        noise_scale=noise_scale, baselines=baselines, executor=executor,
                    )
                    validation_record = {
                        "generation": generation,
                        "validation_score": validation["score"],
                        "per_archetype": validation["per_archetype"],
                    }
                    with (run_path / "validation.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps(validation_record, separators=(",", ":"), allow_nan=False) + "\n")
                        file.flush()
                        os.fsync(file.fileno())
                    if validation["score"] > champion_score:
                        champion_score = validation["score"]
                        champion_genome = best_genome.copy()
                        champion_generation = generation
                        champion_record = validation_record
                        _atomic_text(
                            run_path / "champion_validation.json",
                            json.dumps(validation_record, separators=(",", ":"), allow_nan=False),
                        )
                        _save_checkpoint(
                            checkpoint_dir / "champion.npz", champion_genome,
                            generation=generation, seed=seed, noise_scale=noise_scale,
                            fitness=float(fitness[best_index]), stats=best_stats,
                            extra_metadata={
                                "checkpoint_kind": "validation_champion",
                                "validation_score": champion_score,
                                "validation_generation": generation,
                                "validation_normalized_scores": {
                                    name: validation["per_archetype"][name]["normalized_score"]
                                    for name in ARCHETYPE_NAMES
                                },
                                "fitness_baselines": baselines,
                                "policy_schema_version": 2,
                                "environment_version": ENVIRONMENT_VERSION,
                                "genome_id": int(population_ids[best_index]),
                            },
                        )

                next_population = population
                next_population_ids = population_ids
                next_lineage = lineage_metadata
                if generation < generations:
                    next_population, next_population_ids, next_lineage, next_genome_id = _breed_population_detailed(
                        population, fitness, population_ids, next_genome_id, rng,
                        generation=generation, generations=generations,
                        elite_count=min(ELITE_COUNT, population_size),
                    )
                seconds = time.perf_counter() - generation_started
                row = _log_row(generation, fitness, best_stats, seconds)
                if validation_record is not None:
                    row["validation_score"] = validation_record["validation_score"]
                    row["champion_validation_score"] = champion_score
                    row["champion_generation"] = champion_generation
                writer.writerow(row)
                csv_file.flush()
                os.fsync(csv_file.fileno())
                history.append(row)
                _append_lineage(run_path / "lineage.csv.gz", generation, lineage_metadata, fitness)

                score_text = ",".join(
                    f"{best_stats['by_archetype'][name]['normalized_score']:.2f}"
                    for name in ARCHETYPE_NAMES
                )
                print(
                    f"gen {generation:>4}/{generations} best={fitness[best_index]:8.3f} "
                    f"mean={np.mean(fitness):8.3f} scores={score_text} "
                    f"speed={best_stats['mean_speed_kmh']:.1f} "
                    f"sat={best_stats['action_saturation_share']:.3f} "
                    f"bouncer_no_ball={best_stats['bouncer_no_ball_share']:.3f} "
                    f"bouncers/over={best_stats['bouncers_per_over']:.2f} {seconds:6.1f}s",
                    flush=True,
                )

                if generation < generations:
                    if generation % RESUME_INTERVAL == 0:
                        elite_ids = population_ids[np.argsort(fitness)[::-1][:min(ELITE_COUNT, population_size)]]
                        _save_resume_state(
                            run_path, generation=generation, population=population,
                            fitness=fitness, population_ids=population_ids,
                            next_population=next_population,
                            next_population_ids=next_population_ids,
                            next_lineage=next_lineage, next_genome_id=next_genome_id,
                            rng=rng, champion_genome=champion_genome,
                            champion_score=champion_score,
                            champion_generation=champion_generation,
                            champion_record=champion_record, elite_ids=elite_ids,
                            sigma=mutation_sigma(generation, generations),
                            worker_seeds=scenarios,
                        )
                    population = next_population
                    population_ids = next_population_ids
                    lineage_metadata = next_lineage

        stopped_early = end_generation < generations
        if not stopped_early and champion_genome is None:
            with np.load(checkpoint_dir / f"gen_{generations:04d}.npz", allow_pickle=False) as data:
                champion_genome = data["genome"].astype(float)
                metadata = json.loads(str(data["metadata_json"].item()))
            validation = _run_validation(
                champion_genome, seed_sets=heldout_seeds, noise_scale=noise_scale,
                baselines=baselines, executor=executor,
            )
            champion_score = validation["score"]
            champion_generation = generations
            champion_record = {
                "generation": generations, "validation_score": champion_score,
                "per_archetype": validation["per_archetype"],
            }
            _atomic_text(
                run_path / "champion_validation.json",
                json.dumps(champion_record, separators=(",", ":"), allow_nan=False),
            )
            with (run_path / "validation.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(champion_record, separators=(",", ":"), allow_nan=False) + "\n")
                file.flush()
                os.fsync(file.fileno())
            _save_checkpoint(
                checkpoint_dir / "champion.npz", champion_genome,
                generation=generations, seed=seed, noise_scale=noise_scale,
                fitness=float(metadata.get("fitness", 0.0)), stats=metadata.get("stats", {}),
                extra_metadata={
                    "checkpoint_kind": "validation_champion",
                    "validation_score": champion_score,
                    "validation_generation": generations,
                    "fitness_baselines": baselines,
                    "policy_schema_version": 2,
                    "environment_version": ENVIRONMENT_VERSION,
                },
            )

        if (not stopped_early or save_artifacts) and champion_genome is not None and save_artifacts:
            write_replay_bundle(
                run_path, run_name=run_path.name, label="champion",
                genome=champion_genome, generation=champion_generation,
                fitness=champion_score, noise_scale=noise_scale,
                seed=seed + 99_000_000_000, overs=30,
            )
        run_config["status"] = "stopped_at_gate" if stopped_early else "complete"
        run_config["last_completed_generation"] = end_generation
        run_config["elapsed_seconds_this_session"] = time.perf_counter() - overall_started
        _atomic_text(config_path, json.dumps(run_config, indent=2))
        return {
            "run_dir": str(run_path), "generations": generations,
            "start_generation": start_generation,
            "generations_this_session": len(history),
            "last_completed_generation": end_generation,
            "stopped_early": stopped_early,
            "population_size": population_size, "workers": workers,
            "seconds": time.perf_counter() - overall_started,
            "champion_generation": champion_generation,
            "champion_validation_score": champion_score,
            "final_population": population.copy(),
            "history": history,
        }
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

