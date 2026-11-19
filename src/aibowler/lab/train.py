"""Evolve a bowling plan for one bowler against one batter population.

Same neuroevolution as the main project (elitism, tournament selection, uniform crossover,
cosine-annealed mutation), scaled down so a scenario trains in minutes on a laptop.
Every genome in a generation bowls the same overs (common random numbers), so fitness
differences come from the plan, not the dice.
"""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..evolve import breed_population
from ..policy import MLPPolicy
from .cards import BatterCard, BatterPopulation, BowlerCard, RewardSpec
from .labenv import bowl_overs

_CTX: dict = {}


def _init(card_json: str, params: list, bowler_json: str, reward_json: str) -> None:
    _CTX["pop"] = BatterPopulation(BatterCard.model_validate_json(card_json), params)
    _CTX["bowler"] = BowlerCard.model_validate_json(bowler_json)
    _CTX["reward"] = RewardSpec.model_validate_json(reward_json)


def _score(args) -> dict:
    genome, seeds = args
    policy = MLPPolicy(np.asarray(genome))
    r = bowl_overs(policy.act, _CTX["pop"], _CTX["bowler"], _CTX["reward"], seeds)
    r.pop("balls")
    return r


def train(card: BatterCard, population: BatterPopulation, bowler: BowlerCard, reward: RewardSpec, out_dir: str | Path,
          generations: int = 120, population_size: int = 64, overs_per_eval: int = 16, seed: int = 7,
          workers: int | None = None, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Xavier-scaled start (as in the main project): outputs begin in tanh's linear range, so mutations matter.
    genomes = np.stack([MLPPolicy.random(rng).get_genome() for _ in range(population_size)])
    val_seeds = [10_000_000 + i for i in range(64)]
    history, best_genome, best_val = [], None, -np.inf
    t0 = time.time()
    init = (card.model_dump_json(), population.params, bowler.model_dump_json(), reward.model_dump_json())
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=init) as ex:
        for gen in range(1, generations + 1):
            seeds = [int(s) for s in rng.integers(0, 2**31, size=overs_per_eval)]
            results = list(ex.map(_score, [(g, seeds) for g in genomes], chunksize=2))
            fitness = np.asarray([r["reward_per_over"] for r in results])
            top = int(np.argmax(fitness))
            row = {"gen": gen, "best": float(fitness[top]), "mean": float(fitness.mean()),
                   "wickets_per_over": results[top]["wickets_per_over"], "economy": results[top]["economy"],
                   "seconds": round(time.time() - t0, 1)}
            # Every 10 generations, re-test the top 4 on fixed held-out overs; keep the best on those.
            if gen % 10 == 0 or gen == generations:
                order = np.argsort(fitness)[::-1][:4]
                vals = list(ex.map(_score, [(genomes[i], val_seeds) for i in order]))
                k = int(np.argmax([v["reward_per_over"] for v in vals]))
                row["validation"] = vals[k]["reward_per_over"]
                if vals[k]["reward_per_over"] > best_val:
                    best_val, best_genome = vals[k]["reward_per_over"], genomes[order[k]].copy()
                log(f"gen {gen:4d}  best {row['best']:7.2f}  mean {row['mean']:7.2f}  validation {row['validation']:7.2f}  ({row['seconds']}s)")
            history.append(row)
            if gen < generations:
                genomes = breed_population(genomes, fitness, rng, generation=gen, generations=generations,
                                           elite_count=max(2, population_size // 16))
    np.savez(out / "champion.npz", genome=best_genome)
    with open(out / "log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gen", "best", "mean", "wickets_per_over", "economy", "seconds", "validation"])
        w.writeheader()
        w.writerows(history)
    summary = {"generations": generations, "population_size": population_size, "overs_per_eval": overs_per_eval,
               "seed": seed, "validation_reward_per_over": float(best_val), "seconds": round(time.time() - t0, 1)}
    (out / "train.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {"genome": best_genome, "history": history, **summary}


def load_champion(run_dir: str | Path) -> MLPPolicy:
    return MLPPolicy(np.load(Path(run_dir) / "champion.npz")["genome"])
