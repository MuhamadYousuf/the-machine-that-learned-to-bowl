"""Score the Step 4 17-input champion under the fixed Step 5 environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from aibowler.evolve import (
    ARCHETYPE_NAMES,
    _evaluate_episode,
    load_normalization_baselines,
    normalize_reward,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="ga1500")
    parser.add_argument("--checkpoint", default="champion")
    parser.add_argument("--overs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=202611081)
    args = parser.parse_args()
    if args.overs < 1:
        parser.error("--overs must be positive")

    checkpoint_path = ROOT / "runs" / args.run / "checkpoints" / f"{args.checkpoint}.npz"
    with np.load(checkpoint_path, allow_pickle=False) as data:
        genome = data["genome"].astype(float)
        metadata = json.loads(str(data["metadata_json"].item()))
    if genome.size != 17 * 24 + 24 + 24 * 24 + 24 + 24 * 4 + 4:
        raise SystemExit(f"expected legacy 17-input genome; found {genome.size} parameters")

    baselines = load_normalization_baselines(ROOT / "outputs" / "batter" / "plan_matrix.csv")
    noise_scale = float(metadata.get("noise_scale", 0.05))
    results = {
        "run": args.run,
        "checkpoint": args.checkpoint,
        "source_environment_version": metadata.get("environment_version", 1),
        "evaluated_environment_version": 2,
        "observation_adapter": "new 18-input environment supplies the first 17 values to the legacy policy",
        "overs_per_archetype": args.overs,
        "seed": args.seed,
        "per_archetype": {},
    }
    for index, archetype in enumerate(ARCHETYPE_NAMES):
        rewards = []
        wickets = runs = extras = 0
        for over in range(args.overs):
            episode_seed = args.seed + index * 1_000_000_000 + over * 101
            episode = _evaluate_episode(genome, archetype, episode_seed, noise_scale)
            rewards.append(float(episode["reward"]))
            wickets += int(episode["wickets"])
            runs += int(episode["runs"])
            extras += int(episode["extras"])
        reward = float(np.mean(rewards))
        baseline = baselines[archetype]
        results["per_archetype"][archetype] = {
            "normalized_score": normalize_reward(reward, baseline["random"], baseline["specialist"]),
            "reward_per_over": reward,
            "wickets_per_over": wickets / args.overs,
            "runs_per_over": runs / args.overs,
            "extras_per_over": extras / args.overs,
            "random_reward": baseline["random"],
            "specialist_reward": baseline["specialist"],
        }
        row = results["per_archetype"][archetype]
        print(
            f"{archetype:<12} normalized={row['normalized_score']:.3f} "
            f"reward={reward:.3f}/over wickets={row['wickets_per_over']:.3f}/over "
            f"runs={row['runs_per_over']:.3f}/over extras={row['extras_per_over']:.3f}/over",
            flush=True,
        )

    results["mean_normalized_score"] = float(np.mean([
        row["normalized_score"] for row in results["per_archetype"].values()
    ]))
    output = ROOT / "outputs" / "batter" / "ga1500_champion_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Mean normalized score: {results['mean_normalized_score']:.3f}")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
