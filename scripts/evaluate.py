"""Evaluate a saved genome and write replay JSON for later rendering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aibowler.replay import write_replay_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and save ball-by-ball replay JSON")
    parser.add_argument("--name", required=True, help="run directory name")
    parser.add_argument("--checkpoint", required=True, help="checkpoint filename or path")
    parser.add_argument("--overs", type=int, default=100, help="overs per initial archetype")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--noise", type=float, default=None, help="override checkpoint noise")
    return parser.parse_args()


def load_checkpoint(name: str, checkpoint: str):
    candidate = Path(checkpoint)
    if not candidate.is_absolute():
        candidate = ROOT / "runs" / name / "checkpoints" / candidate
    with np.load(candidate, allow_pickle=False) as data:
        genome = data["genome"].astype(float)
        metadata = json.loads(str(data["metadata_json"].item()))
    return candidate, genome, metadata


def main() -> None:
    args = parse_args()
    if args.overs < 1:
        raise SystemExit("--overs must be positive")
    checkpoint_path, genome, metadata = load_checkpoint(args.name, args.checkpoint)
    noise_scale = metadata.get("noise_scale", 0.05) if args.noise is None else args.noise
    label = "champion" if checkpoint_path.stem == "champion" else f"gen_{metadata.get('generation', 0):04d}"
    paths = write_replay_bundle(
        ROOT / "runs" / args.name,
        run_name=args.name,
        label=label,
        genome=genome,
        generation=metadata.get("generation"),
        fitness=metadata.get("fitness"),
        noise_scale=noise_scale,
        seed=args.seed,
        overs=args.overs,
    )
    for path in paths:
        print(f"Saved schema-v2 replay with <=4 ms trajectory samples: {path}", flush=True)


if __name__ == "__main__":
    main()
