"""Command line entry point for neuroevolution training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aibowler.evolve import run_evolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cricket bowling policy with a genetic algorithm")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--name", help="new run name under runs/")
    target.add_argument("--resume", help="exactly resume runs/<run_name> from its latest durable state")
    parser.add_argument("--gens", type=int, default=1500, help="number of generations")
    parser.add_argument("--pop", type=int, default=192, help="population size")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--workers", type=int, default=None, help="worker processes; default uses up to 8 cores")
    parser.add_argument("--noise", type=float, default=0.05, help="normalized execution noise sigma")
    parser.add_argument("--eval-overs", type=int, default=16, help="evaluation overs per archetype per generation")
    parser.add_argument("--validation-overs", type=int, default=200, help="held-out validation overs per archetype")
    parser.add_argument("--skip-artifacts", action="store_true", help="skip replays and swarm snapshots (for pre-check runs)")
    parser.add_argument("--resume-from", help="warm-resume an interrupted run from a saved checkpoint")
    parser.add_argument("--stop-after-generation", type=int, help="stop after this generation while preserving an exact-resume state")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.resume or args.name
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise SystemExit("run name must be a single directory name")
    run_path = ROOT / "runs" / run_name
    exact_resume = args.resume is not None
    if exact_resume:
        config_path = run_path / "run_config.json"
        if not config_path.exists():
            raise SystemExit(f"run configuration not found: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        args.gens = int(config["generations"])
        args.pop = int(config["population_size"])
        args.seed = int(config["seed"])
        args.workers = int(config.get("workers", 1))
        args.noise = float(config["noise_scale"])
        args.eval_overs = int(config["overs_per_archetype_per_generation"])
        args.validation_overs = int(config["validation_overs_per_archetype"])
        args.skip_artifacts = not bool(config.get("save_artifacts", True))
    result = run_evolution(
        run_path,
        generations=args.gens,
        population_size=args.pop,
        seed=args.seed,
        workers=args.workers,
        noise_scale=args.noise,
        overs_per_archetype=args.eval_overs,
        validation_overs=args.validation_overs,
        save_artifacts=not args.skip_artifacts,
        resume_from=args.resume_from,
        exact_resume=exact_resume,
        stop_after_generation=args.stop_after_generation,
    )
    if exact_resume:
        print(f"Exact-resumed {run_name} at generation {result['start_generation']}.", flush=True)
    if args.resume_from:
        print(
            f"Resumed from {args.resume_from} at generation {result['start_generation']} "
            f"({result['generations_this_session']} generations remain in this session).",
            flush=True,
        )
    status = "Stopped" if result.get("stopped_early") else "Completed"
    print(
        f"{status} at generation {result.get('last_completed_generation', result['generations'])}/"
        f"{result['generations']} with population {result['population_size']} in "
        f"{result['seconds']:.1f} seconds. Champion gen {result['champion_generation']} "
        f"validation score {result['champion_validation_score']:.3f}."
    )


if __name__ == "__main__":
    main()
