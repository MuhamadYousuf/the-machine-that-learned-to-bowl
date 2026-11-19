"""Batter-side data for the documentary: the batter's state on every ball of the replayed overs.

Replays store each ball's outcome but not the batter's internal state. The simulator is fully
seeded, so this script re-bowls each stored over with the same brain and seed, records the
batter's confidence, pattern reading and surprise before and after every ball, and checks every
outcome against the stored replay (any mismatch aborts).

Writes video/public/data/batter_<run>_<label>.json:
  {run, label, archetype, overs: [{over_index, seed, balls: [{ball, legal, result, runs, shot,
   confidence_before, confidence_after, pattern_before, pattern_after, surprise, wicket_prob,
   expected_length_m, length_m, category, speed_kmh, batter}]}]}

Also writes batter_pattern_demo.json: a bouncer/yorker alternation against a fresh batter in the
current (v2) world, showing the batter reading the pattern.

Run:  .venv/Scripts/python.exe scripts/export_batter_data.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataclasses import asdict  # noqa: E402

from aibowler.env import OverEnv  # noqa: E402
from export_video_data import compact_delivery  # noqa: E402
from aibowler.policy import MLPPolicy  # noqa: E402

OUT = ROOT / "video" / "public" / "data"


def champion_genome(run: str, generation: int | None) -> np.ndarray:
    run_dir = ROOT / "runs" / run
    candidates = []
    if generation is not None:
        candidates += [run_dir / "best_genomes" / f"gen_{generation:04d}.npz", run_dir / "checkpoints" / f"gen_{generation:04d}.npz"]
    candidates.append(run_dir / "checkpoints" / "champion.npz")
    for path in candidates:
        if path.exists():
            with np.load(path) as data:
                key = "genome" if "genome" in data else data.files[0]
                return np.asarray(data[key], dtype=float)
    raise FileNotFoundError(f"no genome for {run} gen {generation}")


def resimulate(run: str, label: str, env_version: int) -> dict:
    replay = json.load(gzip.open(ROOT / "runs" / run / "replays" / f"{label}.json.gz"))
    genome = champion_genome(run, replay.get("generation"))
    policy = MLPPolicy(genome)
    overs = []
    for over in replay["overs_data"]:
        env = OverEnv(noise_scale=replay["noise_scale"], environment_version=env_version)
        obs = env.reset(replay["initial_archetype"], np.random.default_rng(over["seed"]))
        balls = []
        for k, stored in enumerate(over["deliveries"]):
            before = env.state
            conf_b, pat_b = float(before.confidence), float(before.pattern_confidence)
            exp_len = float(before.expected_length_m)
            batter = env.current_archetype.name
            obs, _, done, info = env.step(policy.act(obs))
            out = info["outcome"]
            if out.result != stored["outcome"]["result"] or out.runs_conceded != stored["outcome"]["runs_conceded"]:
                raise SystemExit(f"{run}/{label} over {over['over_index']} ball {k}: re-simulation diverged "
                                 f"({out.result} vs {stored['outcome']['result']})")
            after = before if not out.is_wicket else None
            f = info["features"]
            balls.append({
                "ball": k, "legal": out.is_legal_ball, "result": out.result, "runs": out.runs_conceded, "shot": out.shot,
                "batter": batter,
                "confidence_before": conf_b, "confidence_after": None if after is None else float(after.confidence),
                "pattern_before": pat_b, "pattern_after": None if after is None else float(after.pattern_confidence),
                "surprise": float(out.surprise), "wicket_prob": float(out.wicket_prob),
                "expected_length_m": exp_len, "length_m": None if f.length_m is None else float(f.length_m),
                "category": f.length_category, "speed_kmh": float(info["release"].speed_kmh),
            })
            if done:
                break
        overs.append({"over_index": over["over_index"], "seed": over["seed"], "balls": balls})
    return {"run": run, "label": label, "archetype": replay["initial_archetype"], "generation": replay.get("generation"), "overs": overs}


def pattern_demo() -> dict:
    """Bouncer, yorker, bouncer, yorker... against a fresh technician in the v2 world (no execution noise)."""
    env = OverEnv(noise_scale=0.0, environment_version=2, batter_spread=0.0)
    env.reset("technician", np.random.default_rng(7))
    bouncer = np.array([-1.0, -1.0, 0.0, -1.0])  # 115 km/h, steepest release
    yorker = np.array([1.0, 0.33, 0.0, -1.0])     # 150 km/h, full
    balls, deliveries = [], []
    for k in range(6):
        before = env.state
        conf_b, pat_b = float(before.confidence), float(before.pattern_confidence)
        _, _, done, info = env.step(bouncer if k % 2 == 0 else yorker)
        out = info["outcome"]
        deliveries.append(compact_delivery({
            "trajectory_points": info["trajectory_points"], "event_times_s": info["event_times"], "key_points": info["key_points"],
            "features": asdict(info["features"]), "outcome": {k2: v for k2, v in asdict(out).items() if k2 != "probabilities"},
            "release": asdict(info["release"]), "batter_archetype": "technician"}))
        balls.append({"ball": k, "planned": "bouncer" if k % 2 == 0 else "yorker", "category": info["features"].length_category,
                      "result": out.result, "surprise": float(out.surprise), "confidence_before": conf_b,
                      "pattern_before": pat_b, "pattern_after": float(env.state.pattern_confidence) if not out.is_wicket else None,
                      "confidence_after": float(env.state.confidence) if not out.is_wicket else None})
        if done or out.is_wicket:
            break
    return {"note": "Alternating bouncer/yorker against a v2 batter, noise-free", "balls": balls, "deliveries": deliveries}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("ga2", f"champion_{a}", 2) for a in ("slogger", "technician", "tailender")]
    for run, label, version in jobs:
        data = resimulate(run, label, version)
        path = OUT / f"batter_{run}_{label}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        balls = [b for o in data["overs"] for b in o["balls"]]
        print(f"{path.name}: {len(data['overs'])} overs, {len(balls)} balls, all outcomes match the replay")
    demo = pattern_demo()
    deliveries = demo.pop("deliveries")
    (OUT / "batter_pattern_demo.json").write_text(json.dumps(demo, indent=1), encoding="utf-8")
    # Same shapes the renderer already reads: an overs file and a batter file (one over).
    (OUT / "overs_pattern_demo.json").write_text(json.dumps({"run": "demo", "label": "pattern", "generation": None, "archetype": "technician",
        "overs": [{"seed": 7, "summary": None, "deliveries": deliveries}]}), encoding="utf-8")
    hud = [{**b, "legal": d["legal"], "runs": d["runs"], "shot": "leave", "batter": "technician", "wicket_prob": 0.0} for b, d in zip(demo["balls"], deliveries)]
    (OUT / "batter_pattern_hud.json").write_text(json.dumps({"archetype": "technician", "overs": [{"over_index": 0, "balls": hud}]}), encoding="utf-8")
    for b in demo["balls"]:
        print(f"  ball {b['ball'] + 1}: {b['planned']:7s} -> {b['category']:8s} surprise {b['surprise']:.3f} "
              f"pattern {b['pattern_before']:.2f}->{b['pattern_after']}")


if __name__ == "__main__":
    main()
