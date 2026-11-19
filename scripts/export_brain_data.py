"""Export the brain/evolution data behind the 2D video clips (video/public/data/brain_*.json).

Files:
  brain_net_<run>.json      layer names + weights/biases of the best genome over the generations
  brain_cam_<run>_<arch>.json   one champion over, ball by ball: inputs, hidden activations, outputs
  brain_curves.json         learning curves (ga2 and ga1500)
  brain_tree_ga2.json       champion ancestry from the lineage log
  brain_decision.json       decision maps (gen 1 vs champion)
  brain_swarm_behaviour.json   every saved population in behaviour space (mean length, mean speed)

Run: .venv\\Scripts\\python.exe scripts\\export_brain_data.py
"""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aibowler import brain_explorer as be  # noqa: E402
from aibowler.env import OverEnv  # noqa: E402
from aibowler.policy import MLPPolicy  # noqa: E402

OUT = ROOT / "video" / "public" / "data"
ARCH = ("slogger", "technician", "tailender")
R = lambda v, d=4: round(float(v), d)  # noqa: E731


def write(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


def flat_net(bundle: dict) -> dict:
    return {
        "w": [[R(v, 3) for v in np.asarray(m).ravel()] for m in bundle["weights"]],
        "b": [[R(v, 3) for v in np.asarray(b).ravel()] for b in bundle["biases"]],
    }


def export_nets() -> None:
    for run, step in (("ga2", 5), ("ga1500", None)):
        run_dir = ROOT / "runs" / run
        names = sorted({p.stem for folder in ("best_genomes", "checkpoints") for p in (run_dir / folder).glob("gen_*.npz")})
        saved = sorted({int(n.split("_")[1]) for n in names})
        wanted = saved if step is None else sorted({g for g in saved if g % step == 0 or g in (1, 5)} | {1})
        frames = []
        for generation in wanted:
            name, _ = be.nearest_checkpoint_name(run_dir, generation)
            bundle = be.checkpoint_bundle(run_dir, name)
            frames.append({"gen": generation, **flat_net(bundle)})
        champion = be.checkpoint_bundle(run_dir, "champion")
        payload = be.network_payload(run_dir, "champion")
        write(f"brain_net_{run}.json", {
            "run": run,
            "inputs": payload["inputs"],
            "outputs": payload["outputs"],
            "sizes": [payload["input_size"], *payload["hidden_sizes"], len(payload["outputs"])],
            "championGeneration": champion["metadata"].get("validation_generation", champion["metadata"].get("generation")),
            "champion": flat_net(champion),
            "frames": frames,
        })


def pick_seed(run_dir: Path, arch: str, want_wicket: bool = True) -> int:
    """First seed whose champion over contains a wicket (a more interesting over to watch)."""
    for seed in range(1, 400):
        cam = be.brain_cam(run_dir, "champion", arch, seed)
        if not want_wicket or any(d["delivery"]["wicket"] for d in cam["deliveries"]):
            return seed
    return 1


def export_cams() -> None:
    for run in ("ga2", "ga1500"):
        run_dir = ROOT / "runs" / run
        for arch in ARCH:
            seed = pick_seed(run_dir, arch)
            cam = be.brain_cam(run_dir, "champion", arch, seed)
            balls = []
            for d in cam["deliveries"]:
                traj = d["trajectory"]
                bounce = None
                tb = d["event_times"].get("bounce_s")
                if tb is not None:
                    nearest = min(traj, key=lambda p: abs(p["t_s"] - tb))
                    bounce = [R(nearest["x_m"], 3), R(nearest["y_m"], 3)]
                balls.append({
                    "inputs": [R(v) for v in d["observation"]],
                    "h1": [R(v) for v in d["hidden1"]],
                    "h2": [R(v) for v in d["hidden2"]],
                    "raw": [R(v) for v in d["raw_outputs"]],
                    "out": [R(v) for v in d["action"]],
                    "release": {k: R(v, 3) for k, v in d["release"].items()},
                    "length": R(d["delivery"]["length_m"], 3),
                    "line": R(d["delivery"]["line_m"], 3),
                    "category": d["delivery"]["category"],
                    "result": d["delivery"]["result"],
                    "runs": d["delivery"]["runs"],
                    "wicket": d["delivery"]["wicket"],
                    "bounce": bounce,
                    "times": {k: (None if v is None else R(v, 4)) for k, v in d["event_times"].items() if isinstance(v, (int, float)) or v is None},
                })
            write(f"brain_cam_{run}_{arch}.json", {"run": run, "archetype": arch, "seed": seed, "generation": cam["checkpoint"], "balls": balls})


def read_log(run: str) -> list[dict]:
    with (ROOT / "runs" / run / "log.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def export_curves() -> None:
    payload = {}
    for run in ("ga2", "ga1500"):
        rows = read_log(run)
        validation = []
        path = ROOT / "runs" / run / "validation.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    gen = item.get("generation", item.get("gen"))
                    score = item.get("score", item.get("validation_score", item.get("normalized_score")))
                    if gen is not None and score is not None:
                        validation.append([int(gen), R(score)])
        payload[run] = {
            "gen": [int(r["gen"]) for r in rows],
            "best": [R(r["best_fitness"]) for r in rows],
            "mean": [R(r["mean_fitness"]) for r in rows],
            "speed": [R(r["mean_speed_kmh"], 2) for r in rows],
            "validation": validation,
        }
    payload["retest"] = {
        "v1": {"slogger": 0.620, "technician": 0.651, "tailender": 0.664},
        "v2": {"slogger": 1.251, "technician": 1.162, "tailender": 1.179},
    }
    write("brain_curves.json", payload)


def export_tree() -> None:
    tree = be.family_tree(ROOT / "runs" / "ga2", "champion")
    write("brain_tree_ga2.json", tree)


def export_decision() -> None:
    run_dir = ROOT / "runs" / "ga2"
    maps = {}
    seed = 1
    # x: batter confidence, y: length of the previous ball — the pair the champion reacts to most
    # (searched by the entropy of its noise-free length decisions over all input pairs).
    for label, checkpoint in (("gen1", "gen_0001"), ("champion", "champion")):
        for output in ("speed", "length_category"):
            m = be.decision_map(run_dir, checkpoint, "technician", seed, 2, 3, 12, output, resolution=40)
            maps[f"{label}_{output}"] = {"x": m["x_name"], "y": m["y_name"], "values": [[R(v, 3) for v in row] for row in m["values"]]}
    write("brain_decision.json", maps)


def _behaviour(path_text: str) -> dict:
    data = np.load(path_text)
    population = data["population"]
    fitness = data["fitness"]
    points = []
    for genome in population:
        policy = MLPPolicy(genome)
        lengths, speeds = [], []
        for index, arch in enumerate(ARCH):
            rng = np.random.default_rng(424_242 + index)
            env = OverEnv(noise_scale=0.05)
            observation = env.reset(arch, rng)
            done = False
            while not done:
                action = policy.act(observation)
                observation, _, done, info = env.step(action)
                feature = info["features"]
                lengths.append(0.0 if feature.length_m is None else float(feature.length_m))
                speeds.append(float(info["release"].speed_kmh))
        points.append([R(np.mean(lengths), 3), R(np.mean(speeds), 2)])
    generation = int(Path(path_text).stem.split("_")[1])
    order = np.argsort(-fitness)
    return {
        "gen": generation,
        "points": [points[i] for i in order],
        "fitness": [R(fitness[i]) for i in order],
    }


def export_swarm_behaviour() -> None:
    files = sorted((ROOT / "runs" / "ga2" / "populations").glob("population_*.npz"))
    with ProcessPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(_behaviour, [str(f) for f in files]))
    frames.sort(key=lambda f: f["gen"])
    write("brain_swarm_behaviour.json", {"run": "ga2", "frames": frames, "note": "each genome: mean bounce distance from the batter's stumps (m) and mean release speed over one over per batter; sorted by fitness"})


def export_all() -> None:
    export_nets()
    export_cams()
    export_curves()
    export_tree()
    export_decision()
    export_swarm_behaviour()


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    funcs = {"nets": export_nets, "cams": export_cams, "curves": export_curves, "tree": export_tree,
             "decision": export_decision, "swarm": export_swarm_behaviour}
    if only:
        funcs[only]()
    else:
        export_all()
