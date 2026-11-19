"""Export compact JSON data for the Remotion video project (video/public/data).

The Remotion side (video/src) renders the Hawk-Eye replays and the brain
visualisations. Python stays the single source of truth for the simulation;
this script only reshapes saved replays, checkpoints and logs into small,
frame-friendly JSON files.

Usage:
    .venv\\Scripts\\python.exe scripts\\export_video_data.py            # everything
    .venv\\Scripts\\python.exe scripts\\export_video_data.py --only replays
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "video" / "public" / "data"
ARCHETYPES = ("slogger", "technician", "tailender")
SWARM_GENS = (1, 10, 50, 250, 500, 1000, 1500)


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def write(name: str, payload: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


def r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def compact_delivery(delivery: dict[str, Any], stride: int = 1) -> dict[str, Any]:
    """Reduce one replay delivery to what the renderer needs.

    Trajectory points are 4 ms apart. The bounce point is inserted exactly so
    that linear interpolation never cuts the corner at the bounce; between
    samples the chord error is below 0.1 mm (gravity sag over 4 ms).
    """
    points = delivery["trajectory_points"]
    events = delivery["event_times_s"]
    bounce = delivery["key_points"].get("bounce_m")
    bounce_t = events.get("bounce_s")
    samples = [(p["t_s"], p["x_m"], p["y_m"], p["z_m"]) for p in points[::stride]]
    if points and samples[-1][0] != points[-1]["t_s"]:
        p = points[-1]
        samples.append((p["t_s"], p["x_m"], p["y_m"], p["z_m"]))
    if bounce is not None and bounce_t is not None:
        samples = [s for s in samples if abs(s[0] - bounce_t) > 1e-6]
        samples.append((bounce_t, bounce[0], bounce[1], max(0.0, bounce[2])))
        samples.sort(key=lambda s: s[0])
    features = delivery["features"]
    outcome = delivery["outcome"]
    release = delivery["release"]
    return {
        "t": [r(s[0], 5) for s in samples],
        "x": [r(s[1]) for s in samples],
        "y": [r(s[2]) for s in samples],
        "z": [r(s[3]) for s in samples],
        "bounce": None if bounce is None else [r(v) for v in bounce],
        "tb": r(bounce_t, 5),
        "tc": r(events.get("crease_s"), 5),
        "ts": r(events.get("stumps_s"), 5),
        "result": outcome["result"],
        "runs": outcome.get("runs_conceded", 0),
        "wicket": bool(outcome.get("is_wicket")),
        "legal": bool(outcome.get("is_legal_ball", True)),
        "shot": outcome.get("shot"),
        "shotAngle": r(outcome.get("shot_angle_deg"), 2),
        "shotDist": r(outcome.get("shot_distance_m"), 2),
        "length": r(features.get("length_m"), 3),
        "line": r(features.get("line_m"), 3),
        "category": features.get("length_category"),
        "hitsStumps": bool(features.get("hits_stumps")),
        "wide": bool(features.get("is_wide")),
        "beamer": bool(features.get("is_beamer")),
        "speed": r(release.get("speed_kmh"), 2),
        "elev": r(release.get("elevation_deg"), 3),
        "azim": r(release.get("azimuth_deg"), 3),
        "swing": r(release.get("swing"), 3),
        "batter": delivery.get("batter_archetype") or delivery.get("sampled_batter"),
    }


def export_overs(run: str, label: str, overs: int = 30) -> None:
    for archetype in ARCHETYPES:
        path = ROOT / "runs" / run / "replays" / f"{label}_{archetype}.json.gz"
        replay = load(path)
        payload = {
            "run": run,
            "label": label,
            "generation": replay.get("generation"),
            "archetype": archetype,
            "overs": [
                {
                    "seed": over.get("seed"),
                    "summary": over.get("summary"),
                    "deliveries": [compact_delivery(d) for d in over["deliveries"]],
                }
                for over in replay["overs_data"][:overs]
            ],
        }
        write(f"overs_{run}_{label}_{archetype}.json", payload)


def export_typical(run: str = "ga2") -> None:
    """Generation-0 random bowlers, shaped like an overs file (one over per genome/batter)."""
    overs = []
    for index in range(1, 6):
        path = ROOT / "runs" / run / "replays" / f"gen_0000_typical_{index:02d}.json.gz"
        if not path.exists():
            continue
        replay = load(path)
        bundles = replay["archetypes"].values() if "archetypes" in replay else [replay]
        for bundle in bundles:
            for over in bundle["overs_data"][:1]:
                overs.append({"seed": over.get("seed"), "deliveries": [compact_delivery(d) for d in over["deliveries"]]})
    write(f"overs_{run}_typical_gen0.json", {"run": run, "label": "typical", "generation": 0, "archetype": "mixed", "overs": overs})


def export_swarms(run: str = "ga2") -> None:
    for generation in SWARM_GENS:
        path = ROOT / "runs" / run / "replays" / f"swarm_gen_{generation:04d}.json.gz"
        if not path.exists():
            print(f"missing {path}")
            continue
        replay = load(path)
        import numpy as np  # noqa: PLC0415

        population = ROOT / "runs" / run / "populations" / f"population_{generation:04d}.npz"
        fitness = np.load(population)["fitness"] if population.exists() else None
        rows = []
        for delivery in replay["deliveries"]:
            row = compact_delivery(delivery, stride=2)
            row["genome"] = delivery.get("genome_index")
            row["reward"] = r(delivery.get("reward"), 4)
            if fitness is not None and row["genome"] is not None:
                row["fitness"] = r(float(fitness[row["genome"]]), 4)
            rows.append(row)
        write(f"swarm_{run}_{generation:04d}.json", {"run": run, "generation": generation, "deliveries": rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["replays", "swarm", "brain"], default=None)
    args = parser.parse_args()
    if args.only in (None, "replays"):
        export_overs("ga2", "champion")
        export_overs("ga1500", "champion")
        export_typical("ga2")
    if args.only in (None, "swarm"):
        export_swarms("ga2")
    if args.only in (None, "brain"):
        from export_brain_data import export_all as export_brain  # noqa: PLC0415

        export_brain()


if __name__ == "__main__":
    main()
