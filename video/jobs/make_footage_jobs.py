"""Generate jobs/footage.json: every Hawk-Eye footage pass (R1–R6) in both formats.

Run: python jobs/make_footage_jobs.py   (from the video folder), then
     node render.mjs jobs/footage.json --skip-done
"""

import json
from pathlib import Path

FORMATS = ("16x9", "9x16")
ARCH = ("slogger", "technician", "tailender")
# Hero wicket balls from the ga2 champion replays: (over index, ball index, result).
HERO = {
    "slogger": (9, 2, "bowled"),
    "technician": (2, 3, "caught_behind"),
    "tailender": (11, 3, "lbw"),
}
jobs = []


def add(pass_name, name, composition, props, formats=FORMATS):
    for fmt in formats:
        jobs.append({
            "id": f"{pass_name}_{name}_{fmt}",
            "composition": f"{composition}-{fmt}",
            "props": props,
            "out": f"out/renders/{pass_name}/{name}_{fmt}.mp4",
        })


def over_props(file, over, balls=(), count=1, camera="broadcast", speed=1, accumulate=True, band=False):
    return {"file": file, "overIndex": over, "overCount": count, "balls": list(balls), "camera": camera,
            "speed": speed, "accumulate": accumulate, "goodLengthBand": band}


# R1 hero wicket deliveries, several cameras and speeds.
for arch, (over, ball, result) in HERO.items():
    file = f"overs_ga2_champion_{arch}.json"
    for camera, speeds in (("broadcast", (1, 0.25)), ("stumpCam", (1, 0.1)), ("batterEnd", (0.25,)), ("sideOn", (0.25,))):
        for speed in speeds:
            tag = f"{arch}_{result}_{camera}_x{str(speed).replace('.', '')}"
            add("R1_hero", tag, "Over", over_props(file, over, [ball], camera=camera, speed=speed, accumulate=False))

# R2 one full champion over per batter: broadcast with accumulating trails + pitch map.
for arch in ARCH:
    file = f"overs_ga2_champion_{arch}.json"
    add("R2_overs", f"{arch}_broadcast", "Over", over_props(file, 0))
    add("R2_overs", f"{arch}_pitchmap", "Over", over_props(file, 0, camera="pitchMap", band=True))

# R3 the swarm: each saved generation alone, plus the evolution montage (broadcast + orbit).
gens = [1, 10, 50, 250, 500, 1000, 1500]
for g in gens:
    add("R3_swarm", f"gen{g:04d}_technician_broadcast", "Swarm",
        {"generations": [g], "archetype": "technician", "camera": "broadcast", "speed": 0.5, "orbit": False})
for arch in ARCH:
    add("R3_swarm", f"montage_{arch}_broadcast", "Swarm",
        {"generations": gens, "archetype": arch, "camera": "broadcast", "speed": 0.5, "orbit": False})
add("R3_swarm", "montage_technician_orbit", "Swarm",
    {"generations": gens, "archetype": "technician", "camera": "broadcast", "speed": 0.5, "orbit": True})
add("R3_swarm", "gen0001_vs_1500_orbit", "Swarm",
    {"generations": [1, 1500], "archetype": "technician", "camera": "broadcast", "speed": 0.5, "orbit": True})

# R4 v1 (loophole) vs v2, split screen: one over per batter + 3 continuous overs vs the technician.
for arch in ARCH:
    for camera in ("pitchMap", "broadcast"):
        add("R4_v1_vs_v2", f"{arch}_{camera}", "Compare", {
            "left": {"file": f"overs_ga1500_champion_{arch}.json", "overIndex": 0},
            "right": {"file": f"overs_ga2_champion_{arch}.json", "overIndex": 0},
            "overCount": 1, "camera": camera, "speed": 1, "goodLengthBand": camera == "pitchMap"})
add("R4_v1_vs_v2", "technician_pitchMap_3overs", "Compare", {
    "left": {"file": "overs_ga1500_champion_technician.json", "overIndex": 0},
    "right": {"file": "overs_ga2_champion_technician.json", "overIndex": 0},
    "overCount": 3, "camera": "pitchMap", "speed": 1, "goodLengthBand": True})
# v1 alone, one over per batter (side-on shows the bouncer / yorker heights).
for arch in ARCH:
    add("R4_v1_vs_v2", f"v1_only_{arch}_sideOn", "Over", over_props(f"overs_ga1500_champion_{arch}.json", 0, camera="sideOn"),
        formats=("16x9",))

# R5 pitch-map build-up: 30 overs per batter, v2 champion and v1 champion.
for run in ("ga2", "ga1500"):
    for arch in ARCH:
        add("R5_pitchmap", f"{run}_{arch}", "PitchBuild",
            {"file": f"overs_{run}_champion_{arch}.json", "overs": 30, "ballsPerSecond": 8, "colorBy": "outcome"})

# R6 chaos at generation 0: random brains, 3 overs, broadcast + orbit-like side view.
add("R6_chaos", "gen0_broadcast", "Over", over_props("overs_ga2_typical_gen0.json", 0, count=3))
add("R6_chaos", "gen0_batterEnd", "Over", over_props("overs_ga2_typical_gen0.json", 3, count=2, camera="batterEnd"))

Path(__file__).with_name("footage.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")
print(f"{len(jobs)} jobs")
