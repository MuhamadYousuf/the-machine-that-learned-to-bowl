"""Render every 3D shot of the documentary with Blender, then encode each to MP4.

python blender/render_batch.py [--only name] [--skip-done]
Outputs: out/blender/<shot>_<camera>_<16x9|9x16>.mp4 (30 fps, H.264 CRF 16)
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.dirname(HERE)
BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.3\blender.exe"
KITS = {
    "slogger": {"top": [0.62, 0.06, 0.09, 1], "bottom": [0.42, 0.04, 0.06, 1]},
    "technician": {"top": [0.06, 0.16, 0.48, 1], "bottom": [0.05, 0.11, 0.34, 1]},
    "tailender": {"top": [0.82, 0.58, 0.05, 1], "bottom": [0.62, 0.42, 0.04, 1]},
}
# shot, batter, cameras for 16:9, cameras for 9:16
JOBS = [
    ("open_bowled", "slogger", ["release", "tele", "stumpsLow"], ["release", "tele", "stumpsLow"]),
    ("establish", "technician", ["wide", "runSide"], ["runSide"]),
    ("slog_fours", "slogger", ["tele"], ["tele"]),
    ("tech_edge", "technician", ["tele"], ["tele"]),
    ("tail_lbw", "tailender", ["tele", "stumpsLow"], ["tele"]),
]
SIZES = {"16x9": (1920, 1080), "9x16": (1080, 1920)}


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    skip = "--skip-done" in sys.argv
    out_root = os.path.join(VIDEO, "out", "blender")
    os.makedirs(out_root, exist_ok=True)
    log = []
    for shot, batter, cams16, cams9 in JOBS:
        if only and shot != only:
            continue
        for fmt, cams in (("16x9", cams16), ("9x16", cams9)):
            for cam in cams:
                name = f"{shot}_{cam}_{fmt}"
                mp4 = os.path.join(out_root, f"{name}.mp4")
                if skip and os.path.exists(mp4):
                    print("skip", name, flush=True)
                    continue
                frames_dir = os.path.join(out_root, "frames", name)
                w, h = SIZES[fmt]
                env = {**os.environ, "BATTER_KIT": json.dumps(KITS[batter])}
                t0 = time.time()
                r = subprocess.run([BLENDER, "-b", "-P", os.path.join(HERE, "render_shot.py"), "--",
                                    os.path.join(HERE, "shots", shot), cam, str(w), str(h), frames_dir],
                                   env=env, capture_output=True, text=True)
                tail = [l for l in r.stdout.splitlines() if l.startswith(("RENDER", "BUILD", "RELEASE"))]
                if r.returncode != 0 or not any(l.startswith("RENDER") for l in tail):
                    print("FAILED", name, r.stderr[-2000:], flush=True)
                    continue
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", os.path.join(frames_dir, "f_%05d.png"),
                                "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", mp4], check=True)
                secs = time.time() - t0
                print(f"done {name} in {secs / 60:.1f} min  {tail}", flush=True)
                log.append({"name": name, "minutes": round(secs / 60, 1)})
    json.dump(log, open(os.path.join(out_root, "render_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
