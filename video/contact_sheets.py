"""Contact sheets for every rendered clip: a 3×3 grid of evenly spaced frames plus a
"phone row" of three frames downscaled to 360 px wide (what a feed viewer sees).
Also writes out/renders/index.json listing every clip, its duration and sidecar.

Run from the video folder:  python contact_sheets.py
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RENDERS = ROOT / "out" / "renders"


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames,codec_name,profile,pix_fmt:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    return {**stream, "duration": float(data["format"]["duration"])}


def sheet(path: Path, info: dict) -> Path:
    target = path.with_suffix(".sheet.jpg")
    duration = info["duration"]
    portrait = info["height"] > info["width"]
    cell_w = 360 if portrait else 640
    cell_h = int(cell_w * info["height"] / info["width"])
    # 9 frames between the 2 s holds, plus the phone row (3 frames at 360 px wide).
    times = [2 + (duration - 4) * (i + 0.5) / 9 for i in range(9)]
    phone_w = 360
    phone_h = int(phone_w * info["height"] / info["width"])
    inputs = []
    for t in times:
        inputs += ["-ss", f"{max(0, t):.3f}", "-i", str(path)]
    filters = []
    for i in range(9):
        filters.append(f"[{i}:v]scale={cell_w}:{cell_h},setsar=1[c{i}]")
    filters.append("".join(f"[c{i}]" for i in range(9)) + "xstack=inputs=9:layout=" + "|".join(
        f"{(i % 3) * cell_w}_{(i // 3) * cell_h}" for i in range(9)) + "[grid]")
    for k, i in enumerate((1, 4, 7)):
        filters.append(f"[{i}:v]scale={phone_w}:{phone_h},setsar=1,pad={cell_w}:{phone_h}:(ow-iw)/2:0:color=0x060a13[p{k}]")
    filters.append("[p0][p1][p2]hstack=inputs=3[phone]")
    filters.append("[grid][phone]vstack=inputs=2[out]")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(filters),
         "-map", "[out]", "-frames:v", "1", "-q:v", "3", str(target)],
        check=True,
    )
    return target


def main() -> None:
    index = []
    for path in sorted(RENDERS.rglob("*.mp4")):
        info = probe(path)
        sheet_path = sheet(path, info)
        sidecar = path.with_suffix(".json")
        index.append({
            "clip": str(path.relative_to(RENDERS)).replace("\\", "/"),
            "pass": path.parent.name,
            "width": info["width"],
            "height": info["height"],
            "fps": info["r_frame_rate"],
            "codec": f"{info['codec_name']} {info.get('profile', '')} {info['pix_fmt']}",
            "duration_s": round(info["duration"], 3),
            "sidecar": str(sidecar.relative_to(RENDERS)).replace("\\", "/") if sidecar.exists() else None,
            "contact_sheet": str(sheet_path.relative_to(RENDERS)).replace("\\", "/"),
        })
        print(f"{index[-1]['clip']}: {index[-1]['duration_s']} s")
    (RENDERS / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"{len(index)} clips indexed")


if __name__ == "__main__":
    main()
