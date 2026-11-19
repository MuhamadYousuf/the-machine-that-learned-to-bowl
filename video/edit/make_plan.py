"""Edit planner: turns the voiceover timings into a frame-exact edit decision list.

Reads  public/edit/vo_timeline.json  (from audio/make_vo.py)
       out/renders/**/<clip>.json     (render sidecars: release / bounce / wicket frames)
Writes public/edit/plan_<format>.json  shots, overlays, sound-effect cues and music sections
       (both formats share timing; only the clip files differ)

Anchors are strings like "L08" (line start), "L08.end", "L20@edges" (word start),
"L12@Bouncer#2" (second occurrence), with an optional "+0.3" / "-0.2" offset.
A shot either starts its clip at `src` seconds, or `sync`s a clip frame to an anchor.
"""

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDIT = ROOT / "public" / "edit"
CLIPS = ROOT / "public" / "clips"
RENDERS = ROOT / "out" / "renders"
FPS = 60
END_CARD = 4.5

# ---------------------------------------------------------------------------
# The edit. Each scene: shots (in order) + overlays + extra sfx. Shot `at` defaults to the scene start.
# ---------------------------------------------------------------------------
SCENES = [
    {"scene": "hook", "mood": "hook", "shots": [
        {"clip": "R1_hero/slogger_bowled_stumpCam_x01", "sync": [581, "L02.end+0.35"], "zoom": [1.18, 1.0]},
    ], "overlays": [{"type": "title", "at": "L02.end+0.35", "until": "scene.end", "text": "AN AI LEARNS TO BOWL"}], "sfx": [["riser", "L02.end-1.6"], ["boom", "L02.end+0.35"]]},

    {"scene": "chaos", "mood": "chaos", "shots": [
        {"clip": "R6_chaos/gen0_broadcast", "src": 2.2, "zoom": [1.0, 1.08]},
        {"clip": "R6_chaos/gen0_batterEnd", "at": "L04@smashed-1.0", "src": 6.9, "zoom": [1.05, 1.0]},
    ], "overlays": [
        {"type": "gen", "at": "scene", "until": "scene.end", "from": 0, "to": 0},
    ], "sfx": []},

    {"scene": "brain", "mood": "build1", "shots": [
        {"clip": "brain/A1_brain_morph", "src": 2.0, "zoom": [1.0, 1.06]},
    ], "overlays": [
        {"type": "stat", "at": "L06@Eighteen", "until": "L06@Four-0.1", "big": "18", "small": "INPUTS", "sub": "batter · over · last balls", "pos": "top"},
        {"type": "stat", "at": "L06@Four", "until": "scene.end", "big": "4", "small": "OUTPUTS", "pos": "bottom"},
        {"type": "chips", "at": "L06@speed", "until": "scene.end", "items": [["SPEED", "L06@speed"], ["LENGTH", "L06@length"], ["LINE", "L06@line"], ["SWING", "L06@swing"]], "pos": "bottom"},
    ], "sfx": []},

    {"scene": "evolve", "mood": "build1", "shots": [
        {"clip": "brain/A3_behaviour_swarm", "src": 2.0, "zoom": [1.0, 1.05]},
        {"clip": "R3_swarm/montage_technician_broadcast", "at": "L08-0.1", "sync": [150, "L08@ninety+0.15"], "zoom": [1.0, 1.1]},
        {"clip": "brain/A6_family_tree", "at": "L09-0.05", "src": 10.0, "zoom": [1.0, 1.06]},
    ], "overlays": [
        {"type": "stat", "at": "L08@ninety", "until": "L08.end+0.3", "big": "192", "small": "BRAINS", "sub": "every generation", "pos": "top"},
        {"type": "label", "at": "L09@breed", "until": "scene.end", "text": "SURVIVORS → CHILDREN", "pos": "bottom"},
    ], "sfx": []},

    {"scene": "curve", "mood": "build2", "shots": [
        {"clip": "brain/A7a_curve_ga2", "src": 2.0, "rate": 2.6, "zoom": [1.0, 1.05]},
    ], "overlays": [
        {"type": "counter", "at": "L10@Fifteen", "until": "scene.end", "to": 1500, "dur": 1.6, "small": "GENERATIONS", "pos": "top"},
        {"type": "counter", "at": "L10@eighty", "until": "scene.end", "to": 82944000, "dur": 1.6, "small": "DELIVERIES BOWLED", "pos": "bottom"},
    ], "sfx": []},

    {"scene": "twist", "mood": "twist", "shots": [
        {"clip": "brain/A7b_curve_ga1500", "src": 2.0, "rate": 2.2, "zoom": [1.0, 1.12], "grade": "red"},
    ], "overlays": [{"type": "tag", "at": "L11@winning", "until": "scene.end", "text": "VERSION 1", "color": "orange"}], "sfx": [["drop", "scene"]]},

    {"scene": "cheat", "mood": "cheat", "shots": [
        {"clip": "R5_pitchmap/ga1500_technician", "src": 2.0, "rate": 1.4, "zoom": [1.0, 1.08], "grade": "red"},
        {"clip": "brain/A2_braincam_v1_technician", "at": "L13-0.1", "src": 2.4, "zoom": [1.0, 1.05], "grade": "red"},
        {"clip": "brain/A8_v1_vs_v2_technician", "at": "L14-0.1", "src": 4.0, "zoom": [1.05, 1.15], "grade": "red"},
    ], "overlays": [
        {"type": "flash", "at": "L12@Bouncer", "text": "BOUNCER", "side": "a"},
        {"type": "flash", "at": "L12@Yorker", "text": "YORKER", "side": "b"},
        {"type": "flash", "at": "L12@Bouncer#2", "text": "BOUNCER", "side": "a"},
        {"type": "flash", "at": "L12@Yorker#2", "text": "YORKER", "side": "b"},
        {"type": "label", "at": "L13@Flat", "until": "L13.end+0.2", "text": "115 OR 150 KM/H · NOTHING IN BETWEEN", "pos": "top", "color": "orange"},
        {"type": "stamp", "at": "L14@cheat", "until": "scene.end", "text": "REWARD HACKING"},
    ], "sfx": [["hit", "L12@Bouncer"], ["hit", "L12@Yorker"], ["hit", "L12@Bouncer#2"], ["hit", "L12@Yorker#2"], ["stamp", "L14@cheat"]]},

    {"scene": "fix", "mood": "fix", "shots": [
        {"clip": "R2_overs/technician_broadcast", "src": 2.0, "zoom": [1.0, 1.06], "grade": "dim"},
        {"clip": "R6_chaos/gen0_broadcast", "at": "L16-0.05", "src": 12.0, "zoom": [1.1, 1.0], "grade": "dim"},
    ], "overlays": [
        {"type": "rule", "at": "L15@read", "until": "L16-0.1", "n": "01", "text": "Batters spot patterns and get set", "pos": "top"},
        {"type": "rule", "at": "L15@two", "until": "L16-0.1", "n": "02", "text": "Max two bouncers an over", "pos": "bottom"},
        {"type": "gen", "at": "L16@nothing-0.3", "until": "scene.end", "from": 0, "to": 0, "rewind": True},
    ], "sfx": [["rewind", "L16@nothing-0.8"]]},

    {"scene": "rise", "mood": "rise", "shots": [
        {"clip": "R3_swarm/gen0001_vs_1500_orbit", "sync": [371, "L18-0.15"], "zoom": [1.0, 1.06]},
    ], "overlays": [
        {"type": "gen", "at": "scene", "until": "L18-0.15", "from": 1, "to": 1},
        {"type": "gen", "at": "L18-0.15", "until": "scene.end", "from": 1, "to": 1500, "dur": 0.9},
    ], "sfx": [["whoosh", "L18-0.3"]]},

    {"scene": "master", "mood": "hero", "shots": [
        {"clip": "R5_pitchmap/ga2_technician", "src": 4.0, "rate": 1.3, "zoom": [1.0, 1.06]},
    ], "overlays": [{"type": "tag", "at": "L19", "until": "scene.end", "text": "VERSION 2", "color": "teal"}], "sfx": []},

    {"scene": "tech", "mood": "hero", "shots": [
        {"clip": "R1_hero/technician_caught_behind_broadcast_x025", "sync": [282, "L20@edges+0.25"], "zoom": [1.0, 1.12]},
    ], "overlays": [
        {"type": "label", "at": "L20", "until": "L20@edges+0.2", "text": "vs THE TECHNICIAN", "pos": "top"},
        {"type": "wicket", "at": "L20@edges+0.25", "until": "scene.end", "text": "CAUGHT BEHIND"},
    ], "sfx": []},

    {"scene": "tail", "mood": "hero", "shots": [
        {"clip": "R1_hero/tailender_lbw_batterEnd_x025", "sync": [289, "L21@Trapped+0.1"], "zoom": [1.0, 1.1]},
    ], "overlays": [
        {"type": "label", "at": "L21", "until": "L21@Trapped", "text": "vs THE TAILENDER", "pos": "top"},
        {"type": "wicket", "at": "L21@Trapped+0.1", "until": "scene.end", "text": "LBW"},
    ], "sfx": []},

    {"scene": "slog", "mood": "hero", "shots": [
        {"clip": "R1_hero/slogger_bowled_broadcast_x025", "sync": [326, "L22@stumps"], "zoom": [1.0, 1.1]},
        {"clip": "R1_hero/slogger_bowled_stumpCam_x1", "at": "L22.end+0.15", "sync": [198, "L22.end+0.45"], "zoom": [1.15, 1.0]},
    ], "overlays": [
        {"type": "label", "at": "L22", "until": "L22@stumps-0.1", "text": "vs THE SLOGGER", "pos": "top"},
        {"type": "wicket", "at": "L22@stumps", "until": "scene.end", "text": "BOWLED"},
    ], "sfx": []},

    {"scene": "think", "mood": "think", "shots": [
        {"clip": "brain/A2_braincam_technician", "src": 2.6, "zoom": [1.0, 1.05]},
    ], "overlays": [{"type": "label", "at": "L23@think", "until": "scene.end", "text": "BRAIN CAM · LIVE ACTIVATIONS", "pos": "top"}], "sfx": []},

    {"scene": "result", "mood": "result", "shots": [
        {"clip": "@bars", "zoom": [1.0, 1.03]},
    ], "overlays": [], "sfx": [["boom", "L25@close"]]},

    {"scene": "outro", "mood": "outro", "shots": [
        {"clip": "R3_swarm/montage_technician_orbit", "src": 24.8, "rate": 0.65, "zoom": [1.0, 1.08]},
        {"clip": "@endcard", "at": "L27.end+0.6"},
    ], "overlays": [], "sfx": [["boom", "L27.end+0.6"]]},
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


class Timeline:
    def __init__(self, vo: dict, scene_bounds: dict):
        self.lines = {l["id"]: l for l in vo["lines"]}
        self.bounds = scene_bounds

    def resolve(self, anchor: str | float, scene: str) -> float:
        if isinstance(anchor, (int, float)):
            return float(anchor)
        m = re.match(r"^([^+\-]+?)([+\-][\d.]+)?$", anchor.replace(" ", ""))
        base, off = m.group(1), float(m.group(2) or 0)
        if base == "scene":
            t = self.bounds[scene][0]
        elif base == "scene.end":
            t = self.bounds[scene][1]
        elif "@" in base:
            lid, word = base.split("@")
            nth = 1
            if "#" in word:
                word, n = word.split("#")
                nth = int(n)
            target = norm(word)
            hits = [w for w in self.lines[lid]["words"] if norm(w["w"]) == target]
            if len(hits) < nth:
                # multi-word targets ("ninety two") or tokenizer splits: match by prefix
                hits = [w for w in self.lines[lid]["words"] if norm(w["w"]).startswith(target[:5])]
            if len(hits) < nth:
                raise KeyError(f"word {word!r} #{nth} not in {lid}: {[w['w'] for w in self.lines[lid]['words']]}")
            t = hits[nth - 1]["s"]
        elif base.endswith(".end"):
            t = self.lines[base[:-4]]["end"]
        else:
            t = self.lines[base]["start"]
        return t + off


def sidecar(clip: str, fmt: str) -> dict | None:
    p = RENDERS / f"{clip}_{fmt}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def clip_duration(clip: str, fmt: str) -> float:
    meta = sidecar(clip, fmt)
    return meta["durationInFrames"] / FPS


def ball_sfx(meta: dict, src0: float, rate: float, t0: float, t1: float) -> list:
    """Bounce / bat / wicket sounds for the balls visible in a shot (time in film seconds)."""
    out = []
    balls = meta.get("balls") if isinstance(meta.get("balls"), list) else None
    if not balls:
        return out
    slow = meta.get("props", {}).get("speed", 1) < 0.5

    def film(frame):
        return t0 + (frame / FPS - src0) / rate

    for b in balls:
        if b.get("bounceFrame") is not None:
            t = film(b["bounceFrame"])
            if t0 <= t < t1:
                out.append({"t": t, "type": "bounce_slow" if slow else "bounce"})
        t = film(b["endFrame"])
        if not (t0 <= t < t1):
            continue
        r = b["result"]
        if r == "bowled":
            out.append({"t": t, "type": "stumps"})
        elif r == "lbw":
            out.append({"t": t, "type": "pad"})
        elif r in ("caught_behind", "caught_field"):
            out.append({"t": t, "type": "edge"})
        elif r == "runs":
            out.append({"t": t, "type": "bat"})
    return out


def build(fmt: str, vo: dict) -> dict:
    lines = {l["id"]: l for l in vo["lines"]}
    order = [s["scene"] for s in SCENES]
    first_line = {}
    for l in vo["lines"]:
        first_line.setdefault(l["scene"], l)
    starts = []
    for i, name in enumerate(order):
        starts.append(0.0 if i == 0 else max(0.0, first_line[name]["start"] - 0.22))
    total = vo["duration"] + END_CARD
    bounds = {name: (starts[i], starts[i + 1] if i + 1 < len(order) else total) for i, name in enumerate(order)}
    tl = Timeline(vo, bounds)

    shots, overlays, sfx, sections = [], [], [], []
    for spec in SCENES:
        name = spec["scene"]
        s0, s1 = bounds[name]
        sections.append({"scene": name, "mood": spec["mood"], "start": round(s0, 3), "end": round(s1, 3)})
        starts_in = [s0 if "at" not in sh else tl.resolve(sh["at"], name) for sh in spec["shots"]]
        for k, sh in enumerate(spec["shots"]):
            t0 = starts_in[k]
            t1 = starts_in[k + 1] if k + 1 < len(spec["shots"]) else s1
            rate = sh.get("rate", 1.0)
            shot = {"scene": name, "clip": sh["clip"], "start": round(t0, 3), "end": round(t1, 3), "rate": rate,
                    "zoom": sh.get("zoom", [1, 1]), "grade": sh.get("grade")}
            if not sh["clip"].startswith("@"):
                if "sync" in sh:
                    frame, anchor = sh["sync"]
                    src0 = frame / FPS - (tl.resolve(anchor, name) - t0) * rate
                else:
                    src0 = sh["src"]
                dur = clip_duration(sh["clip"], fmt)
                if src0 < 0:
                    print(f"  ! {fmt} {name}: {sh['clip']} in-point {src0:.2f}s clamped to 0")
                    src0 = 0.0
                if src0 + (t1 - t0) * rate > dur + 1e-3:
                    print(f"  ! {fmt} {name}: {sh['clip']} runs out ({src0 + (t1 - t0) * rate:.2f}s > {dur:.2f}s); last frame holds")
                shot["src"] = round(src0, 3)
                shot["file"] = f"clips/{sh['clip'].replace('/', '__')}_{fmt}.mp4"
                meta = sidecar(sh["clip"], fmt)
                if meta:
                    sfx += ball_sfx(meta, src0, rate, t0, t1)
            shots.append(shot)
            if k > 0 or name != order[0]:
                sfx.append({"t": t0, "type": "whoosh_soft" if k > 0 else "whoosh"})
        for ov in spec["overlays"]:
            o = dict(ov)
            o["start"] = round(tl.resolve(ov["at"], name), 3)
            o["end"] = round(tl.resolve(ov.get("until", ov["at"] + "+0.55" if isinstance(ov["at"], str) else 0.55), name), 3)
            if ov["type"] == "chips":
                o["items"] = [[label, round(tl.resolve(a, name), 3)] for label, a in ov["items"]]
            if ov["type"] in ("stat", "counter", "rule", "wicket", "stamp", "tag"):
                sfx.append({"t": o["start"], "type": {"wicket": "impact", "stamp": "stamp"}.get(ov["type"], "tick")})
            if ov["type"] == "chips":
                sfx += [{"t": t, "type": "blip"} for _, t in o["items"]]
            overlays.append(o)
        for kind, anchor in spec["sfx"]:
            sfx.append({"t": round(tl.resolve(anchor, name), 3), "type": kind})

    # Result bars and end card are drawn in Remotion; give them their key times.
    result_marks = {"v2": tl.resolve("L24@beats", "result"), "v1": tl.resolve("L25@cheater", "result"), "close": tl.resolve("L25@close", "result")}
    sfx += [{"t": result_marks["v2"], "type": "rise_hit"}, {"t": result_marks["v1"], "type": "tick"}]

    sfx = sorted(({**s, "t": round(s["t"], 3)} for s in sfx), key=lambda s: s["t"])
    return {"format": fmt, "fps": FPS, "duration": round(total, 3), "durationInFrames": int(round(total * FPS)),
            "sections": sections, "shots": shots, "overlays": overlays, "sfx": sfx, "result": result_marks,
            "captions": [{"id": l["id"], "start": l["start"], "end": l["end"], "emph": l["emph"], "words": l["words"], "scene": l["scene"]} for l in vo["lines"]]}


def main() -> None:
    vo = json.loads((EDIT / "vo_timeline.json").read_text(encoding="utf-8"))
    CLIPS.mkdir(parents=True, exist_ok=True)
    for fmt in ("9x16", "16x9"):
        plan = build(fmt, vo)
        for shot in plan["shots"]:
            if "file" in shot:
                src = RENDERS / f"{shot['clip']}_{fmt}.mp4"
                dst = ROOT / "public" / shot["file"]
                if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)
        (EDIT / f"plan_{fmt}.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
        print(f"{fmt}: {plan['duration']:.1f} s, {len(plan['shots'])} shots, {len(plan['overlays'])} overlays, {len(plan['sfx'])} sfx")


if __name__ == "__main__":
    main()
