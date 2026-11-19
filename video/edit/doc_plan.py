"""Edit planner for the two documentary cuts ("short" ~3 min, "long" ~8 min).

Reads  public/edit/<cut>/vo_timeline.json            narration timings (audio/make_vo.py)
       out/renders/**/<clip>_<fmt>.json               Hawk-Eye / brain / duel sidecars (60 fps)
       blender/shots/<shot>/shot.json                  Blender shot events (30 fps)
Writes public/edit/<cut>/plan_<fmt>.json, copies clips to public/clips/

Shots can start at `src` seconds or `sync` a clip event to a narration anchor:
  "sync": ["@end:0", "S25@Edged"]     the end of ball 0 (contact/stumps) lands on the word "Edged"
  "sync": [371, "S22"]                 clip frame 371 (at the clip's own fps) lands on line S22
Clip ids: "R1_hero/...", "brain/...", "duel/<name>" (out/renders/...), "blender/<shot>_<camera>" (out/blender/...).
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_plan as mp  # noqa: E402  (Timeline, ball_sfx, sidecar helpers)

ROOT = mp.ROOT
FPS = 60

# ---------------------------------------------------------------- clip registry

def clip_source(clip: str, fmt: str) -> Path:
    if clip.startswith("blender/"):
        name = clip.split("/", 1)[1]
        p = ROOT / "out" / "blender" / f"{name}_{fmt}.mp4"
        if not p.exists():  # portrait fallback: centre-crop the landscape render
            p = ROOT / "out" / "blender" / f"{name}_16x9.mp4"
        return p
    p = ROOT / "out" / "renders" / f"{clip}_{fmt}.mp4"
    if not p.exists():
        p = ROOT / "out" / "renders" / f"{clip}_16x9.mp4"
    return p


_dur_cache: dict[Path, float] = {}


def duration(p: Path) -> float:
    if p not in _dur_cache:
        if not p.exists():
            _dur_cache[p] = 999.0  # not rendered yet: planning still works; render is checked later
        else:
            out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                                 capture_output=True, text=True)
            _dur_cache[p] = float(out.stdout.strip() or 0)
    return _dur_cache[p]


def clip_fps(clip: str) -> int:
    return 30 if clip.startswith("blender/") else FPS


def blender_meta(clip: str) -> dict | None:
    shot = clip.split("/", 1)[1].rsplit("_", 1)[0]
    p = ROOT / "blender" / "shots" / shot / "shot.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    spec = json.loads((ROOT / "blender" / "shots" / f"{shot}.json").read_text(encoding="utf-8"))
    return {"events": d["events"], "fps": d["fps"], "spec": spec}


def events(clip: str, fmt: str) -> list[dict]:
    """Per-ball events in clip seconds: start, release, end, result, runs."""
    if clip.startswith("blender/"):
        m = blender_meta(clip)
        if not m:
            return []
        f = m["fps"]
        return [{"start": e["startFrame"] / f, "release": e["releaseFrame"] / f, "end": e["endFrame"] / f, "result": e["result"], "runs": e["runs"], "bounce": None}
                for e in m["events"]]
    meta = mp.sidecar(clip, fmt) or mp.sidecar(clip, "16x9")
    if not meta or not isinstance(meta.get("balls"), list):
        return []
    return [{"start": b.get("startFrame", b["releaseFrame"]) / FPS, "release": b["releaseFrame"] / FPS, "end": b["endFrame"] / FPS,
             "result": b["result"], "runs": b.get("runs", 0), "bounce": (b["bounceFrame"] / FPS) if b.get("bounceFrame") else None} for b in meta["balls"]]


def clip_time(clip: str, fmt: str, ref) -> float:
    if isinstance(ref, (int, float)):
        return ref / clip_fps(clip)
    m = re.match(r"@(start|release|end|bounce):(\d+)([+\-][\d.]+)?$", ref)
    if not m:
        raise ValueError(ref)
    ev = events(clip, fmt)[int(m.group(2))]
    return ev[m.group(1)] + float(m.group(3) or 0)


def ball_sounds(clip: str, fmt: str, src0: float, rate: float, t0: float, t1: float, slow: bool) -> list[dict]:
    out = []
    for e in events(clip, fmt):
        film = lambda s: t0 + (s - src0) / rate  # noqa: E731
        if e["bounce"] is not None and t0 <= film(e["bounce"]) < t1:
            out.append({"t": film(e["bounce"]), "type": "bounce_slow" if slow else "bounce"})
        t = film(e["end"])
        if not (t0 <= t < t1):
            continue
        r = e["result"]
        kind = {"bowled": "stumps", "lbw": "pad", "caught_behind": "edge", "caught_field": "edge"}.get(r, "bat" if r == "runs" else None)
        if kind:
            out.append({"t": t, "type": kind})
    return out


# ---------------------------------------------------------------- the two cuts

def B(name):
    return f"blender/{name}"


SHORT = [
    {"scene": "open", "mood": "hook", "shots": [
        {"clip": B("open_bowled_release"), "src": 0.6, "zoom": [1.08, 1.0]},
        {"clip": B("open_bowled_tele"), "at": "S02-0.2", "sync": ["@release:0", "S02-0.05"], "zoom": [1.0, 1.08]},
        {"clip": B("open_bowled_stumpsLow"), "at": "S03-0.1", "sync": ["@end:0", "S03.end+0.45"], "zoom": [1.0, 1.06]},
    ], "overlays": [{"type": "title", "at": "S03.end+0.5", "until": "scene.end", "text": "THE MACHINE THAT LEARNED TO BOWL"}],
        "sfx": [["riser", "S03.end-1.15"], ["boom", "S03.end+0.45"], ["crowd", "scene"]]},
    {"scene": "chaos", "mood": "chaos", "shots": [{"clip": "R6_chaos/gen0_broadcast", "src": 2.2, "zoom": [1.0, 1.08]}],
     "overlays": [{"type": "gen", "at": "scene", "until": "scene.end", "from": 0, "to": 0}], "sfx": []},
    {"scene": "batters", "mood": "build1", "shots": [{"clip": B("establish_wide"), "src": 0.4, "zoom": [1.0, 1.1]}],
     "overlays": [{"type": "label", "at": "S06@three", "until": "scene.end", "text": "3 BATTERS · 3 HIDDEN WEAKNESSES", "pos": "top"}], "sfx": []},
    {"scene": "slogger", "mood": "build1", "shots": [{"clip": B("slog_fours_tele"), "sync": ["@end:0", "S07@short+0.2"]}],
     "overlays": [{"type": "card", "at": "S07", "until": "scene.end", "name": "THE SLOGGER", "colour": "red",
                   "lines": ["Aggression 94 / 100", "Punishes anything full or short", "Weak: yorker on the stumps, slower ball"]}], "sfx": []},
    {"scene": "technician", "mood": "build1", "shots": [{"clip": "duel/tech_over5", "src": 2.2}],
     "overlays": [{"type": "card", "at": "S08", "until": "scene.end", "name": "THE TECHNICIAN", "colour": "blue",
                   "lines": ["Skill 91 / 100", "Leaves wide, blocks straight", "Weak: good length, moving away late"]}], "sfx": []},
    {"scene": "tailender", "mood": "build1", "shots": [{"clip": B("tail_lbw_tele"), "src": 0.4}],
     "overlays": [{"type": "card", "at": "S09", "until": "scene.end", "name": "THE TAILENDER", "colour": "yellow",
                   "lines": ["Skill 30 / 100", "Brave against the short ball", "Weak: full and straight"]}], "sfx": []},
    {"scene": "learn", "mood": "build1", "shots": [{"clip": "duel/tech_over5", "sync": ["@start:2", "S10"]}],
     "overlays": [{"type": "label", "at": "S10@ready", "until": "scene.end", "text": "CONFIDENCE RISES WITH REPETITION", "pos": "bottom"}], "sfx": []},
    {"scene": "brain", "mood": "build1", "shots": [{"clip": "brain/A1_brain_morph", "src": 2.0, "zoom": [1.0, 1.06]}],
     "overlays": [
         {"type": "stat", "at": "S11@Eighteen", "until": "S11@Four-0.1", "big": "18", "small": "INPUTS", "sub": "batter · over · last balls", "pos": "top"},
         {"type": "stat", "at": "S11@Four", "until": "scene.end", "big": "4", "small": "OUTPUTS", "pos": "bottom"},
         {"type": "chips", "at": "S11@speed", "until": "scene.end", "items": [["SPEED", "S11@speed"], ["LENGTH", "S11@length"], ["LINE", "S11@line"], ["SWING", "S11@swing"]], "pos": "bottom"}],
     "sfx": []},
    {"scene": "evolve", "mood": "build1", "shots": [
        {"clip": "R3_swarm/montage_technician_broadcast", "sync": [150, "S12@ninety+0.15"], "zoom": [1.0, 1.1]},
        {"clip": "brain/A6_family_tree", "at": "S12@survive", "src": 10.0},
        {"clip": "brain/A7a_curve_ga2", "at": "S13-0.1", "src": 2.0, "rate": 2.6}],
     "overlays": [
         {"type": "stat", "at": "S12@ninety", "until": "S12@survive-0.1", "big": "192", "small": "BRAINS", "sub": "every generation", "pos": "top"},
         {"type": "counter", "at": "S13@Fifteen", "until": "scene.end", "to": 1500, "dur": 1.4, "small": "GENERATIONS", "pos": "top"},
         {"type": "counter", "at": "S13@eighty", "until": "scene.end", "to": 82944000, "dur": 1.4, "small": "DELIVERIES BOWLED", "pos": "bottom"}],
     "sfx": []},
    {"scene": "twist", "mood": "twist", "shots": [{"clip": "brain/A7b_curve_ga1500", "src": 2.0, "rate": 2.2, "zoom": [1.0, 1.12], "grade": "red"}],
     "overlays": [{"type": "tag", "at": "S14", "until": "scene.end", "text": "VERSION 1", "color": "orange"}], "sfx": [["drop", "scene"]]},
    {"scene": "cheat", "mood": "cheat", "shots": [
        {"clip": "R5_pitchmap/ga1500_technician", "src": 2.0, "rate": 1.4, "zoom": [1.0, 1.08], "grade": "red"},
        {"clip": "brain/A2_braincam_v1_technician", "at": "S16-0.1", "src": 2.4, "grade": "red"}],
     "overlays": [
         {"type": "flash", "at": "S15@Bouncer", "text": "BOUNCER", "side": "a"}, {"type": "flash", "at": "S15@Yorker", "text": "YORKER", "side": "b"},
         {"type": "flash", "at": "S15@Bouncer#2", "text": "BOUNCER", "side": "a"}, {"type": "flash", "at": "S15@Yorker#2", "text": "YORKER", "side": "b"},
         {"type": "label", "at": "S16@Flat", "until": "scene.end", "text": "115 OR 150 KM/H · 3 BOUNCERS AN OVER", "pos": "top", "color": "orange"}],
     "sfx": [["hit", "S15@Bouncer"], ["hit", "S15@Yorker"], ["hit", "S15@Bouncer#2"], ["hit", "S15@Yorker#2"]]},
    {"scene": "cheat2", "mood": "cheat", "shots": [{"clip": "brain/A8_v1_vs_v2_technician", "src": 4.0, "zoom": [1.05, 1.15], "grade": "red"}],
     "overlays": [{"type": "stamp", "at": "S17@fool", "until": "scene.end", "text": "REWARD HACKING"}], "sfx": [["stamp", "S17@fool"]]},
    {"scene": "fix", "mood": "fix", "shots": [{"clip": "duel/pattern", "sync": ["@end:3", "S19@fourth+0.15"]}],
     "overlays": [{"type": "label", "at": "S18@patterns", "until": "scene.end", "text": "BATTERS NOW READ PATTERNS", "pos": "bottom"}], "sfx": []},
    {"scene": "fix2", "mood": "fix", "shots": [{"clip": "R2_overs/technician_broadcast", "src": 2.0, "grade": "dim"}],
     "overlays": [{"type": "rule", "at": "S20@two", "until": "scene.end", "n": "02", "text": "Max two bouncers an over", "pos": "top"}], "sfx": []},
    {"scene": "reset", "mood": "fix", "shots": [{"clip": "R6_chaos/gen0_broadcast", "src": 12.0, "zoom": [1.1, 1.0], "grade": "dim"}],
     "overlays": [{"type": "gen", "at": "S21@deleted-0.2", "until": "scene.end", "from": 0, "to": 0, "rewind": True}], "sfx": [["rewind", "S21@deleted-0.6"]]},
    {"scene": "rise", "mood": "rise", "shots": [{"clip": "R3_swarm/gen0001_vs_1500_orbit", "sync": [371, "S22-0.1"], "zoom": [1.0, 1.06]}],
     "overlays": [{"type": "gen", "at": "S22-0.1", "until": "scene.end", "from": 1, "to": 1500, "dur": 0.9}], "sfx": [["whoosh", "S22-0.3"]]},
    {"scene": "tech", "mood": "hero", "shots": [
        {"clip": "duel/tech_over5", "src": 2.3},
        {"clip": B("tech_edge_tele"), "at": "S24-0.1", "sync": ["@end:0", "S25@Edged+0.05"], "zoom": [1.0, 1.08]}],
     "overlays": [{"type": "stat", "at": "S23@seventy", "until": "S24-0.2", "big": "73%", "small": "GOOD LENGTH", "pos": "bottom"},
                  {"type": "wicket", "at": "S25@Gone", "until": "scene.end", "text": "CAUGHT BEHIND"}], "sfx": []},
    {"scene": "slog", "mood": "hero", "shots": [
        {"clip": B("slog_fours_tele"), "sync": ["@end:0", "S26@Four"]},
        {"clip": B("slog_fours_tele"), "at": "S26@Four#2-0.25", "sync": ["@end:1", "S26@Four#2"]},
        {"clip": B("slog_fours_tele"), "at": "S26@Four#3-0.25", "sync": ["@end:2", "S26@Four#3"]},
        {"clip": "duel/slog_over17", "at": "S26@confidence-0.2", "sync": ["@start:5", "S26@confidence+0.1"]},
        {"clip": B("open_bowled_stumpsLow"), "at": "S27@good-0.6", "sync": ["@end:0", "S27@good+0.25"]}],
     "overlays": [{"type": "flash", "at": "S26@Four", "text": "FOUR", "side": "a"}, {"type": "flash", "at": "S26@Four#2", "text": "FOUR", "side": "b"},
                  {"type": "flash", "at": "S26@Four#3", "text": "FOUR", "side": "a"},
                  {"type": "wicket", "at": "S27@good+0.25", "until": "scene.end", "text": "BOWLED"}], "sfx": [["crowd", "S26@Four"]]},
    {"scene": "tail", "mood": "hero", "shots": [{"clip": B("tail_lbw_tele"), "sync": ["@end:0", "S28@Trapped+0.05"], "zoom": [1.0, 1.08]}],
     "overlays": [{"type": "wicket", "at": "S28@Trapped+0.05", "until": "scene.end", "text": "LBW"}], "sfx": []},
    {"scene": "think", "mood": "think", "shots": [{"clip": "brain/A2_braincam_technician", "src": 2.6}],
     "overlays": [{"type": "label", "at": "S29@decide", "until": "scene.end", "text": "BRAIN CAM · LIVE ACTIVATIONS", "pos": "top"}], "sfx": []},
    {"scene": "result", "mood": "result", "shots": [{"clip": "@bars"}], "overlays": [], "sfx": [["boom", "S31@close"]]},
    {"scene": "outro", "mood": "outro", "shots": [
        {"clip": B("establish_runSide"), "src": 0.3, "rate": 0.8},
        {"clip": "@endcard", "at": "S33.end+0.6"}], "overlays": [], "sfx": [["boom", "S33.end+0.6"]]},
]
SHORT_RESULT = {"v2": "S30@beats", "v1": "S31@cheating", "close": "S31@close"}


LONG = [
    {"scene": "open", "mood": "hook", "shots": [
        {"clip": B("open_bowled_release"), "src": 0.3, "zoom": [1.08, 1.0]},
        {"clip": B("open_bowled_tele"), "at": "L002-0.2", "sync": ["@release:0", "L002"]},
        {"clip": B("open_bowled_stumpsLow"), "at": "L003-0.1", "sync": ["@end:0", "L003.end+0.4"]},
        {"clip": B("establish_wide"), "at": "L004-0.2", "src": 0.0, "rate": 0.7}],
     "overlays": [{"type": "title", "at": "L004.end+0.6", "until": "scene.end", "text": "THE MACHINE THAT LEARNED TO BOWL"}],
     "sfx": [["riser", "L003.end-1.1"], ["boom", "L003.end+0.4"], ["crowd", "scene"], ["boom", "L004.end+0.6"]]},
    {"scene": "c1_problem", "mood": "build1", "chapter": ["01", "THE PROBLEM"], "shots": [{"clip": B("establish_runSide"), "src": 0.0, "rate": 0.85}],
     "overlays": [{"type": "chips", "at": "L011@fast", "until": "scene.end", "items": [["SPEED", "L011@fast"], ["LENGTH", "L011@full"], ["LINE", "L011@line"], ["SWING", "L011@swings"]], "pos": "bottom"}], "sfx": []},
    {"scene": "c1_physics", "mood": "build1", "shots": [{"clip": "R1_hero/slogger_bowled_sideOn_x025", "src": 2.4}],
     "overlays": [{"type": "chips", "at": "L012@Gravity", "until": "scene.end", "items": [["GRAVITY", "L012@Gravity"], ["AIR DRAG", "L012@air"], ["SWING", "L012@seam"], ["BOUNCE", "L012@pitch"]], "pos": "bottom"}], "sfx": []},
    {"scene": "c1_noise", "mood": "build1", "shots": [{"clip": "R2_overs/technician_pitchmap", "src": 2.0}],
     "overlays": [{"type": "label", "at": "L014@wobble", "until": "scene.end", "text": "EVERY RELEASE GETS A RANDOM WOBBLE", "pos": "top"}], "sfx": []},
    {"scene": "c1_sim", "mood": "build1", "shots": [
        {"clip": "R1_hero/technician_caught_behind_broadcast_x025", "src": 2.2},
        {"clip": B("slog_fours_tele"), "at": "L015-0.1", "src": 0.5}],
     "overlays": [{"type": "label", "at": "L014b@milliseconds", "until": "L015-0.2", "text": "STEPPED EVERY 2 MS", "pos": "top"}], "sfx": []},
    {"scene": "c2_intro", "mood": "build1", "chapter": ["02", "MEET THE BATTERS"], "shots": [{"clip": B("establish_wide"), "src": 0.0}], "overlays": [], "sfx": []},
    {"scene": "c2_slogger", "mood": "build1", "shots": [
        {"clip": B("slog_fours_tele"), "sync": ["@end:0", "L021@ropes"]},
        {"clip": "R5_pitchmap/ga2_slogger", "at": "L022", "src": 6.0}],
     "overlays": [{"type": "card", "at": "L021", "until": "L022.end", "name": "THE SLOGGER", "colour": "red",
                   "lines": ["Aggression 94 / 100", "Punishes full and short", "Weak: yorker, slower ball"]}], "sfx": []},
    {"scene": "c2_technician", "mood": "build1", "shots": [
        {"clip": "duel/tech_over5", "src": 2.2},
        {"clip": "R1_hero/technician_caught_behind_sideOn_x025", "at": "L024", "src": 2.4}],
     "overlays": [{"type": "card", "at": "L023", "until": "L024.end", "name": "THE TECHNICIAN", "colour": "blue",
                   "lines": ["Skill 91 / 100", "Leaves wide, blocks straight", "Weak: good length, moving away late"]}], "sfx": []},
    {"scene": "c2_tailender", "mood": "build1", "shots": [{"clip": B("tail_lbw_tele"), "src": 0.4}],
     "overlays": [{"type": "card", "at": "L025", "until": "scene.end", "name": "THE TAILENDER", "colour": "yellow",
                   "lines": ["Skill 30 / 100", "Hops around against the short ball", "Weak: full and straight"]}], "sfx": []},
    {"scene": "c2_mind", "mood": "build1", "shots": [{"clip": "duel/tech_over5", "src": 1.6}],
     "overlays": [{"type": "label", "at": "L027@confidence", "until": "scene.end", "text": "CONFIDENCE RISES WITH REPETITION", "pos": "bottom"}], "sfx": []},
    {"scene": "c3_brain", "mood": "think", "chapter": ["03", "THE BRAIN"], "shots": [
        {"clip": "brain/A1_brain_morph", "src": 2.0},
        {"clip": "brain/A4_weight_heatmaps", "at": "L031@Four", "src": 3.0},
        {"clip": "brain/A1_brain_milestones", "at": "L032", "src": 2.0}],
     "overlays": [{"type": "stat", "at": "L030@Eighteen", "until": "L031-0.1", "big": "18", "small": "INPUTS", "pos": "top"},
                  {"type": "stat", "at": "L031@Four", "until": "L031@One-0.1", "big": "4", "small": "OUTPUTS", "pos": "top"},
                  {"type": "counter", "at": "L031@One", "until": "L032-0.1", "to": 1156, "dur": 1.2, "small": "NUMBERS", "pos": "top"}], "sfx": []},
    {"scene": "c4_evolve", "mood": "build1", "chapter": ["04", "EVOLUTION"], "shots": [{"clip": "brain/A3_behaviour_swarm", "src": 2.0}], "overlays": [], "sfx": []},
    {"scene": "c4_chaos", "mood": "chaos", "shots": [{"clip": "R6_chaos/gen0_broadcast", "src": 2.2}],
     "overlays": [{"type": "gen", "at": "scene", "until": "scene.end", "from": 0, "to": 0}], "sfx": []},
    {"scene": "c4_swarm", "mood": "build1", "shots": [{"clip": "R3_swarm/montage_technician_broadcast", "sync": [150, "L041@ninety+0.15"]}],
     "overlays": [{"type": "stat", "at": "L041@ninety", "until": "scene.end", "big": "192", "small": "BRAINS", "sub": "48 overs each", "pos": "top"}], "sfx": []},
    {"scene": "c4_tree", "mood": "build1", "shots": [{"clip": "brain/A6_family_tree", "src": 8.0}],
     "overlays": [{"type": "label", "at": "L042@best", "until": "scene.end", "text": "TOP 8 SURVIVE · THE REST ARE BRED", "pos": "bottom"}], "sfx": []},
    {"scene": "c4_curve", "mood": "build2", "shots": [{"clip": "brain/A7a_curve_ga2", "src": 2.0, "rate": 2.2}],
     "overlays": [{"type": "counter", "at": "L043@fifteen", "until": "scene.end", "to": 1500, "dur": 1.4, "small": "GENERATIONS", "pos": "top"},
                  {"type": "counter", "at": "L043@eighty", "until": "scene.end", "to": 82944000, "dur": 1.4, "small": "DELIVERIES", "pos": "bottom"}], "sfx": []},
    {"scene": "c4_montage", "mood": "build2", "shots": [
        {"clip": "R3_swarm/montage_technician_orbit", "src": 2.0},
        {"clip": "R3_swarm/montage_technician_orbit", "at": "L045@fifty-0.3", "sync": [646, "L045@fifty"]},
        {"clip": "R3_swarm/montage_technician_orbit", "at": "L045@five-0.3", "sync": [1139, "L045@five"]},
        {"clip": "R3_swarm/montage_technician_orbit", "at": "L046-0.2", "sync": [1621, "L046"]}],
     "overlays": [{"type": "gen", "at": "L044", "until": "L045@fifty-0.3", "from": 1, "to": 1},
                  {"type": "gen", "at": "L045@fifty-0.3", "until": "L045@five-0.3", "from": 50, "to": 50},
                  {"type": "gen", "at": "L045@five-0.3", "until": "L046-0.2", "from": 500, "to": 500},
                  {"type": "gen", "at": "L046-0.2", "until": "scene.end", "from": 1500, "to": 1500}], "sfx": []},
    {"scene": "c4_family", "mood": "build2", "shots": [{"clip": "brain/A6_family_tree", "src": 12.0, "rate": 0.8}],
     "overlays": [{"type": "label", "at": "L047@born", "until": "scene.end", "text": "CHAMPION · GENERATION 1,425", "pos": "bottom"}], "sfx": []},
    {"scene": "c5_rise", "mood": "twist", "chapter": ["05", "THE CHEAT"], "shots": [{"clip": "brain/A7b_curve_ga1500", "src": 2.0, "rate": 2.2, "grade": "red"}],
     "overlays": [{"type": "tag", "at": "L050", "until": "scene.end", "text": "VERSION 1", "color": "orange"}], "sfx": [["drop", "scene"]]},
    {"scene": "c5_cheat", "mood": "cheat", "shots": [
        {"clip": "R5_pitchmap/ga1500_technician", "src": 2.0, "rate": 1.4, "grade": "red"},
        {"clip": "brain/A2_braincam_v1_technician", "at": "L052-0.1", "src": 2.4, "grade": "red"}],
     "overlays": [{"type": "flash", "at": "L051@Bouncer", "text": "BOUNCER", "side": "a"}, {"type": "flash", "at": "L051@Yorker", "text": "YORKER", "side": "b"},
                  {"type": "flash", "at": "L051@Bouncer#2", "text": "BOUNCER", "side": "a"}, {"type": "flash", "at": "L051@Yorker#2", "text": "YORKER", "side": "b"},
                  {"type": "label", "at": "L052", "until": "scene.end", "text": "115 OR 150 KM/H · NOTHING IN BETWEEN", "pos": "top", "color": "orange"}],
     "sfx": [["hit", "L051@Bouncer"], ["hit", "L051@Yorker"], ["hit", "L051@Bouncer#2"], ["hit", "L051@Yorker#2"]]},
    {"scene": "c5_cheat2", "mood": "cheat", "shots": [{"clip": "R4_v1_vs_v2/v1_only_technician_sideOn", "src": 2.0, "grade": "red"}],
     "overlays": [{"type": "stat", "at": "L053@Three", "until": "scene.end", "big": "3", "small": "BOUNCERS AN OVER", "sub": "illegal", "pos": "top"}], "sfx": []},
    {"scene": "c5_why", "mood": "cheat", "shots": [{"clip": "brain/A8_v1_vs_v2_technician", "src": 3.0, "grade": "red"}],
     "overlays": [{"type": "label", "at": "L054", "until": "scene.end", "text": "SURPRISE = DISTANCE FROM EXPECTATION", "pos": "top", "color": "orange"}], "sfx": []},
    {"scene": "c5_name", "mood": "cheat", "shots": [{"clip": "R4_v1_vs_v2/technician_pitchMap", "src": 2.0, "grade": "red"}],
     "overlays": [{"type": "stamp", "at": "L057@Reward", "until": "scene.end", "text": "REWARD HACKING"}], "sfx": [["stamp", "L057@Reward"]]},
    {"scene": "c6_fix", "mood": "fix", "chapter": ["06", "REWRITING THE RULES"], "shots": [{"clip": "R2_overs/technician_broadcast", "src": 2.0, "grade": "dim"}], "overlays": [], "sfx": []},
    {"scene": "c6_pattern", "mood": "fix", "shots": [{"clip": "duel/pattern", "sync": ["@end:3", "L061@fourth+0.15"]}],
     "overlays": [{"type": "label", "at": "L061@patterns", "until": "scene.end", "text": "PATTERN READ BY BALL 4", "pos": "bottom"}], "sfx": []},
    {"scene": "c6_fade", "mood": "fix", "shots": [{"clip": "R2_overs/slogger_pitchmap", "src": 2.0}],
     "overlays": [{"type": "stat", "at": "L061b@quarter", "until": "L061b@sixth-0.1", "big": "28%", "small": "SHOCK BALL · BALL 1", "pos": "top"},
                  {"type": "stat", "at": "L061b@sixth", "until": "scene.end", "big": "2%", "small": "SAME BALL · BALL 6", "pos": "top"}], "sfx": []},
    {"scene": "c6_laws", "mood": "fix", "shots": [{"clip": "R2_overs/tailender_broadcast", "src": 2.0, "grade": "dim"}],
     "overlays": [{"type": "rule", "at": "L062@capped", "until": "scene.end", "n": "02", "text": "Max two bouncers an over", "pos": "top"}], "sfx": []},
    {"scene": "c6_reset", "mood": "fix", "shots": [{"clip": "R6_chaos/gen0_broadcast", "src": 12.0, "grade": "dim"}],
     "overlays": [{"type": "gen", "at": "L063@threw", "until": "scene.end", "from": 0, "to": 0, "rewind": True}], "sfx": [["rewind", "L063@threw-0.4"]]},
    {"scene": "c7_rise", "mood": "rise", "chapter": ["07", "THE DUEL"], "shots": [{"clip": "R3_swarm/gen0001_vs_1500_orbit", "sync": [371, "L070"]}],
     "overlays": [{"type": "gen", "at": "L070", "until": "scene.end", "from": 1, "to": 1500, "dur": 0.9}], "sfx": []},
    {"scene": "c7_tech", "mood": "hero", "shots": [
        {"clip": "duel/tech_over5", "sync": ["@end:0", "L071@Blocked"]},
        {"clip": B("tech_edge_tele"), "at": "L073-0.2", "sync": ["@end:0", "L074@Edged"]}],
     "overlays": [{"type": "wicket", "at": "L074@Edged", "until": "scene.end", "text": "CAUGHT BEHIND"}], "sfx": []},
    {"scene": "c7_slog", "mood": "hero", "shots": [
        {"clip": "duel/slog_over17", "sync": ["@end:1", "L075@dots"]},
        {"clip": B("slog_fours_tele"), "at": "L075@Four-0.3", "sync": ["@end:0", "L075@Four"]},
        {"clip": B("slog_fours_tele"), "at": "L075@Four#2-0.25", "sync": ["@end:1", "L075@Four#2"]},
        {"clip": B("slog_fours_tele"), "at": "L075@Four#3-0.25", "sync": ["@end:2", "L075@Four#3"]},
        {"clip": "duel/slog_over17", "at": "L076", "sync": ["@start:5", "L076+0.3"]},
        {"clip": B("open_bowled_tele"), "at": "L077@doesn-0.2", "sync": ["@end:0", "L077@Bowled"]},
        {"clip": B("open_bowled_stumpsLow"), "at": "L077.end+0.3", "sync": ["@end:0", "L077.end+0.9"]}],
     "overlays": [{"type": "flash", "at": "L075@Four", "text": "FOUR", "side": "a"}, {"type": "flash", "at": "L075@Four#2", "text": "FOUR", "side": "b"},
                  {"type": "flash", "at": "L075@Four#3", "text": "FOUR", "side": "a"}, {"type": "wicket", "at": "L077@Bowled", "until": "scene.end", "text": "BOWLED"}],
     "sfx": [["crowd", "L075@Four"]]},
    {"scene": "c7_tail", "mood": "hero", "shots": [
        {"clip": B("tail_lbw_tele"), "sync": ["@end:0", "L079@Trapped"]},
        {"clip": B("tail_lbw_stumpsLow"), "at": "L079.end+0.2", "sync": ["@end:0", "L079.end+0.8"]}],
     "overlays": [{"type": "wicket", "at": "L079@Trapped", "until": "scene.end", "text": "LBW"}], "sfx": []},
    {"scene": "c7_stats", "mood": "hero", "shots": [{"clip": "R5_pitchmap/ga2_slogger", "src": 8.0}],
     "overlays": [{"type": "stat", "at": "L080@six", "until": "L081-0.1", "big": "5.7", "small": "RUNS / OVER", "sub": "the Slogger still scores", "pos": "top"},
                  {"type": "stat", "at": "L081@wicket", "until": "L081@one-0.1", "big": "0.44", "small": "WICKETS / OVER", "pos": "top"},
                  {"type": "stat", "at": "L081@one", "until": "scene.end", "big": "1.2", "small": "RUNS / OVER vs TECHNICIAN", "pos": "top"}], "sfx": []},
    {"scene": "c8_cam", "mood": "think", "chapter": ["08", "INSIDE ITS HEAD"], "shots": [{"clip": "brain/A2_braincam_technician", "src": 2.6}],
     "overlays": [{"type": "label", "at": "L090@brain", "until": "scene.end", "text": "BRAIN CAM · LIVE ACTIVATIONS", "pos": "top"}], "sfx": []},
    {"scene": "c8_map", "mood": "think", "shots": [{"clip": "brain/A2_braincam_slogger", "src": 1.0}],
     "overlays": [{"type": "label", "at": "L091@confidence", "until": "scene.end", "text": "PLAN SHIFTS AS CONFIDENCE CHANGES", "pos": "top"}], "sfx": []},
    {"scene": "c8_swing", "mood": "think", "shots": [{"clip": "R1_hero/technician_caught_behind_batterEnd_x025", "src": 2.2}],
     "overlays": [{"type": "label", "at": "L092@always", "until": "scene.end", "text": "ALWAYS OUTSWING", "pos": "top"}], "sfx": []},
    {"scene": "c9_expert", "mood": "result", "chapter": ["09", "THE VERDICT"], "shots": [
        {"clip": "R2_overs/technician_pitchmap", "src": 2.0},
        {"clip": "R5_pitchmap/ga2_tailender", "at": "L096", "src": 6.0},
        {"clip": "R5_pitchmap/ga2_technician", "at": "L097", "src": 6.0}],
     "overlays": [{"type": "label", "at": "L096@threw", "until": "L097-0.1", "text": "AI: 0 BOUNCERS TO THE TAILENDER", "pos": "top"},
                  {"type": "label", "at": "L097@good", "until": "scene.end", "text": "73% GOOD LENGTH vs THE TECHNICIAN", "pos": "top"}], "sfx": []},
    {"scene": "c9_verdict", "mood": "result", "shots": [{"clip": "@bars"}], "overlays": [], "sfx": [["boom", "L102@six"]]},
    {"scene": "c10_limits", "mood": "think", "chapter": ["10", "WHAT'S NEXT"], "shots": [
        {"clip": B("establish_wide"), "src": 0.0, "rate": 0.6},
        {"clip": "R3_swarm/montage_technician_orbit", "at": "L104", "src": 26.6, "rate": 0.7}], "overlays": [], "sfx": []},
    {"scene": "c10_next", "mood": "outro", "shots": [{"clip": B("establish_runSide"), "src": 0.0, "rate": 0.7}], "overlays": [], "sfx": []},
    {"scene": "outro", "mood": "outro", "shots": [
        {"clip": B("open_bowled_release"), "src": 0.0, "rate": 0.85},
        {"clip": "@endcard", "at": "L111.end+0.6"}], "overlays": [], "sfx": [["boom", "L111.end+0.6"]]},
]
LONG_RESULT = {"v2": "L101", "v1": "L102@cheater", "close": "L102@six"}


# ---------------------------------------------------------------- build

def build(scenes: list, vo: dict, fmt: str, result: dict, end_card: float) -> dict:
    order = [s["scene"] for s in scenes]
    first = {}
    for l in vo["lines"]:
        first.setdefault(l["scene"], l)
    missing = [n for n in order if n not in first]
    if missing:
        raise SystemExit(f"scenes without narration: {missing}")
    starts = [0.0 if i == 0 else max(0.0, first[n]["start"] - (2.6 if scenes[i].get("chapter") else 0.22)) for i, n in enumerate(order)]
    total = vo["duration"] + end_card
    bounds = {n: (starts[i], starts[i + 1] if i + 1 < len(order) else total) for i, n in enumerate(order)}
    tl = mp.Timeline(vo, bounds)
    shots, overlays, sfx, sections = [], [], [], []
    for spec in scenes:
        name = spec["scene"]
        s0, s1 = bounds[name]
        sections.append({"scene": name, "mood": spec["mood"], "start": round(s0, 3), "end": round(s1, 3)})
        if spec.get("chapter"):
            n_, title = spec["chapter"]
            overlays.append({"type": "chapter", "n": n_, "title": title, "start": round(s0, 3), "end": round(min(s0 + 2.5, s1), 3)})
            sfx.append({"t": s0, "type": "boom"})
        starts_in = [s0 if "at" not in sh else tl.resolve(sh["at"], name) for sh in spec["shots"]]
        for k, sh in enumerate(spec["shots"]):
            t0 = starts_in[k]
            t1 = starts_in[k + 1] if k + 1 < len(spec["shots"]) else s1
            rate = sh.get("rate", 1.0)
            shot = {"scene": name, "clip": sh["clip"], "start": round(t0, 3), "end": round(t1, 3), "rate": rate,
                    "zoom": sh.get("zoom", [1, 1]), "grade": sh.get("grade")}
            if not sh["clip"].startswith("@"):
                if "sync" in sh:
                    ref, anchor = sh["sync"]
                    src0 = clip_time(sh["clip"], fmt, ref) - (tl.resolve(anchor, name) - t0) * rate
                else:
                    src0 = sh["src"]
                src = clip_source(sh["clip"], fmt)
                dur = duration(src)
                if src0 < 0:
                    print(f"  ! {fmt} {name}: {sh['clip']} in-point {src0:.2f}s clamped to 0")
                    src0 = 0.0
                if src0 + (t1 - t0) * rate > dur - 0.05:
                    # never run past the end of a clip (that would show black)
                    if "sync" in sh:
                        ev = clip_time(sh["clip"], fmt, sh["sync"][0])
                        anchor = tl.resolve(sh["sync"][1], name)
                        if t1 > anchor:
                            rate = min(rate, max(0.25, (dur - 0.05 - ev) / (t1 - anchor)))
                        src0 = max(0.0, ev - (anchor - t0) * rate)
                    else:
                        src0 = max(0.0, dur - 0.05 - (t1 - t0) * rate)
                    if src0 + (t1 - t0) * rate > dur - 0.05:
                        rate = max(0.2, (dur - 0.05 - src0) / (t1 - t0))
                    shot["rate"] = round(rate, 4)
                    print(f"  ~ {fmt} {name}: {sh['clip']} fitted (src {src0:.2f}s, rate {rate:.2f})")
                shot["src"] = round(src0, 3)
                shot["file"] = f"clips/{sh['clip'].replace('/', '__')}_{fmt}.mp4"
                shot["source_path"] = str(src)
                slow = "x01" in sh["clip"] or "x025" in sh["clip"]
                if sh["clip"].startswith("blender/"):
                    bm = blender_meta(sh["clip"])
                    slow = bool(bm and bm["spec"].get("slowmo"))
                sfx += ball_sounds(sh["clip"], fmt, src0, rate, t0, t1, slow)
            shots.append(shot)
            if k > 0 or name != order[0]:
                sfx.append({"t": t0, "type": "whoosh_soft" if k > 0 else "whoosh"})
        for ov in spec["overlays"]:
            o = dict(ov)
            o["start"] = round(tl.resolve(ov["at"], name), 3)
            o["end"] = round(tl.resolve(ov["until"], name), 3) if "until" in ov else round(o["start"] + 0.8, 3)
            if ov["type"] == "chips":
                o["items"] = [[label, round(tl.resolve(a, name), 3)] for label, a in ov["items"]]
                sfx += [{"t": t, "type": "blip"} for _, t in o["items"]]
            if ov["type"] in ("stat", "counter", "rule", "wicket", "stamp", "tag", "card"):
                sfx.append({"t": o["start"], "type": {"wicket": "impact", "stamp": "stamp"}.get(ov["type"], "tick")})
            overlays.append(o)
        for kind, anchor in spec["sfx"]:
            sfx.append({"t": round(tl.resolve(anchor, name), 3), "type": kind})
    marks = {k: round(tl.resolve(a, "result"), 3) for k, a in result.items()}
    sfx += [{"t": marks["v2"], "type": "rise_hit"}, {"t": marks["v1"], "type": "tick"}]
    sfx = sorted(({**s, "t": round(s["t"], 3)} for s in sfx), key=lambda s: s["t"])
    return {"format": fmt, "fps": FPS, "duration": round(total, 3), "durationInFrames": int(round(total * FPS)),
            "sections": sections, "shots": shots, "overlays": overlays, "sfx": sfx, "result": marks,
            "captions": [{"id": l["id"], "start": l["start"], "end": l["end"], "emph": l["emph"], "words": l["words"], "scene": l["scene"]} for l in vo["lines"]]}


CUTS = {"short": (SHORT, SHORT_RESULT, 5.0, ("9x16", "16x9")), "long": (LONG, LONG_RESULT, 7.0, ("16x9",))}


def main() -> None:
    copy = "--copy" in sys.argv
    for cut, (scenes, result, end_card, fmts) in CUTS.items():
        if len(sys.argv) > 1 and sys.argv[1] in CUTS and sys.argv[1] != cut:
            continue
        vo = json.loads((ROOT / "public" / "edit" / cut / "vo_timeline.json").read_text(encoding="utf-8"))
        for fmt in fmts:
            plan = build(scenes, vo, fmt, result, end_card)
            plan["cut"] = cut
            missing = sorted({s["source_path"] for s in plan["shots"] if "source_path" in s and not Path(s["source_path"]).exists()})
            if copy:
                (ROOT / "public" / "clips").mkdir(parents=True, exist_ok=True)
                for s in plan["shots"]:
                    if "source_path" in s and Path(s["source_path"]).exists():
                        dst = ROOT / "public" / s["file"]
                        src = Path(s["source_path"])
                        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                            shutil.copy2(src, dst)
            for s in plan["shots"]:
                s.pop("source_path", None)
            out = ROOT / "public" / "edit" / cut / f"plan_{fmt}.json"
            out.write_text(json.dumps(plan, indent=1), encoding="utf-8")
            print(f"{cut} {fmt}: {plan['duration']:.1f} s, {len(plan['shots'])} shots, {len(plan['overlays'])} overlays, "
                  f"{len(plan['sfx'])} sfx; not rendered yet: {len(missing)}")
            for m in missing:
                print("   waiting for", Path(m).name)


if __name__ == "__main__":
    main()
