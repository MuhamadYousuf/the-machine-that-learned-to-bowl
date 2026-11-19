"""Music + SFX plan for the reel, timed to the voice timeline and the same event maths as src/reel/*.tsx.

Writes public/reel/plan_9x16.json, then run:  audio/.venv/Scripts/python.exe audio/make_audio.py ../reel
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "public"
TL = json.loads((PUB / "reel" / "vo_timeline.json").read_text(encoding="utf-8"))
TAIL = 3.2
lines = {l["id"]: l for l in TL["lines"]}
SCENES = ["hook", "sees", "brain", "gen1", "cheat", "fix", "rise", "plans", "outro"]


def at(i):
    return lines[i]["start"]


def end(i):
    return lines[i]["end"]


def word(i, w, nth=0):
    n = lambda s: "".join(c for c in s.lower() if c.isalnum())
    hits = [x for x in lines[i]["words"] if n(x["w"]) == n(w)]
    return hits[min(nth, len(hits) - 1)]["s"] if hits else at(i)


def overs(name):
    return json.loads((PUB / "data" / name).read_text(encoding="utf-8"))["overs"]


def flight_end(d):
    last = d["t"][-1]
    stumps = d["result"] in ("bowled", "lbw") or d.get("wide") or (not d["legal"] and d["result"] != "no_ball")
    return min(d["ts"] if stumps else d["tc"], last)


starts = []
for i, s in enumerate(SCENES):
    if i == 0:
        starts.append(0.0)
    else:
        starts.append(max(0.0, min(l["start"] for l in TL["lines"] if l["scene"] == s) - 0.3))
total = TL["duration"] + TAIL
sc = {s: (starts[i], starts[i + 1] if i + 1 < len(starts) else total) for i, s in enumerate(SCENES)}

sfx = []


def cue(t, kind):
    sfx.append({"t": round(t, 3), "type": kind})


slog = overs("overs_ga2_champion_slogger.json")
tech = overs("overs_ga2_champion_technician.json")
tail = overs("overs_ga2_champion_tailender.json")
v1 = overs("overs_ga1500_champion_technician.json")

# hook
d = slog[9]["deliveries"][2]
hit = at("h3") - 0.15
release = hit - flight_end(d) / 0.3
cue(1.0, "crowd")
cue(release - 0.3, "whoosh")
cue(hit - 1.4, "riser")
cue(hit, "stumps")
cue(hit, "boom")
cue(end("h3") + 0.2, "impact")

# sees
for w, li in [("numbers", "s2"), ("batting", "s3"), ("confident", "s3"), ("last", "s3"), ("before", "s3")]:
    cue(word(li, w), "blip")
cue(at("s4"), "tick")
cue(at("s5"), "blip")
d = slog[11]["deliveries"][4]
rel = word("s5", "speed")
cue(rel, "whoosh")
cue(rel + flight_end(d) / 0.5, "stumps")
cue(rel + flight_end(d) / 0.5, "hit")

# brain
for k in range(4):
    cue(at("b1") + (word("b1", "weights") - at("b1")) * k / 4, "blip")
cue(word("b2", "change"), "whoosh_soft")
cue(at("b4"), "drop")

# gen 1
cue(at("g1"), "riser")
cue(word("g1", "random"), "rise_hit")
for k in range(10):
    cue(at("g2") - 0.4 + k * 0.45, "bat" if k % 3 else "crowd")
cue(word("g2", "carnage"), "stamp")
cue(word("g3", "few"), "blip")
cue(word("g4", "children"), "whoosh")
for w in ("worse", "better", "best", "repeat"):
    cue(word("g5", w), "tick")

# cheat: v1 over 1, six balls
start = sc["cheat"][0] + 0.2
for i, dd in enumerate(v1[1]["deliveries"]):
    r = start + i * 2.05 + 0.75
    if r > at("c4"):
        break
    cue(r, "whoosh_soft")
    if dd["bounce"]:
        cue(r + dd["tb"], "bounce")
    cue(r + flight_end(dd), "bat" if dd["runs"] else "pad")
cue(word("c3", "legal"), "stamp")
cue(word("c4", "cheat"), "boom")

# fix
cue(word("f1", "patterns") - 0.6, "blip")
cue(word("f1", "bouncers") - 0.3, "blip")
cue(word("f2", "again"), "rewind")
cue(word("f2", "nothing") + 0.3, "impact")

# rise
for li in ("r2", "r3", "r4"):
    cue(at(li) - 0.15, "rise_hit")

# plans
p2, p3, p4, p5 = (at(x) - 0.2 for x in ("p2", "p3", "p4", "p5"))
d = slog[17]["deliveries"][5]
cue(p2 + 0.7, "whoosh")
cue(p2 + 1.0 + flight_end(d), "stumps")
cue(p2 + 1.0 + flight_end(d), "crowd")
ds = [tech[6]["deliveries"][k] for k in (0, 2, 3, 4)]
acc = p3
for i, dd in enumerate(ds):
    r = acc + 0.55
    cue(r, "whoosh_soft")
    cue(r + (dd["tc"] if i < 3 else dd["tc"]), "edge" if i == 3 else "bat")
    acc += 0.55 + flight_end(dd) + 0.55
cue(acc - 0.55 - flight_end(ds[3]) + ds[3]["tc"] + 0.45, "crowd")
d = tail[9]["deliveries"][1]
cue(p4 + 0.9 + flight_end(d), "stumps")
cue(p4 + 0.9 + flight_end(d), "crowd")
cue(word("p4", "zero"), "blip")
cue(word("p4", "two"), "blip")
cue(word("p5", "beat") + 0.5, "boom")

# outro
for w in ("random", "generations", "deliveries"):
    cue(word("o1", w), "rise_hit")
cue(word("o3", "learned"), "boom")

moods = {"hook": "hook", "sees": "think", "brain": "build1", "gen1": "chaos", "cheat": "cheat", "fix": "fix", "rise": "build2", "plans": "result", "outro": "outro"}
sections = [{"scene": s, "mood": moods[s], "start": round(sc[s][0], 3), "end": round(sc[s][1], 3)} for s in SCENES]
plan = {"duration": round(total, 3), "fps": 30, "sections": sections, "sfx": sorted(sfx, key=lambda c: c["t"])}
(PUB / "reel" / "plan_9x16.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
print(f"plan: {total:.1f}s, {len(sfx)} cues")
