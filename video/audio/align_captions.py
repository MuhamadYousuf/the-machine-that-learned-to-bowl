"""Caption words = the script's own words, timed by Whisper.

Whisper's transcript can differ from the script ("1500" vs "fifteen hundred", "bounces" vs "bouncers").
We expand both into normalised tokens, match them with difflib, and give each script word the time of its
matched tokens (unmatched words are interpolated between neighbours). Rewrites public/reel/vo_timeline.json.
"""

import difflib
import json
import re
from pathlib import Path

from num2words import num2words

ROOT = Path(__file__).resolve().parents[1]
TLP = ROOT / "public" / "reel" / "vo_timeline.json"
SCRIPT = json.loads((ROOT / "reel" / "script.json").read_text(encoding="utf-8"))


def toks(w):
    w = w.lower().replace("-", " ").replace(",", "")
    out = []
    for p in re.sub(r"[^a-z0-9' ]", " ", w).split():
        if p.isdigit():
            out += re.sub(r"[^a-z ]", " ", num2words(int(p))).split()
        else:
            out.append(p)
    return out


tl = json.loads(TLP.read_text(encoding="utf-8"))
texts = {l["id"]: l["text"] for l in SCRIPT["lines"]}
for line in tl["lines"]:
    hyp = line["words"]
    # hypothesis tokens, each with a time slice of its source word
    ht, hs, he = [], [], []
    for w in hyp:
        ts = toks(w["w"]) or ["_"]
        for k, tk in enumerate(ts):
            ht.append(tk)
            hs.append(w["s"] + (w["e"] - w["s"]) * k / len(ts))
            he.append(w["s"] + (w["e"] - w["s"]) * (k + 1) / len(ts))
    words = texts[line["id"]].split()
    st, owner = [], []
    for i, w in enumerate(words):
        for tk in toks(w) or ["_"]:
            st.append(tk)
            owner.append(i)
    sm = difflib.SequenceMatcher(a=st, b=ht, autojunk=False)
    tok_s = [None] * len(st)
    tok_e = [None] * len(st)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            tok_s[a + k] = hs[b + k]
            tok_e[a + k] = he[b + k]
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "replace" and b1 > b0:
            span_s, span_e = hs[b0], he[b1 - 1]
            for k in range(a0, a1):
                tok_s[k] = span_s + (span_e - span_s) * (k - a0) / (a1 - a0)
                tok_e[k] = span_s + (span_e - span_s) * (k - a0 + 1) / (a1 - a0)
    out = []
    for i, w in enumerate(words):
        idx = [k for k, o in enumerate(owner) if o == i]
        ss = [tok_s[k] for k in idx if tok_s[k] is not None]
        ee = [tok_e[k] for k in idx if tok_e[k] is not None]
        out.append({"w": w, "s": min(ss) if ss else None, "e": max(ee) if ee else None})
    # interpolate gaps
    t0, t1 = line["start"], line["end"]
    for i, w in enumerate(out):
        if w["s"] is None:
            prev = next((out[j]["e"] for j in range(i - 1, -1, -1) if out[j]["e"] is not None), t0)
            nxt = next((out[j]["s"] for j in range(i + 1, len(out)) if out[j]["s"] is not None), t1)
            w["s"], w["e"] = prev, max(prev + 0.05, nxt)
    for w in out:
        w["s"] = round(w["s"], 3)
        w["e"] = round(max(w["e"], w["s"] + 0.05), 3)
    line["words"] = out
    line["text"] = texts[line["id"]]
TLP.write_text(json.dumps(tl, indent=1), encoding="utf-8")
print("captions aligned to script wording")
for l in tl["lines"]:
    if l["id"] in ("c3", "f1", "o1", "b1"):
        print(l["id"], [(w["w"], w["s"]) for w in l["words"]])
