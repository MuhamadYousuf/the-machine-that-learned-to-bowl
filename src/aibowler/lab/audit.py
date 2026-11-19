"""Audit a trained plan before anyone trusts it.

* **Baseline** - an international-standard bowler bowling a normal, format-typical mix at the
  same batter. A plan is only interesting if it beats this.
* **Robustness** - the plan against every calibrated version of the batter and against a
  *stress* population with twice the uncertainty. A plan that only works on one version
  is a guess, not a plan.
* **Exploit detectors** - the failure modes our own v1 bowler showed (pinned outputs,
  bang-bang alternation, riding the bouncer limit) plus no-ball habits, overfitting, and
  one-trick plans.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .calibrate import MIX, reference_pool
from .cards import BatterCard, BatterPopulation, BowlerCard, RewardSpec
from .labenv import bowl_overs
from .vocab import LINES

LENGTH_ORDER = ("full_toss", "yorker", "full", "good", "short", "bouncer")


def reference_policy(fmt: str, seed: int = 0):
    pool = reference_pool()
    mix = {k: v for k, v in MIX[fmt].items() if pool.get(k)}
    cats, probs = list(mix), np.asarray(list(mix.values()))
    probs = probs / probs.sum()
    rng = np.random.default_rng(seed)

    def act(_obs):
        bucket = pool[cats[int(rng.choice(len(cats), p=probs))]]
        return bucket[int(rng.integers(0, len(bucket)))]

    return act


def nearest_line(line_m: float, hand: str) -> str:
    v = line_m if hand == "right" else -line_m
    return min(LINES, key=lambda k: abs(LINES[k][0] - v))


def summarize_plan(balls: list[dict], hand: str) -> dict:
    legal = [b for b in balls if b["legal"]]
    n = max(1, len(legal))
    lengths = Counter(b["category"] for b in legal)
    lines = Counter(nearest_line(b["line_m"], hand) for b in legal)
    by_ball = []
    for i in range(6):
        sel = [b for b in balls if b["ball"] == i]
        if sel:
            c = Counter(b["category"] for b in sel).most_common(1)[0]
            l = Counter(nearest_line(b["line_m"], hand) for b in sel).most_common(1)[0]
            by_ball.append({"ball": i + 1, "length": c[0], "length_share": c[1] / len(sel), "line": l[0],
                            "speed_kmh": float(np.mean([b["speed_kmh"] for b in sel])),
                            "wicket_prob": float(np.mean([b["wicket_prob"] for b in sel])),
                            "confidence": float(np.mean([b["confidence"] for b in sel]))})
    # Where the wickets come from: expected wickets by (length, line) cell.
    cells: dict[tuple[str, str], list[float]] = {}
    for b in legal:
        cells.setdefault((b["category"], nearest_line(b["line_m"], hand)), []).append(b["wicket_prob"])
    hot = sorted(({"length": k[0], "line": k[1], "share": len(v) / n, "wicket_prob": float(np.mean(v))}
                  for k, v in cells.items() if len(v) >= max(5, n * 0.03)), key=lambda c: -c["wicket_prob"] * c["share"])
    shots = Counter(b["shot"] for b in legal)
    outs = Counter(b["result"] for b in legal if b["wicket"])
    return {
        "length_mix": {k: lengths.get(k, 0) / n for k in LENGTH_ORDER},
        "line_mix": {k: v / n for k, v in lines.most_common()},
        "speed": {"mean": float(np.mean([b["speed_kmh"] for b in legal])) if legal else 0.0,
                  "p10": float(np.percentile([b["speed_kmh"] for b in legal], 10)) if legal else 0.0,
                  "p90": float(np.percentile([b["speed_kmh"] for b in legal], 90)) if legal else 0.0},
        "by_ball": by_ball,
        "hot_zones": hot[:5],
        "shots": {k: v / n for k, v in shots.most_common()},
        "dismissals": dict(outs),
    }


def detectors(result: dict, balls: list[dict], train_gain: float, stress_gain: float, robustness: float,
              max_bouncers: int) -> list[dict]:
    checks = []

    def add(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})

    sat = result["saturation_share"]
    add("Pinned outputs", "fail" if sat > 0.5 else "warn" if sat > 0.3 else "ok",
        f"{sat:.0%} of control outputs sit at their limits (v1's cheat was ~100%).")
    seq = []
    for seed in sorted({b["seed"] for b in balls}):
        lens = [0.0 if b["length_m"] is None else b["length_m"] for b in balls if b["seed"] == seed]  # every delivery; full toss = 0
        seq += list(zip(lens[:-1], lens[1:]))
    if len(seq) > 20:
        a, b = np.asarray(seq).T
        r = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-6 and np.std(b) > 1e-6 else 0.0
        spread = float(np.std(np.concatenate([a, b])))
        status = "fail" if (r < -0.5 and spread > 3.0) else "warn" if r < -0.35 else "ok"
        add("Bang-bang alternation", status, f"Ball-to-ball length correlation {r:+.2f}, length spread {spread:.1f} m.")
    bpo = result["bouncers_per_over"]
    add("Riding the bouncer limit", "warn" if max_bouncers and bpo >= max_bouncers - 0.2 else "ok",
        f"{bpo:.2f} counted bouncers per over (limit {max_bouncers}).")
    ext = result["extras_per_over"]
    add("Extras habit", "fail" if ext > 0.8 else "warn" if ext > 0.3 else "ok", f"{ext:.2f} wides/no-balls per over.")
    top = max(Counter(b["category"] for b in balls if b["legal"]).values(), default=0) / max(1, sum(b["legal"] for b in balls))
    add("One-trick plan", "warn" if top > 0.85 else "ok", f"Most common length is {top:.0%} of balls.")
    if train_gain > 0:
        keep = stress_gain / train_gain
        add("Overfitting", "fail" if keep < 0.3 else "warn" if keep < 0.6 else "ok",
            f"Keeps {keep:.0%} of its advantage over normal bowling when the batter is twice as uncertain.")
    else:
        add("Beats normal bowling", "fail", "The trained plan does not beat a normal international bowling mix.")
    add("Robustness", "fail" if robustness < 0.6 else "warn" if robustness < 0.8 else "ok",
        f"Beats normal bowling against {robustness:.0%} of plausible versions of this batter.")
    return checks


def audit(policy, card: BatterCard, population: BatterPopulation, bowler: BowlerCard, reward: RewardSpec,
          overs: int = 300, members: int = 40, overs_per_member: int = 25, seed: int = 123) -> dict:
    fmt = reward.format
    seeds = [seed * 1000 + i for i in range(overs)]
    plan = bowl_overs(policy, population, bowler, reward, seeds, record=True)
    base = bowl_overs(reference_policy(fmt, seed), population, bowler, reward, seeds, record=True)
    stress_card = card.model_copy(update={"uncertainty": "high"})
    stress_pop = BatterPopulation(stress_card, population.params)
    stress_seeds = [seed * 7000 + i for i in range(overs)]
    plan_stress = bowl_overs(policy, stress_pop, bowler, reward, stress_seeds)
    base_stress = bowl_overs(reference_policy(fmt, seed + 1), stress_pop, bowler, reward, stress_seeds)

    wins, gaps = 0, []
    count = min(members, len(population))
    for i in range(count):
        single = BatterPopulation(card, [population.params[i]])
        s = [seed * 31 + i * 1000 + k for k in range(overs_per_member)]
        p = bowl_overs(policy, single, bowler, reward, s)["reward_per_over"]
        b = bowl_overs(reference_policy(fmt, seed + i), single, bowler, reward, s)["reward_per_over"]
        gaps.append(p - b)
        wins += int(p > b)
    robustness = wins / max(1, count)
    train_gain = plan["reward_per_over"] - base["reward_per_over"]
    stress_gain = plan_stress["reward_per_over"] - base_stress["reward_per_over"]
    strip = lambda r: {k: v for k, v in r.items() if k != "balls"}
    return {
        "plan": strip(plan),
        "baseline": strip(base),
        "stress": {"plan": strip(plan_stress), "baseline": strip(base_stress)},
        "robustness": {"share": robustness, "members": count, "gaps": [float(g) for g in gaps]},
        "summary": summarize_plan(plan["balls"], card.hand),
        "baseline_summary": summarize_plan(base["balls"], card.hand),
        "checks": detectors(plan, plan["balls"], train_gain, stress_gain, robustness, reward.resolved()["max_bouncers"]),
        "balls": plan["balls"],
    }
