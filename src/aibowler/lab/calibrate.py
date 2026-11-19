"""Calibration gate: make the simulated batter reproduce the real one before training against him.

Method (approximate Bayesian computation, rejection form):
1. Draw candidate parameter sets around the card (skill, aggression, starting confidence,
   overall weakness and strength scales).
2. Bowl each candidate a few hundred balls of *reference bowling*: a format-typical mix of
   lengths from an international-standard bowler. Real stats were made against real bowling,
   so the sim has to face something comparable, not our trained AI.
3. Keep candidates whose strike rate, dot %, boundary %, balls per dismissal and dismissal
   mix all land within tolerance of the real profile. Those survivors are the batter
   *population* that training and robustness checks use.
4. Gate: at least ``min_accepted`` survivors, then check the population against the
   held-out (most recent) matches.

If the gate fails, the analyst gets the closest candidates and the reason, and training
should not be trusted.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..batter import BatterState, DeliveryFeatures
from ..outcome import resolve
from ..physics import PitchConditions, release_from_action, simulate_delivery
from .cards import BatterCard, BatterPopulation, BowlerCard
from .cricsheet import Profile
from .labenv import map_command

STATS = ("strike_rate", "dot_pct", "boundary_pct", "balls_per_dismissal", "caught_share")
# How far off a stat may be and still count as "reproduced". Relative stats use log ratios.
TOLERANCE = {"strike_rate": ("rel", 0.12), "dot_pct": ("abs", 6.0), "boundary_pct": ("abs", 3.0),
             "balls_per_dismissal": ("rel", 0.35), "caught_share": ("abs", 0.18)}

MIX = {  # share of deliveries by length, typical of each format (approximate)
    "t20": {"yorker": 0.08, "full": 0.25, "good": 0.33, "short": 0.22, "bouncer": 0.08, "full_toss": 0.04},
    "odi": {"yorker": 0.05, "full": 0.25, "good": 0.42, "short": 0.20, "bouncer": 0.05, "full_toss": 0.03},
    "test": {"yorker": 0.02, "full": 0.22, "good": 0.52, "short": 0.18, "bouncer": 0.05, "full_toss": 0.01},
}
REFERENCE_BOWLER = BowlerCard(name="Reference international seamer", min_kmh=122, max_kmh=145, accuracy="international", swing="some")

_POOL: dict[str, list[np.ndarray]] | None = None


def reference_pool(size: int = 3000, seed: int = 99) -> dict[str, list[np.ndarray]]:
    """Random policy actions bucketed by the length they produce (noise-free), for resampling to a mix."""
    global _POOL
    if _POOL is not None:
        return _POOL
    rng = np.random.default_rng(seed)
    pool: dict[str, list[np.ndarray]] = {}
    for _ in range(size):
        a = rng.uniform(-1, 1, 4)
        traj = simulate_delivery(release_from_action(map_command(a, REFERENCE_BOWLER), rng), PitchConditions(seam_sd_deg=0.0), rng)
        pool.setdefault(DeliveryFeatures.from_trajectory(traj).length_category, []).append(a)
    _POOL = pool
    return pool


def simulate_profile(card: BatterCard, params: dict, fmt: str, balls: int, seed: int) -> dict[str, float]:
    """Bowl `balls` reference deliveries at one candidate batter; return the same rates Cricsheet gives."""
    pool = reference_pool()
    mix = {k: v for k, v in MIX[fmt].items() if pool.get(k)}
    cats, probs = list(mix), np.asarray(list(mix.values()))
    probs = probs / probs.sum()
    rng = np.random.default_rng(seed)
    archetype = card.to_archetype(params)
    state = BatterState.for_archetype(archetype)
    faced = runs = dots = bounds = 0
    outs = {"caught": 0, "bowled_lbw": 0}
    noise = REFERENCE_BOWLER.noise_scale()
    in_over = 0
    while faced < balls:
        cat = cats[int(rng.choice(len(cats), p=probs))]
        bucket = pool[cat]
        a = np.clip(bucket[int(rng.integers(0, len(bucket)))] + rng.normal(0, noise, 4), -1, 1)
        traj = simulate_delivery(release_from_action(map_command(a, REFERENCE_BOWLER), rng), PitchConditions(), rng)
        out = resolve(DeliveryFeatures.from_trajectory(traj), archetype, state, rng)
        if out.result == "wide":
            continue
        faced += 1
        in_over += 1
        runs += out.runs_scored
        dots += int(out.runs_scored == 0)
        bounds += int(out.runs_scored >= 4)
        if out.is_wicket:
            outs["caught" if out.result.startswith("caught") else "bowled_lbw"] += 1
            state = BatterState.for_archetype(archetype)  # new innings, same player
            in_over = 0
    total_outs = sum(outs.values())
    return {
        "strike_rate": 100.0 * runs / faced,
        "dot_pct": 100.0 * dots / faced,
        "boundary_pct": 100.0 * bounds / faced,
        "balls_per_dismissal": faced / total_outs if total_outs else float(faced * 2),
        "caught_share": outs["caught"] / total_outs if total_outs else float("nan"),
    }


def scaled_errors(sim: dict, real: dict, real_outs: int) -> dict[str, float]:
    errs = {}
    for k in STATS:
        if k == "caught_share" and (real_outs < 10 or math.isnan(sim.get(k, float("nan")))):
            continue
        r, s = real[k], sim[k]
        if not np.isfinite(r):
            continue
        kind, tol = TOLERANCE[k]
        errs[k] = (math.log(max(s, 1e-6) / max(r, 1e-6)) / math.log(1 + tol)) if kind == "rel" else (s - r) / tol
    return errs


def _candidate(args):
    card_json, params, fmt, balls, seed = args
    card = BatterCard.model_validate_json(card_json)
    return params, simulate_profile(card, params, fmt, balls, seed)


def draw_params(card: BatterCard, rng: np.random.Generator) -> dict[str, float]:
    base = card.base_params()
    return {
        "skill": float(np.clip(base["skill"] + rng.uniform(-0.2, 0.2), 0.2, 0.98)),
        "aggression": float(np.clip(base["aggression"] + rng.uniform(-0.25, 0.25), 0.05, 1.0)),
        "starting_confidence": float(np.clip(base["starting_confidence"] * math.exp(rng.uniform(-0.7, 0.7)), 0.02, 0.6)),
        "weakness_scale": float(math.exp(rng.uniform(math.log(0.4), math.log(2.5)))),
        "strength_scale": float(math.exp(rng.uniform(math.log(0.5), math.log(2.0)))),
        "strike_rotation": float(np.clip(base["strike_rotation"] + rng.uniform(-0.35, 0.35), 0.0, 1.0)),
        "aerial_risk": float(np.clip(base["aerial_risk"] + rng.uniform(-0.35, 0.35), 0.0, 1.0)),
    }


@dataclass
class CalibrationResult:
    status: str  # "passed", "failed", "uncalibrated"
    reason: str
    fmt: str
    accepted: list[dict]
    closest: list[dict]
    real_fit: dict | None
    real_holdout: dict | None
    sim_population: dict | None
    holdout_check: dict | None

    def population(self, card: BatterCard) -> BatterPopulation:
        params = [a["params"] for a in (self.accepted or self.closest[:10])] or [{}]
        return BatterPopulation(card, params)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def uncalibrated(card: BatterCard, fmt: str = "odi", n: int = 40, seed: int = 0) -> CalibrationResult:
    """No real data: a population spread around the card, clearly labelled as uncalibrated."""
    rng = np.random.default_rng(seed)
    base = card.base_params()
    params = []
    for _ in range(n):
        p = draw_params(card, rng)
        # shrink towards the card: without data we trust the card's words more than the wide prior
        params.append({k: base.get(k, 1.0) + 0.4 * (v - base.get(k, 1.0)) for k, v in p.items()})
    return CalibrationResult("uncalibrated", "No ball-by-ball data supplied; population is the card's prior only.",
                             fmt, [{"params": p} for p in params], [], None, None, None, None)


def calibrate(card: BatterCard, fit: Profile, holdout: Profile | None, fmt: str = "odi", candidates: int = 240,
              balls: int = 720, min_accepted: int = 10, min_balls: int = 300, seed: int = 0,
              workers: int | None = None) -> CalibrationResult:
    real = fit.rates()
    real_fit = fit.to_dict()
    real_hold = holdout.to_dict() if holdout and holdout.balls else None
    if fit.balls < min_balls:
        return CalibrationResult("failed", f"Only {fit.balls} balls faced in the fit set; need {min_balls}+ to calibrate.",
                                 fmt, [], [], real_fit, real_hold, None, None)
    rng = np.random.default_rng(seed)
    card_json = card.model_dump_json()
    reference_pool()

    def score(results):
        out = []
        for params, sim in results:
            errs = scaled_errors(sim, real, fit.outs)
            worst = max(abs(e) for e in errs.values()) if errs else float("inf")
            rms = math.sqrt(sum(e * e for e in errs.values()) / max(1, len(errs)))
            out.append({"params": params, "sim": sim, "errors": errs, "worst": worst, "rms": rms})
        return out

    with ProcessPoolExecutor(max_workers=workers) as ex:
        # Round 1: broad draws around the card.
        first = candidates // 2
        jobs = [(card_json, draw_params(card, rng), fmt, balls, seed * 100003 + i) for i in range(first)]
        scored = score(ex.map(_candidate, jobs, chunksize=4))
        # Round 2 (ABC-SMC style): jitter the closest 20% to search where the real batter lives.
        scored.sort(key=lambda s: s["rms"])
        elite = [s["params"] for s in scored[: max(4, first // 5)]]
        keys = list(elite[0])
        spread = {k: max(0.03, 0.6 * float(np.std([e[k] for e in elite]))) for k in keys}
        bounds = {"skill": (0.2, 0.98), "aggression": (0.05, 1.0), "starting_confidence": (0.02, 0.6),
                  "weakness_scale": (0.3, 3.0), "strength_scale": (0.4, 2.5), "strike_rotation": (0.0, 1.0), "aerial_risk": (0.0, 1.0)}
        jobs = []
        for i in range(candidates - first):
            parent = elite[i % len(elite)]
            child = {k: float(np.clip(parent[k] + rng.normal(0, spread[k]), *bounds.get(k, (-np.inf, np.inf)))) for k in keys}
            jobs.append((card_json, child, fmt, balls, seed * 100003 + first + i))
        scored += score(ex.map(_candidate, jobs, chunksize=4))
    scored.sort(key=lambda s: s["rms"])
    accepted = [s for s in scored if s["worst"] <= 1.0]
    closest = scored[:40]
    group = accepted or closest[:min_accepted]
    # Diagnosis for the analyst: which statistic the near-misses fail most often, and in which direction.
    misses = {}
    for row in closest:
        for k, e in row["errors"].items():
            if abs(e) > 1.0:
                misses.setdefault(k, []).append(e)
    diagnosis = "; ".join(f"{k} {'too high' if np.mean(v) > 0 else 'too low'} in {len(v)} of {len(closest)} near-misses"
                          for k, v in sorted(misses.items(), key=lambda kv: -len(kv[1])))
    sim_pop = {k: {"mean": float(np.nanmean([g["sim"][k] for g in group])), "sd": float(np.nanstd([g["sim"][k] for g in group]))}
               for k in STATS}
    hold_check = None
    if real_hold and holdout.balls >= 60:
        hr = holdout.rates()
        hold_check = {}
        for k in ("strike_rate", "dot_pct", "boundary_pct"):
            # widen tolerance by the holdout's own sampling error (binomial on per-ball rates)
            p = hr[k] / 100.0 if k != "strike_rate" else None
            se = 100.0 * math.sqrt(max(p * (1 - p), 1e-4) / holdout.balls) if p is not None else hr[k] * 1.2 / math.sqrt(holdout.balls)
            kind, tol = TOLERANCE[k]
            allowed = (hr[k] * tol if kind == "rel" else tol) + 2 * se
            diff = sim_pop[k]["mean"] - hr[k]
            hold_check[k] = {"real": hr[k], "sim": sim_pop[k]["mean"], "allowed": allowed, "ok": abs(diff) <= allowed}
        if holdout.outs >= 5:
            real_bpd = hr["balls_per_dismissal"]
            allowed = real_bpd * TOLERANCE["balls_per_dismissal"][1] + 2 * real_bpd / math.sqrt(holdout.outs)
            diff = sim_pop["balls_per_dismissal"]["mean"] - real_bpd
            hold_check["balls_per_dismissal"] = {"real": real_bpd, "sim": sim_pop["balls_per_dismissal"]["mean"],
                                                 "allowed": allowed, "ok": abs(diff) <= allowed}
    if len(accepted) >= min_accepted:
        status, reason = "passed", f"{len(accepted)} of {candidates} candidate batters reproduce the real record."
        if hold_check and not all(v["ok"] for v in hold_check.values()):
            status, reason = "failed", "Fits the older matches but misses the held-out recent matches (form change or overfitting)."
    else:
        status = "failed"
        reason = (f"Only {len(accepted)} of {candidates} candidates reproduce the real record (need {min_accepted}). "
                  "The card's weaknesses/strengths probably do not match how this batter actually scores and gets out."
                  + (f" Simulated {diagnosis}." if diagnosis else ""))
    strip = lambda rows: [{"params": r["params"], "sim": r["sim"], "errors": r["errors"], "worst": r["worst"]} for r in rows]
    return CalibrationResult(status, reason, fmt, strip(accepted), strip(closest), real_fit, real_hold, sim_pop, hold_check)
