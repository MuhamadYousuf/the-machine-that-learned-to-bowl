"""Report and enforce ga2's generation-100 go/no-go conditions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHETYPES = ("slogger", "technician", "tailender")
CATEGORIES = ("yorker", "full", "good", "short", "bouncer", "full_toss")


def js_divergence(first: dict[str, float], second: dict[str, float]) -> float:
    p = [max(0.0, float(first.get(name, 0.0))) for name in CATEGORIES]
    q = [max(0.0, float(second.get(name, 0.0))) for name in CATEGORIES]
    p_total, q_total = sum(p), sum(q)
    p = [v / max(1e-12, p_total) for v in p]
    q = [v / max(1e-12, q_total) for v in q]
    middle = [(a + b) / 2.0 for a, b in zip(p, q)]
    def kl(values):
        return sum(v * math.log2(v / m) for v, m in zip(values, middle) if v > 0 and m > 0)
    return 0.5 * (kl(p) + kl(q))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="ga2")
    parser.add_argument("--generation", type=int, default=100)
    args = parser.parse_args()
    run_dir = ROOT / "runs" / args.run
    log_path = run_dir / "log.csv"
    with log_path.open(newline="", encoding="utf-8") as file:
        rows = {int(row["gen"]): row for row in csv.DictReader(file) if row.get("gen", "").isdigit()}
    if args.generation not in rows:
        raise SystemExit(f"generation {args.generation} is not in {log_path}")
    row = rows[args.generation]
    champion_path = run_dir / "champion_validation.json"
    if not champion_path.exists():
        raise SystemExit("champion validation record has not been written")
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    champion_saturation = sum(
        float(champion["per_archetype"][name].get("action_saturation_share", 1.0))
        for name in ARCHETYPES
    ) / len(ARCHETYPES)
    pair_scores = {}
    distributions = {
        name: {
            category: float(row.get(f"{name}_length_{category}_share", 0.0) or 0.0)
            for category in CATEGORIES
        }
        for name in ARCHETYPES
    }
    for i, first in enumerate(ARCHETYPES):
        for second in ARCHETYPES[i + 1:]:
            pair_scores[f"{first} / {second}"] = js_divergence(distributions[first], distributions[second])
    varied = {
        name: sum(distributions[name][category] for category in ("good", "full", "short"))
        for name in ARCHETYPES
    }
    genbest_no_ball_share = float(row.get("bouncer_no_ball_share", 0.0) or 0.0)
    conditions = {
        "bouncer_rule_no_balls_below_5_percent": genbest_no_ball_share < 0.05,
        "good_full_short_above_15_percent_for_one_archetype": max(varied.values()) > 0.15,
        "champion_action_saturation_below_60_percent": champion_saturation < 0.60,
        "at_least_one_pairwise_js_divergence_above_0_1": max(pair_scores.values()) > 0.1,
    }
    print(f"ga2 generation {args.generation} gate")
    print(f"  gen-best bouncer-rule no-ball share: {genbest_no_ball_share:.3%}")
    print(f"  champion action-output saturation share: {champion_saturation:.3%}")
    for name in ARCHETYPES:
        print(
            f"  {name}: normalized {float(row[f'{name}_normalized_score']):.3f}; "
            f"good+full+short {varied[name]:.3%}"
        )
    for pair, value in pair_scores.items():
        print(f"  Jensen-Shannon divergence {pair}: {value:.4f}")
    ordered_rows = [rows[g] for g in sorted(rows)]
    first_ten = ordered_rows[:10]
    last_ten = ordered_rows[-10:]
    normalized_trends = {}
    print("  normalized trend (first 10 generation-best mean -> last 10):")
    for name in ARCHETYPES:
        start = sum(float(item[f"{name}_normalized_score"]) for item in first_ten) / max(1, len(first_ten))
        end = sum(float(item[f"{name}_normalized_score"]) for item in last_ten) / max(1, len(last_ten))
        normalized_trends[name] = (start, end)
        print(f"    {name}: {start:.3f} -> {end:.3f}")
    start_speed = sum(float(item["mean_speed_kmh"]) for item in first_ten) / max(1, len(first_ten))
    end_speed = sum(float(item["mean_speed_kmh"]) for item in last_ten) / max(1, len(last_ten))
    speed_delta = end_speed - start_speed
    generation_best_speed = float(row["mean_speed_kmh"])
    print(
        f"  mean release speed trend (first 10 -> last 10, diagnostic only): "
        f"{start_speed:.2f} -> {end_speed:.2f} km/h ({speed_delta:+.2f})"
    )
    print(
        f"  gen-best mean release speed at generation {args.generation}: "
        f"{generation_best_speed:.2f} km/h (required >= 125.00)"
    )
    conditions["gen_best_mean_speed_at_least_125_kmh"] = generation_best_speed >= 125.0
    conditions["all_archetype_scores_rising"] = all(
        end > start for start, end in normalized_trends.values()
    )
    print("Gate checks:")
    for name, passed in conditions.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    passed = all(conditions.values())
    print(f"OVERALL: {'PASS - exact-resume to generation 1500' if passed else 'FAIL - stop at generation 100'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
