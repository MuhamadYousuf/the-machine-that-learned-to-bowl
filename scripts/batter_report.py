"""Validate batter calibration, scripted plans, and smooth matchup maps."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aibowler.batter import (
    ARCHETYPES,
    BatterArchetype,
    BatterState,
    DeliveryFeatures,
    SLOGGER,
    TAILENDER,
    TECHNICIAN,
    delivery_features,
    sample_batter,
)
from aibowler.constants import (
    BATTER_POPPING_CREASE_X_M,
    BATTER_STUMPS_X_M,
    RELEASE_POSITION_M,
    STUMPS_WIDTH_M,
    WIDE_GUIDELINE_M,
    LEG_SIDE_WIDE_GUIDELINE_M,
    SPEED_RANGE_KMH,
)
from aibowler.env import OverEnv
from aibowler.outcome import calculate_probabilities, resolve
from aibowler.physics import PitchConditions, Release, release_from_action, simulate_delivery


OUTPUT_DIR = ROOT / "outputs" / "batter"
DARK = "#0b1020"
GRID = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
ARCHETYPE_ORDER = (SLOGGER, TECHNICIAN, TAILENDER)
PLAN_ORDER = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "ORACLE")
OWN_PLAN = {"slogger": "P3", "technician": "P4", "tailender": "P5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-overs", type=int, default=2000)
    parser.add_argument("--calibration-overs", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20261106)
    return parser.parse_args()


def length_category(length: float) -> str:
    if length <= 2.0:
        return "yorker"
    if length <= 6.0:
        return "full"
    if length <= 8.0:
        return "good"
    if length <= 10.5:
        return "short"
    return "bouncer"


def approximate_height_at_crease(length: float) -> float:
    """Interpolate the calibrated 140 km/h height profile for pitch maps."""
    known_lengths = np.array([0.0, 2.0, 6.5, 9.0, 11.5, 14.0])
    known_heights = np.array([0.10, 0.25, 0.68, 1.034, 1.561, 1.95])
    return float(np.interp(length, known_lengths, known_heights))


def features_for_map(length: float, line: float) -> DeliveryFeatures:
    """Make a consistent regular-pace, late-outswing feature slice."""
    height = approximate_height_at_crease(length)
    late = 0.18
    center_y = line + STUMPS_WIDTH_M / 2.0
    is_wide = center_y > WIDE_GUIDELINE_M or center_y < -LEG_SIDE_WIDE_GUIDELINE_M
    hits_stumps = -STUMPS_WIDTH_M <= line <= 0.06 and height <= 0.72
    pitched_outside_leg = center_y + 0.036 < -STUMPS_WIDTH_M / 2.0
    return DeliveryFeatures(
        length_m=length,
        line_m=line,
        height_at_crease_m=height,
        speed_at_crease_kmh=118.0,
        late_movement_m=late,
        hits_stumps=hits_stumps,
        pitched_outside_leg=pitched_outside_leg,
        is_wide=is_wide,
        is_beamer=False,
        length_category=length_category(length),
    )


def figure_axis(ax) -> None:
    ax.set_facecolor(DARK)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def heatmap_values(archetype: BatterArchetype, metric: str) -> np.ndarray:
    lengths = np.linspace(0.0, 14.0, 81)
    lines = np.linspace(-0.8, 1.0, 91)
    state = BatterState.for_archetype(
        archetype, expected_speed_kmh=135.0
    )
    values = np.zeros((len(lines), len(lengths)), dtype=float)
    for row, line in enumerate(lines):
        for column, length in enumerate(lengths):
            features = features_for_map(float(length), float(line))
            distribution = calculate_probabilities(features, archetype, state)
            values[row, column] = (
                distribution.wicket_prob
                if metric == "wicket"
                else distribution.expected_runs
            )
    return values


def draw_heatmap(
    ax,
    archetype: BatterArchetype,
    metric: str,
    *,
    values: np.ndarray | None = None,
    colorbar: bool = True,
):
    lengths = np.linspace(0.0, 14.0, 81)
    lines = np.linspace(-0.8, 1.0, 91)
    if values is None:
        values = heatmap_values(archetype, metric)
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=(0.0, 14.0, -0.8, 1.0),
        cmap="magma" if metric == "wicket" else "viridis",
        vmin=0.0,
        vmax=0.30 if metric == "wicket" else 3.5,
        interpolation="bilinear",
    )
    ax.axhline(0.0, color="#f8fafc", alpha=0.82, lw=1.0)
    ax.vlines(0.0, -STUMPS_WIDTH_M, 0.0, color="#f8fafc", alpha=0.95, lw=4.0, zorder=5)
    ax.set_xlim(0.0, 14.0)
    ax.set_ylim(-0.8, 1.0)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_yticks([-0.8, -0.4, 0.0, 0.4, 0.8, 1.0])
    ax.set_xlabel("Bounce distance from stumps (m)")
    ax.set_ylabel("Line relative to off stump (m)")
    ax.set_title(archetype.name.title())
    ax.grid(color=GRID, alpha=0.22, linewidth=0.45)
    figure_axis(ax)
    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.ax.tick_params(colors=TEXT, labelsize=8)
        bar.set_label("Wicket probability" if metric == "wicket" else "Expected runs", color=TEXT)
    return image


def save_heatmaps(rng: np.random.Generator) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    map_cache = {
        (archetype.name, metric): heatmap_values(archetype, metric)
        for metric in ("wicket", "runs")
        for archetype in ARCHETYPE_ORDER
    }
    for metric, prefix in (("wicket", "weakness"), ("runs", "runs")):
        for archetype in ARCHETYPE_ORDER:
            fig, ax = plt.subplots(figsize=(8.2, 5.6), facecolor=DARK, constrained_layout=True)
            draw_heatmap(ax, archetype, metric, values=map_cache[(archetype.name, metric)])
            fig.suptitle("Neutral state · 118 km/h ball · 135 km/h initial expectation · 0.18 m late outswing slice", color=MUTED, fontsize=9)
            fig.savefig(OUTPUT_DIR / f"{prefix}_{archetype.name}.png", dpi=180, facecolor=DARK)
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=DARK, constrained_layout=True, sharey=True)
        for index, (ax, archetype) in enumerate(zip(axes, ARCHETYPE_ORDER)):
            draw_heatmap(
                ax, archetype, metric,
                values=map_cache[(archetype.name, metric)],
                colorbar=(index == 2),
            )
        fig.suptitle("Smooth outcome pitch maps · neutral batter states", color=TEXT, fontsize=15)
        fig.savefig(OUTPUT_DIR / f"{prefix}_combined.png", dpi=180, facecolor=DARK)
        plt.close(fig)

    sampled = [sample_batter(SLOGGER, rng, spread=0.15) for _ in range(3)]
    sampled_maps = [heatmap_values(batter, "wicket") for batter in sampled]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=DARK, constrained_layout=True, sharey=True)
    for index, (ax, batter) in enumerate(zip(axes, sampled)):
        draw_heatmap(ax, batter, "wicket", values=sampled_maps[index], colorbar=(index == 2))
        ax.set_title(f"Sampled slogger {index + 1}")
    fig.suptitle("Sampled Slogger variants · yorker weakness remains distinct", color=TEXT, fontsize=15)
    fig.savefig(OUTPUT_DIR / "sampled_sloggers.png", dpi=180, facecolor=DARK)
    plt.close(fig)


def calibrated_features(
    length: float | None,
    line: float,
    *,
    height: float,
    speed: float = 118.0,
    late: float = 0.0,
    hits: bool = False,
    category: str | None = None,
    beamer: bool = False,
) -> DeliveryFeatures:
    return DeliveryFeatures(
        length_m=length,
        line_m=line,
        height_at_crease_m=height,
        speed_at_crease_kmh=speed,
        late_movement_m=late,
        hits_stumps=hits,
        pitched_outside_leg=False,
        is_wide=False,
        is_beamer=beamer,
        length_category=category or ("full_toss" if length is None else length_category(length)),
    )


def exploit_features(archetype: BatterArchetype) -> DeliveryFeatures:
    if archetype.name == "slogger":
        return calibrated_features(1.0, -0.04, height=0.18, hits=True, category="yorker")
    if archetype.name == "technician":
        return calibrated_features(7.0, 0.25, height=0.75, late=0.08, category="good")
    return calibrated_features(12.0, -0.10, height=1.55, category="bouncer")


def setup_surprising_state(archetype: BatterArchetype) -> BatterState:
    if archetype.name == "slogger":
        expected = (12.5, 0.55, 0.25, 140.0)
    elif archetype.name == "technician":
        expected = (11.5, -0.55, -0.25, 138.0)
    else:
        expected = (2.0, 0.70, -0.35, 138.0)
    state = BatterState.for_archetype(
        archetype,
        confidence=archetype.starting_confidence,
        expected_length_m=expected[0],
        expected_line_m=expected[1],
        expected_late_movement_m=expected[2],
        expected_speed_kmh=expected[3],
    )
    state.balls_faced = 2
    state.last_speed_kmh = expected[3]
    return state


def random_over_metrics(
    archetype: BatterArchetype, rng: np.random.Generator, overs: int
) -> tuple[float, float]:
    wicket_probability = 0.0
    legal_balls = 0
    expected_runs = 0.0
    for _ in range(overs):
        batter = sample_batter(archetype, rng)
        state = BatterState.for_archetype(batter)
        legal_in_over = 0
        illegals = 0
        while legal_in_over < 6 and illegals < 6:
            raw_action = rng.uniform(-1.0, 1.0, size=4)
            release = release_from_action(raw_action, rng, noise_scale=0.05)
            trajectory = simulate_delivery(release, PitchConditions(), rng)
            features = delivery_features(trajectory)
            outcome = resolve(features, batter, state, rng)
            expected_runs += outcome.expected_runs
            if outcome.is_legal_ball:
                legal_in_over += 1
                legal_balls += 1
                wicket_probability += outcome.wicket_prob
            else:
                illegals += 1
            if outcome.is_wicket:
                batter = sample_batter(archetype, rng)
                state = BatterState.for_archetype(batter)
    return wicket_probability / max(1, legal_balls), expected_runs / overs


def calibration_table(rng: np.random.Generator, overs: int) -> None:
    print("\nBatter calibration summary")
    print("-" * 106)
    print(f"{'Case':<34} {'Wicket %':>11} {'Expected runs':>15} {'Target':>39}")
    print("-" * 106)
    for archetype in ARCHETYPE_ORDER:
        wicket_rate, economy = random_over_metrics(archetype, rng, overs)
        economy_target = "1-3% / 9-14 per over" if archetype is not TAILENDER else "1-3% / 5-8 per over"
        print(
            f"{('Random - ' + archetype.name):<34} {100*wicket_rate:>10.2f}% "
            f"{economy:>13.2f}/over {economy_target:>39}"
        )

    good = calibrated_features(6.5, 0.0, height=0.68, speed=120.0, hits=True, category="good")
    for archetype in ARCHETYPE_ORDER:
        distribution = calculate_probabilities(
            good, archetype, BatterState.for_archetype(archetype)
        )
        print(
            f"{('Generic good - ' + archetype.name):<34} "
            f"{100*distribution.wicket_prob:>10.2f}% "
            f"{distribution.expected_runs:>13.2f} {'2-5% / 0.3-0.9':>39}"
        )

    for archetype in ARCHETYPE_ORDER:
        features = exploit_features(archetype)
        state = setup_surprising_state(archetype)
        first = calculate_probabilities(features, archetype, state).wicket_prob
        repeated_state = setup_surprising_state(archetype)
        repeated_rng = np.random.default_rng(810 + len(archetype.name))
        for _ in range(6):
            outcome = resolve(features, archetype, repeated_state, repeated_rng)
        print(
            f"{('Exploit - ' + archetype.name):<34} {100*first:>10.2f}% "
            f"{'n/a':>15} {('ball 6: ' + format(100*outcome.wicket_prob, '.2f') + '%'):>39}"
        )

    full_toss = calibrated_features(None, -0.10, height=0.55, speed=115.0, category="full_toss")
    slogger_ev = calculate_probabilities(
        full_toss, SLOGGER, BatterState.for_archetype(SLOGGER)
    ).expected_runs
    print(f"{'Full toss - slogger':<34} {'n/a':>11} {slogger_ev:>13.2f} {'at least 3 runs':>39}")
    print("-" * 106)


def fresh_batter_table() -> list[dict[str, float | str]]:
    """Report the outcome model's first-ball response by pitch length."""
    categories = (
        ("full_toss", None, 0.52),
        ("yorker", 1.0, 0.18),
        ("full", 4.0, 0.42),
        ("good", 7.0, 0.78),
        ("short", 9.5, 1.18),
        ("bouncer", 12.0, 1.58),
    )
    rows = []
    print("\nFresh-batter, single-ball outcomes (straight line, 140 km/h, no history)")
    print(f"{'Length':<12} {'Archetype':<12} {'Wicket %':>10} {'Expected runs':>14} {'Reward':>10}")
    for category, length, height in categories:
        for archetype in ARCHETYPE_ORDER:
            feature = calibrated_features(
                length, 0.0, height=height, speed=140.0,
                hits=category == "yorker", category=category,
            )
            outcome = calculate_probabilities(
                feature, archetype, BatterState.for_archetype(archetype)
            )
            reward = 25.0 * outcome.wicket_prob - outcome.expected_runs
            row = {
                "length": category,
                "archetype": archetype.name,
                "wicket_probability": outcome.wicket_prob,
                "expected_runs": outcome.expected_runs,
                "expected_reward": reward,
            }
            rows.append(row)
            print(
                f"{category:<12} {archetype.name:<12} "
                f"{100*outcome.wicket_prob:>9.2f}% {outcome.expected_runs:>14.3f} "
                f"{reward:>10.3f}"
            )
    with (OUTPUT_DIR / "fresh_batter_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for archetype in ARCHETYPE_ORDER:
        candidates = [row for row in rows if row["archetype"] == archetype.name]
        best = max(candidates, key=lambda row: float(row["expected_reward"]))
        print(f"  Best single ball vs {archetype.name}: {best['length']} ({best['expected_reward']:.3f})")
    return rows


def benchmark_full_ball(iterations: int = 2000) -> float:
    rng = np.random.default_rng(450)
    state = BatterState.for_archetype(SLOGGER)
    action = np.array([0.0, -0.05, 0.0, 0.0])
    started = time.perf_counter()
    for _ in range(iterations):
        release = release_from_action(action, rng, noise_scale=0.05)
        trajectory = simulate_delivery(release, PitchConditions(), rng)
        features = delivery_features(trajectory)
        outcome = resolve(features, SLOGGER, state, rng)
        if outcome.is_wicket:
            state = BatterState.for_archetype(SLOGGER)
    return (time.perf_counter() - started) / iterations


def _target_line_azimuth(line_m: float) -> float:
    x0, y0, _ = RELEASE_POSITION_M
    desired_y = STUMPS_WIDTH_M / 2.0 + line_m
    return math.degrees(math.atan2(desired_y - y0, BATTER_POPPING_CREASE_X_M - x0))


@lru_cache(maxsize=None)
def scripted_action(length_m: float, line_m: float, speed_kmh: float, swing: float) -> tuple[float, ...]:
    """Invert the action map once for a named pitch-plan delivery target."""
    x0, y0, _ = RELEASE_POSITION_M
    aimed = math.degrees(math.atan2(-y0, BATTER_STUMPS_X_M - x0))
    azimuth = _target_line_azimuth(line_m)
    action0 = float(np.clip((speed_kmh - sum(SPEED_RANGE_KMH) / 2.0) / ((SPEED_RANGE_KMH[1] - SPEED_RANGE_KMH[0]) / 2.0), -1.0, 1.0))
    action2 = float(np.clip((azimuth - aimed) / 2.0, -1.0, 1.0))
    rng = np.random.default_rng(0)
    lo, hi = -12.0, 2.0
    for _ in range(18):
        elevation = (lo + hi) / 2.0
        release = Release(speed_kmh, elevation, aimed + action2 * 2.0, swing)
        trajectory = simulate_delivery(release, PitchConditions(seam_sd_deg=0.0), rng)
        bounce_length = trajectory.length_from_stumps
        if bounce_length is None or bounce_length < length_m:
            hi = elevation
        else:
            lo = elevation
    elevation = (lo + hi) / 2.0
    action = np.array([action0, (elevation + 5.0) / 7.0, action2, swing], dtype=float)

    # Swing moves the line, so correct azimuth against the simulated crease.
    for _ in range(4):
        release = release_from_action(action, rng, noise_scale=0.0)
        trajectory = simulate_delivery(release, PitchConditions(seam_sd_deg=0.0), rng)
        if trajectory.state_at_popping_crease is None:
            break
        actual_line = trajectory.state_at_popping_crease.pos[1] - STUMPS_WIDTH_M / 2.0
        error = line_m - actual_line
        action[2] = float(np.clip(action[2] + error / 0.65, -1.0, 1.0))
        if abs(error) < 0.005:
            break
    return tuple(float(value) for value in np.clip(action, -1.0, 1.0))


def action_for_plan(
    plan: str,
    legal_ball_index: int,
    rng: np.random.Generator,
    archetype_name: str | None = None,
) -> np.ndarray:
    if plan == "ORACLE":
        if archetype_name is None:
            raise ValueError("ORACLE requires the initial archetype name")
        plan = OWN_PLAN[archetype_name]
    if plan == "P1":
        return np.asarray(scripted_action(7.0, 0.0, 140.0, 0.0))
    if plan == "P2":
        if legal_ball_index % 2 == 0:
            return np.asarray(scripted_action(12.0, -0.10, 140.0, 0.0))
        return np.asarray(scripted_action(1.0, 0.0, 140.0, 0.0))
    if plan == "P3":
        if legal_ball_index % 2 == 0:
            return np.asarray(scripted_action(1.0, 0.0, 150.0, 0.0))
        return np.asarray(scripted_action(6.5, 0.0, 116.0, 0.0))
    if plan == "P4":
        if legal_ball_index % 3 < 2:
            return np.asarray(scripted_action(7.0, 0.0, 150.0, 0.70))
        return np.asarray(scripted_action(7.0, 0.25, 115.0, -0.80))
    if plan == "P5":
        # Respect the two above-shoulder bouncers per over. Two body-line
        # bouncers target the Tailender weakness; four straight yorkers keep
        # the remaining balls difficult without repeating an illegal quota.
        if legal_ball_index % 6 in (0, 3):
            return np.asarray(scripted_action(12.0, -0.10, 140.0, -1.0))
        return np.asarray(scripted_action(1.0, 0.0, 140.0, -1.0))
    if plan == "P6":
        return rng.uniform(-1.0, 1.0, size=4)
    if plan == "P7":
        return np.asarray(scripted_action(1.0, 0.0, 140.0, 0.0))
    if plan == "P8":
        return np.asarray(scripted_action(1.0, 0.0, 117.0, 0.0))
    if plan == "P9":
        if legal_ball_index % 2 == 0:
            return np.asarray(scripted_action(12.0, 0.0, 150.0, -1.0))
        return np.asarray(scripted_action(1.0, 0.0, 115.0, -1.0))
    raise ValueError(f"unknown plan {plan}")


def run_plan_matrix(overs: int, seed: int) -> list[dict[str, float | str]]:
    print(f"\nScripted-plan matrix: {overs:,} sampled overs per plan and initial archetype", flush=True)
    rows: list[dict[str, float | str]] = []
    for archetype_index, archetype in enumerate(ARCHETYPE_ORDER):
        for plan in PLAN_ORDER:
            wickets = runs = extras = fitness = expected_fitness = 0.0
            for over_index in range(overs):
                over_seed = seed + archetype_index * 10_000_000 + over_index * 101
                rng = np.random.default_rng(over_seed)
                environment = OverEnv(noise_scale=0.05, batter_spread=0.15)
                observation = environment.reset(archetype, rng)
                done = False
                while not done:
                    ball_index = environment.legal_balls
                    action = action_for_plan(plan, ball_index, rng, archetype.name)
                    observation, reward, done, info = environment.step(action)
                    expected_fitness += reward
                summary = environment.summary()
                wickets += summary["wickets"]
                runs += summary["runs_conceded"]
                extras += summary["extras"]
                fitness += 25.0 * summary["wickets"] - summary["runs_conceded"] - 3.0 * summary["extras"]
            row = {
                "plan": plan,
                "archetype": archetype.name,
                "wickets_per_over": wickets / overs,
                "runs_per_over": runs / overs,
                "extras_per_over": extras / overs,
                "fitness": fitness / overs,
                "expected_fitness": expected_fitness / overs,
            }
            rows.append(row)
            print(
                f"  {archetype.name:<12} {plan}: W/O {row['wickets_per_over']:.3f}, "
                f"R/O {row['runs_per_over']:.2f}, X/O {row['extras_per_over']:.2f}, "
                f"fitness {row['fitness']:.2f}, expected-reward {row['expected_fitness']:.2f}",
                flush=True,
            )

    path = OUTPUT_DIR / "plan_matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nPlan winners and pace checks by initial archetype:")
    pass_all = True
    pace_pass = True
    for archetype in ARCHETYPE_ORDER:
        candidates = [row for row in rows if row["archetype"] == archetype.name]
        winner = max(candidates, key=lambda row: float(row["fitness"]))
        own = OWN_PLAN[archetype.name]
        own_row = next(row for row in candidates if row["plan"] == own)
        p9 = next(row for row in candidates if row["plan"] == "P9")
        p7 = next(row for row in candidates if row["plan"] == "P7")
        p8 = next(row for row in candidates if row["plan"] == "P8")
        own_wins = winner["plan"] == own
        p9_ok = float(p9["fitness"]) < float(own_row["fitness"])
        pass_all = pass_all and own_wins and p9_ok
        pace_ok = float(p8["fitness"]) < float(p7["fitness"])
        pace_pass = pace_pass and pace_ok
        print(
            f"  {archetype.name}: best {winner['plan']} ({winner['fitness']:.2f}); "
            f"own {own} {own_row['fitness']:.2f}; P7 {p7['fitness']:.2f}; "
            f"P8 {p8['fitness']:.2f}; P9 {p9['fitness']:.2f} "
            f"(P8<P7 {'PASS' if pace_ok else 'FAIL'}, P9<own {'PASS' if p9_ok else 'FAIL'})"
        )
    print(f"Specialist winners: {'PASS' if pass_all else 'FAIL'}")
    print(f"P8 worse than P7 for all archetypes: {'PASS' if pace_pass else 'FAIL'}")
    return rows


def main() -> None:
    args = parse_args()
    if args.plan_overs < 1 or args.calibration_overs < 1:
        raise SystemExit("overs must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    save_heatmaps(rng)
    calibration_table(rng, args.calibration_overs)
    fresh_batter_table()
    rows = run_plan_matrix(args.plan_overs, args.seed + 1000)
    average_seconds = benchmark_full_ball()
    print(f"\nFull-ball benchmark (physics + features + outcome): {average_seconds*1000:.3f} ms average")
    print("\nWrote batter figures and plan matrix to:")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {path} ({path.stat().st_size:,} bytes)")
    print(f"  {OUTPUT_DIR / 'plan_matrix.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
