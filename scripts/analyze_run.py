"""Plot training progress and characterize the final policy by batter type."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aibowler.env import OverEnv
from aibowler.dashboard_data import read_jsonl_tolerant
from aibowler.evolve import LENGTH_CATEGORIES, normalize_reward
from aibowler.policy import MLPPolicy


DARK = "#0b1020"
PANEL = "#111827"
GRID = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
ARCHETYPES = ("slogger", "technician", "tailender")
CATEGORY_COLORS = {
    "yorker": "#f472b6",
    "full": "#fb923c",
    "good": "#38bdf8",
    "short": "#a3e635",
    "bouncer": "#c084fc",
    "full_toss": "#facc15",
}
ACTION_NAMES = ("speed", "elevation", "azimuth", "swing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--analysis-overs", type=int, default=500)
    return parser.parse_args()


def style_axes(ax, title: str | None = None) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if title:
        ax.set_title(title, color=TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, alpha=0.32, linewidth=0.6)


def load_log(path: Path) -> list[dict[str, str]]:
    from aibowler.dashboard_data import read_csv_tolerant

    return read_csv_tolerant(path)


def load_plan_matrix(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    if not path.exists():
        return {}
    plans: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            plans.setdefault(row["archetype"], {})[row["plan"]] = {
                key: float(row[key])
                for key in ("fitness", "expected_fitness", "wickets_per_over", "runs_per_over", "extras_per_over")
                if row.get(key) not in (None, "")
            }
    return plans


def normalized_plan_references(
    plan_matrix: dict[str, dict[str, dict[str, float]]],
    baselines: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    references: dict[str, dict[str, float]] = {}
    baselines = baselines or {}
    plan_names = sorted({plan for archetype in plan_matrix.values() for plan in archetype})
    for plan in plan_names:
        scores = []
        for archetype, rows in plan_matrix.items():
            if plan not in rows or "P6" not in rows or "ORACLE" not in rows:
                continue
            stored = baselines.get(archetype, {})
            random_reward = stored.get(
                "random", rows["P6"].get("expected_fitness", rows["P6"]["fitness"])
            )
            specialist = stored.get(
                "specialist", rows["ORACLE"].get("expected_fitness", rows["ORACLE"]["fitness"])
            )
            if plan == "P6":
                reward = random_reward
            elif plan == "ORACLE":
                reward = specialist
            else:
                reward = rows[plan].get("expected_fitness", rows[plan]["fitness"])
            scores.append(normalize_reward(reward, random_reward, specialist))
        if scores:
            references[plan] = {"fitness": float(np.mean(scores))}
    return references


def jensen_shannon_divergence(first: dict[str, float], second: dict[str, float]) -> float:
    """Return the base-2 Jensen-Shannon divergence for two categorical distributions."""
    categories = tuple(dict.fromkeys((*first.keys(), *second.keys())))
    if not categories:
        return 0.0
    p = np.asarray([max(0.0, float(first.get(key, 0.0))) for key in categories])
    q = np.asarray([max(0.0, float(second.get(key, 0.0))) for key in categories])
    if p.sum() == 0.0 or q.sum() == 0.0:
        raise ValueError("Jensen-Shannon distributions must have positive mass")
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)

    def kl_bits(distribution: np.ndarray) -> float:
        nonzero = distribution > 0.0
        return float(np.sum(distribution[nonzero] * np.log2(distribution[nonzero] / midpoint[nonzero])))

    return 0.5 * (kl_bits(p) + kl_bits(q))


def load_checkpoint(run_dir: Path):
    champion = run_dir / "checkpoints" / "champion.npz"
    paths = sorted((run_dir / "checkpoints").glob("gen_*.npz"))
    if champion.exists():
        paths = [champion]
    if not paths:
        raise FileNotFoundError(f"no checkpoints under {run_dir / 'checkpoints'}")
    path = paths[-1]
    with np.load(path, allow_pickle=False) as data:
        genome = data["genome"].astype(float)
        metadata = json.loads(str(data["metadata_json"].item()))
    return path, genome, metadata


def plot_learning_curve(
    rows: list[dict[str, str]],
    plans: dict[str, dict[str, float]],
    validations: list[dict],
    output: Path,
    power_cut_event: dict | None = None,
) -> None:
    generations = [int(row["gen"]) for row in rows]
    best = [float(row["best_fitness"]) for row in rows]
    mean = [float(row["mean_fitness"]) for row in rows]
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=DARK, constrained_layout=True)
    style_axes(ax, "Neuroevolution · normalized fitness")
    ax.plot(generations, best, color="#22d3ee", lw=2.0, label="Generation best")
    ax.plot(generations, mean, color="#a78bfa", lw=1.4, alpha=0.90, label="Population mean")
    validation_points = [
        (int(row["generation"]), float(row["validation_score"]))
        for row in validations
        if row.get("generation") is not None and row.get("validation_score") is not None
    ]
    if validation_points:
        ax.scatter(
            [point[0] for point in validation_points],
            [point[1] for point in validation_points],
            color="#5ee0a0", edgecolor="#07111f", linewidth=0.7,
            s=42, zorder=5, label="Held-out validation",
        )
    if power_cut_event:
        cut_generation = int(power_cut_event.get("generation", 0))
        ax.axvline(cut_generation, color="#fb7185", lw=1.4, ls=(0, (5, 4)), alpha=0.95)
        ax.annotate(
            "POWER CUT · gen 520\nresumed 521 from gen 500 champion",
            xy=(cut_generation, 0.98), xycoords=("data", "axes fraction"),
            xytext=(8, -8), textcoords="offset points", va="top", ha="left",
            color="#ffd4dc", fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": PANEL, "edgecolor": "#fb7185", "alpha": 0.92},
        )
    if plans:
        for plan, data in sorted(plans.items()):
            ax.axhline(data["fitness"], color="#fbbf24", lw=0.8, alpha=0.38, ls="--")
            ax.text(
                generations[-1], data["fitness"], f" {plan}",
                color="#fbbf24", fontsize=8, va="center", ha="left",
            )
    ax.set_xlabel("Generation")
    ax.axhline(0.0, color="#64748b", lw=0.8, alpha=0.5)
    ax.axhline(1.0, color="#5ee0a0", lw=0.9, alpha=0.5, ls=":")
    ax.set_ylabel("Normalized over fitness · 0 random / 1 oracle")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, loc="best")
    fig.savefig(output, dpi=180, facecolor=DARK)
    plt.close(fig)


def plot_normalized_scores(rows: list[dict[str, str]], output: Path) -> None:
    generations = np.asarray([int(row["gen"]) for row in rows])
    fig, ax = plt.subplots(figsize=(12, 6.2), facecolor=DARK, constrained_layout=True)
    style_axes(ax, "Generation-best normalized scores by archetype")
    colors = {"slogger": "#fb7185", "technician": "#5ee0a0", "tailender": "#fbbf24"}
    has_scores = False
    for name in ARCHETYPES:
        key = f"{name}_normalized_score"
        if key not in rows[0]:
            continue
        values = np.asarray([float(row[key]) if row.get(key) else np.nan for row in rows])
        ax.plot(generations, values, color=colors[name], lw=1.6, label=name.title())
        has_scores = True
    ax.axhline(0.0, color="#94a3b8", lw=0.8, ls=":", label="Random = 0")
    ax.axhline(1.0, color="#fbbf24", lw=1.0, ls="--", label="Specialist = 1")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized reward")
    if not has_scores:
        ax.text(0.5, 0.5, "This run has no normalized score columns", color=MUTED, ha="center", transform=ax.transAxes)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, loc="best")
    fig.savefig(output, dpi=180, facecolor=DARK)
    plt.close(fig)


def plot_wickets_economy(
    rows: list[dict[str, str]], validations: list[dict], output: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), facecolor=DARK, constrained_layout=True)
    validation_generations = [int(row["generation"]) for row in validations]
    for ax, name in zip(axes, ARCHETYPES):
        if validation_generations:
            wickets = np.asarray([
                row["per_archetype"][name]["wickets_per_over"]
                for row in validations
            ], dtype=float)
            economy = np.asarray([
                row["per_archetype"][name]["economy"]
                for row in validations
            ], dtype=float)
            generations = np.asarray(validation_generations)
            series_label = "Held-out validation"
        else:
            generations = np.asarray([int(row["gen"]) for row in rows])
            wickets = np.asarray([float(row[f"{name}_wickets_per_over"]) for row in rows])
            economy = np.asarray([float(row[f"{name}_economy"]) for row in rows])
            window = max(3, min(25, len(generations) // 12))
            if len(generations) >= window:
                kernel = np.ones(window) / window
                wickets = np.convolve(wickets, kernel, mode="same")
                economy = np.convolve(economy, kernel, mode="same")
            series_label = f"Rolling mean ({window})"
        style_axes(ax, name.title())
        ax.plot(generations, wickets, color="#fb7185", lw=1.7, label=series_label)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Wickets / over", color="#fb7185")
        twin = ax.twinx()
        twin.plot(generations, economy, color="#34d399", lw=1.5)
        twin.set_ylabel("Runs / over", color="#34d399")
        twin.tick_params(colors=TEXT, labelsize=9)
        for spine in twin.spines.values():
            spine.set_color(GRID)
    fig.suptitle("Validated wickets and economy by batter archetype", color=TEXT, fontsize=15)
    fig.savefig(output, dpi=180, facecolor=DARK)
    plt.close(fig)


def evaluate_policy(
    genome: np.ndarray,
    archetype_name: str,
    *,
    seed: int,
    overs: int,
    noise_scale: float,
) -> tuple[list[dict], dict[str, float]]:
    policy = MLPPolicy(genome)
    deliveries = []
    total_runs = total_wickets = total_extras = 0
    total_reward = 0.0
    length_counts = Counter()
    speed_values = []
    swing_values = []
    saturated_actions = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    negative_saturated_actions = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    positive_saturated_actions = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    action_count = 0
    for over_index in range(overs):
        rng = np.random.default_rng(seed + over_index * 101)
        env = OverEnv(noise_scale=noise_scale)
        observation = env.reset(archetype_name, rng)
        while True:
            ball_index = env.legal_balls
            action = policy.act(observation)
            negative_saturated_actions += action <= -0.95
            positive_saturated_actions += action >= 0.95
            saturated_actions += np.abs(action) >= 0.95
            action_count += 1
            observation, reward, done, info = env.step(action)
            total_reward += reward
            release = info["release"]
            features = info["features"]
            outcome = info["outcome"]
            length_counts[features.length_category] += 1
            speed_values.append(release.speed_kmh)
            swing_values.append(release.swing)
            if len(deliveries) < 500:
                deliveries.append(
                    {
                        "over": over_index,
                        "ball_index": ball_index,
                        "length_m": 0.0 if features.length_m is None else features.length_m,
                        "line_m": features.line_m,
                        "late_movement_m": features.late_movement_m,
                        "category": features.length_category,
                        "speed_kmh": release.speed_kmh,
                        "swing": release.swing,
                        "result": outcome.result,
                        "runs": outcome.runs_conceded,
                        "wicket": outcome.is_wicket,
                        "is_legal": outcome.is_legal_ball,
                    }
                )
            if done:
                break
        summary = env.summary()
        total_runs += summary["runs_conceded"]
        total_wickets += summary["wickets"]
        total_extras += summary["extras"]
    stats = {
        "wickets_per_over": total_wickets / overs,
        "runs_per_over": total_runs / overs,
        "extras_per_over": total_extras / overs,
        "fitness": total_reward / overs,
        "sampled_fitness": (25.0 * total_wickets - total_runs - 3.0 * total_extras) / overs,
        "mean_release_speed": float(np.mean(speed_values)) if speed_values else 0.0,
        "mean_swing": float(np.mean(swing_values)) if swing_values else 0.0,
        "action_saturation": {
            name: {
                "absolute": float(saturated_actions[index] / max(1, action_count)),
                "negative": float(negative_saturated_actions[index] / max(1, action_count)),
                "positive": float(positive_saturated_actions[index] / max(1, action_count)),
            }
            for index, name in enumerate(ACTION_NAMES)
        },
        "length_distribution": {
            category: length_counts[category] / max(1, sum(length_counts.values()))
            for category in LENGTH_CATEGORIES
        },
    }
    return deliveries, stats


def _describe_style(deliveries: list[dict]) -> str:
    counts = Counter(row["category"] for row in deliveries)
    total = max(1, len(deliveries))
    dominant, count = counts.most_common(1)[0]
    swing = float(np.mean([row["swing"] for row in deliveries]))
    swing_word = "outswing-biased" if swing < -0.15 else "inswing-biased" if swing > 0.15 else "mostly straight"
    per_ball = []
    for ball_index in range(6):
        subset = [row["category"] for row in deliveries if row["ball_index"] == ball_index]
        per_ball.append(Counter(subset).most_common(1)[0][0] if subset else "n/a")
    return (
        f"{dominant}-heavy ({100*count/total:.0f}% of {total} sampled deliveries), {swing_word}; "
        f"modal balls 1-6: {' -> '.join(per_ball)}"
    )


def plot_strategy(archetype: str, deliveries: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=DARK, constrained_layout=True)
    pitch_ax, speed_ax, category_ax, sequence_ax = axes.ravel()
    style_axes(pitch_ax, f"Pitch map · first {len(deliveries)} deliveries")
    color_for_result = {
        "runs": "#34d399",
        "dot": "#94a3b8",
        "wide": "#fbbf24",
        "no_ball": "#fb923c",
        "bowled": "#fb7185",
        "lbw": "#f43f5e",
        "caught_behind": "#c084fc",
        "caught_field": "#ef4444",
    }
    for result, color in color_for_result.items():
        subset = [row for row in deliveries if row["result"] == result]
        if subset:
            pitch_ax.scatter(
                [row["length_m"] for row in subset],
                [row["line_m"] for row in subset],
                s=17, alpha=0.72, c=color, label=result.replace("_", " "),
                edgecolors="none",
            )
    pitch_ax.axhline(0.0, color=TEXT, alpha=0.55, lw=0.8)
    pitch_ax.set_xlim(0.0, 14.0)
    pitch_ax.set_ylim(-0.8, 1.0)
    pitch_ax.set_xlabel("Bounce distance from stumps (m)")
    pitch_ax.set_ylabel("Line relative to off stump (m)")
    pitch_ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7, ncol=2)

    style_axes(speed_ax, "Release speed and swing")
    speed_ax.hist([row["speed_kmh"] for row in deliveries], bins=18, color="#38bdf8", alpha=0.82)
    speed_ax.set_xlabel("Release speed (km/h)")
    speed_ax.set_ylabel("Deliveries")
    swing_ax = speed_ax.twiny()
    swing_ax.hist([row["swing"] for row in deliveries], bins=14, range=(-1, 1), color="#a78bfa", alpha=0.45)
    swing_ax.set_xlabel("Swing command (-1 outswing, +1 inswing)", color=TEXT)
    swing_ax.tick_params(colors=TEXT)

    style_axes(category_ax, "Length categories")
    category_counts = Counter(row["category"] for row in deliveries)
    category_ax.bar(
        list(LENGTH_CATEGORIES),
        [category_counts[category] for category in LENGTH_CATEGORIES],
        color=[CATEGORY_COLORS[category] for category in LENGTH_CATEGORIES],
    )
    category_ax.tick_params(axis="x", rotation=22)
    category_ax.set_ylabel("Deliveries")

    style_axes(sequence_ax, "Ball-by-ball over pattern")
    means_length = []
    means_line = []
    modal_category = []
    for ball_index in range(6):
        subset = [row for row in deliveries if row["ball_index"] == ball_index]
        means_length.append(float(np.mean([row["length_m"] for row in subset])) if subset else np.nan)
        means_line.append(float(np.mean([row["line_m"] for row in subset])) if subset else np.nan)
        modal_category.append(Counter(row["category"] for row in subset).most_common(1)[0][0] if subset else "")
    positions = np.arange(1, 7)
    sequence_ax.plot(positions, means_length, marker="o", color="#22d3ee", label="Mean length")
    sequence_ax.set_xticks(positions, [f"Ball {i}" for i in positions])
    sequence_ax.set_ylabel("Mean bounce distance (m)")
    twin = sequence_ax.twinx()
    twin.plot(positions, means_line, marker="s", color="#fb923c", label="Mean line")
    twin.set_ylabel("Mean line from off stump (m)", color="#fb923c")
    twin.tick_params(colors=TEXT)
    for x, y, label in zip(positions, means_length, modal_category):
        if np.isfinite(y):
            sequence_ax.annotate(label, (x, y), xytext=(0, 8), textcoords="offset points", color=TEXT, fontsize=7, ha="center")
    handles1, labels1 = sequence_ax.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    sequence_ax.legend(handles1 + handles2, labels1 + labels2, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    fig.suptitle(f"Champion policy against {archetype.title()}", color=TEXT, fontsize=16)
    fig.savefig(output, dpi=180, facecolor=DARK)
    plt.close(fig)


def plot_plan_comparison(
    champion_stats: dict[str, dict[str, float]],
    plan_matrix: dict[str, dict[str, dict[str, float]]],
    output: Path,
) -> None:
    own_plan = {"slogger": "P3", "technician": "P4", "tailender": "P5"}
    labels = ["Champion", "Specialist", "Random", "P9 clone"]
    metrics = [
        ("wickets_per_over", "Wickets / over"),
        ("runs_per_over", "Runs / over"),
        ("fitness", "Fitness / over · sampled"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), facecolor=DARK, constrained_layout=True)
    colors = ["#42d9e8", "#5ee0a0", "#a78bfa", "#fb7185"]
    x = np.arange(len(ARCHETYPES))
    width = 0.24
    for ax, (metric, title) in zip(axes, metrics):
        style_axes(ax, title)
        for offset, label in enumerate(labels):
            values = []
            for name in ARCHETYPES:
                if label == "Champion":
                    key = "sampled_fitness" if metric == "fitness" else metric
                    values.append(champion_stats[name][key])
                else:
                    plan_name = (
                        own_plan[name] if label == "Specialist"
                        else "P9" if label == "P9 clone"
                        else "P6"
                    )
                    values.append(plan_matrix[name][plan_name]["fitness" if metric == "fitness" else metric])
            ax.bar(
                x + (offset - (len(labels) - 1) / 2) * width,
                values, width, label=label, color=colors[offset], alpha=.88,
            )
        ax.set_xticks(x, [name.title() for name in ARCHETYPES])
        ax.set_ylabel(title)
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    fig.suptitle("Champion vs specialist oracle vs random baseline", color=TEXT, fontsize=15)
    fig.savefig(output, dpi=180, facecolor=DARK)
    plt.close(fig)


def plot_ga1500_ga2_comparison(
    first_deliveries: dict[str, list[dict]],
    first_stats: dict[str, dict[str, float]],
    second_deliveries: dict[str, list[dict]],
    second_stats: dict[str, dict[str, float]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), facecolor=DARK, constrained_layout=True)
    run_labels = ("ga1500 champion · v2", "ga2 champion · v2")
    run_deliveries = (first_deliveries, second_deliveries)
    run_stats = (first_stats, second_stats)
    x = np.arange(2)
    for row_index, archetype in enumerate(ARCHETYPES):
        length_ax, speed_ax = axes[row_index]
        style_axes(length_ax, f"{archetype.title()} · length distribution")
        bottom = np.zeros(2)
        for category in LENGTH_CATEGORIES:
            values = [stats[archetype]["length_distribution"][category] for stats in run_stats]
            length_ax.bar(
                x, values, bottom=bottom, width=0.58,
                color=CATEGORY_COLORS[category], label=category.replace("_", " "),
            )
            bottom += np.asarray(values)
        length_ax.set_xticks(x, run_labels, rotation=10, ha="right")
        length_ax.set_ylim(0.0, 1.0)
        length_ax.set_ylabel("Share of deliveries")
        length_ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7, ncol=2)

        style_axes(speed_ax, f"{archetype.title()} · release speed")
        speeds = [
            [delivery["speed_kmh"] for delivery in deliveries[archetype]]
            for deliveries in run_deliveries
        ]
        speed_ax.hist(
            speeds, bins=np.linspace(115, 150, 22), color=("#a78bfa", "#42d9e8"),
            alpha=0.72, label=run_labels, histtype="stepfilled",
        )
        speed_ax.set_xlabel("Release speed (km/h)")
        speed_ax.set_ylabel("Sampled deliveries")
        speed_ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    fig.suptitle(
        "Legacy ga1500 vs ga2 · both champions re-evaluated under environment v2",
        color=TEXT, fontsize=16,
    )
    fig.savefig(output, dpi=180, facecolor=DARK)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = ROOT / "runs" / args.name
    log_path = run_dir / "log.csv"
    if not log_path.exists():
        raise SystemExit(f"training log not found: {log_path}")
    rows = load_log(log_path)
    if not rows:
        raise SystemExit(f"training log is empty: {log_path}")
    plan_path = ROOT / "outputs" / "batter" / "plan_matrix.csv"
    plan_matrix = load_plan_matrix(plan_path)
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    plans = normalized_plan_references(plan_matrix, config.get("fitness_baselines"))
    validations = read_jsonl_tolerant(run_dir / "validation.jsonl")
    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    power_cut_event = next(
        (event for event in config.get("events", []) if event.get("type") == "power_cut"),
        None,
    )
    if power_cut_event is None and config.get("recovery_history"):
        old = config["recovery_history"][0]
        power_cut_event = {
            "type": "power_cut",
            "generation": old.get("interrupted_after_generation", 0),
            "note": "resumed at generation 521 from generation 500 champion; population reconstructed",
        }
    plot_learning_curve(rows, plans, validations, output_dir / "learning_curve.png", power_cut_event)
    plot_normalized_scores(rows, output_dir / "normalized_scores.png")
    plot_wickets_economy(rows, validations, output_dir / "wickets_economy.png")

    checkpoint_path, genome, metadata = load_checkpoint(run_dir)
    noise_scale = float(metadata.get("noise_scale", 0.05))
    print(f"Champion checkpoint: {checkpoint_path}")
    print(f"Evaluating champion for {args.analysis_overs} overs per archetype.")
    all_beaten = True
    champion_stats = {}
    champion_deliveries = {}
    for archetype_index, archetype in enumerate(ARCHETYPES):
        deliveries, stats = evaluate_policy(
            genome,
            archetype,
            seed=args.seed + archetype_index * 1_000_000,
            overs=args.analysis_overs,
            noise_scale=noise_scale,
        )
        plot_strategy(archetype, deliveries, output_dir / f"strategy_{archetype}.png")
        own_plan = {"slogger": "P3", "technician": "P4", "tailender": "P5"}[archetype]
        own_row = plan_matrix.get(archetype, {}).get(own_plan)
        random_row = plan_matrix.get(archetype, {}).get("P6")
        if own_row is None or random_row is None:
            raise SystemExit(f"plan matrix missing specialist/random rows for {archetype}")
        beats = stats["sampled_fitness"] > own_row["fitness"]
        all_beaten = all_beaten and beats
        normalizer = metadata.get("fitness_baselines", {}).get(archetype)
        if normalizer is None:
            normalizer = {
                "random": random_row.get("expected_fitness", random_row["fitness"]),
                "specialist": own_row.get("expected_fitness", own_row["fitness"]),
            }
        stats["normalized_score"] = normalize_reward(
            stats["fitness"], normalizer["random"], normalizer["specialist"]
        )
        champion_stats[archetype] = stats
        champion_deliveries[archetype] = deliveries
        status = "beats" if beats else "below"
        if not beats and own_row["fitness"] - stats["sampled_fitness"] <= 0.25:
            status = "within 0.25 of"
        print(
            f"{archetype.title():<12} normalized {stats['normalized_score']:.2f}; sampled "
            f"fitness {stats['sampled_fitness']:.2f} vs {own_plan} {own_row['fitness']:.2f} "
            f"({status} specialist); random P6 {random_row['fitness']:.2f}; "
            f"W/O {stats['wickets_per_over']:.3f}, R/O {stats['runs_per_over']:.2f}, "
            f"X/O {stats['extras_per_over']:.2f}"
        )
        print(f"  Plan found: {_describe_style(deliveries)}")
        saturation = ", ".join(
            f"{name} |x| {100 * value['absolute']:.1f}% "
            f"(negative {100 * value['negative']:.1f}%, positive {100 * value['positive']:.1f}%)"
            for name, value in stats["action_saturation"].items()
        )
        print(f"  Action saturation (|tanh output| >= 0.95): {saturation}")
    print("Pairwise Jensen-Shannon divergence of champion length distributions (bits):")
    for first_index, first in enumerate(ARCHETYPES):
        for second in ARCHETYPES[first_index + 1 :]:
            divergence = jensen_shannon_divergence(
                champion_stats[first]["length_distribution"],
                champion_stats[second]["length_distribution"],
            )
            print(f"  {first.title()} / {second.title()}: {divergence:.4f}")
    plot_plan_comparison(champion_stats, plan_matrix, output_dir / "plan_comparison.png")
    if args.name == "ga2":
        legacy_dir = ROOT / "runs" / "ga1500"
        if (legacy_dir / "checkpoints" / "champion.npz").exists():
            _, legacy_genome, legacy_metadata = load_checkpoint(legacy_dir)
            legacy_deliveries, legacy_stats = {}, {}
            print("Evaluating the ga1500 champion again under environment v2 for comparison.")
            for archetype_index, archetype in enumerate(ARCHETYPES):
                deliveries, stats = evaluate_policy(
                    legacy_genome,
                    archetype,
                    seed=args.seed + 9_000_000 + archetype_index * 1_000_000,
                    overs=args.analysis_overs,
                    noise_scale=float(legacy_metadata.get("noise_scale", 0.05)),
                )
                legacy_deliveries[archetype] = deliveries
                legacy_stats[archetype] = stats
            plot_ga1500_ga2_comparison(
                legacy_deliveries, legacy_stats,
                champion_deliveries, champion_stats,
                output_dir / "ga1500_vs_ga2.png",
            )
            print(f"Comparison figure saved: {output_dir / 'ga1500_vs_ga2.png'}")
    print(f"Beats the specialist plan against all three: {'YES' if all_beaten else 'NO'}")
    print(f"Analysis figures saved under {output_dir}")


if __name__ == "__main__":
    main()
