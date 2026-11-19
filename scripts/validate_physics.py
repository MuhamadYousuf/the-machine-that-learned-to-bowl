"""Generate visual and timing checks for the Step 1 delivery simulator."""

from __future__ import annotations

import time
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aibowler.constants import (
    BATTER_POPPING_CREASE_X_M,
    BATTER_STUMPS_X_M,
    LEG_SIDE_WIDE_GUIDELINE_M,
    RELEASE_POSITION_M,
    STUMPS_HEIGHT_M,
    STUMPS_WIDTH_M,
    WIDE_GUIDELINE_M,
)
from aibowler.physics import PitchConditions, Release, release_from_action, simulate_delivery


OUTPUT_DIR = ROOT / "outputs" / "validation"
DARK = "#0b1020"
GRID = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"


def aimed_azimuth_deg() -> float:
    x0, y0, _ = RELEASE_POSITION_M
    return float(np.degrees(np.arctan2(-y0, BATTER_STUMPS_X_M - x0)))


def release_at_elevation(elevation_deg: float, swing: float = 0.0) -> Release:
    return Release(
        speed_kmh=140.0,
        elevation_deg=elevation_deg,
        azimuth_deg=aimed_azimuth_deg(),
        swing=swing,
    )


def elevation_for_length(target_length_m: float, rng: np.random.Generator) -> float:
    """Numerically tune an angle for a desired bounce distance."""
    low, high = -14.0, 1.5
    for _ in range(32):
        middle = (low + high) / 2.0
        result = simulate_delivery(
            release_at_elevation(middle),
            PitchConditions(seam_sd_deg=0.0),
            rng,
        )
        length = result.length_from_stumps
        if length is None or length < target_length_m:
            high = middle
        else:
            low = middle
    return (low + high) / 2.0


def make_figure():
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=DARK, constrained_layout=True)
    ax.set_facecolor(DARK)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, alpha=0.5, linewidth=0.7)
    return fig, ax


def draw_side_stumps(ax) -> None:
    for x in (0.0, BATTER_STUMPS_X_M):
        ax.plot([x, x], [0.0, STUMPS_HEIGHT_M], color="#f8fafc", lw=3.0, zorder=5)
        ax.plot([x - 0.14, x + 0.14], [STUMPS_HEIGHT_M, STUMPS_HEIGHT_M], color="#f8fafc", lw=1.2)


def draw_top_stumps(ax) -> None:
    for x in (0.0, BATTER_STUMPS_X_M):
        ax.plot(
            [x, x],
            [-STUMPS_WIDTH_M / 2.0, STUMPS_WIDTH_M / 2.0],
            color="#f8fafc",
            lw=4.0,
            solid_capstyle="round",
            zorder=5,
        )


def format_speed(trajectory) -> str:
    if trajectory.state_at_stumps is None:
        return "not reached"
    return f"{np.linalg.norm(trajectory.state_at_stumps.vel) * 3.6:.1f} km/h"


def build_side_view(samples: list[tuple[str, Release, str]], rng: np.random.Generator) -> None:
    fig, ax = make_figure()
    colors = {
        "yorker": "#facc15",
        "good-length": "#4ade80",
        "short": "#fb923c",
        "bouncer": "#f472b6",
        "full toss": "#60a5fa",
    }
    for label, release, color_key in samples:
        trajectory = simulate_delivery(release, PitchConditions(), rng)
        color = colors[color_key]
        ax.plot(trajectory.pos[:, 0], trajectory.pos[:, 2], label=label, color=color, lw=2.5)
        if trajectory.bounce_point is not None:
            ax.scatter(
                [trajectory.bounce_point[0]],
                [trajectory.bounce_point[2]],
                s=48,
                color=color,
                edgecolor="#f8fafc",
                linewidth=0.8,
                zorder=6,
            )
    draw_side_stumps(ax)
    ax.axvline(BATTER_POPPING_CREASE_X_M, color=MUTED, ls=":", lw=1.0, alpha=0.8)
    ax.set(
        title="Delivery height profile",
        xlabel="Distance from bowler's stumps (m)",
        ylabel="Ball-centre height (m)",
        xlim=(-1.6, BATTER_STUMPS_X_M + 1.3),
        ylim=(0, 3.4),
    )
    ax.legend(
        facecolor="#111827",
        edgecolor=GRID,
        labelcolor=TEXT,
        framealpha=0.96,
        ncols=2,
        loc="upper left",
    )
    fig.savefig(OUTPUT_DIR / "side_view.png", dpi=190, facecolor=DARK)
    plt.close(fig)


def build_top_view(rng: np.random.Generator) -> None:
    fig, ax = make_figure()
    examples = [
        ("Outswing", -1.0, "#38bdf8"),
        ("Straight", 0.0, "#e2e8f0"),
        ("Inswing", 1.0, "#f472b6"),
    ]
    angle = elevation_for_length(6.5, rng)
    for label, swing, color in examples:
        trajectory = simulate_delivery(
            release_at_elevation(angle, swing),
            PitchConditions(seam_sd_deg=0.0),
            rng,
        )
        ax.plot(trajectory.pos[:, 0], trajectory.pos[:, 1], label=label, color=color, lw=2.5)
        if trajectory.bounce_point is not None:
            ax.scatter(
                trajectory.bounce_point[0],
                trajectory.bounce_point[1],
                s=42,
                color=color,
                edgecolor="#f8fafc",
                linewidth=0.8,
                zorder=6,
            )
    ax.axhline(WIDE_GUIDELINE_M, color="#f87171", ls="--", lw=1.1, alpha=0.9)
    ax.axhline(-LEG_SIDE_WIDE_GUIDELINE_M, color="#fb923c", ls="--", lw=1.1, alpha=0.9)
    draw_top_stumps(ax)
    ax.axvline(BATTER_POPPING_CREASE_X_M, color=MUTED, ls=":", lw=1.0, alpha=0.8)
    ax.set(
        title="Lateral movement before and after bounce",
        xlabel="Distance from bowler's stumps (m)",
        ylabel="Lateral position (m; positive is off side)",
        xlim=(-1.6, BATTER_STUMPS_X_M + 1.3),
        ylim=(-1.55, 1.55),
    )
    ax.legend(facecolor="#111827", edgecolor=GRID, labelcolor=TEXT, framealpha=0.96)
    ax.text(
        0.01,
        0.02,
        "Off-side wide +0.89 m (red) · leg-side wide -0.45 m (orange)",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9,
    )
    fig.savefig(OUTPUT_DIR / "top_view.png", dpi=190, facecolor=DARK)
    plt.close(fig)


def build_noise_scatter(rng: np.random.Generator) -> None:
    fig, ax = make_figure()
    base_elevation = elevation_for_length(6.5, rng)
    normalized_elevation = 2.0 * (base_elevation - (-12.0)) / (2.0 - (-12.0)) - 1.0
    intended_action = np.array([0.0, normalized_elevation, 0.0, 0.0])
    point_x: list[float] = []
    point_y: list[float] = []
    categories: list[str] = []
    full_tosses = 0
    for _ in range(500):
        release = release_from_action(intended_action, rng, noise_scale=0.06)
        trajectory = simulate_delivery(release, PitchConditions(), rng)
        if trajectory.bounce_point is None:
            full_tosses += 1
            continue
        point_x.append(float(trajectory.bounce_point[0]))
        point_y.append(float(trajectory.bounce_point[1]))
        categories.append(trajectory.length_category)

    ax.axhspan(-1.32, 1.32, color="#14532d", alpha=0.55, zorder=0)
    ax.axhline(0.0, color="#86efac", lw=0.8, alpha=0.5)
    for x in (0.0, BATTER_STUMPS_X_M):
        ax.plot([x, x], [-STUMPS_WIDTH_M / 2, STUMPS_WIDTH_M / 2], color="#f8fafc", lw=4.0, zorder=5)
    ax.axvline(BATTER_POPPING_CREASE_X_M, color="#f8fafc", lw=1.0, ls="--", alpha=0.8)
    colors = {
        "yorker": "#facc15",
        "full": "#fb923c",
        "good": "#4ade80",
        "short": "#38bdf8",
        "bouncer": "#f472b6",
    }
    for category, color in colors.items():
        mask = [item == category for item in categories]
        if any(mask):
            ax.scatter(
                np.asarray(point_x)[mask],
                np.asarray(point_y)[mask],
                s=25,
                alpha=0.82,
                c=color,
                label=category.replace("_", " "),
                edgecolors="#0b1020",
                linewidths=0.25,
            )
    ax.set(
        title="500 noisy releases around one intended good length",
        xlabel="Bounce distance from bowler's stumps (m)",
        ylabel="Lateral bounce position (m)",
        xlim=(-0.2, BATTER_STUMPS_X_M + 0.2),
        ylim=(-1.5, 1.5),
    )
    ax.legend(facecolor="#111827", edgecolor=GRID, labelcolor=TEXT, framealpha=0.96, title="Length")
    if ax.get_legend() is not None:
        ax.get_legend().get_title().set_color(TEXT)
    ax.text(
        0.01,
        0.02,
        f"Execution noise σ=0.06 normalized action units; omitted full tosses: {full_tosses}",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9,
    )
    fig.savefig(OUTPUT_DIR / "noise_scatter.png", dpi=190, facecolor=DARK)
    plt.close(fig)


def print_sample_table(samples: list[tuple[str, Release, str]], rng: np.random.Generator) -> None:
    print("\nSample delivery summary")
    print("-" * 145)
    print(
        f"{'Sample':<13} {'Release':>11} {'At stumps':>13} {'Flight':>9} "
        f"{'Bounce m':>9} {'Category':>10} {'Crease z':>10} {'Stumps z':>10} {'Hits':>7}"
    )
    print("-" * 145)
    for label, release, _ in samples:
        trajectory = simulate_delivery(release, PitchConditions(), rng)
        time_s = (
            f"{trajectory.flight_time_to_stumps:.3f} s"
            if trajectory.flight_time_to_stumps is not None
            else "n/a"
        )
        bounce = (
            f"{trajectory.length_from_stumps:.2f}"
            if trajectory.length_from_stumps is not None
            else "none"
        )
        crease_height = (
            f"{trajectory.state_at_popping_crease.pos[2]:.2f}"
            if trajectory.state_at_popping_crease is not None
            else "n/a"
        )
        stumps_height = (
            f"{trajectory.state_at_stumps.pos[2]:.2f}"
            if trajectory.state_at_stumps is not None
            else "n/a"
        )
        print(
            f"{label:<13} {release.speed_kmh:>8.1f} km/h "
            f"{format_speed(trajectory):>13} {time_s:>9} "
            f"{bounce:>9} {trajectory.length_category:>10} "
            f"{crease_height:>9} m {stumps_height:>9} m "
            f"{str(trajectory.hits_stumps):>7}"
        )
    print("-" * 145)


def benchmark(release: Release) -> None:
    rng = np.random.default_rng(912)
    conditions = PitchConditions()
    durations_ms: list[float] = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        simulate_delivery(release, conditions, rng)
        durations_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    values = np.asarray(durations_ms)
    print(
        "Timing benchmark (1000 deliveries): "
        f"mean={values.mean():.3f} ms, median={np.median(values):.3f} ms, "
        f"p95={np.percentile(values, 95):.3f} ms"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261106)
    samples = [
        ("Yorker", release_at_elevation(elevation_for_length(1.0, rng)), "yorker"),
        ("Good length", release_at_elevation(elevation_for_length(6.5, rng)), "good-length"),
        ("Short ball", release_at_elevation(elevation_for_length(10.0, rng)), "short"),
        ("Bouncer", release_at_elevation(elevation_for_length(12.5, rng)), "bouncer"),
        ("Full toss", release_at_elevation(1.5), "full toss"),
    ]
    build_side_view(samples, rng)
    build_top_view(rng)
    build_noise_scatter(rng)
    print_sample_table(samples, rng)
    benchmark(samples[1][1])
    print("\nWrote:")
    for image_name in ("side_view.png", "top_view.png", "noise_scatter.png"):
        image_path = OUTPUT_DIR / image_name
        print(f"  {image_path} ({image_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
