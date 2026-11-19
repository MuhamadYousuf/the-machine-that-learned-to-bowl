"""Checks for the video data export (scripts/export_video_data.py) and rendered sidecars."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from aibowler.constants import PITCH_LENGTH_M
from aibowler.physics import PitchConditions, release_from_action, simulate_delivery

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_video_data import compact_delivery  # noqa: E402

DATA = ROOT / "video" / "public" / "data"
RENDERS = ROOT / "video" / "out" / "renders"


def _fake_delivery(trajectory, stride_ms=4):
    """Build a replay-shaped delivery from a fine physics trajectory, sampled every 4 ms."""
    step = max(1, int(round(stride_ms / 1000 / float(trajectory.t[1] - trajectory.t[0]))))
    points = [
        {"t_s": float(t), "x_m": float(p[0]), "y_m": float(p[1]), "z_m": float(p[2])}
        for t, p in zip(trajectory.t[::step], trajectory.pos[::step])
    ]
    bounce_index = int(np.argmin(trajectory.pos[:, 2]))
    bounce_t = float(trajectory.t[bounce_index])
    return {
        "trajectory_points": points,
        "event_times_s": {"bounce_s": bounce_t, "crease_s": float(trajectory.t[-1]), "stumps_s": float(trajectory.t[-1])},
        "key_points": {"bounce_m": [float(v) for v in trajectory.pos[bounce_index]]},
        "features": {"length_m": trajectory.length_from_stumps, "length_category": trajectory.length_category},
        "outcome": {"result": "dot", "runs_conceded": 0, "is_wicket": False, "is_legal_ball": True},
        "release": {"speed_kmh": 140, "elevation_deg": -5, "azimuth_deg": 0, "swing": 0},
    }, bounce_t


def test_linear_interpolation_of_4ms_samples_is_sub_millimetre():
    """The renderer interpolates linearly between 4 ms samples (bounce inserted exactly).
    Compared with the 2 ms integrator output, the error must stay below 1 mm."""
    rng = np.random.default_rng(3)
    for action in ([0.4, -0.2, 0.0, -0.8], [0.9, 0.6, 0.1, 0.5], [-0.5, -0.9, -0.2, 0.0]):
        release = release_from_action(np.asarray(action), rng, noise_scale=0.0)
        trajectory = simulate_delivery(release, PitchConditions(seam_sd_deg=0.0), np.random.default_rng(1))
        delivery, bounce_t = _fake_delivery(trajectory)
        compact = compact_delivery(delivery)
        ts = np.asarray(compact["t"])
        assert np.all(np.diff(ts) > 0), "times must be strictly increasing"
        assert any(abs(t - bounce_t) < 1e-5 for t in ts), "bounce sample must be present"
        worst = 0.0
        for t, p in zip(trajectory.t, trajectory.pos):
            interp = np.array([np.interp(t, ts, compact[k]) for k in ("x", "y", "z")])
            worst = max(worst, float(np.linalg.norm(interp - p)))
        assert worst < 1e-3, f"interpolation error {worst * 1000:.2f} mm"


@pytest.mark.skipif(not (DATA / "overs_ga2_champion_technician.json").exists(), reason="video data not exported")
def test_exported_bounce_points_match_lengths():
    """Pitch-map dots (length from the batter's stumps) and the 3D bounce point must agree."""
    for name in ("overs_ga2_champion_technician.json", "overs_ga1500_champion_slogger.json"):
        data = json.loads((DATA / name).read_text(encoding="utf-8"))
        checked = 0
        for over in data["overs"]:
            for d in over["deliveries"]:
                if d["bounce"] is None or d["length"] is None:
                    continue
                assert abs(PITCH_LENGTH_M - d["bounce"][0] - d["length"]) < 2e-3
                assert d["bounce"][2] >= 0
                checked += 1
        assert checked > 50


@pytest.mark.skipif(not RENDERS.exists(), reason="footage not rendered")
def test_sidecar_events_match_delivery_times():
    """Every Over sidecar: bounce/crease frames follow from the delivery's event times and playback speed."""
    sidecars = [p for p in RENDERS.rglob("*.json") if p.name != "index.json"]
    checked = 0
    for path in sidecars:
        meta = json.loads(path.read_text(encoding="utf-8"))
        if not meta.get("composition", "").startswith("Over-"):
            continue
        props = meta["props"]
        data = json.loads((DATA / props["file"]).read_text(encoding="utf-8"))
        overs = data["overs"][props["overIndex"]: props["overIndex"] + props.get("overCount", 1)]
        deliveries = [data["overs"][props["overIndex"]]["deliveries"][i] for i in props["balls"]] if props["balls"] else [
            d for o in overs for d in o["deliveries"]]
        assert len(meta["balls"]) == len(deliveries)
        for event, d in zip(meta["balls"], deliveries):
            if d["tb"] is not None:
                expected = event["releaseFrame"] + round(d["tb"] / props["speed"] * meta["fps"])
                assert event["bounceFrame"] == expected
            assert event["releaseFrame"] < event["endFrame"] <= meta["durationInFrames"] - meta["holdFrames"]
            assert event["result"] == d["result"]
        checked += 1
    assert checked > 0
