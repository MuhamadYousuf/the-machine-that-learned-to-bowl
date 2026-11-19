"""Tactics lab: cards, bowler limits, Cricsheet parsing, the calibration gate, exploit detectors, LLM wiring, report."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aibowler.lab import vocab
from aibowler.lab.audit import audit, detectors
from aibowler.lab.calibrate import calibrate, simulate_profile, uncalibrated
from aibowler.lab.cards import BatterCard, BatterPopulation, BowlerCard, RewardSpec, Zone, load_batter
from aibowler.lab.cricsheet import Profile, profile, split_profiles
from aibowler.lab.labenv import LabOverEnv, bowl_overs, map_command
from aibowler.constants import SPEED_RANGE_KMH

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "lab" / "opener_short_ball.yaml"


def test_vocabulary_rejects_unknown_words_and_maps_known_ones():
    centres, widths = vocab.zone_for({"length": "good", "line": "fourth_stump"})
    assert centres == {"length_m": 7.0, "line_m": 0.25} and widths["length_m"] > 0
    with pytest.raises(ValueError):
        vocab.zone_for({"length": "half_volley_ish"})
    with pytest.raises(ValueError):
        vocab.zone_for({"spin": "doosra"})


def test_left_hander_mirrors_lines_and_movement():
    zone = Zone(length="good", line="fourth_stump", movement="away")
    right = BatterCard(name="R", weaknesses=[{"zone": zone, "dismissal": "caught_behind", "severity": "high"}])
    left = right.model_copy(update={"hand": "left"})
    wr, wl = right.to_archetype().weakness_zones[0], left.to_archetype().weakness_zones[0]
    assert wr.centres["line_m"] == -wl.centres["line_m"] > 0
    assert wr.centres["late_movement_m"] == -wl.centres["late_movement_m"] > 0


def test_bowler_card_keeps_speed_inside_their_band():
    bowler = BowlerCard(min_kmh=118, max_kmh=128, swing="none")
    lo, hi = SPEED_RANGE_KMH
    for cmd, expect in ((-1.0, 118.0), (1.0, 128.0)):
        mapped = map_command(np.array([cmd, 0.0, 0.0, 1.0]), bowler)
        kmh = lo + (mapped[0] + 1) / 2 * (hi - lo)
        assert kmh == pytest.approx(expect)
        assert mapped[3] == 0.0  # no swing ability -> no swing


def test_reward_spec_uses_format_defaults_and_rejects_silly_weights():
    assert RewardSpec(format="t20").resolved()["max_bouncers"] == 1
    assert RewardSpec(format="test").resolved()["wicket"] == 30.0
    with pytest.raises(ValueError):
        RewardSpec(wicket_value=500).resolved()


def test_bowler_without_a_bouncer_gets_no_balled_for_one():
    card = load_batter(DEMO)
    env = LabOverEnv(BatterPopulation(card), BowlerCard(bouncer=False, accuracy="elite"), RewardSpec())
    env.reset(rng=np.random.default_rng(0))
    bouncer = np.array([0.6, -1.0, 0.0, 0.0])  # steepest release = bouncer
    _, _, _, info = env.step(bouncer)
    if info["features"].length_category == "bouncer" and info["features"].height_at_crease_m >= 1.25:
        assert info["bouncer_no_ball"] and info["outcome"].result == "no_ball"


def _match(date, balls):
    return {"info": {"dates": [date]}, "innings": [{"team": "A", "overs": [{"over": 0, "deliveries": balls}]}]}


def test_cricsheet_profile_counts_like_a_scorer():
    b = "Test Batter"
    balls = [
        {"batter": b, "runs": {"batter": 4, "extras": 0, "total": 4}},
        {"batter": b, "runs": {"batter": 0, "extras": 1, "total": 1}, "extras": {"wides": 1}},  # wide: not faced
        {"batter": b, "runs": {"batter": 0, "extras": 0, "total": 0}},
        {"batter": b, "runs": {"batter": 4, "extras": 0, "total": 4, "non_boundary": True}},  # all-run four
        {"batter": b, "runs": {"batter": 1, "extras": 0, "total": 1},
         "wickets": [{"player_out": b, "kind": "run out"}]},  # run outs are not the bowler's
        {"batter": b, "runs": {"batter": 0, "extras": 0, "total": 0},
         "wickets": [{"player_out": b, "kind": "caught", "fielders": [{"name": "X"}]}]},
    ]
    p = profile([_match("2024-01-01", balls)], b)
    assert (p.balls, p.runs, p.dots, p.fours, p.sixes) == (5, 9, 2, 1, 0)
    assert p.dismissals == {"caught": 1}
    fit, hold = split_profiles([_match(f"2024-0{i}-01", balls) for i in range(1, 9)], b)
    assert fit.matches == 6 and hold.matches == 2 and hold.first_date > fit.last_date


def _profile_from(stats: dict, balls: int = 2400) -> Profile:
    outs = int(round(balls / stats["balls_per_dismissal"]))
    caught = int(round(outs * (stats["caught_share"] if np.isfinite(stats["caught_share"]) else 0.5)))
    return Profile(player="synthetic", matches=60, balls=balls, runs=int(round(stats["strike_rate"] * balls / 100)),
                   dots=int(round(stats["dot_pct"] * balls / 100)), fours=int(round(stats["boundary_pct"] * balls / 100)),
                   dismissals={"caught": caught, "bowled": outs - caught}, first_date="2020-01-01", last_date="2023-01-01")


def test_calibration_recovers_a_known_batter_and_rejects_a_wrong_card():
    card = load_batter(DEMO)
    truth = {**card.base_params(), "weakness_scale": 1.0, "strength_scale": 1.0}
    real = _profile_from(simulate_profile(card, truth, "odi", 6000, seed=1))
    ok = calibrate(card, real, None, fmt="odi", candidates=80, balls=500, min_accepted=5, seed=3, workers=4)
    assert ok.status == "passed", ok.reason
    # A dour blocker's record (strike rate 35, rarely out) cannot be reproduced by an aggressive-hitter card.
    dour = Profile(player="dour", matches=50, balls=3000, runs=1050, dots=2300, fours=60, sixes=0,
                   dismissals={"caught": 10, "bowled": 5}, first_date="2020-01-01", last_date="2023-01-01")
    bad = calibrate(card, dour, None, fmt="odi", candidates=80, balls=500, min_accepted=5, seed=3, workers=4)
    assert bad.status == "failed"
    thin = calibrate(card, Profile(player="new", balls=40), None)
    assert thin.status == "failed" and "40 balls" in thin.reason


def test_detectors_catch_the_v1_style_cheat():
    card = load_batter(DEMO)
    pop = uncalibrated(card, n=5).population(card)
    state = {"i": 0}

    def bang_bang(_obs):  # pinned outputs, alternating extremes (the v1 exploit)
        state["i"] += 1
        return np.array([1.0 if state["i"] % 2 else -1.0, -0.75 if state["i"] % 2 else 0.6, 1.0, -1.0])

    res = bowl_overs(bang_bang, pop, BowlerCard(), RewardSpec(), list(range(60)), record=True)
    checks = {c["name"]: c["status"] for c in detectors(res, res["balls"], 1.0, 1.0, 1.0, 2)}
    assert checks["Pinned outputs"] == "fail"
    assert checks["Bang-bang alternation"] in ("warn", "fail")


def test_llm_builder_uses_schema_fallbacks_and_web_search(monkeypatch):
    from aibowler.lab import llm

    calls = []
    card = load_batter(DEMO)

    class FakeMessages:
        def create(self, **kw):
            calls.append(("create", kw))
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="- note (https://example.org)")])

        def parse(self, **kw):
            calls.append(("parse", kw))
            return SimpleNamespace(stop_reason="end_turn", parsed_output=card)

    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    drafted, notes = llm.build_card("Some Batter", client=client)
    assert drafted.name == card.name and "example.org" in notes
    (_, research), (_, draft) = calls
    assert research["model"] == draft["model"] == "claude-opus-5-5"
    assert research["fallbacks"] == "default" and draft["fallbacks"] == "default"
    assert research["tools"][0]["type"] == "web_search_20260209"
    assert draft["output_format"] is BatterCard
    # The structured-output schema only allows the coach vocabulary.
    schema = json.dumps(BatterCard.model_json_schema())
    assert '"fourth_stump"' in schema and '"caught_behind"' in schema


def test_report_renders_with_verdict_checks_and_evidence():
    from aibowler.lab.report import render

    card = load_batter(DEMO)
    cal = uncalibrated(card, n=4)
    pop = cal.population(card)
    rng = np.random.default_rng(0)
    a = audit(lambda o: rng.uniform(-0.3, 0.3, 4), card, pop, BowlerCard(), RewardSpec(), overs=30, members=3, overs_per_member=4)
    html = render(card, BowlerCard(), RewardSpec(), cal.to_dict(), a, {"generations": 0})
    for needle in ("Scouting report", "Checks", "Weaknesses", "UNCALIBRATED", "assumption", "data:image/png;base64"):
        assert needle in html
