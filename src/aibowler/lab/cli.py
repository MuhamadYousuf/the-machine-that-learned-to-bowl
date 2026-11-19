"""aibowler-lab: describe a batter → calibrate → train a plan for your bowler → audit → scouting report.

    aibowler-lab card draft "Babar Azam" --out cards/babar.yaml        # LLM research + draft (needs ANTHROPIC_API_KEY)
    aibowler-lab card check cards/babar.yaml                           # validate, show evidence quality
    aibowler-lab data find t20s_json.zip "Babar"                       # find the Cricsheet name
    aibowler-lab run cards/babar.yaml --bowler examples/lab/bowler_express.yaml \
        --data t20s_json.zip --player "Babar Azam" --format t20 --out runs/lab/babar
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from .cards import BatterCard, load_batter, load_bowler, load_reward, save_card


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def cmd_card_draft(args) -> int:
    from . import llm
    from .cricsheet import load_matches, profile

    prof = None
    if args.data and args.player:
        prof = profile(load_matches(args.data, args.player), args.player).to_dict()
    card, notes = llm.build_card(args.name, args.context or "", prof)
    save_card(card, args.out)
    Path(args.out).with_suffix(".notes.md").write_text(notes, encoding="utf-8")
    print(f"Draft card → {args.out}  (research notes → {Path(args.out).with_suffix('.notes.md')})")
    print(f"{card.assumption_share():.0%} of claims are assumptions. Review the YAML before training.")
    return 0


def cmd_card_check(args) -> int:
    card = load_batter(args.card)
    arch = card.to_archetype()
    print(f"{card.name}: {card.hand}-handed {card.level} {card.style}; {len(card.weaknesses)} weaknesses, {len(card.strengths)} strengths")
    for w in card.weaknesses:
        kinds = ",".join(e.kind for e in w.evidence) or "none"
        print(f"  weak   {w.zone.describe():45s} → {w.dismissal:14s} {w.severity:6s} evidence: {kinds}")
    for s in card.strengths:
        kinds = ",".join(e.kind for e in s.evidence) or "none"
        print(f"  strong {s.zone.describe():45s}   {'':14s} {s.severity:6s} evidence: {kinds}")
    print(f"Assumed: {card.assumption_share():.0%} of claims. Compiles to {len(arch.weakness_zones)} weakness and {len(arch.scoring_zones)} scoring zones.")
    return 0


def cmd_data_find(args) -> int:
    from .cricsheet import find_players

    for name, balls in find_players(args.data, args.query):
        print(f"{balls:6d} balls  {name}")
    return 0


def cmd_run(args) -> int:
    from .audit import audit
    from .calibrate import calibrate, uncalibrated
    from .report import render
    from .train import train
    from ..policy import MLPPolicy

    card = load_batter(args.card)
    bowler = load_bowler(args.bowler)
    reward = load_reward(args.reward)
    if args.format:
        reward = reward.model_copy(update={"format": args.format})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("1/4 calibration")
    if args.data and args.player:
        from .cricsheet import load_matches, split_profiles

        matches = load_matches(args.data, args.player)
        fit, hold = split_profiles(matches, args.player)
        print(f"    {args.player}: {fit.balls} balls to fit, {hold.balls} held out")
        cal = calibrate(card, fit, hold, fmt=reward.format, candidates=args.candidates, seed=args.seed)
    else:
        cal = uncalibrated(card, reward.format, seed=args.seed)
    print(f"    {cal.status.upper()}: {cal.reason}")
    (out / "calibration.json").write_text(json.dumps(cal.to_dict(), indent=1, default=_json_default), encoding="utf-8")
    if cal.status == "failed" and not args.force:
        print("    Stopping: the simulated batter does not match the real one. Fix the card or pass --force.")
        return 2
    population = cal.population(card)

    print(f"2/4 training ({args.generations} generations × {args.population} brains)")
    t = train(card, population, bowler, reward, out, generations=args.generations, population_size=args.population,
              overs_per_eval=args.overs, seed=args.seed)

    print("3/4 audit (baseline, robustness, exploit detectors)")
    a = audit(MLPPolicy(t["genome"]).act, card, population, bowler, reward)
    for c in a["checks"]:
        print(f"    [{c['status']:4s}] {c['name']}: {c['detail']}")
    (out / "audit.json").write_text(json.dumps({k: v for k, v in a.items() if k != "balls"}, indent=1, default=_json_default), encoding="utf-8")

    print("4/4 report")
    html_text = render(card, bowler, reward, cal.to_dict(), a, t)
    report = out / "report.html"
    report.write_text(html_text, encoding="utf-8")
    print(f"    → {report}")
    if args.open:
        webbrowser.open(report.resolve().as_uri())
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="aibowler-lab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    card = sub.add_parser("card", help="draft or check batter cards").add_subparsers(dest="card_cmd", required=True)
    d = card.add_parser("draft", help="research a player with Claude and draft a card")
    d.add_argument("name")
    d.add_argument("--out", required=True)
    d.add_argument("--context", help="extra description (role, format, what you have seen)")
    d.add_argument("--data", help="Cricsheet zip/folder to add real numbers")
    d.add_argument("--player", help="exact Cricsheet name")
    d.set_defaults(func=cmd_card_draft)
    c = card.add_parser("check", help="validate a card and show evidence quality")
    c.add_argument("card")
    c.set_defaults(func=cmd_card_check)

    data = sub.add_parser("data", help="Cricsheet helpers").add_subparsers(dest="data_cmd", required=True)
    f = data.add_parser("find", help="find a player's Cricsheet name")
    f.add_argument("data")
    f.add_argument("query")
    f.set_defaults(func=cmd_data_find)

    r = sub.add_parser("run", help="calibrate, train, audit and write the scouting report")
    r.add_argument("card")
    r.add_argument("--bowler", required=True)
    r.add_argument("--reward", help="reward spec YAML (default: ODI weights)")
    r.add_argument("--format", choices=["test", "odi", "t20"])
    r.add_argument("--data", help="Cricsheet zip/folder for calibration")
    r.add_argument("--player", help="exact Cricsheet name")
    r.add_argument("--out", required=True)
    r.add_argument("--generations", type=int, default=120)
    r.add_argument("--population", type=int, default=64)
    r.add_argument("--overs", type=int, default=16, help="overs per brain per generation")
    r.add_argument("--candidates", type=int, default=240, help="calibration candidates")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--force", action="store_true", help="train even if calibration fails")
    r.add_argument("--open", action="store_true", help="open the report when done")
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
