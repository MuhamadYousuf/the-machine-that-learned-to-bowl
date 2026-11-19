# AI Bowler Lab — describe a batter, get a bowling plan you can check

The lab turns the research project into a tool. You describe a batter, calibrate a simulated
version of him against his real record, train a plan **for your bowler**, audit the plan for the
ways AI plans fail, and get a one-page scouting report.

```
card (words + evidence) ──► calibration gate (Cricsheet) ──► batter population
                                                                  │
bowler card (what your bowler can do) ──► training (neuroevolution) ◄── reward spec (format + laws)
                                                                  │
                                       audit: baseline · robustness · exploit detectors
                                                                  │
                                                         scouting report (HTML)
```

## Install

    pip install -e .            # core: numpy, matplotlib, pyyaml, pydantic
    pip install -e .[llm]       # + the Anthropic SDK for drafting cards with Claude

## Quick start (no API key, no data)

    aibowler-lab card check examples/lab/opener_short_ball.yaml
    aibowler-lab run examples/lab/opener_short_ball.yaml --bowler examples/lab/bowler_express.yaml \
        --out runs/lab/demo --generations 60 --population 48 --open

About 1.5 minutes on an 8-core laptop. The report will say **UNCALIBRATED**: without real data the batter is only
the card, and the report says so.

## With real data (recommended)

1. Download ball-by-ball JSON from [cricsheet.org/downloads](https://cricsheet.org/downloads/)
   (for example `t20s_json.zip` or `odis_json.zip`; open data under the ODC-BY licence).
2. Find the player's Cricsheet name: `aibowler-lab data find t20s_json.zip "Babar"`.
3. Run with the data:

       aibowler-lab run cards/babar.yaml --bowler my_bowler.yaml --format t20 \
           --data t20s_json.zip --player "Babar Azam" --out runs/lab/babar --open

The run stops before training if the gate fails (override with `--force`, and the report will say so).

## Drafting a card with Claude

    set ANTHROPIC_API_KEY=...      (or: ant auth login)
    aibowler-lab card draft "Babar Azam" --context "T20I opener, facing a right-arm quick" \
        --data t20s_json.zip --player "Babar Azam" --out cards/babar.yaml

Claude (`claude-opus-5-5`) first researches the player with web search and writes sourced notes
(saved next to the card as `.notes.md`). A second call turns the notes into a card using
**structured outputs** against the card schema, so it can only use the coach vocabulary
(`good` length, `fourth_stump`, `slower_ball`, ...) and must label every claim `data`, `source`
(with URL) or `assumption`. Then **you review the YAML**: it is a draft, not a verdict.
The LLM never writes reward code or simulator numbers.

## The files

| File | What you decide | Example |
|---|---|---|
| Batter card | level, style, how he starts, straight defence, leaving, weaknesses and strengths in coach words, each with evidence | `examples/lab/opener_short_ball.yaml` |
| Bowler card | pace band (km/h), accuracy, swing, whether he bowls bouncers | `examples/lab/bowler_express.yaml`, `bowler_club_seamer.yaml` |
| Reward spec | format (`test`/`odi`/`t20`), optional weights within safe bounds | defaults are sensible; `--format t20` |

## What the gate and the audit check

| Check | Why |
|---|---|
| **Calibration gate** | Candidate batters (skill, aggression, starting confidence, weakness/strength scales) face format-typical international bowling; only those reproducing the real strike rate, dot %, boundary %, balls per dismissal and caught share survive. Needs 300+ balls. |
| **Held-out matches** | The most recent 25% of matches are kept back; the survivors must also fit those. |
| **Beats normal bowling** | The plan is compared with a normal international bowling mix against the same batter. |
| **Robustness** | Plan vs normal bowling against each calibrated version of the batter; reports the share it beats. |
| **Overfitting** | Advantage kept when the batter is twice as uncertain. |
| **Pinned outputs / bang-bang alternation / riding the bouncer limit** | The exact signatures of our v1 bowler's reward hack. |
| **Extras habit / one-trick plan** | Plans built on no-balls or a single delivery. |

## Limits (read before using a report)

* The simulator models right-arm pace and seam only. No spin yet.
* Cricsheet has no ball tracking, so *where* a batter is weak still comes from the card's evidence;
  calibration constrains *how much* and *how often*. The report shows which claims are assumed.
* A report is a hypothesis for the nets, not a fact about the player.

## Code map

`src/aibowler/lab/`: `vocab.py` (coach words → zones), `cards.py` (batter/bowler cards, reward spec, populations),
`labenv.py` (over environment with bowler limits), `cricsheet.py` (profiles), `calibrate.py` (ABC gate),
`train.py` (neuroevolution), `audit.py` (baseline, robustness, detectors), `report.py` (HTML), `llm.py` (Claude), `cli.py`.
Tests: `tests/test_lab.py`.
