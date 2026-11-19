# Vision: from "an AI that learned to bowl" to an open tactics lab

Status (2026-11-17): **v0.2–0.4 core built** as `aibowler-lab` (see [LAB.md](LAB.md)): batter/bowler cards with evidence labels, Claude card drafting with web search + structured outputs, Cricsheet profiles, ABC calibration gate with held-out matches, batter populations, training, exploit detectors, robustness, HTML scouting report. Not yet: vectorised sim, self-play, field placements, second sport.

## The idea, sharpened

> Describe an opponent in plain words. The tool builds a simulated version of them,
> trains an AI to beat that simulation, and hands the analyst a plan they can check and coach.

The user's original pitch was: give an LLM a prompt ("an Australian batter who…"), it researches
the player online, builds an AI batter, then an RL bowler masters him and the analyst reads off
the weaknesses. Then generalise to *any* analysis problem "just by one prompt".

The core is strong: **reward design and environment setup are the parts that stop people using RL**,
and LLMs are now good at drafting both. But as stated, the pitch has five flaws that would make it
produce confident nonsense. Each has a fix.

## Flaws and fixes

### 1. The bowler can only find weaknesses that you put into the batter (circularity)
Our batters have hand-designed weakness zones. The bowler "discovers" them, which is impressive to
watch but is not discovery about a real player. If an LLM writes the batter from web articles
("struggles against the short ball"), training a bowler just reflects the article back with a
fancy chart. And our own v1 run proves the worse case: the bowler also finds *bugs* in the batter
model (the bouncer/yorker cheat) and reports them as tactics.

**Fix — calibrate before you train.** The LLM only proposes *priors*. Real data sets the parameters:
- Free ball-by-ball data exists for nearly every international and franchise match: **Cricsheet**
  (runs, dismissal type, bowler, over). It has no line/length/speed, so it constrains *outcomes*
  (strike rate, dismissal mix, by phase, by bowler type), not ball-tracking zones.
- Analysts who do have tracking or video-tag data (clubs, academies) load it through a documented CSV.
- **Calibration gate:** the simulated batter must reproduce the real player's held-out numbers
  (dismissal mix, strike rate per phase, dot-ball %) within tolerance before any bowler is trained.
  If it can't, the tool says so instead of training.

### 2. One batter model is one guess
Web text and thin stats fit many different batters equally well.

**Fix — train against a population of plausible batters, not one.** Sample 50–200 batters from the
calibrated uncertainty (the same trick as our ±15% `sample_batter`, but driven by data). A plan is
only reported if it works across most of them. The report shows a **robustness score** and which
conclusions depend on which assumption ("this plan relies on him being weak to full balls outside
off; that comes from 3 of 41 dismissals — low confidence").

### 3. A plan the AI can execute may be one your bowler can't
Our bowler can bowl 150 km/h yorkers on demand. A real 128 km/h club seamer cannot.

**Fix — a bowler card too.** Max/typical pace, accuracy (execution noise), which deliveries they
have (swing, cutters, bouncer). Then the output is a plan **for this bowler against this batter**,
which is exactly what a coach can use. This is the product's sharpest version.

### 4. LLM-written rewards get hacked
Free-form reward code from an LLM is the v1 cheat waiting to happen (this is a known result:
NVIDIA's *Eureka* and *Text2Reward* work, and their failure cases).

**Fix — the LLM fills a typed spec, it never writes the reward code.**
- Rewards are composed from vetted building blocks (wickets, runs conceded, extras, legality rules)
  with weights, in a readable YAML contract the analyst approves.
- Automatic **exploit detectors** run during training: saturated outputs, bang-bang alternation,
  rule-limit riding, behaviour that only works against one sampled batter. These are the exact
  checks that would have caught v1.
- A human reviews a 30-second replay of the champion before the report is produced.

### 5. "Any analysis problem with one prompt" has no simulator
RL needs an environment that behaves like reality. For cricket we built one (ball physics plus a
batter model). For "any business or analysis problem" there usually is no simulator, and an LLM
inventing one produces fiction with a chart. Many of those problems are also better served by
plain optimisation, bandits or forecasting than by RL.

**Fix — narrow, then widen by plugins.**
- v1 scope: **bat-and-ball duels**, where the physics is known and outcome data is public:
  cricket (bowling plans, later field placements), **baseball pitch sequencing** (public
  Statcast pitch-tracking data is far richer than cricket's), tennis serve placement,
  football penalties.
- Architecture: a **sport pack** = environment + data adapter + calibration targets + report
  template. The LLM layer is a *config compiler* (prompt → batter card, bowler card, reward spec),
  not a general problem solver.
- Generalising beyond sports is a research question. Leave it as a stated long-term direction,
  not a launch promise.

### Smaller but real issues
| Issue | Fix |
|---|---|
| LLM hallucinates player facts | Every claim on a card carries a source link; unsourced claims are marked as assumptions |
| Scraping terms of service and player data | Use open data (Cricsheet, attribution licence); no scraping of sites that forbid it; the user brings any private data |
| Training takes about 6 hours on CPU (our sim is Python loops) | Vectorise the simulator (NumPy/JAX across the whole population): realistically 20–100× faster, minutes per scenario, GPU optional |
| API cost and privacy | Bring your own key; support local models (Ollama) so a club can run it offline |
| Real batters adapt mid-match | Already partly modelled (confidence, pattern memory); add self-play: evolve the batter's counter-strategy too, so plans hold up against a batter who adjusts |

## Ideas worth adding
1. **Scouting report as the output, with replays.** PDF/dashboard: pitch maps, the over-by-over plan,
   confidence, and our Hawk-Eye renderer showing the key deliveries. No other open-source tool has
   that visual layer, so it is our differentiator.
2. **Two-sided reports.** Run self-play and the same batter yields a report for the *batting* coach:
   "the plan bowlers will use against you, and the counter that worked in simulation."
3. **Batter and bowler cards as shareable files.** A versioned community registry on GitHub,
   with sources and calibration scores, like model cards.
4. **Benchmark and leaderboard.** Standard batter suites; people submit bowler agents. This is cheap
   to run and brings contributors.
5. **Field placement head.** A second output (9 fielder positions) trained jointly; analysts care
   about this as much as line and length.

## Roadmap
| Version | Delivers |
|---|---|
| 0.1 | Open-source the current project cleanly: `pip install`, CLI (`aibowler train/replay/render`), YAML configs for batters, docs, the renderer, the video pipeline |
| 0.2 | **Batter card** spec (YAML) and an LLM card builder: description + Cricsheet stats → card with citations → human review |
| 0.3 | Calibration gate against Cricsheet; batter populations from uncertainty; robustness score |
| 0.4 | Bowler card (capabilities), exploit detectors, scouting report with replays; vectorised sim |
| 1.0 | Self-play two-sided reports, field placements, benchmark; second sport pack (baseball, using Statcast) |

## Further improvements (added after the first build)
1. **Conditions-aware plans.** The physics already has pitch conditions (bounce, seam). Let the analyst set
   "green top / flat deck / dew" and compare plans across conditions; coaches plan per venue.
2. **Live re-calibration.** Feed the ball-by-ball of the current innings into the gate between overs and
   re-rank plans: a batter's form on the day matters more than his career average.
3. **Plan robustness to *your* execution.** The bowler card's accuracy already adds noise; add a
   "bad day" stress test (accuracy one level worse) so a plan isn't fragile to a slightly off bowler.
4. **Uncertainty-aware wording.** When calibration is thin, the report should soften its recommendations
   automatically ("worth trying" vs "strong").
5. **Video in the report.** Render the top 3 plan deliveries with the Hawk-Eye renderer and embed them.
