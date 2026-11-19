# Progress

## Step 1 — project skeleton and ball physics

Built the src-layout package, SI-unit constants, deterministic action mapping,
single-bounce trajectory simulator, focused physics tests, and three dark
Matplotlib validation plots. The current action maps speed to 115-150 km/h,
elevation to -12 through +2 degrees, azimuth offset to -2 through +2 degrees
around the line from release to middle stump, and swing directly to [-1, 1].
For a right-handed batter, negative swing is outswing to positive y and
positive swing is inswing to negative y.

Design choices and tuned constants:

- Corrected release position: (-1.22, +0.30, 2.20) m, off side of the line.
- Semi-implicit Euler, 2 ms fixed step.
- Drag coefficient: 0.40; air density: 1.2 kg/m^3.
- Swing acceleration coefficient: 0.0023 /m times speed squared, active only
  before bounce.
- Restitution base: 0.67 at 0.15 rad impact angle, with a 0.55/rad impact
  slope; horizontal retention: 0.96.
- Seam deviation: normal angle with standard deviation 0.25 degrees.
- Length thresholds from the batter's stumps: yorker <=2 m, full <=6 m,
  good <=8 m, short <=10.5 m, bouncer >10.5 m.
- Wide checks: off side y > +0.89 m and leg side y < -0.45 m at the
  batter's popping crease; beamer: no bounce and height above 1.0 m there.

Validation results:

- Python 3.14.4 venv created at .venv. NumPy, Matplotlib, and pytest installed
  from Python 3.14 wheels; no Python 3.12 fallback was needed.
- Original Step 1 test result: 8 passed in 0.19 s; replaced by the combined
  Step 2 run below.
- The timing benchmark and PNGs are superseded by the calibrated Step 2
  validation recorded below.

Commands (run from the cricket_ai project root):

    .venv\Scripts\python.exe -m pytest
    .venv\Scripts\python.exe scripts\validate_physics.py

Known limitations and questions for architect review:

- The wide rules use asymmetric centre-line thresholds and still ignore
  batter reach and umpire judgement.
- Swing is a tuned lateral force rather than a seam, spin, or atmospheric
  flow simulation. The same scalar swing is used on every pre-bounce step.
- After the first bounce, only one flight arc is modelled; later ground
  contact terminates the trajectory rather than simulating rolling or hops.
- The beamer rule is an approximate waist-height flag, not a full no-ball rule.
- The architect's revised length thresholds are implemented in Step 2. An
  explicit front-foot/no-ball check remains open for a later step.

## Step 2 — physics corrections, batter model, and outcomes

### Part A: architect-directed physics corrections

| Setting | Before review | Current |
|---|---:|---:|
| Release lateral position | -0.30 m | +0.30 m (off side) |
| Drag coefficient | 0.36 | 0.40 |
| Restitution | 0.50 constant | 0.67 base at 0.15 rad, plus 0.55/rad impact-angle slope |
| Horizontal retention at bounce | 0.90 | 0.96 |
| Azimuth action range | +/-4 degrees | +/-2 degrees |
| Length bands (from stumps) | full through 4 m; short through 11 m | full through 6 m; short through 10.5 m |
| Leg-side wide line | not implemented | y < -0.45 m at popping crease |

Off-side wide remains y > +0.89 m. Zero azimuth still aims from the
off-side release at the middle stump line. Narrowing the azimuth action range
keeps uniformly random actions from producing an excessive wide rate while
leaving enough range to learn line control.

Bounce calibration, 140 km/h with zero swing and zero seam:

| Bounce distance | Category | Height at crease | Height at stumps | Speed at crease | Cross-pitch time | Release-to-stumps |
|---:|---|---:|---:|---:|---:|---:|
| 6.5 m | good | 0.681 m | 0.794 m | 118.4 km/h (84.6%) | 0.566 s | 0.598 s |
| 9.0 m | short | 1.034 m | 1.141 m | 117.8 km/h | 0.572 s | 0.604 s |
| 11.5 m | bouncer | 1.561 m | 1.679 m | 116.8 km/h | 0.580 s | 0.612 s |

Cross-pitch time is measured from x=0, the bowler's stumps plane, to x=20.12,
the batter's stumps plane. Trajectory sample times remain release-relative;
the API also reports release-to-stumps elapsed time. The fixed 0.40 drag
coefficient was not used as a calibration variable. The search script found
the current restitution and retention settings within all requested windows.

### Part B: batter and outcome model

Added DeliveryFeatures, Gaussian WeaknessZone and ScoringZone records,
BatterArchetype, BatterState, and BallOutcome. Archetype behavior is
data-driven; outcome resolution has no archetype-name branches. Each zone
matches as the product of per-feature Gaussian bumps. Full-toss length maps to
zero. sample_batter varies numeric parameters, centres, and widths up to 15%,
while preserving each archetype's dominant weakness.

BatterState tracks EMA expectations for length, line, late movement, and
crease speed (alpha=0.5), plus confidence, balls faced, and the previous
delivery. Surprise is the clipped RMS distance from those four expectations,
scaled by 7 m, 0.85 m, 0.45 m, and 15 km/h. Zone strength combines with
surprise to increase dismissal probability; confidence reduces risk. Smooth
ScoringZones modify run quality. The resolver remains hand-tuned rather than
learned.

Late movement measures the final six metres before the bounce (or before the
popping crease for a full toss). This retains an outswing signal because the
physics model disables swing after bounce.

Latest calibration from scripts/batter_report.py (320 random-action overs per
archetype, uniform actions plus 0.05 normalized execution noise; rates use
expected probabilities and runs, including extras):

| Case | Wicket probability | Expected runs |
|---|---:|---:|
| Random - Slogger | 2.60% per legal ball | 14.06 per over |
| Random - Technician | 1.27% per legal ball | 9.48 per over |
| Random - Tailender | 2.92% per legal ball | 7.93 per over |
| Generic good - Slogger | 2.55% | 0.89 |
| Generic good - Technician | 2.00% | 0.88 |
| Generic good - Tailender | 3.11% | 0.57 |
| High-surprise exploit - Slogger | 28.98% | — |
| High-surprise exploit - Technician | 28.23% | — |
| High-surprise exploit - Tailender | 28.17% | — |
| Same exploit repeated, ball 6 | 2.66% / 1.46% / 2.05% | — |
| Full toss to Slogger | — | 3.30 |

The repeat-ball-6 values correspond to Slogger, Technician, and Tailender.
Slogger random economy is 0.06 runs/over above the requested upper bound of
14; the other random, generic-good, exploit, repetition, and full-toss targets
are within range. The flight-time window uses the documented x=0-to-x=20.12
cross-pitch measure (0.566 s); release-to-stumps time is 0.598 s.

Validation:

- Combined test result: 29 passed in 0.48 s, including environment, genome,
  evolution smoke, deterministic replay, and common-random-number checks.
- Bounce calibration selected Cd=0.40, restitution=0.67, impact slope=0.55/rad,
  and horizontal retention=0.96; all five target windows passed.
- Latest physics validation benchmark, 1,000 good-length deliveries: mean
  0.564 ms, median 0.522 ms, p95 0.810 ms.
- Physics figures regenerated: side_view.png (226,350 bytes),
  top_view.png (122,850 bytes), and noise_scatter.png (114,609 bytes).
- Batter report completed with the calibration table above, the 2,000-over
  scripted-plan matrix, and a 1.037 ms average full-ball benchmark (physics,
  feature extraction, and outcome). It generated smooth wicket/run maps,
  combined 1x3 figures, three sampled-slogger maps, and plan_matrix.csv under
  outputs/batter/.

Commands (run from the cricket_ai project root):

    .venv\Scripts\python.exe -m pytest
    .venv\Scripts\python.exe scripts\calibrate_bounce.py
    .venv\Scripts\python.exe scripts\validate_physics.py
    .venv\Scripts\python.exe scripts\batter_report.py

Limitations and open questions for the architect:

- Restitution and swing are calibrated analytic approximations; the bounce
  search hits the requested points but is not fitted to instrumented cricket
  ball data. A later pitch model may need surface-specific parameters.
- Late movement compares the final six metres with a straight extrapolation
  from the preceding six metres. It is a useful signal, but it does not
  distinguish seam, swing, and bounce deflection.
- No front-foot no-ball, standard LBW law, batter reach, running between
  wickets, or additional ball movement/fielding is modelled. The current
  beamer no-ball is the only no-ball trigger.
- The outcome probabilities are manually tuned. Should dismissal state
  reset confidence only for a new batter, or carry any over-level batter
  memory across wickets?

## Step 3 — scripted plans, over environment, and neuroevolution

### Part A: batter variation and plan calibration

Retained the zone-based batter model from Step 2 and added per-feature
numeric variation to `sample_batter(..., spread=0.15)`. The calibration report
now runs each of six scripted plans for 2,000 overs against each sampled
archetype. The matrix reports wickets, runs, extras, and sampled fitness
(`25 * wickets - runs - 3 * extras`, averaged per over). The best plan for
each archetype is its own specialist plan: P3 for Slogger, P4 for Technician,
and P5 for Tailender. P2 is not the universal winner.

| Archetype | Plan | Wickets/over | Runs/over | Extras/over | Fitness/over |
|---|---:|---:|---:|---:|---:|
| Slogger | P1 | 0.046 | 5.865 | 0.000 | -4.715 |
| Slogger | P2 | 0.288 | 10.026 | 0.000 | -2.826 |
| Slogger | P3 | 0.512 | 7.752 | 0.000 | 5.049 |
| Slogger | P4 | 0.192 | 6.050 | 0.000 | -1.262 |
| Slogger | P5 | 0.349 | 9.647 | 0.000 | -0.935 |
| Slogger | P6 | 0.162 | 13.929 | 2.036 | -15.998 |
| Technician | P1 | 0.041 | 4.942 | 0.000 | -3.929 |
| Technician | P2 | 0.126 | 1.930 | 0.000 | 1.221 |
| Technician | P3 | 0.222 | 4.059 | 0.000 | 1.491 |
| Technician | P4 | 0.345 | 4.481 | 0.000 | 4.145 |
| Technician | P5 | 0.130 | 5.003 | 0.000 | -1.753 |
| Technician | P6 | 0.085 | 9.413 | 2.105 | -13.615 |
| Tailender | P1 | 0.078 | 3.498 | 0.000 | -1.561 |
| Tailender | P2 | 0.338 | 2.963 | 0.000 | 5.475 |
| Tailender | P3 | 0.263 | 3.206 | 0.000 | 3.357 |
| Tailender | P4 | 0.115 | 3.832 | 0.000 | -0.969 |
| Tailender | P5 | 0.577 | 4.295 | 0.000 | 10.131 |
| Tailender | P6 | 0.179 | 8.157 | 2.009 | -9.709 |

The matrix also stores the environment's expected reward (`expected_fitness`)
for comparing against training. Sampled and expected values differ because
the model samples shot outcomes. `outputs/batter/` contains per-archetype and
combined smooth heatmaps, sampled Slogger maps, and the full CSV matrix.

### Part B: over environment and genetic algorithm

Added `OverEnv` with six legal balls per over, a cap of six illegal deliveries,
and a 17-value observation: batter type (3 one-hot values), confidence, balls
faced, legal balls, runs, and two recent deliveries (length, line, late
movement, speed, valid flag). The policy is a NumPy MLP with architecture
17-24-24-4. Fitness is mean environment reward per over, with wicket value 25,
expected batter runs charged, and a 3-point extra penalty.

The genetic algorithm uses a 128-member population, 8 elites, tournament size
4, uniform crossover, mutation rate 0.10, and mutation sigma annealed from
0.08 to 0.02. Each candidate sees the same seeded scenarios within a
generation. Independent per-delivery random streams keep action noise,
physics seam, outcome sampling, and incoming-batter draws aligned across
policies even when their over transitions differ. Evaluation uses eight overs
per archetype per candidate and multiprocessing. CSV logs include fitness,
outcome rates, release speed, length distribution, and generation time;
checkpoints are saved at generations 1, 2, 5, 10, 25, 50, 100, 150, and 200.

Training and evaluation artifacts:

- A 10-generation, 16-member smoke run completed in 10.3 seconds; best fitness
  improved from -2.602 to -1.993.
- The required 200-generation, 128-member run completed in 929.7 seconds
  (mean 4.64 seconds/generation). Per-generation best fitness rose from
  -1.284 at generation 1 to 2.730 at generation 200; its highest observed
  value was 3.444 at generation 194. Generation 194 was not a checkpoint, so
  the final saved candidate is the generation 200 best.
- The generation 200 candidate was evaluated for 100 overs per archetype and
  replay snapshots were saved as JSON. Wickets/runs per over: Slogger
  0.300/6.86, Technician 0.180/1.88, Tailender 0.370/3.16.
- A separate 500-over-per-archetype analysis compared sampled learned fitness
  with the matrix's sampled fitness for its strongest specialist plan:
  Slogger 0.44 vs P3 5.05, Technician 2.77 vs P4 4.14, and Tailender 5.48 vs
  P5 10.13. The learned policy does not beat any of the three scripted
  specialists. It uses a different full-heavy style against Slogger, but
  leans heavily on repeated yorkers against Technician and Tailender; it has
  not learned distinct specialist sequencing. Learning-curve reference lines
  use the matrix's expected reward values.
- `runs/ga200/analysis/` contains learning, wickets/economy, and per-archetype
  strategy plots. The replay JSON contains sampled 10 ms trajectory points,
  key points, delivery features, outcomes, and over summaries. No renderer or
  video output was in this milestone's scope.

Commands (run from the cricket_ai project root):

    .venv\Scripts\python.exe -m pytest
    .venv\Scripts\python.exe scripts\calibrate_bounce.py
    .venv\Scripts\python.exe scripts\validate_physics.py
    .venv\Scripts\python.exe scripts\batter_report.py
    .venv\Scripts\python.exe scripts\train.py --name smoke10_crn --gens 10 --pop 16 --seed 31 --workers 4 --noise 0.05 --eval-overs 8
    .venv\Scripts\python.exe scripts\train.py --name ga200 --gens 200 --pop 128 --seed 20261107 --workers 8 --noise 0.05 --eval-overs 8
    .venv\Scripts\python.exe scripts\evaluate.py --name ga200 --checkpoint gen_0200.npz --overs 100 --seed 20261108
    .venv\Scripts\python.exe scripts\analyze_run.py --name ga200 --seed 20261109 --analysis-overs 500

### Part C: remaining model limits

- Random-action Slogger economy is 14.06 runs/over, exceeding the requested
  maximum by 0.06; this small miss remains visible in the calibration table.
- Training fitness improved substantially, but the learned controller remains
  below the hand-authored specialist plans and overuses yorkers against two
  archetypes. Further policy inputs/reward shaping or more robust training
  would be needed before claiming specialist strategy discovery.
- The environment still omits front-foot no-balls, batter reach, running
  between wickets, field placement, and a renderer. Replay data is available
  for a later visualization step.

## Step 4 — training fix, long run, and dashboard

### Part A: pace and missed-length corrections

BatterState now starts with an expected crease speed of 135 km/h. Pace
surprise feeds the same all-archetype miss, edge, and scoring-control logic:
fast balls are harder to time, repeated slow balls are familiar and easier to
score, and a fast-to-slow delivery remains a change-up. Missed yorkers that
become full tosses or half-volleys are scoring opportunities for all three
archetypes; the Technician drives them well and the Tailender can still hit
some boundaries. Correctly landed yorkers remain difficult to score from.
The scripted batting plans and archetype weaknesses were not weakened to
improve the learned policy.

During calibration, the shared run-distribution scoring-probability
intercept was moved from 0.18 to 0.14 to bring random-action economy into the
specified calibration ranges. It applies uniformly to every archetype and
plan; specialist plans and weakness zones received no AI-specific adjustment.

The updated 2,000-over-per-plan matrix below uses sampled fitness per over
(`25 * wickets - runs - 3 * extras`). ORACLE selects the corresponding
specialist plan for each initial archetype.

| Archetype | Random P6 | Fast yorker P7 | Slow yorker P8 | Own specialist / ORACLE |
|---|---:|---:|---:|---:|
| Slogger | -16.10 | -6.02 | -9.02 | P3: 3.48 |
| Technician | -16.09 | -3.39 | -6.72 | P4: 4.25 |
| Tailender | -9.75 | 2.72 | -0.67 | P5: 9.20 |

P8 is worse than P7 against all three types, neither yorker plan beats its
specialist, and P3/P4/P5 remain their own archetype's best plan. The complete
27-row matrix is in `outputs/batter/plan_matrix.csv`.

The Step 2 calibration targets were rerun at 2,000 overs for each random
archetype and remain in range:

| Case | Wicket probability | Expected runs / over | Target |
|---|---:|---:|---|
| Random Slogger | 2.62% per legal ball | 13.95 | 1–3%; 9–14 |
| Random Technician | 1.27% per legal ball | 11.67 | 1–3%; 9–14 |
| Random Tailender | 2.90% per legal ball | 7.83 | 1–3%; 5–8 |
| Generic good, Slogger | 3.04% | 0.64 expected runs | 2–5%; 0.3–0.9 |
| Generic good, Technician | 2.01% | 0.68 expected runs | 2–5%; 0.3–0.9 |
| Generic good, Tailender | 3.11% | 0.33 expected runs | 2–5%; 0.3–0.9 |
| High-surprise exploit | 28.08% / 26.17% / 26.08% | — | about 25–30% |
| Same exploit repeated on ball 6 | 2.50% / 1.51% / 2.05% | — | under 3% |
| Full toss to Slogger | — | 3.22 | at least 3 |

Archetype values in the two exploit rows are ordered Slogger, Technician,
Tailender. The mean full-ball benchmark was 1.051 ms; the 1,000-delivery
physics validation benchmark was 0.483 ms mean and 0.849 ms p95.

### Part B: normalized genetic search and results

Genome fitness is the unweighted mean over the three archetypes of
`(reward - random) / (specialist - random)`, using the large-sample P6 and
ORACLE expected-reward rows from the plan matrix. The run config freezes these
normalizers: Slogger -16.7633 / 3.5827, Technician -17.0254 / 4.5133, and
Tailender -10.8410 / 8.2804 (random / specialist). Thus 0 is random bowling,
1 is the specialist reference, and values above 1 exceed it. Every generation
logs per-archetype scores. The population was 192, elitism 8, tournament size
4, uniform crossover, mutation rate 0.10, and cosine mutation sigma 0.08 to
0.015. All elites were reevaluated on each generation's common seeds; each
genome received 16 overs per archetype. Fixed validation seeds are outside
the training seed range; the seed-disjointness test passes. Validation runs
every 25 generations for 200 overs per archetype and selects the saved
champion by held-out score.

The required 100-generation pre-check completed in 1,390.9 seconds. Mean
speed at generation 100 was 131.57 km/h. The mean of each archetype's
generation-best normalized score rose from the first 10 generations to the
last 10 as follows: Slogger 0.616 to 1.122, Technician 0.599 to 0.890, and
Tailender 0.565 to 1.189. The gen100 candidate's scores were 1.166, 0.901,
and 1.243; its held-out champion score was 1.066. It was not pinned to the
old ~115 km/h behavior. Its later inspected delivery pattern alternated
bouncer/full-toss/yorker lengths and pace changes, rather than repeating
P7/P8's constant yorker. No new loophole was found in the pre-check.

The 1,500-generation `ga1500` run finished. Summed generation timings are
21,794.1 seconds (6 h 3 min 14 s; 14.53 seconds per generation). The best
held-out champion is generation 1275 with normalized score 1.124271. On its
200-over-per-archetype validation:

| Archetype | Normalized score | Reward / over | Wickets / over | Runs / over |
|---|---:|---:|---:|---:|
| Slogger | 1.196 | 7.570 | 0.600 | 7.105 |
| Technician | 0.905 | 2.475 | 0.245 | 3.270 |
| Tailender | 1.271 | 13.471 | 0.675 | 3.085 |

An independent 500-over replay analysis compared sampled results with the
same seeded plan matrix. The champion scored 9.18 vs P3's 3.48 against
Slogger, 2.90 vs P4's 4.25 against Technician, and 13.25 vs P5's 9.20
against Tailender. It beats two specialists and remains below the Technician
specialist. The champion's broad learned style is bouncer-heavy, outswing-
biased, and alternates bouncers with yorkers; it is not a distinct specialist
plan for each archetype. The validation champion was used for the report,
not the noisy final generation best.

The computer lost power after generation 520, before an exact population
snapshot for that generation existed. The run resumed at generation 521 from
the gen500 validation champion, reconstructing 75% local mutations and 25%
fresh genomes. The run completed and all scheduled files are present, but the
post-restart population is a documented warm restart, not a bit-for-bit
continuation of the missing gen520 population. The resume process itself
took 13,976.8 seconds; the summed per-generation total above includes both
sessions and excludes the outage gap.

Checkpoints exist at generations 1, 5, 10, 25, 50, 100, 250, 500, 750,
1000, 1250, and 1500, plus `champion.npz`. Best-policy replay bundles exist
for 1, 10, 50, 100, 250, 500, 1000, 1500, and champion: 30 overs for each
archetype. Five median-fitness genomes each are recorded at generations 0
and 1 (10 overs per archetype). Full 192-genome populations and common-seed
swarm replays are present at generations 1, 10, 50, 250, 1000, and 1500.
The replay audit checked all 60 validations, replay counts, event times, and
trajectory spacing (4 ms, within the 5 ms limit). Replay JSON is gzip
compressed using schema version 2; event metadata labels bat contact as a
crease-time proxy because bat collision is not simulated.

### Part C: dashboard, analysis, and final verification

`runs/ga1500/analysis/` contains `learning_curve.png` (P1–P8 and ORACLE
references plus validation points), `normalized_scores.png`, validation-based
`wickets_economy.png`, `plan_comparison.png`, and champion `strategy_*.png`
figures. The plan-reference normalization uses the exact frozen run-config
baselines. `scripts/analyze_run.py` prints the champion's plan summary and
specialist comparison. The dashboard at `http://localhost:8050` was active
while training and is still serving the finished run; its API reported
generation 1500/1500 and the gen1275 champion. Start it with:

    .venv\Scripts\python.exe scripts\dashboard.py --run ga1500

Final verification:

- `.venv\Scripts\python.exe -m pytest`: 40 passed.
- `scripts/calibrate_bounce.py`, `scripts/validate_physics.py`, and
  `scripts/batter_report.py --plan-overs 2000 --calibration-overs 2000`
  completed. The matrix confirmed each specialist winner and P8 < P7 for all
  archetypes.
- `scripts/analyze_run.py --name ga1500 --analysis-overs 500` completed,
  reporting champion outcomes above. `scripts/evaluate.py` also wrote three
  schema-v2 smoke replays from the champion checkpoint.
- The live dashboard partial-line CSV/JSONL tests and validation seed
  disjointness test pass.

Limitations remain visible: the Technician score is below its specialist,
the champion repeats a similar bouncer/yorker rhythm against the three types,
and the batting/contact, front-foot, fielding, and running models remain
approximations. No renderer, video, or social-media output was created in
this step.

## Step 5 — fixed environment, crash-safe GA, and Brain Explorer

### Model and calibration

The Step 4 loophole is addressed in environment v2. A shoulder-height bouncer is a short-pitched ball in the bouncer length category at or above 1.25 m at the popping crease; the first two are legal and later ones are one-run no-balls, with no dismissal. The 18th observation input reports the bouncers already used. Old 17-input policies still load through the versioned adapter.

Batter surprise tracks EMA means and variances over length, line, movement, and speed, plus lag-2 similarity and pattern confidence. It reads a bouncer/yorker alternation by ball 4 in the seeded example. Constant slow pace is easier to score for all batters; the expected-speed baseline is 135 km/h. The fastness cost is now 0.60 (up from 0.25) after the first 100-generation run drifted down in speed. The stronger pace cost brought random Slogger calibration back under its 14-run ceiling without moving scripted plans or weakening the batters.

The final 2,000-over plan matrix remains archetype-specific. Expected reward per over:

| Plan | Slogger | Technician | Tailender |
|---|---:|---:|---:|
| P6 random | -18.687 | -18.128 | -12.768 |
| Specialist / ORACLE | P3 -0.802 | P4 +0.708 | P5 +3.972 |
| P7 fast yorker | -5.001 | -2.856 | +2.566 |
| P8 slow yorker | -8.996 | -7.056 | -1.669 |
| P9 exploit clone | -34.416 | -33.667 | -30.235 |

Each specialist remains best in its own column, P8 is worse than P7 in all columns, and P9 is below every specialist. The single-ball table is in SPEC.md and `outputs/batter/fresh_batter_table.csv`; the correctly landed yorker is the best isolated ball for each fresh batter. No bouncer retuning was needed.

Final Step 2 calibration (2,000 overs): random wicket rates / runs per over are Slogger 2.16% / 13.33, Technician 1.12% / 11.30, Tailender 2.43% / 7.34. Generic-good wicket rates / expected runs are 3.90% / 0.48, 2.55% / 0.52, and 3.81% / 0.18. High-surprise exploit first-ball wicket rates are 28.06%, 29.63%, and 26.08%, dropping by ball 6 to 2.29%, 1.48%, and 1.98%. Full-toss Slogger scoring is 3.04 runs. Bounce calibration selected restitution 0.67, impact-angle slope 0.55/rad, and horizontal retention 0.96; physics validation averaged 0.471 ms per delivery (p95 0.785 ms).

The legacy ga1500 champion, evaluated for 2,000 overs per archetype in the fixed environment through its 17-input adapter, scored normalized 0.620 / 0.651 / 0.664 (mean 0.645). Rewards / wickets / runs per over were Slogger -7.593 / 0.290 / 9.085, Technician -5.873 / 0.170 / 4.518, Tailender -1.657 / 0.374 / 3.957. Results: `outputs/batter/ga1500_champion_v2.json`.

### Crash safety, lineage, and dashboard

The GA uses population 192, elitism 8, tournament k=4, uniform crossover, cosine mutation sigma 0.08 to 0.015, and 16 overs per archetype per generation. It reevaluates elites on each generation's common seeds and freezes normalized-fitness baselines from the matrix: Slogger random/specialist -18.686/-0.802, Technician -18.128/+0.708, Tailender -12.768/+3.972. Held-out validation runs every 25 generations for 200 overs per archetype with disjoint seeds. Exact resume snapshots include populations, fitness, champion, IDs, RNG states, and worker seeds every 10 generations; lineage records child and parent IDs and mutation/crossover details. All 46 tests pass, including the 30-generation deterministic crash/resume comparison and held-out seed disjointness.

The historical ga1500 power-cut event remains recorded at generation 520; its chart marks the warm restart at 521 from the gen500 champion. `scripts/dashboard.py` serves the root training page and `/brain` at localhost:8050. Brain Explorer screenshots D1–D9 are in `docs/screens/`; panels include the network, weight heatmaps, Brain Cam, decision map, sensitivity, neuron diagnostics, population PCA, weight statistics, and lineage.

### ga2 pre-check status

The earlier seed-20261108 pre-check reached generation 100 but failed only the rising-speed condition: first-10 mean 139.66 km/h versus last-10 136.00 km/h. All normalized-score trends rose (Slogger 0.746 to 1.107, Technician 0.937 to 1.135, Tailender 0.792 to 1.130); the other gate checks passed. Its pre-check folder is retained as `runs/ga2_precheck_failed_partial`; a mistaken restart initialized its config and truncated its CSV before it was stopped, so it is explicitly a partial audit archive.

A clean full-artifact pre-check ran as `ga2`, seed 20261109, and stopped at generation 100 in 1,469.3 seconds (14.69 s/generation average). The gen-best bouncer-rule no-ball share was 0%; the validation champion action saturation share was 24.701%; good+full+short shares were Slogger 91.667%, Technician 96.875%, Tailender 70.833%; pairwise Jensen-Shannon divergences were Slogger/Technician 0.5839, Slogger/Tailender 0.0888, Technician/Tailender 0.6704. Each condition passed. First-10 to last-10 normalized generation-best means rose Slogger 0.808 to 1.103, Technician 0.669 to 1.046, Tailender 0.879 to 1.106. Mean release speed fell from 140.88 to 136.21 km/h (-4.67), although it remained far above the old 115 km/h exploit and above the 120 km/h floor. This was the only failed gate, so the required stop rule was honored and the 1,500-generation continuation was not launched.

The held-out champion is generation 75 (validation mean 1.084):

| Batter | Normalized | Reward / over | Wickets / over | Runs / over | Specialist reward / over | Random reward / over |
|---|---:|---:|---:|---:|---:|---:|
| Slogger | 1.094 | 0.884 | 0.300 | 4.920 | P3 -0.802 | P6 -18.687 |
| Technician | 1.061 | 1.855 | 0.120 | 1.595 | P4 +0.708 | P6 -18.128 |
| Tailender | 1.097 | 5.604 | 0.265 | 1.370 | P5 +3.972 | P6 -12.768 |

An independent 500-over analysis of this champion beat the matching specialist in all three samples: Slogger sampled fitness 2.02 vs P3 -0.97, Technician 1.95 vs P4 0.68, Tailender 5.27 vs P5 4.30. Styles differed by type: Slogger full-heavy (62%), Technician good-heavy (69%), Tailender full-heavy (47%) with a yorker/full sequence. The v2 ga1500 legacy champion remains lower at normalized 0.620 / 0.651 / 0.664 in its 2,000-over adapter evaluation. `scripts/analyze_run.py --name ga2 --analysis-overs 500` generated all requested plots, including `ga1500_vs_ga2.png`; `scripts/evaluate.py --name ga2 --checkpoint champion.npz --overs 30` wrote schema-v2 champion replay bundles. These results characterize the stopped pre-check champion, not a 1,500-generation model.

The dashboard remains available at `http://localhost:8050`, serving the stopped `ga2` run and historical `ga1500`; `/brain` and the root page both returned HTTP 200. Use `.venv\Scripts\python.exe scripts\dashboard.py --run ga2` to start it again. The long-run checkpoint and replay schedule beyond generation 100 is not present because the go/no-go condition failed.


## Prompt 5B — Evolution Theater, offline clip export, and ga2 continuation (in progress)

The architect clarified the speed gate's intent: prevent a collapse toward the former 115 km/h exploit, rather than require speed to keep rising. The exact gen100 state measured 137.47 km/h, so `run_config.json` records a `gate_override` with a 125 km/h floor and the subsequent exact `resume` event. Training continues in the background with `scripts/train.py --resume ga2`; it has not been restarted.

Generation 345 was the latest completed generation at the time of this update. Its generation-best normalized scores were Slogger 1.226, Technician 1.122, Tailender 1.147; mean release speed was 136.55 km/h. The latest 50-generation timing mean was 15.18 seconds per generation, with an estimated 4.87 hours remaining. The held-out champion was generation 325 (validation mean 1.159; normalized Slogger 1.186, Technician 1.138, Tailender 1.155).

Live-state audit: `log.csv` contains 345 unique contiguous rows from generation 1 through 345: **passed**. `lineage.csv.gz` covers generations 0–345 with no missing generations: **passed**. Generation-best genome files are saved every generation from 101 onward; the legacy pre-resume span has only its original checkpoints and cannot be reconstructed. Population snapshots are currently at 1, 10, 50, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325; generation checkpoints are at 1, 5, 10, 25, 50, 100, 250. New generation snapshots and the 25-generation population/checkpoint schedule are running alongside training.

The updated local dashboard is active at `http://localhost:8050`, including `/brain` and `/theater`; the new routes returned HTTP 200 while training was running. The dashboard command is `.venv\Scripts\python.exe scripts\dashboard.py --run ga2`. It reads the live files without writing to the trainer. The cached noise-free length-category decision map is implemented. Brain Explorer initialization and the offline swarm/comparison clips were audited and corrected: the swarm now highlights saved IDs on the champion's primary ancestry, and the v1/v2 clip follows a full same-seed over. A tiny ffmpeg export smoke test produced both aspect ratios, a 241-frame H.264 clip, timing sidecars and a contact sheet.

Final 1,500-generation analysis, full clip export, saturation audit, and champion-based screenshot set are pending until ga2 finishes. The report below will supersede this in-progress snapshot.


## Prompt 5B — resumed after the gen460 PC restart (in progress)

The saved `resume_state.npz` was gen460. `train.py --resume ga2` resumed exactly at gen461 and logged the new `resume` event at generation 460. The stale gen461 log/lineage tail from before the power-off was truncated, then rebuilt from the saved RNG and next-population state. At this update the run has reached generation 471/1500. The latest best normalized scores are Slogger 1.232, Technician 1.150, Tailender 1.142; mean speed is 137.45 km/h. Last-50 timing averages 15.67 sec/gen, about 4.48 hours remaining. The current validation champion is gen425 with score 1.172.

Post-resume continuity check: log rows 1–471 are contiguous and unique (**passed**); lineage generations 0–471 are complete (**passed**) and child IDs are unique. Around the boundary, gens459, 460 and 461 each have one log row and 192 lineage children. Generation-best snapshots are dense from gen101 onward; population snapshots now run through gen450 and checkpoints through gen250.

The dashboard was restarted and is active at `http://localhost:8050`; `/` and `/theater` returned HTTP 200 and the live API reports gen471/1500. Start it with `.venv\Scripts\python.exe scripts\dashboard.py --run ga2`. The trainer remains running in the background. Final 500-over champion analysis, final evaluation replays, full clip export/contact sheets, champion and Theater screenshots, final pytest/physics validation, and the final report are scheduled after generation 1500.


## Prompt 5B final closeout — ga2 complete

This final record supersedes the earlier Prompt 5B sections marked in progress.

`ga2` resumed from the saved generation 460 and completed generation 1500. The gen460 `resume` event is in `runs/ga2/run_config.json`; resumed evaluation began at gen461. Two later exact-state recoveries were also recorded: gen600 after the trainer process exited, and gen670 after a Windows sharing violation while publishing gen678 snapshots. The snapshot writer now retries transient Windows sharing violations and publishes NPZ snapshots atomically. The run finished without further interruption.

The final audit is `runs/ga2/resume_audit.json`, generated by `scripts/audit_resume.py --run ga2`. It verifies all 1,500 log generations are unique and contiguous; lineage covers generations 0–1500, has exactly 192 child records per generation, and has no duplicate child IDs. At the gen460 restart boundary, generations 459, 460 and 461 each have exactly one log row and 192 lineage children. The same row counts hold around the gen600 and gen670 recoveries. There are 1,500 log rows and 288,192 lineage records.

The validation champion is gen1425 with mean normalized score 1.1974. The gen1500 generation best scored 1.1897 and released at a mean 136.42 km/h. Mean normalized generation-best scores rose from the first ten to the last ten generations: Slogger 0.808→1.223, Technician 0.669→1.165, Tailender 0.879→1.181. Mean speed was 140.88 km/h in the first ten and 137.12 km/h in the last ten, staying well clear of the former 115 km/h exploit.

The table below uses the gen1425 champion's held-out validation (200 overs per archetype), the frozen specialist rewards from the plan matrix, and the legacy ga1500 champion adapted to environment v2 (2,000 overs per archetype). Runs are runs per over.

| Batter | ga2 normalized | ga2 reward/over | ga2 wickets/over | ga2 runs/over | Specialist | Specialist reward/over | ga1500-v2 normalized | ga1500-v2 reward/over | ga1500-v2 wickets/over | ga1500-v2 runs/over |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Slogger | 1.251 | 3.694 | 0.440 | 5.665 | P3 | -0.802 | 0.620 | -7.593 | 0.290 | 9.085 |
| Technician | 1.162 | 3.763 | 0.210 | 1.165 | P4 | +0.708 | 0.651 | -5.873 | 0.170 | 4.518 |
| Tailender | 1.179 | 6.964 | 0.340 | 1.520 | P5 | +3.972 | 0.664 | -1.657 | 0.375 | 3.957 |

The ga2 champion exceeds its matching specialist's reward against all three archetypes. The independent 500-over analysis also beat each specialist (sampled fitness: Slogger 4.40 vs P3 -0.97; Technician 3.99 vs P4 0.68; Tailender 6.62 vs P5 4.30). Its plans differ by batter: Slogger is full-heavy (45%; modal sequence full/full/yorker/yorker/full/full), Technician is good-heavy (72%; good/short/good/good/good/good), and Tailender is full-heavy (58%; full/full/yorker/full-toss/yorker/full). Swing is outswing-biased for all three. Length-distribution Jensen–Shannon divergences are 0.5240 bits for Slogger/Technician, 0.0260 for Slogger/Tailender and 0.6674 for Technician/Tailender; the first and last styles are similar, while the Technician style differs.

The 500-over action saturation audit (absolute tanh output ≥0.95) found speed saturation of 68.6% for Slogger (24.2% negative, 44.4% positive), 48.1% for Technician (0.7% negative, 47.4% positive), and 61.1% for Tailender (14.0% negative, 47.1% positive). Elevation and azimuth saturation are 0% for every batter. Swing is still fully saturated negative at 100% for all three, so the gen75 swing behavior persists at gen1500; this remains a known policy limitation.

`runs/ga2/analysis/` contains `learning_curve.png`, `normalized_scores.png`, `strategy_slogger.png`, `strategy_technician.png`, `strategy_tailender.png`, `plan_comparison.png`, `wickets_economy.png`, and `ga1500_vs_ga2.png`. Final champion replays were regenerated with `.venv\Scripts\python.exe scripts\evaluate.py --name ga2 --checkpoint champion.npz --overs 30`; each archetype has 30 overs in schema v2 with 4 ms trajectory samples. The replay folder contains the eight scheduled checkpoint replay sets, champion replays, five median-fitness typical genomes at gen0 and gen1 (10 overs per archetype each), and seven population swarms. Each swarm covers 192 genomes × three archetypes (576 deliveries) with shared archetype seeds and 4 ms trajectories.

### Clip exports and screenshots

`outputs/clips/index.json` lists each clip, its source data, duration and both formats. All 11 clips have 9:16 (1080×1920) and 16:9 (1920×1080) H.264 High, CRF16, yuv420p encodes at 60 fps, timing sidecars, two-second opening and closing holds, and a 3×3 contact sheet. `ffprobe` verified codec, profile, pixel format, frame rate and resolutions for all 22 MP4 files.

| Clip | Duration | Contact sheet |
|---|---:|---|
| C1 network morph | 20.0 s | `outputs/clips/C1_network_morph/contact_sheet.png` |
| C1 milestone holds | 12.0 s | `outputs/clips/C1_network_milestones/contact_sheet.png` |
| C2 Brain Cam — Slogger | 7.683 s | `outputs/clips/C2_cam_slogger/contact_sheet.png` |
| C2 Brain Cam — Technician | 7.700 s | `outputs/clips/C2_cam_technician/contact_sheet.png` |
| C2 Brain Cam — Tailender | 7.650 s | `outputs/clips/C2_cam_tailender/contact_sheet.png` |
| C3 population swarm | 24.0 s | `outputs/clips/C3_population_swarm/contact_sheet.png` |
| C4 weight morph | 24.0 s | `outputs/clips/C4_weight_morph/contact_sheet.png` |
| C5 decision map | 14.0 s | `outputs/clips/C5_decision_map/contact_sheet.png` |
| C6 family tree | 14.0 s | `outputs/clips/C6_family_tree/contact_sheet.png` |
| C7 learning curve | 16.0 s | `outputs/clips/C7_learning_curve/contact_sheet.png` |
| C8 v1 vs v2 | 7.9 s | `outputs/clips/C8_v1_vs_v2/contact_sheet.png` |

Champion-based Brain Explorer screenshots D1–D9 are in `docs/screens/` and show validation champion gen1425. Evolution Theater screenshots are `docs/screens/Theater_ga2_gen_0001.png`, `Theater_ga2_gen_0100.png`, `Theater_ga2_gen_0500.png`, and `Theater_ga2_gen_1500.png`.

### Validation, timing and dashboard

The full suite passed: **52 passed** with `.venv\Scripts\python.exe -m pytest -q`. `scripts/validate_physics.py` completed successfully: the sample yorker hit the stumps; good, short, bouncer and full-toss samples were classified as expected; the 1,000-delivery timing benchmark averaged 0.476 ms (p95 0.692 ms). It regenerated the side, top and noise-scatter figures in `outputs/validation/`.

Summed committed `seconds_per_generation` values are 23,036.94 seconds (about 6 h 24 min, mean 15.36 s/gen) across the 1,500 logged generations. From gen461 onward they total 15,622.18 seconds; from gen671 onward, 12,399.25 seconds. The final resumed session reports 12,464.87 seconds in `run_config.json` including startup and session overhead.

The local dashboard is available at `http://localhost:8050`; `.venv\Scripts\python.exe scripts\dashboard.py --run ga2` is the launch command. Its API currently reports gen1500/1500 complete and champion gen1425 at 1.1974. An additional stop-and-relaunch attempt at finalization was blocked by command policy, so I left the already-running dashboard process in place; `/`, `/brain`, and `/theater` all returned HTTP 200 and the API reported the completed run. Historical model snapshots for gens2–99 were not present in the original run; the Theater shows the nearest saved generation over that span instead of inventing genomes.

## Step 6 — renderer and video-grade clips

- New `video/` Remotion + three.js project; GPU rendering via ANGLE (about 20–35
  frames/s at 1080p). Data exporters: `scripts/export_video_data.py`,
  `scripts/export_brain_data.py`.
- **Hawk-Eye footage (R1–R6, 105 clips):** hero wicket balls (real time, 0.25×, 0.1×;
  broadcast, stump cam, batter's end, side-on), full champion overs (broadcast with
  accumulating trails + pitch map), the swarm (all 192 genomes bowling the same first
  ball at gens 1–1500, single gens + montages + orbit), v1 vs v2 split screens, 30-over
  pitch-map build-ups for both runs, and generation-0 chaos.
- **Brain clips (A1–A9, 40 clips):** brain morph gen 1→1500 (+ milestones), Brain Cam per
  batter (real time and 0.5×, plus the v1 brain), behaviour swarm, weight heatmaps,
  decision maps, family tree, learning curves (ga2 and ga1500), re-test bars, v1 vs v2
  brains (gauges, landing spots, speed histogram).
- Findings while building: in behaviour space the ga2 population's spread in mean
  length fell from 2.71 m (gen 1) to 0.57 m (gen 1500) and in speed from 5.6 to 1.0 km/h.
  Over the batter-confidence × last-length sweep, the gen-1 brain bowls a yorker
  everywhere; the champion splits the map into full, good and short regions.
- The v1 brain cam shows the loophole plainly: bouncers at 115 km/h, full balls/yorkers
  at 150 km/h, every output pinned to its limit.
- Tests: `tests/test_video_data.py` (sub-millimetre interpolation, pitch-map/3D bounce
  agreement, sidecar frames vs replay event times).
