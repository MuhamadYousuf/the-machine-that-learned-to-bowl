# AI Bowler project specification

## Project goal

Build an AI cricket bowler that learns through neuroevolution: a genetic
algorithm evolves the weights of a small neural network without gradients.
The agent learns to take wickets against simulated batters. The final
deliverable is a polished, physically believable broadcast/Hawk-Eye-style
video for LinkedIn showing the change from early deliveries such as wides and
full tosses to planned yorkers and outswingers.

## Milestones

1. Ball physics engine and validation.
2. Batter model and delivery outcome resolution.
3. Six-ball-over environment and neuroevolution trainer.
4. Pseudo-3D Hawk-Eye renderer in pygame-ce.
5. Full training run and highlight video: generation 1 versus generation N
   split-screen, with a pitch-map heatmap tightening over generations.
6. README, GIFs, and post.

## Coordinate system and bowling setup

Use SI units. The x axis runs along the pitch from the bowler's stumps at
x=0 to the batter's stumps at x=20.12 m. Positive y is the off side for a
right-handed batter; z is height above the ground. The bowler delivers over
the wicket from the right arm. The release is approximately 1.22 m behind the
bowler's stumps, 0.30 m to the off side of the stumps line, and 2.20 m high.
Bowling around the wicket is out of scope for the initial project.

Use y > +0.89 m at the batter's popping crease for the approximate one-day
off-side wide guideline. Use y < -0.45 m for the simplified leg-side wide
limit beyond the batter's pads. The simulator uses ball-centre thresholds and
does not model umpire judgement or the batter's reach.

## Bowler action

Each delivery uses four continuous network outputs in [-1, 1], mapped to
physical controls:

- Release speed: 115-150 km/h.
- Vertical release angle.
- Horizontal release angle and direction, relative to a line aimed at the
  middle stumps.
- Swing amount and direction, with -1 outswing and +1 inswing for a
  right-handed batter.

With positive y toward off side, outswing curves toward positive y and
inswing curves toward negative y. Zero azimuth offset aims at the middle
stumps from the off-side release position. The current azimuth action range is
-2 to +2 degrees. Gaussian execution noise is added to the release so the
bowler has to learn a robust aim.

## Batter archetypes (Milestone 2)

- Slogger: weakest to on-stump yorkers and slower balls at full/good length;
  punishes full tosses, half-volleys, and short balls.
- Technician: weakest to good-length outswing in the outside-off corridor;
  defends straight balls and leaves balls wider than 0.5 m outside off.
- Tailender: weakest to body-line short balls/bouncers, with a secondary
  weakness to straight full balls; scores less overall.

WeaknessZone and ScoringZone records hold Gaussian centres, widths, and
strengths/bonuses for any subset of length, line, late movement, crease height,
and speed delta. Full toss length is treated as zero for matching. Each batter
is sampled with parameter and zone variation up to 15%. Batter expectations
track length, line, late movement, and crease speed with EMA alpha 0.5;
surprise includes all four values. Repetition and successful scoring increase
confidence, while misses, edges, and wickets reduce it.

## Milestone 3: over environment and neuroevolution

An episode is six legal balls. Wides and beamer no-balls are re-bowled; six
illegal balls end an episode with a further -10 reward. The incoming batter is
sampled from the requested archetype. A wicket brings in a fresh, uniformly
sampled archetype and batter. Strike rotation is omitted. Observation history
resets for the incoming batter.

Per-delivery reward is Rao-Blackwellised as `25*wicket_prob - expected_runs -
3*extra`, where extra is one for a wide or no-ball. Batter and episode state
still transition from the sampled result. Sampled runs, wickets, and extras are
tracked separately for reporting.

The 17-element observation is normalized to approximately [-1, 1]:

| Index | Value |
|---:|---|
| 0-2 | Current batter one-hot: Slogger, Technician, Tailender |
| 3 | Batter confidence, [0, 1] |
| 4 | Current batter deliveries faced / 12, clipped to [0, 1] |
| 5 | Legal balls already bowled in this over / 6 |
| 6 | Runs conceded this over / 12, clipped to [0, 1] |
| 7-11 | Older of last two deliveries to this batter: length/14 (full toss=0), line/1 m, late movement/0.5 m, crease speed scaled from 115-150 km/h, valid flag |
| 12-16 | Most recent delivery to this batter, with the same five fields |

Missing history is five zeros, including a zero valid flag. Line and movement
are clipped to [-1, 1].

The policy is a pure-NumPy MLP, 17 -> 24 -> 24 -> 4, with tanh activations and
one flat genome vector. Outputs are speed, elevation, azimuth, and swing
controls in [-1, 1].

The Step 4 GA uses a population of 192, copies the top 8 genomes, selects by
4-way tournaments, uniform crossover at p=0.5, and mutates 10% of genes with
Gaussian sigma cosine-annealed from 0.08 to 0.015. Each generation evaluates
16 overs per archetype (48 balanced overs per genome) with common scenario
seeds across genomes; the seeds change each generation and all elites are
re-evaluated. Evaluation is parallelized with spawn-safe worker processes.

Fitness is normalized per archetype using the 2,000-over plan matrix:
`(mean_reward - P6_random_reward) / (ORACLE_specialist_reward - P6_random_reward)`.
The final genome fitness is the mean of the three normalized scores, where 0
matches random bowling and 1 matches the hand-authored specialist plan. The
run's `run_config.json` stores those baselines, all fixed validation seeds,
hyperparameters, and checkpoint schedules. `log.csv` records the total and
per-archetype normalized scores, sampled wickets/runs/extras, speed, length
mix, and seconds per generation. A held-out check runs every 25 generations
for 200 overs per archetype. Its scores, pitch-map points, and ball-by-ball
length patterns are appended to `validation.jsonl`; `champion.npz` is the
candidate with the highest held-out normalized score. Its matching validation
record is atomically mirrored to `champion_validation.json` for the dashboard;
the JSONL file remains the complete validation history.

Generation checkpoints are saved at 1, 5, 10, 25, 50, 100, 250, 500, 750,
1000, 1250, and 1500 (plus the configured final generation). Whole populations
and common-seed swarm replays are recorded at generations 1, 10, 50, 250,
1000, and 1500. Median-fitness chaos replays are saved for generations 0 and
1. The champion and selected generation-best checkpoints receive 30 overs per
archetype of replay.

`scripts/evaluate.py` writes gzip-compressed replay JSON at
`runs/<name>/replays/gen_XXXX_<archetype>.json.gz` (or
`champion_<archetype>.json.gz`). Schema version 2 stores the checkpoint and
run metadata, per-over sampled summaries, and each delivery's noisy action,
release, key trajectory points, features, outcome, reward, and trajectory
samples every 4 ms (the 2 ms integrator contributes every second sample).
Event times include release, bounce, crease, stumps, and a batting-contact
proxy at the popping crease. The proxy is labelled because this model does not
simulate a bat collision. Positions are metres, time is seconds, speed is
km/h, and angles are degrees. `aibowler.replay.load_replay()` reads gzip
schema-v2 and legacy plain schema-v1 files.

`scripts/analyze_run.py` writes the normalized learning curve with P1-P8 and
ORACLE references, per-archetype normalized scores, validation-based
wickets/economy, champion strategy maps, and champion/specialist/random
comparisons. Strategy figures summarize 500 sampled deliveries per batter.

`scripts/dashboard.py --run ga1500` serves a read-only, self-contained local
dashboard at `http://localhost:8050`. It refreshes every five seconds, reads
partial log writes safely, shows training and validation charts, renders the
latest champion pitch maps and ball patterns, and can evaluate any two saved
checkpoints side-by-side. It works with `ga200` logs and their schema-v1
replays as well as current runs.

## Visual target (Milestones 4-5)

Use a night-match broadcast look with a green pitch, white crease lines, a
glowing ball trail, a pitch-map with dots colored by outcome, and a HUD with
generation, wickets, and economy. The highlight video compares the first and
final generations and shows the pitch-map heatmap tightening over generations.

## Physics requirements (Milestone 1)

Use a fixed integration step no larger than 2 ms and RK4 or semi-implicit
Euler. Model gravity, quadratic drag (nominal Cd about 0.4), and lateral swing
force proportional to speed squared before the first bounce. Tune maximum
realistic swing to roughly 0.3-0.6 m over flight. At bounce, reflect vertical
velocity using pitch restitution with a small impact-angle dependency, retain
horizontal speed according to the calibrated pitch value, and add a small
random seam deviation from an angle drawn from N(0, seam_sd). Current
calibration uses Cd=0.40, base restitution=0.67 at a 0.15 rad impact angle,
restitution slope=0.55/rad, and horizontal retention=0.96. Track at most one
meaningful bounce and stop behind the stumps or when the ball stops.

For each trajectory report sampled time, position, and velocity; bounce
point or no bounce for a full toss; state at the batter's popping crease and
stumps plane; stumps contact; wide and beamer flags; bounce distance from the
batter's stumps; and length category. A beamer is a no-bounce delivery above
approximately 1.0 m at the batter's popping crease.

Length categories use the distance from the batter's stumps to the bounce:
yorker at 2 m or less; full above 2 m through 6 m; good above 6 m through
8 m; short above 8 m through 10.5 m; bouncer above 10.5 m. A delivery with no
bounce before the batter's stumps is a full toss.

Cross-pitch flight time is measured from the ball crossing x=0 (the bowler's
stumps plane) to x=20.12 (the batter's stumps plane). Sampled trajectory time
still starts at release; the API also reports release-to-stumps elapsed time.

Keep the API friendly to later batters and fast enough for large training
runs. The initial implementation reports timing, but the Python loop and
per-delivery arrays still need profiling in the actual training workload.

## Step 1 validation

The physics tests cover flight time, straight-line lateral motion, opposite
swing directions, length categories, bouncer height, yorker contact, and
deterministic execution noise. The validation script writes side, top, and
noisy pitch-map views to outputs/validation/.

## Batter and outcome model (Milestone 2)

DeliveryFeatures contains bounce length, line at the batter's popping crease
relative to off stump, height and speed at that crease, late lateral movement,
stumps contact, pitching outside leg, wide/beamer flags, and length category.
Late movement is the lateral difference over the last six metres from a
straight extrapolation of the preceding six-metre path.

BatterState tracks confidence, expected length, expected line, expected late
movement, expected crease speed (initialized at 135 km/h), balls faced, and
the previous delivery. Expectations use an EMA with alpha=0.5. Similar
deliveries increase confidence and set the batter; surprise includes pace as
well as length, line, and late movement. Fast balls raise miss/edge risk and
reduce scoring control. A repeated, expected slow pace is easier to score from;
a fast-to-slow change remains surprising and can exploit the Slogger's
mistiming zone. All three archetypes score off full tosses and half-volleys.
Confidence reduces wicket probability, while surprising sequences increase
it.

Slogger is aggressive, scores heavily from full tosses, half-volleys,
short-wide and leg-side balls, and is most vulnerable to yorkers on the
stumps. Technician defends straight balls and leaves wide off-side balls;
good-length outswing in the outside-off corridor creates edge risk.
Tailender scores less and is most vulnerable to body-line short balls and
bouncers, with a secondary weakness to straight full balls.

Outcome resolution is explicit probability logic, not a learned model. It
samples dots, runs, wides, beamer no-balls, bowled, LBW, caught-behind, and
caught-field outcomes. Wides and no-balls are illegal deliveries; a beamer
no-ball includes one automatic run plus sampled batting runs. The outcome
includes the full categorical probability vector, expected runs, wicket
probability, sampled shot, and wagon-wheel angle/distance for scoring shots.

## Step 5 — fixed environment, crash-safe GA, and Brain Explorer

### Versioned policy and bouncer rule

Environment v1 retains the original 17-value observation and the 17→24→24→4
policy for historical checkpoints. Current environment v2 appends input 17,
`bouncers_this_over / 2` clipped to [0, 1], and uses an 18→24→24→4 policy.
Loading a v1 checkpoint automatically adapts the observation to its first 17
values. The Brain Cam replays old checkpoints under their recorded environment
version. To compare the old ga1500 champion under v2, use
`scripts/evaluate_legacy_champion.py`, which runs v2 but feeds only the first
17 inputs to the legacy policy.

A short-pitched delivery counts toward the over's bouncer quota only when its
length category is `bouncer` and its height at the batter's popping crease is
at least `BOUNCER_SHOULDER_HEIGHT_M = 1.25 m`. The first two count normally.
The third and later are forced no-balls: one automatic extra run, zero batter
runs, no wicket, and no legal-ball advancement. The existing reward charges
`-3 × extras` for each one.

### Pace and pattern-aware batter expectations

The batter begins with an expected crease speed of 135 km/h. Pace contributes
to both the four-dimensional variance surprise and the established
fast-to-slow change-of-pace term. Faster deliveries raise miss and edge risk
and reduce scoring control. A slow delivery near an established slow-speed
expectation is easier to score from; a large negative `speed_delta_kmh` after
fast bowling can remain surprising.

The settled-slow scoring opportunity is
`slow_expected = clip((135 - speed_kmh) / 20, 0, 1) ×
exp(-0.5 × (speed_delta_kmh / 10)²)`. The pace contribution to batting
quality is `1.50 × slow_expected - 0.60 × fastness`, with
`fastness = clip((speed_kmh - 100) / 45, 0, 1)`. This reserves the easy-scoring
bonus for an expected ball below 135 km/h; a sudden slow ball after fast
bowling has a much smaller bonus and can still surprise the batter.

For each measurement `x` (length, line, late movement, crease speed), the
model maintains an EMA mean `μ` and variance `v`, with `α = 0.5`:

```text
v' = (1 - α) × (v + α × (x - μ)²)
μ' = α × x + (1 - α) × μ
z = |x - μ| / sqrt(v + floor²)
s(x) = 1 - exp(-z² / 2)
```

The floors are respectively 1.5 m, 0.15 m, 0.10 m, and 6 km/h. The base
surprise is the mean of `s(x)` over those dimensions, plus the small
state-relative pace-change term, clipped to [0, 1]. Lag-2 memory predicts the
ball two deliveries back. Its similarity uses the same length/line/movement/
speed metric as `repetition_similarity`; pattern confidence is updated as
`c' = clip(0.65 × lag2_similarity + 0.35 × c, 0, 1)`. Final surprise is the
minimum of base surprise and `(1-c) × base_surprise + c × (1-lag2_similarity)`.

For a strict bouncer/yorker alternation, starting from a fresh Slogger state,
the measured per-ball surprise is:

| Sequence | B | Y | B | Y | B | Y |
|---|---:|---:|---:|---:|---:|---:|
| Surprise | 0.437 | 0.605 | 0.386 | 0.158 | 0.048 | 0.018 |

The alternating pattern is read by ball 4, and confidence reaches 0.985 after
ball 6 in this worked seeded example.

### Scripted-plan matrix and calibration

`outputs/batter/plan_matrix.csv` contains the 2,000-over-per-plan, per-batter
matrix for P1–P9 and ORACLE. P9 reproduces the Step 4 bouncer@150/yorker@115,
outswing alternation. P5 uses two shoulder-height bouncers and four yorkers
per over so its Tailender specialist plan respects the new legal quota. The
fresh single-ball table is in `outputs/batter/fresh_batter_table.csv`; yorkers
remain the best single ball against all three archetypes, so no bouncer
retuning was needed.

| Archetype | P6 random | Specialist / ORACLE | P7 fast yorker | P8 slow yorker | P9 champion clone |
|---|---:|---:|---:|---:|---:|
| Slogger | -18.687 | P3 -0.802 | -5.001 | -8.996 | -34.416 |
| Technician | -18.128 | P4 +0.708 | -2.856 | -7.056 | -33.667 |
| Tailender | -12.768 | P5 +3.972 | +2.566 | -1.669 | -30.235 |

The required ranking holds: P8 < P7 for every archetype; neither yorker-only
plan or P9 beats any matching specialist; and P3/P4/P5 remain the best
specialist against Slogger/Technician/Tailender respectively. Training
normalizers are re-frozen from each archetype's P6 and ORACLE expected reward
in this v2 matrix. The Step 2 calibration rerun remains in range at 2,000
overs: random expected wickets/runs per over are Slogger 2.16% / 13.33,
Technician 1.12% / 11.30, Tailender 2.43% / 7.34; generic-good rates are
3.90% / 2.55% / 3.81% wickets and 0.48 / 0.52 / 0.18 expected runs.
High-surprise exploit first-ball wicket rates are 28.06% / 29.63% / 26.08%,
falling by ball 6 to 2.29% / 1.48% / 1.98%. A Slogger full toss yields 3.04
expected runs. The fresh-batter single-ball table (wicket probability / runs
/ reward) is:

| Length | Slogger | Technician | Tailender |
|---|---:|---:|---:|
| Full toss | 1.92% / 2.563 / -2.084 | 0.87% / 1.860 / -1.644 | 1.48% / 0.339 / +0.030 |
| Yorker | 5.76% / 0.625 / +0.816 | 3.67% / 0.028 / +0.889 | 5.13% / 0.028 / +1.254 |
| Full | 1.14% / 0.954 / -0.668 | 0.83% / 1.764 / -1.557 | 2.37% / 0.197 / +0.394 |
| Good | 1.00% / 0.204 / +0.047 | 0.79% / 0.096 / +0.101 | 1.26% / 0.029 / +0.286 |
| Short | 1.27% / 0.949 / -0.632 | 0.85% / 0.029 / +0.184 | 1.84% / 0.029 / +0.433 |
| Bouncer | 1.62% / 0.776 / -0.370 | 0.87% / 0.029 / +0.187 | 3.65% / 0.028 / +0.883 |

Correctly landed yorkers remain the best isolated single ball against each
fresh batter; full tosses and half-volleys have meaningful scoring outcomes.

The Step 4 ga1500 champion was evaluated for 2,000 overs per archetype in
environment v2 with its 17-input adapter. It scored 0.620 normalized against
Slogger, 0.651 against Technician, and 0.664 against Tailender (mean 0.645).
Its per-over reward / wickets / runs were respectively -7.593 / 0.290 / 9.085,
-5.873 / 0.170 / 4.518, and -1.657 / 0.374 / 3.957. Full results are stored
in `outputs/batter/ga1500_champion_v2.json`.

### Crash-safe evolution and lineage

New runs write `resume_state.npz` every 10 generations. The file contains the
evaluated and next populations, fitness, generation and mutation position,
champion, elite IDs, next genome ID, NumPy and Python RNG states, and common
scenario seeds. It is written through `resume_state.tmp.npz`, flushed and
fsynced, then atomically replaced; `resume_state_prev.npz` is retained as a
fallback. `train.py --resume <name>` restores the next population and RNG,
truncates later log, validation, lineage, checkpoint, population, and replay
outputs, and records a timestamped resume event in run config. New runs assign
unique genome IDs and append parent, crossover, elitism, mutation, and fitness
records to `lineage.csv.gz`.

### Brain Explorer and ga2

`scripts/dashboard.py --run ga2` serves the live training dashboard at
`http://localhost:8050` and `/brain`. It remains read-only and tolerates a
partial final log record. Brain Explorer includes the policy graph and weight
maps, ball-by-ball Brain Cam, decision map, permutation importance over 2,000
states, neuron diagnostics, a shared PCA population view, checkpoint weight
statistics, and the ga2 family tree. Screenshot exports are saved under
`docs/screens/`. The page and charting use local Python/Canvas code and work
offline. ga1500 has no lineage log and displays that limitation explicitly.

The clean `ga2` pre-check (seed 20261109) completed 100 generations in
1,469.3 seconds. No-balls were 0%, champion action saturation was 24.701%,
all normalized-score windows rose (Slogger 0.808 to 1.103, Technician 0.669
to 1.046, Tailender 0.879 to 1.106), and all three archetypes used good/full/
short deliveries over 15%. Pairwise Jensen-Shannon divergences were 0.5839,
0.0888, and 0.6704. The only failed condition was the required speed trend:
140.88 km/h in generations 1-10 versus 136.21 km/h in generations 91-100
(-4.67 km/h). Speed remained well above 120 km/h. Per the go/no-go rule, the
continuation had not started at that review point. Prompt 5B later clarified
the gate intent as preventing a return to the old 115 km/h exploit, replacing
the trend test with a generation-best mean-speed floor of 125 km/h. Generation
100 measured 137.47 km/h and passed the revised gate; the exact-saved-state
continuation is recorded below and in PROGRESS.md.

### Prompt 5B: dense snapshots, Theater, and clip export

Before freezing the v2 normalizers, the pace coefficient in batting quality
was increased from `0.25` to `0.60`:

```text
fastness = clip((crease_speed_kmh - 100) / 45, 0, 1)
pace_quality = 1.50 × slow_expected × pace_repeat - λ × fastness
Δpace_quality = -(0.60 - 0.25) × fastness = -0.35 × fastness
```

`pace_quality` is added to batter scoring quality, so the stronger negative
fastness term makes a fast ball harder to score from. At 140 km/h it changes
batter quality by about -0.311; at 150 km/h by -0.350. This addresses the
old model's nearly free slow delivery while preserving a sudden slow ball as a
change of pace through `slow_expected`. The P6 random and ORACLE samples and
the plan matrix were then measured with the final coefficient; the stored
normalizers therefore include this correction.

The P9 bouncer/yorker exploit's `expected_fitness` near -34 is the sampled
environment reward, not the matrix's `fitness` column (which omits the
sixth-illegal-ball terminal penalty). The plan repeats its third above-shoulder
bouncer because an illegal ball does not advance the legal-ball count. In a
seeded 200-over audit for each batter it produced 6 no-balls and 10 total
deliveries per over: 4 legal deliveries followed by 6 illegal bouncer attempts.
The reward breakdown is approximately -6 runs, -18 from the six -3 extra
penalties, and -10 from the six-illegal-ball cap. `MAX_ILLEGAL_BALLS = 6`
ends the over, so this failure mode is bounded; even a worst-case over can
contain at most 6 legal plus 6 illegal deliveries (12 total).

The exact resume from generation 100 writes a small `best_genomes/gen_XXXX.npz`
for each new generation and a complete population snapshot every 25
generations. Snapshot writes do not consume random values. The original
generation 1–100 run predates dense snapshots, so only its available saved
checkpoints could be backfilled; missing intermediate genomes are shown by
nearest-saved lookup rather than invented. Exact resume preserves its old
state schema and logs `gate_override` and `resume` events.

`scripts/dashboard.py --run ga2` serves the live dashboard at port 8050,
`/brain` for the Brain Explorer, and `/theater` for synchronized network,
validation curve, sampled over, and population views. Brain Cam accepts a
seeded simulation or a saved replay and animates the recorded 4 ms trajectory
through release, bounce, crease, and outcome times. Shared drawing primitives
live under `src/aibowler/viz/`; `scripts/export_brain_clips.py` imports those
primitives and encodes caption-free 60 fps H.264 clips with two-second holds,
frame timing JSON sidecars, and contact sheets. Use `--labels` for reference
labels or `--png-sequence --transparent` for transparent PNG frame sequences.


### Prompt 5B live continuation

The `ga2` generation-100 speed gate is overridden as intended: gen-best mean release speed must remain at least 125 km/h. `run_config.json` contains both the override and exact-state resume events. Training is continuing in background from the existing state. As of this documentation update, generation 345 is complete at 136.55 km/h; the best normalized generation scores are 1.226 / 1.122 / 1.147. Last-50 timing averages 15.18 sec/gen; estimated time to gen1500 is 4.87 hours. Current validation champion generation 325 has mean normalized score 1.159 (Slogger 1.186, Technician 1.138, Tailender 1.155).

The live log is contiguous through generation 275 and lineage generations 0–275 are present. Dense per-generation genome snapshots are complete from generation 101 onward. Missing gen2–99 genomes predate the snapshot feature; UI lookup selects the nearest saved model and does not fabricate them. Population snapshots currently cover 1, 10, 50, 100, 125, 150, 175, 200, 225, 250, 275, with 1, 5, 10, 25, 50, 100, 250 as per-run generation checkpoints. Dashboard routes are live at port 8050, including the new Evolution Theater at `/theater`. The final training, analysis, validation, replay, clip, screenshot and saturation results remain pending until generation 1500.


### Prompt 5B — gen460 exact resume

The saved `resume_state.npz` was gen460. `train.py --resume ga2` resumed exactly at gen461 and logged the new `resume` event at generation 460. The stale gen461 log/lineage tail from before the power-off was truncated, then rebuilt from the saved RNG and next-population state. At this update the run has reached generation 471/1500. The latest best normalized scores are Slogger 1.232, Technician 1.150, Tailender 1.142; mean speed is 137.45 km/h. Last-50 timing averages 15.67 sec/gen, about 4.48 hours remaining. The current validation champion is gen425 with score 1.172.

Post-resume continuity check: log rows 1–471 are contiguous and unique (**passed**); lineage generations 0–471 are complete (**passed**) and child IDs are unique. Around the boundary, gens459, 460 and 461 each have one log row and 192 lineage children. Generation-best snapshots are dense from gen101 onward; population snapshots now run through gen450 and checkpoints through gen250.

The live read-only dashboard is running at port 8050. Full final analysis and clip/screenshot artifacts remain pending until generation 1500.


### Prompt 5B final artifact record — ga2 generation 1500

This final artifact record supersedes the earlier Prompt 5B notes marked in progress.

The final `ga2` state is `status: complete`, `last_completed_generation: 1500`. It resumed from `resume_state.npz` at generation 460, continued with gen461, and logged the resume in `runs/ga2/run_config.json`. Recovery events at gen600 and gen670 are also recorded; the latter followed a Windows sharing violation while publishing a best-genome snapshot. NPZ snapshot writes now retry transient sharing violations and use atomic replacement. A fresh audit is reproducible with `.venv\Scripts\python.exe scripts\audit_resume.py --run ga2` and is stored in `runs/ga2/resume_audit.json`.

The audit finds 1,500 unique contiguous rows in `log.csv`, lineage generations 0–1500 with 192 children at every generation, and unique child IDs. Gens459–461, 599–601 and 669–671 each contain one log row and 192 lineage children. The dataset contains 288,192 lineage rows. This confirms that the gen460 restart produced no gap or duplicate around the saved generation.

The final held-out champion is gen1425, mean normalized validation score 1.19743. Its per-batter normalized scores are 1.25135 / 1.16223 / 1.17869; reward per over is 3.69379 / 3.76330 / 6.96408; wickets per over are 0.440 / 0.210 / 0.340; and runs per over are 5.665 / 1.165 / 1.520. Each champion reward exceeds the corresponding matrix specialist P3 / P4 / P5 reward (-0.80160 / +0.70833 / +3.97192). Full champion-versus-specialist and legacy ga1500-v2 results are recorded in PROGRESS.md. The gen1500 generation best has mean release speed 136.417 km/h and normalized archetype scores 1.25561 / 1.15573 / 1.15790.

The 500-over champion style report found full-heavy Slogger (45%), good-heavy Technician (72%), and full-heavy Tailender (58%) plans. Length-distribution Jensen–Shannon divergences (bits) are Slogger/Technician 0.5240, Slogger/Tailender 0.0260, Technician/Tailender 0.6674. At |tanh output| ≥0.95, swing is 100% negative for every batter; elevation and azimuth are 0% saturated; speed saturation is 68.6%, 48.1%, and 61.1% for Slogger, Technician, and Tailender. Negative swing saturation remains a learned limitation.

`runs/ga2/analysis/` contains all requested final plots: `learning_curve.png`, `normalized_scores.png`, the three `strategy_<archetype>.png` files, `plan_comparison.png`, validation-based `wickets_economy.png`, and `ga1500_vs_ga2.png`. The final champion's three replay files in `runs/ga2/replays/` each contain 30 overs per archetype. Schema v2 stores sampled deliveries and event times; trajectory samples are spaced at 4 ms, including release, bounce, crease, stumps and bat-contact proxy timing fields. Typical gen0/gen1 replays retain five median-fitness genomes each and 10 overs per archetype; swarms save 192 genomes × three archetypes (576 first-ball trajectories) at the scheduled generations with shared seeds.

The offline clip exporter produced 11 clips in `outputs/clips/`, with both 9:16 and 16:9 MP4s, timing sidecars and one 3×3 contact sheet per clip. `outputs/clips/index.json` is the authoritative index of source data, files, dimensions and durations. All 22 MP4s were checked with ffprobe: H.264 High profile, CRF16, yuv420p, 60 fps, at 1080×1920 or 1920×1080. Videos use the dark background, have no labels by default and include two-second stills at each end. The optional `--labels` variant and `--png-sequence --transparent` mode remain available.

The final champion screenshots D1–D9 and Theater captures for generations 1, 100, 500 and 1500 are in `docs/screens/`. Dashboard routes `/`, `/brain`, and `/theater` are served from port 8050; the `ga2` API reports generation 1500 complete and champion generation 1425 with score 1.19743. Use `.venv\Scripts\python.exe scripts\dashboard.py --run ga2` to start the read-only dashboard.

The complete test suite passed (52 tests). `scripts/validate_physics.py` completed with expected sample categories and an average 0.476 ms / p95 0.692 ms over 1,000 deliveries. Generation snapshots for original gens2–99 do not exist; the Theater uses nearest-saved genomes across that legacy sparse span rather than fabricating checkpoints. See the final closeout section of PROGRESS.md for per-archetype tables, complete clip durations and timing totals.

## Rendering (Step 6)

Rendering moved from the planned pygame/moderngl renderer to a Remotion + three.js
project in `video/` (see `video/README.md`). The same project will hold the final
edit, so footage never leaves the pipeline that edits it.

- **Data flow:** `scripts/export_video_data.py` (replays, swarms) and
  `scripts/export_brain_data.py` (network weights over generations, brain-cam overs,
  learning curves, lineage, decision maps, behaviour swarm) write compact JSON to
  `video/public/data/`. Rendering never re-runs the simulation.
- **Output contract:** 1920×1080 and 1080×1920, 60 fps, H.264 High, CRF 16, yuv420p
  BT.709, 2 s still holds at both ends, no text. One JSON sidecar per clip with
  frame-exact events.
- **Scene:** pitch 3.05 m wide with a procedural texture; popping creases 1.22 m in
  front of the stumps (3.66 m long), bowling creases 2.64 m, return creases ±1.32 m;
  stumps 0.711 m tall, 22.86 cm wide with bails; good-length band 6–8 m from the
  batter's stumps; outfield disc fading into a dark void. Ball drawn 1.2–2.4× real size
  depending on camera distance, with a height-dependent shadow. Trails: cyan before
  the bounce, gold after. Bowled: stumps tilt and bails fly. LBW: dashed red predicted
  path into the stumps plus a pulsing wicket zone. Bat contact (crease-time proxy):
  small white flash.
- **Palette:** background #060a13, teal #49debe (positive weights / v2), pink #fb7185
  (negative weights / wickets), amber #fbbf24, orange #fb923c (v1 / pinned outputs),
  violet #a78bfa. Fitness ramp: dull blue → violet → teal → near-white.
- **Brain clips:** networks draw only a stable top-K edge set (union of each key
  generation's top edges, so morphs don't flicker), edge width/opacity by |w| or
  |w × source activation|, nodes glowing by activation. The behaviour swarm places
  each genome at (mean bounce length, mean release speed) over one over per batter,
  dots interpolated by fitness rank between saved populations. The decision map sweeps
  batter confidence × last length (the input pair with the most varied champion
  decisions).
