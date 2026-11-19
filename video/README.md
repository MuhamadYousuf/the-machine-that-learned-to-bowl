# Video project (Remotion + three.js)

Everything visual for the final video lives here: the Hawk-Eye-style 3D replays,
the 2D brain/evolution clips, and later the edited video itself. Python stays the
single source of truth for the simulation; this project only renders data that
`scripts/export_video_data.py` and `scripts/export_brain_data.py` write to
`public/data/`.

## Setup

    npm install
    ..\.venv\Scripts\python.exe ..\scripts\export_video_data.py     # replays, swarms (+ brain data)
    ..\.venv\Scripts\python.exe ..\scripts\export_brain_data.py     # brain / evolution data

Preview interactively with `npm run studio`.

## Rendering

    python jobs/make_footage_jobs.py                     # writes jobs/footage.json
    node render.mjs jobs/footage.json --skip-done        # 3D footage passes R1–R6
    node render.mjs jobs/brain_clips.json --skip-done    # 2D brain clips A1–A9
    python contact_sheets.py                             # contact sheets + out/renders/index.json

`render.mjs` bundles once and renders each job with GPU WebGL (ANGLE). Every MP4
is 60 fps, H.264 High, CRF 16, yuv420p (BT.709), with 2 s of still frames at the
start and end. Next to each MP4 is a JSON **sidecar** with frame-exact events
(release, bounce, crease, end of each ball; generation segments for swarms) for
syncing captions, sound effects and cuts. All footage is text-free.

## Layout

| Path | What |
|---|---|
| `src/hawkeye/` | 3D scene: `Stadium` (pitch, creases, stumps), `BallFlight` (ball, shadow, trails, bounce ring, LBW predicted path), `cameras.ts`, compositions `Over`, `Swarm`, `Compare`, `PitchBuild` |
| `src/brain/` | 2D SVG clips: `NetworkView`, gauges, pitch map, `BrainMorph`, `BrainCam`, `EvolutionClips` (swarm, heatmaps, decision map, family tree, curves, re-test bars, v1 vs v2) |
| `src/lib/` | data loading, trajectory interpolation, palette |
| `jobs/` | render job lists |
| `out/renders/` | rendered footage, sidecars, contact sheets, `index.json` |

## Coordinates

Simulation: x along the pitch from the bowler's stumps (0) to the batter's (20.12 m),
y = off side for a right-hander, z = up. Scene (three.js): X = y, Y = z, Z = x
(a cyclic permutation, so handedness is preserved). From the broadcast camera
behind the bowler, the off side appears on the left, as on TV.

## Cameras

`broadcast` (high, behind the bowler), `batterEnd` (low, looking back up the pitch),
`sideOn` (square leg; pans with the ball in portrait), `sideHalf` (side view of the
batter's half), `topDown`, `pitchMap` (straight down on the batter's half; pitch runs
left→right in landscape panels), `stumpCam` (low behind the batter's stumps), `orbit`.
Poses are aspect-aware so split screens frame correctly.

## The film (`src/reel/`)

`Reel-9x16` (Reels) and `Reel-4x5` (LinkedIn), 30 fps, about 2:46. One script drives voice, picture, captions, music and sound:

    audio\.venv_cb\Scripts\python.exe audio\make_vo_cb.py --takes 3    # Chatterbox VO, Whisper-checked takes → public/reel/vo.wav, vo_timeline.json
    audio\.venv_cb\Scripts\python.exe audio\align_captions.py          # caption words = script words, timed by Whisper
    audio\.venv\Scripts\python.exe audio\reel_plan.py                 # music moods + sound cues → public/reel/plan_9x16.json
    audio\.venv\Scripts\python.exe audio\make_audio.py ../reel        # score + SFX + ducking + -14 LUFS → public/reel/mix.wav
    node render.mjs jobs/reel.json                                     # → out/film/reel_{9x16,4x5}.mp4

| File | What |
|---|---|
| `reel/script.json` | Narration, one entry per line: scene, `exag` (emotion intensity), `cfg` (pacing) and the pause after it |
| `audio/make_vo_cb.py` | Chatterbox TTS cloned from `reel/voice_ref.wav` (a Kokoro `bm_george` read). Each line gets several takes; faster-whisper transcribes them and the take closest to the script wins; a light cinematic voice chain |
| `src/reel/World.tsx` | The daylight stage: sky, sun with soft shadows, low-poly stadium (instanced crowd, floodlights, trees), pitch and stumps that can be knocked over |
| `src/reel/Mannequin.tsx` | Procedural bowler (run-up, bound, windmill arm, follow-through) and batter (two-bone IK, bat keyframes per shot), keeper and fielders |
| `src/reel/Nets.tsx` | 16 practice nets seen from above, one real genome per lane; result flashes and labels projected from 3D |
| `src/reel/ActOne/Two/Three.tsx` | The nine scenes (hook, sees, brain, gen1, cheat, fix, rise, plans, outro), each timed to words in the narration |
| `src/reel/ui.tsx` | Captions (spoken word in yellow, key words boxed), panels, meters, generation badge, stamps, title card |

The voice lives in its own Python 3.12 venv (`audio/.venv_cb`: torch CUDA, chatterbox-tts, faster-whisper). Set `HF_HOME` to a drive with ~3 GB free for the model download.

## Earlier documentary cuts

The first edit and the documentary cuts (Blender stadium, Kokoro voice) are kept for reference. The first edit (`Film-9x16`, `Film-16x9`, ~108 s) is built from one script and one timeline,
so voice, picture, captions, music and sound effects can never drift apart:

    audio\.venv\Scripts\python.exe audio\make_vo.py      # Kokoro voiceover + word timings  → public/edit/vo.wav, vo_timeline.json
    python edit\make_plan.py                             # edit decision list per format     → public/edit/plan_*.json (+ copies clips to public/clips)
    audio\.venv\Scripts\python.exe audio\make_audio.py   # original score + SFX + final mix  → public/edit/mix.wav (-14 LUFS)
    node render.mjs jobs/film.json                       # → out/film/AI_learns_to_bowl_{9x16,16x9}.mp4

| File | What |
|---|---|
| `edit/script.json` | Narration: one entry per line with speaking speed, pause after, scene, and words to emphasise in captions |
| `edit/make_plan.py` | The edit: shots per scene, anchored to words in the narration (`"L20@edges"`), overlays and sound cues. Ball sounds (bounce, bat, edge, pad, stumps) are added automatically from each clip's sidecar |
| `audio/make_vo.py` | Kokoro TTS (voice `af_heart`, change with `--voice` or in the script); trims and polishes each line; timings come from Kokoro itself |
| `audio/make_audio.py` | Synthesised score that follows the story (hook drone → chaos → build → hard stop on the cheat → dark pulse → main theme → piano outro), synthesised SFX, sidechain ducking under the voice, loudness normalisation |
| `src/edit/` | `Film` (shots, zoom/blur transitions, shake, grain), `Captions` (word-synced, emphasis in amber), `Overlays` (generation counter, stats, chips, flashes, stamp, rules, wicket slams), `ResultBars`, `EndCard` |

The voice lives in its own Python 3.12 venv (`audio/.venv`: torch CUDA + kokoro) because Kokoro does not support Python 3.14 yet.
To use a licensed music track instead of the generated score, drop it in and mix it in `make_audio.py` in place of `music(plan)`.

## Slow motion

Replays store trajectory samples every 4 ms. The renderer interpolates linearly,
with the bounce sample inserted exactly; against the 2 ms integrator output the
error stays below 1 mm (`tests/test_video_data.py`), so 0.25× and 0.1× slow motion
need no re-simulation.
