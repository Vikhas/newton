# Newton

Newton watches a phone video of a thrown ball and works out the physics from only the first quarter-second of flight. It marks where the ball will land before it gets there, then answers what-if questions by voice, such as "what if this were on the Moon?", and replays the same throw under the new physics.

Entry for the Meta Global AI Developer Hackathon: Indie Developer track, theme *Push the Frontier*.

**Demo video (2:11):** _[link ](https://youtu.be/OknQUSb4C2s?si=nCqFxBmWjIrat1Af)_

> **Built during the hackathon.** All code in this repository was written during the event (Oct 27 – Nov 4, 2026). The first commit is dated Nov 4, 2026; see `git log`.

## What it does

1. **Upload** a slow-motion clip filmed side-on from a tripod, with a ball of known size (a tennis ball, 6.7 cm).
2. **Click** the ball, then the floor.
3. Newton tracks the ball with **SAM 3.1**, then **freezes** the video 0.25 s after release. The label reads *"Prediction made from frames A–B only"*.
4. A glowing **predicted arc** and a target ring appear where the ball will land.
5. Playback **resumes**, the real ball lands, and a readout shows the **error in cm**.
6. Ask **"What if this were on the Moon?"** by voice or text. A violet ghost ball replays the same launch under lunar gravity.

## How it works

```
phone clip ──► SAM 3.1 (video tracking) ──► ball path + ball-size scale
                                                │   frames ≤ cutoff only
                                                ▼
                     ┌──── Muse Spark: plans the physics model ◄─────┐
                     │            (projectile / drag / fit g / drop frames)
                     ▼                                               │
               Python fit tools ──► Muse Spark Open: critiques fit ──┘  (max 4 rounds, stop < 1 cm RMS)
                     │
                     ▼
           landing prediction + arc ──► web player: freeze, arc, reveal, error
                     ▲
Muse Voice Transcribe ──► Muse Spark parses the what-if ──► re-simulate ──► ghost ball
```

| Meta model | Where it runs | Job in Newton |
|---|---|---|
| **SAM 3.1** (`sam-3.1`) | Meta Model API, Responses + Files API, video mode | Tracks the ball through the whole clip (stable object ids, box-local masks). The mask gives the centre and pixel diameter for every frame. A second pass on "hand" finds the release frame. |
| **Muse Spark** (`muse-spark-1.3`) | Meta Model API | The physicist. It reads three keyframes and the tracked path, chooses the model and fitting plan, writes the explanation, and turns a spoken what-if into a parameter change. |
| **Muse Spark Open** (`meta-models/muse-spark-open`, open weights) | **Baseten** Model APIs | The critic. It checks each fit's residuals and asks to drop blurred frames, add drag, or recheck the scale. |
| **Muse Voice Transcribe** (`muse-voice-transcribe-1.0`) | Meta Model API `/asr/transcribe` | Turns the spoken what-if into text. |

Request formats follow Meta's official cookbook. See [`docs/meta-api-notes.md`](docs/meta-api-notes.md), which also records the live checks.

### Physics

- **Scale from the ball itself.** Scale = 6.7 cm ÷ the median mask diameter, so no calibration sheet is needed.
- **Fit.** Least squares on `x = x0 + vx·t` and `y = y0 + vy·t − ½gt²`, with optional linear drag. A second fit also solves for g. A result near 9.8 m/s² is the honesty check that the scale and frame rate are right, and the UI warns when it isn't.
- **Landing.** Where the ball's centre reaches the floor plus one radius. The target ring's size comes from the fit's parameter covariance.

### The honesty guarantee

The prediction uses **only frames up to the cutoff** (release + 0.25 s). The code enforces this, not the prompt:

- `AgentTools` holds only the sliced points, and the scale and blur cleaning use only those frames.
- The keyframes sent to Muse Spark are release, mid and cutoff.
- `tests/test_agent.py::test_tools_never_see_future` and `test_no_post_cutoff_data_in_prompts` pin this down.

## Run it

Requirements: Python ≥ 3.11 and ffmpeg. If ffmpeg isn't installed, the bundled `imageio-ffmpeg` build is used.

```bash
python3 -m venv .venv            # or: uv venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env             # add MODEL_API_KEY and BASETEN_API_KEY
.venv/bin/uvicorn backend.app:app --port 8811
# open http://localhost:8811
```

**No keys? Use offline mode.** `NEWTON_OFFLINE=1` swaps every model for a local stand-in: colour tracking and canned replies. The whole flow then runs on a synthetic throw:

```bash
.venv/bin/python -c "from tests.fixtures.synth import make_synthetic_clip as m; m('cache/synth.mp4')"
NEWTON_OFFLINE=1 .venv/bin/uvicorn backend.app:app --port 8811
```

Other commands:

```bash
.venv/bin/pytest                          # 77 tests: physics, tracking, API clients, agent loop, what-if, end to end
.venv/bin/python scripts/smoke_meta.py    # one live call per Meta model
.venv/bin/python scripts/batch_eval.py    # every take in clips/manifest.json -> results/takes.csv + error_hist.png
```

Every model response is cached in `cache/`, so re-running a clip is instant and costs nothing.

## Results

**Rendered throw (a 240 fps clip where the true landing is known):** in offline mode, with colour tracking, the error is 0.3 cm from 61 frames before the cutoff, with a gravity check of 9.79 m/s². With the live models, the error is **3.7 cm** from 60 frames (release frame 16 to cutoff 76) and the gravity check reads 9.52 m/s². SAM 3.1's masks on the rendered ball come out about 3% wide, which skews the scale taken from the ball's size. The release frame was entered by hand, because SAM doesn't recognise the rendered hand.

**Real footage: a public sample clip.** `BallTossOut.mp4` comes from the Open Source Physics Tracker [mechanics videos](https://opensourcephysics.github.io/tracker-website/download/mechanics_videos.zip). It is a real hand toss at 30 fps (not slow motion), with a small white ball and a meter stick in the throw plane. This run used every live model: SAM 3.1, Muse Spark and Muse Spark Open.

| | |
|---|---|
| Frames Newton saw | 8 (release frame 10 to cutoff 18, about 0.27 s) |
| Scale from the ball's size | 0.2515 cm/px. The meter stick reads 0.2500 cm/px. |
| Gravity check | 9.85 m/s² |
| Predicted vs actual crossing of the landing line | 109.8 cm vs 110.6 cm: **0.7 cm off** (2.9 px), predicted 0.37 s ahead |
| Agent | Planner chose a gravity-only fit. The critic accepted it at 0.07 cm RMS. The blur filter dropped frame 17, where the ball passes behind the stick. |

There was no floor in view, so the "landing surface" was a line at y = 420 px. The ball's diameter, 2.75 cm, was entered from the meter stick. With the default tennis-ball size, the gravity check reads 24 m/s² and the UI warns before trusting the prediction. To reproduce, upload the clip, then:
- click the ball near the hand (≈ 82, 272) and the line at y ≈ 420
- set capture fps 30, ball size 2.75 cm, ball "white ball"

**Not measured yet:** a batch of our own slow-motion throws. When there is one, report the median and spread over every take, not just the best one.

`scripts/batch_eval.py` writes the per-take table to `results/takes.csv` and the error histogram to `results/error_hist.png`.

## Demo video

The submission video is generated from this repo: Kokoro text-to-speech for the narration, Playwright recording the running app, an animated HTML page fed the real `result.json` data, and ffmpeg for assembly. The narration script is in `video/narration.json`. Output goes to `video/out/`, which is gitignored.

```bash
python video/tts.py kokoro-v1.0.onnx voices-v1.0.bin          # in a venv with kokoro-onnx + soundfile
node video/record.mjs <playwright-core dir> <chromium> hero second whatif
PYTHONPATH=. .venv/bin/python video/assemble.py plan
node video/record.mjs <playwright-core dir> <chromium> title pipe agent outro
PYTHONPATH=. .venv/bin/python video/assemble.py build       # -> video/out/newton-demo.mp4
```

The app has to be running on port 8811 during recording. The what-if scene feeds the spoken question through Chromium's fake microphone, so it goes through Muse Voice Transcribe for real.

## Filming tips

- Phone on a tripod, square to the throw, 3–4 m away, slow-motion mode (120 or 240 fps).
- A bright ball against a plain wall, in good light.
- Start recording with the ball in your hand, and keep the landing in frame.
- If the gravity check is far from 9.8, the capture fps is usually wrong (some exports play slow-mo back at 30 fps). Set it in the capture-fps field.

## Roadmap

Pendulums, springs, collisions and bounces. Then any experiment you can film. Muse Image could restyle the background for the Moon replay.

## Layout

```
backend/   app.py (API) · pipeline.py · track.py · trajectory.py · physics.py · agent.py · prompts.py · voice.py
           clients/ meta.py (SAM 3.1, Muse Spark, Muse Voice) · open.py (Muse Spark Open on Baseten) · fake.py (offline)
web/       index.html (player, canvas overlay, mic, reasoning panel)
scripts/   smoke_meta.py · batch_eval.py
tests/     pytest suite + synthetic clip generator
docs/      meta-api-notes.md (Meta Model API request formats and live checks)
video/     narration.json · tts.py · record.mjs · scenes.html · assemble.py (demo video pipeline)
```
