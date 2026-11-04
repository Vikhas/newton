"""Constants and environment for Newton."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parents[1]
load_dotenv(ROOT / ".env")
CACHE_DIR = ROOT / "cache"

BALL_DIAM_M = 0.067        # tennis ball
G_EARTH = 9.81
CUTOFF_S = 0.25            # fit only on this much flight after release
PLAYBACK_FPS = 30          # player video always re-encoded at this rate
MAX_ROUNDS = 4
RMS_TARGET_CM = 1.0
MIN_FIT_POINTS = 6
BLUR_JUMP = 0.30           # drop frames whose mask area jumps more than this
G_SANE = (8.5, 11.0)
MAX_FRAMES = 1500          # ~6 s at 240 fps; longer clips are cut and flagged

# Model endpoints (per https://github.com/meta-models/meta-model-cookbook). Each can be overridden by an env var of the same name.
META_BASE_URL = os.getenv("META_BASE_URL", "https://api.meta.ai/v1")
META_ASR_URL = os.getenv("META_ASR_URL", "https://api.meta.ai/v1/asr/transcribe")
SAM_MODEL = os.getenv("SAM_MODEL", "sam-3.1")
SPARK_MODEL = os.getenv("SPARK_MODEL", "muse-spark-1.3")
VOICE_MODEL = os.getenv("VOICE_MODEL", "muse-voice-transcribe-1.0")
OPEN_BASE_URL = os.getenv("OPEN_BASE_URL", "https://inference.baseten.co/v1")
OPEN_MODEL = os.getenv("OPEN_MODEL", "meta-models/muse-spark-open")
# SAM needs a noun phrase that matches the ball; tried in order until one tracks (a plain "ball" can miss).
BALL_PROMPTS = list(dict.fromkeys([os.getenv("BALL_PROMPT", "ball"), "tennis ball", "white ball", "yellow ball",
                                   "orange ball", "small ball"]))
HAND_PROMPT = os.getenv("HAND_PROMPT", "hand")
OFFLINE = os.getenv("NEWTON_OFFLINE") == "1"
