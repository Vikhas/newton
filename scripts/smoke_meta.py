"""One live call per model. Run: .venv/bin/python scripts/smoke_meta.py

Needs MODEL_API_KEY (or META_API_KEY) and BASETEN_API_KEY in the environment or .env.
"""
import io
import json
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend import config  # noqa: E402
from backend.clients.meta import MetaClient  # noqa: E402
from backend.clients.open import OpenClient  # noqa: E402
from backend.track import encode_video  # noqa: E402


def tiny_clip(path: Path):
    """Two seconds of a yellow ball rolling across a grey wall."""
    frames = path.parent / "frames"
    frames.mkdir()
    for i in range(60):
        img = np.full((360, 640, 3), 120, np.uint8)
        cv2.circle(img, (60 + 8 * i, 200), 18, (0, 220, 230), -1)
        cv2.imwrite(str(frames / f"{i:05d}.jpg"), img)
    encode_video(frames, path, 30)


def silent_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 24000)
    return buf.getvalue()


def check(name, fn):
    try:
        print(f"OK   {name}: {fn()}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}: {e}")


def main():
    meta, open_ = MetaClient(), OpenClient()
    ping = [{"role": "user", "content": 'Reply with exactly {"ok": true}'}]
    check(config.SPARK_MODEL, lambda: meta.chat(ping))
    check(config.OPEN_MODEL, lambda: open_.chat(ping))
    with tempfile.TemporaryDirectory() as d:
        clip = Path(d) / "smoke.mp4"
        tiny_clip(clip)
        check(config.SAM_MODEL, lambda: json.dumps(
            {k: len(v) for k, v in meta.segment_video(clip, "ball", {"smoke": 1}).items()}) + " frames per object")
    check(config.VOICE_MODEL, lambda: repr(meta.transcribe(silent_wav())))


if __name__ == "__main__":
    main()
