"""Narration -> one WAV per line with Kokoro (offline TTS), plus timings.json.

Run with a venv that has kokoro-onnx + soundfile:
    python video/tts.py MODEL.onnx VOICES.bin
Lines starting with "@ASK " are the on-screen question, spoken by the asker voice.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

HERE = Path(__file__).parent
OUT = HERE / "out" / "voice"


def main(model, voices):
    spec = json.loads((HERE / "narration.json").read_text())
    k = Kokoro(model, voices)
    OUT.mkdir(parents=True, exist_ok=True)
    timings = {}
    for scene in spec["scenes"]:
        rows = []
        for i, line in enumerate(scene["lines"]):
            ask = line.startswith("@ASK ")
            text = line.removeprefix("@ASK ")
            voice = spec["asker"] if ask else spec["narrator"]
            audio, sr = k.create(text, voice=voice, speed=spec["speed"], lang="en-us")
            loud = np.flatnonzero(np.abs(audio) > 1e-3)  # trim leading/trailing silence only
            audio = audio[loud[0]:loud[-1] + 1]
            path = OUT / f"{scene['id']}_{i}.wav"
            sf.write(path, audio, sr)
            rows.append({"file": path.name, "ask": ask, "text": text, "dur": round(len(audio) / sr, 3)})
        timings[scene["id"]] = rows
        print(scene["id"], [r["dur"] for r in rows])
    (OUT / "timings.json").write_text(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
