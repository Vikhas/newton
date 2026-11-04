"""Spoken what-if -> text (Muse Voice Transcribe) -> a physics change (Muse Spark)."""
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from backend import prompts
from backend.agent import AgentError, ask
from backend.track import ffmpeg_exe

UNSUPPORTED = "I can change gravity or throw speed — try 'on the Moon' or 'twice as hard'"


class WhatIfError(ValueError):
    """The question could not be turned into a supported physics change."""


@dataclass
class WhatIf:
    g: float | None
    speed_scale: float | None
    label: str


def to_wav(audio: bytes, mime: str) -> bytes:
    """Mono 16-bit 24 kHz WAV, the format Muse Voice Transcribe expects."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in", Path(d) / "out.wav"
        src.write_bytes(audio)
        try:
            subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                            "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(dst)], check=True)
        except subprocess.CalledProcessError as e:
            raise WhatIfError("I didn't catch that — hold the button while you speak") from e
        return dst.read_bytes()


def transcribe(audio: bytes, mime: str, meta) -> str:
    return meta.transcribe(to_wav(audio, mime or ""))


def parse_whatif(text: str, spark) -> WhatIf:
    try:
        reply = ask(spark, prompts.WHATIF, {"question": text}, prompts.WHATIF_SCHEMA)
    except AgentError as e:
        raise WhatIfError(UNSUPPORTED) from e
    g, speed = reply.get("g"), reply.get("speed_scale")
    ok_g = g is None or (isinstance(g, (int, float)) and 0.1 <= g <= 300)
    ok_s = speed is None or (isinstance(speed, (int, float)) and 0.1 <= speed <= 10)
    if not (ok_g and ok_s) or (g is None and speed is None):
        raise WhatIfError(UNSUPPORTED)
    return WhatIf(g=None if g is None else float(g), speed_scale=None if speed is None else float(speed),
                  label=str(reply.get("label") or "What-if"))
