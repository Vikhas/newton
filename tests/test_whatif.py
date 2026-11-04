import io
import json
import subprocess
import wave

import pytest
from fastapi.testclient import TestClient

from backend import config, pipeline
from backend.track import ffmpeg_exe
from backend.voice import WhatIf, WhatIfError, parse_whatif, to_wav
from tests.fixtures.synth import make_synthetic_clip
from tests.test_agent import ScriptedClient


@pytest.fixture(scope="module")
def synth_run(tmp_path_factory):
    """One offline pipeline run shared by the what-if tests (each test still gets its own CACHE_DIR)."""
    d = tmp_path_factory.mktemp("whatif")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "CACHE_DIR", d / "cache")
    mp.setattr(config, "OFFLINE", True)
    truth = make_synthetic_clip(d / "t.mp4")
    meta = pipeline.ingest(d / "t.mp4")
    result = pipeline.run(meta["clip_id"], truth["click_xy"], truth["surface_y_px"])
    mp.undo()
    return d / "cache", result


@pytest.fixture
def cache(synth_run, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", synth_run[0])
    monkeypatch.setattr(config, "OFFLINE", True)
    return synth_run[1]


def test_parse_whatif_moon():
    spark = ScriptedClient([json.dumps({"g": 1.62, "speed_scale": None, "label": "Moon"})])
    assert parse_whatif("what if this were on the moon", spark) == WhatIf(g=1.62, speed_scale=None, label="Moon")


def test_parse_whatif_rejects_out_of_range():
    with pytest.raises(WhatIfError, match="gravity or throw speed"):
        parse_whatif("zero gravity", ScriptedClient([json.dumps({"g": 0, "speed_scale": None, "label": "Zero g"})]))


def test_parse_whatif_rejects_unsupported():
    with pytest.raises(WhatIfError):
        parse_whatif("what if it were red", ScriptedClient([json.dumps({"g": None, "speed_scale": None, "label": ""})]))


def test_parse_whatif_bad_json_is_whatif_error():
    with pytest.raises(WhatIfError):
        parse_whatif("moon", ScriptedClient(["??", "??"]))


def test_whatif_moon_lands_farther(cache):
    out = pipeline.whatif(cache["clip_id"], text="What if this were on the Moon?")
    assert out["change"] == {"g": 1.62, "speed_scale": None} and out["label"] == "Moon"
    ghost = out["ghost"]
    assert ghost["landing_distance_m"] > 5 * ghost["real_distance_m"]
    assert ghost["arc_px"][0][2] == pytest.approx(cache["release_frame"])
    assert ghost["landing_frame"] > cache["prediction"]["landing_frame"]


def test_whatif_offscreen_flag(cache):
    assert pipeline.whatif(cache["clip_id"], text="on the moon")["ghost"]["offscreen"] is True
    assert pipeline.whatif(cache["clip_id"], text="on jupiter")["ghost"]["offscreen"] is False


def test_whatif_text_route(cache):
    from backend.app import app
    r = TestClient(app).post(f"/api/clips/{cache['clip_id']}/whatif", json={"text": "twice as hard"})
    assert r.status_code == 200 and r.json()["change"]["speed_scale"] == 2.0


def test_whatif_route_unsupported_is_422(cache):
    from backend.app import app
    r = TestClient(app).post(f"/api/clips/{cache['clip_id']}/whatif", json={"text": "make it purple"})
    assert r.status_code == 422 and "gravity or throw speed" in r.json()["detail"]


def test_whatif_audio_route_offline(cache):
    from backend.app import app
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000), w.writeframes(b"\0\0" * 2400)
    r = TestClient(app).post(f"/api/clips/{cache['clip_id']}/whatif",
                             files={"audio": ("q.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200 and r.json()["transcript"] == "what if this were on the moon"


def test_to_wav_converts_webm(tmp_path):
    src = tmp_path / "q.webm"
    subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                    "-c:a", "libopus", str(src)], check=True)
    wav = to_wav(src.read_bytes(), "audio/webm;codecs=opus")
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)


def test_empty_recording_is_whatif_error():
    with pytest.raises(WhatIfError, match="didn't catch"):
        to_wav(b"not audio", "audio/webm")


def test_model_busy_is_not_reported_as_unsupported():
    from backend.clients import ModelError

    class Busy:
        def chat(self, *a, **k):
            raise ModelError("429 rate limited")

    with pytest.raises(ModelError):
        parse_whatif("on the moon", Busy())
