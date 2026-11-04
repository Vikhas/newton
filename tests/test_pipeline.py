import time

import pytest
from fastapi.testclient import TestClient

from backend import config, pipeline
from tests.fixtures.synth import make_synthetic_clip


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(config, "OFFLINE", True)


@pytest.fixture
def synth(tmp_path):
    src = tmp_path / "throw.mp4"
    return src, make_synthetic_clip(src)


def test_run_synthetic_end_to_end(synth):
    src, truth = synth
    meta = pipeline.ingest(src)
    assert meta["capture_fps"] == pytest.approx(240) and meta["n_frames"] == truth["n_frames"]
    r = pipeline.run(meta["clip_id"], truth["click_xy"], truth["surface_y_px"])
    assert 10 <= r["release_frame"] <= 20
    assert r["cutoff_frame"] == r["release_frame"] + 60
    assert max(r["fit_frames"]) <= r["cutoff_frame"] and min(r["fit_frames"]) >= r["release_frame"]
    assert r["error_cm"] is not None and r["error_cm"] < 2.0
    assert abs(r["prediction"]["landing_px"][0] - truth["landing_x_px"]) < 8
    assert r["fit"]["g_warning"] is False and abs(r["fit"]["g_check"] - 9.81) < 0.8
    assert (config.CACHE_DIR / "clips" / meta["clip_id"] / "result.json").exists()


def test_ingest_is_idempotent(synth):
    src, _ = synth
    assert pipeline.ingest(src)["clip_id"] == pipeline.ingest(src)["clip_id"]


def test_result_flags_bad_g(synth):
    src, truth = synth
    meta = pipeline.ingest(src)
    r = pipeline.run(meta["clip_id"], truth["click_xy"], truth["surface_y_px"], capture_fps=30)
    assert r["fit"]["g_warning"] is True


def test_landing_not_visible(tmp_path):
    src = tmp_path / "short.mp4"
    truth = make_synthetic_clip(src, dur=0.6)
    meta = pipeline.ingest(src)
    r = pipeline.run(meta["clip_id"], truth["click_xy"], truth["surface_y_px"])
    assert r["actual"] is None and r["error_cm"] is None
    assert r["prediction"]["landing_px"][0] > truth["click_xy"][0]


def test_ball_not_found_is_clear_error(tmp_path):
    src = tmp_path / "empty.mp4"
    truth = make_synthetic_clip(src, dur=0.3, blank_frames=set(range(100)))
    meta = pipeline.ingest(src)
    with pytest.raises(ValueError, match="ball"):
        pipeline.run(meta["clip_id"], truth["click_xy"], truth["surface_y_px"])


def test_api_flow(synth):
    from backend.app import app

    src, truth = synth
    c = TestClient(app)
    with open(src, "rb") as f:
        meta = c.post("/api/clips", files={"file": ("throw.mp4", f, "video/mp4")}).json()
    cid = meta["clip_id"]
    assert c.get(meta["first_frame_url"]).status_code == 200
    assert c.post(f"/api/clips/{cid}/run", json={"ball_xy": truth["click_xy"],
                                                  "surface_y_px": truth["surface_y_px"]}).json() == {"status": "queued"}
    deadline = time.time() + 60
    while time.time() < deadline:
        s = c.get(f"/api/clips/{cid}/status").json()
        if s["stage"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert s["stage"] == "done", s
    r = c.get(f"/api/clips/{cid}/result").json()
    assert r["error_cm"] < 2.0
    assert c.get("/").status_code == 200
    assert c.get(f"/media/{cid}/../../etc/passwd").status_code == 404


def test_clip_ending_before_cutoff_is_clear_error(tmp_path):
    src = tmp_path / "cut.mp4"
    truth = make_synthetic_clip(src, dur=0.3)
    meta = pipeline.ingest(src)
    with pytest.raises(ValueError, match="ends less than"):
        pipeline.run(meta["clip_id"], truth["click_xy"], truth["surface_y_px"])


def test_ingest_flags_truncated_clip(synth, monkeypatch):
    src, _ = synth
    monkeypatch.setattr(config, "MAX_FRAMES", 50)
    meta = pipeline.ingest(src)
    assert meta["n_frames"] == 50 and meta["truncated"] is True


def test_run_with_custom_ball_size(tmp_path):
    src = tmp_path / "pingpong.mp4"
    truth = make_synthetic_clip(src, ball_diam_m=0.04)
    meta = pipeline.ingest(src)
    r = pipeline.run(meta["clip_id"], truth["click_xy"], truth["surface_y_px"], ball_diam_m=0.04)
    assert r["error_cm"] < 2.0 and r["ball_diam_m"] == 0.04 and r["fit"]["g_warning"] is False
    wrong = pipeline.run(meta["clip_id"], truth["click_xy"], truth["surface_y_px"])  # assumes a tennis ball
    assert wrong["fit"]["g_warning"] is True


def test_api_accepts_ball_size_and_description(synth):
    from backend.app import RunRequest

    req = RunRequest(ball_xy=[1, 2], surface_y_px=3, ball_diam_cm=4.0, ball_prompt="ping pong ball")
    assert req.ball_diam_cm == 4.0 and req.ball_prompt == "ping pong ball"
