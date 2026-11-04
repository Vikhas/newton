import json
import re

import pytest

from backend import config, pipeline
from backend.agent import AgentError, AgentTools, run_agent
from backend.clients.fake import FakeClient
from backend.trajectory import TrackPoint
from tests.fixtures.synth import make_synthetic_clip

SCALE, H, FPS, RELEASE, CUTOFF, SURFACE = 0.0025, 720, 240, 10, 70, 680


def make_points(n=200, vx=2.5, vy=3.5, glitch=None, glitch_px=25):
    pts = []
    for i in range(n):
        t = max(0, i - RELEASE) / FPS
        x = 150 + vx * t / SCALE
        y = 450 - (vy * t - 4.905 * t * t) / SCALE
        if glitch and i == glitch:
            y += glitch_px  # a bad mask
        pts.append(TrackPoint(frame=i, x_px=x, y_px=y, diam_px=26.8))
    return pts


def tools(points=None):
    return AgentTools(points or make_points(), RELEASE, CUTOFF, SCALE, H, FPS, SURFACE)


class ScriptedClient(FakeClient):
    """Segments like FakeClient but answers chat from a script, recording every message."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    def chat(self, messages, images=(), json_schema=None):
        self.sent.append(messages)
        assert self.responses, "unexpected extra chat call"
        return self.responses.pop(0)


def plan(**kw):
    return json.dumps({"model": "projectile", "fit_g": False, "drop_frames": [], "recheck_scale": False,
                       "note": "plan", **kw})


def crit(verdict, frames=()):
    return json.dumps({"verdict": verdict, "frames": list(frames), "reason": f"{verdict} because"})


EXPLAIN = json.dumps({"lines": ["Projectile model.", "4.3 m/s at 54°.", "Dropped one bad frame."]})


def test_tools_never_see_future():
    t = tools()
    assert max(p.frame for p in t.points) <= CUTOFF and min(p.frame for p in t.points) >= RELEASE
    assert max(t.keyframe_ids()) <= CUTOFF


def test_drop_frames_ignores_post_cutoff():
    assert tools().drop_frames([CUTOFF + 5]) == []


def test_drop_frames_keeps_minimum():
    t = tools()
    keep = [p.frame for p in t.points][:3]
    assert t.drop_frames([p.frame for p in t.points if p.frame not in keep]) == []
    assert len(t.active()) == len(t.points)


def test_recheck_scale_ignores_bad_diameters():
    pts = make_points()
    for p in pts[RELEASE:RELEASE + 20]:
        p.diam_px = 40  # blurred, oversized masks
    t = tools(pts)
    r = t.recheck_scale()
    assert r["scale_new"] == pytest.approx(0.067 / 26.8) and t.scale == r["scale_new"]


def test_critic_drop_then_accept_two_rounds():
    bad = RELEASE + 30
    spark = ScriptedClient([plan(), plan(drop_frames=[bad]), EXPLAIN])
    critic = ScriptedClient([crit("drop_outliers", [bad]), crit("accept")])
    r = run_agent(tools(make_points(glitch=bad)), [b"jpg"] * 3, spark, critic)
    assert r.rounds == 2 and bad not in r.fit.frames
    roles = [x["role"] for x in r.reasoning]
    assert {"planner", "critic", "tool", "summary"} <= set(roles)


def test_never_accept_stops_at_four():
    spark = ScriptedClient([plan()] * 4 + [EXPLAIN])
    critic = ScriptedClient([crit("drop_outliers")] * 4)
    assert run_agent(tools(), [b"jpg"] * 3, spark, critic).rounds == 4


def test_add_drag_switches_model():
    spark = ScriptedClient([plan(), plan(model="projectile_drag"), EXPLAIN])
    critic = ScriptedClient([crit("add_drag"), crit("accept")])
    assert run_agent(tools(), [b"jpg"] * 3, spark, critic).fit.drag is True


def test_accept_with_high_rms_keeps_going():
    spark = ScriptedClient([plan(), plan(drop_frames=[RELEASE + 30]), EXPLAIN])
    critic = ScriptedClient([crit("accept"), crit("accept")])
    r = run_agent(tools(make_points(glitch=RELEASE + 30, glitch_px=80)), [b"jpg"] * 3, spark, critic)
    assert r.rounds == 2 and r.fit.rms_cm < config.RMS_TARGET_CM


def test_explain_failure_is_not_fatal():
    spark = ScriptedClient([plan(), "nope", "nope"])
    critic = ScriptedClient([crit("accept")])
    r = run_agent(tools(), [b"jpg"] * 3, spark, critic)
    assert r.rounds == 1 and "summary" not in [x["role"] for x in r.reasoning]


def test_no_post_cutoff_data_in_prompts():
    spark = ScriptedClient([plan(), EXPLAIN])
    critic = ScriptedClient([crit("accept")])
    run_agent(tools(), [b"jpg"] * 3, spark, critic)
    for msgs in spark.sent + critic.sent:
        text = json.dumps(msgs)
        frames = [int(m) for m in re.findall(r'\\"frame\\": (\d+)', text)]
        assert all(f <= CUTOFF for f in frames)


def test_malformed_json_raises_agent_error():
    with pytest.raises(AgentError):
        run_agent(tools(), [b"jpg"] * 3, ScriptedClient(["not json", "not json"]), ScriptedClient([]))


def test_malformed_json_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OFFLINE", True)
    src = tmp_path / "t.mp4"
    truth = make_synthetic_clip(src)
    meta = pipeline.ingest(src)
    r = pipeline.run(meta["clip_id"], truth["click_xy"], truth["surface_y_px"],
                     clients=(ScriptedClient(["not json", "not json"]), FakeClient()))
    assert any(x["role"] == "system" and "Agent unavailable" in x["text"] for x in r["reasoning"])
    assert r["error_cm"] < 2.0


def test_pipeline_uses_agent_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OFFLINE", True)
    src = tmp_path / "t.mp4"
    truth = make_synthetic_clip(src)
    meta = pipeline.ingest(src)
    r = pipeline.run(meta["clip_id"], truth["click_xy"], truth["surface_y_px"])
    roles = {x["role"] for x in r["reasoning"]}
    assert {"planner", "critic", "summary"} <= roles and r["fit"]["rounds"] == 1


def test_wrongly_typed_reply_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OFFLINE", True)
    src = tmp_path / "t.mp4"
    truth = make_synthetic_clip(src)
    meta = pipeline.ingest(src)
    bad = json.dumps({"model": "projectile", "fit_g": False, "drop_frames": "12, 13", "recheck_scale": False,
                      "note": "x"})
    r = pipeline.run(meta["clip_id"], truth["click_xy"], truth["surface_y_px"],
                     clients=(ScriptedClient([bad]), FakeClient()))
    assert any(x["role"] == "system" and "Agent unavailable" in x["text"] for x in r["reasoning"])
