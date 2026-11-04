import json

import cv2
import httpx
import numpy as np
import pytest

from backend.clients import ModelError
from backend.clients.cache import cached_json
from backend.clients.fake import FakeClient
from backend.clients.meta import MetaClient
from backend.clients.open import OpenClient
from tests.fixtures.sam_stream import sam_events, sam_line, sse

CHAT_OK = {"id": "c1", "object": "chat.completion", "created": 0, "model": "m",
           "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "{\"ok\": true}"}}]}


def test_cache_calls_fn_once(tmp_path):
    calls = []
    fn = lambda: calls.append(1) or {"v": 1}
    assert cached_json("ns", {"a": 1, "b": b"\x00\x01"}, fn) == {"v": 1}
    assert cached_json("ns", {"a": 1, "b": b"\x00\x01"}, fn) == {"v": 1}
    assert len(calls) == 1


def test_meta_chat_request_shape():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=CHAT_OK)

    c = MetaClient(api_key="LLM|1|k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = c.chat([{"role": "system", "content": "PLANNER"}, {"role": "user", "content": "hi"}],
                 images=[b"\xff\xd8jpeg"], json_schema={"type": "object"})
    assert out == "{\"ok\": true}"
    assert seen["url"] == "https://api.meta.ai/v1/chat/completions"
    assert seen["auth"] == "Bearer LLM|1|k"
    body = seen["body"]
    assert body["model"] == "muse-spark-1.3"
    assert body["response_format"]["type"] == "json_schema"
    user = body["messages"][-1]
    assert user["role"] == "user" and user["content"][1]["type"] == "image_url"
    assert user["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_meta_segment_video_parses_stream(tmp_path):
    mask = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], bool)
    events = sam_events([sam_line(0, 0, 10, 20, mask), sam_line(1, 0, 11, 20, mask)])
    calls = []

    def handler(req: httpx.Request):
        calls.append(req.url.path)
        if req.url.path.endswith("/files"):
            return httpx.Response(200, json={"id": "file-abc", "object": "file", "bytes": 3, "created_at": 0,
                                             "filename": "sam.mp4", "purpose": "user_data", "status": "processed"})
        body = json.loads(req.content)
        assert body["model"] == "sam-3.1" and body["stream"] is True
        content = body["input"][0]["content"]
        assert content[0] == {"type": "input_text", "text": "ball"}
        assert content[1] == {"type": "input_video", "file_id": "file-abc"}
        return httpx.Response(200, content=sse(events), headers={"content-type": "text/event-stream"})

    video = tmp_path / "sam.mp4"
    video.write_bytes(b"mp4")
    c = MetaClient(api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    objs = c.segment_video(video, "ball", {"clip": "abc"})
    assert set(objs) == {"0"} and set(objs["0"]) == {0, 1}
    inst = objs["0"][1]
    assert inst.left == 11 and inst.top == 20 and inst.mask.shape == (3, 4) and inst.mask.sum() == 8
    n = len(calls)
    c.segment_video(video, "ball", {"clip": "abc"})
    assert len(calls) == n  # served from cache


def test_transcribe_request_shape():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = req.content
        return httpx.Response(200, json={"sessionId": "s", "transcript": "what if this were on the moon",
                                         "audioDurationMs": 1000})

    c = MetaClient(api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.transcribe(b"RIFF....WAVE") == "what if this were on the moon"
    assert seen["url"] == "https://api.meta.ai/v1/asr/transcribe" and seen["auth"] == "Bearer k"
    assert b'"audioEncoding": "WAV"' in seen["body"] and b'"model": "muse-voice-transcribe-1.0"' in seen["body"]
    assert b'name="audio"' in seen["body"] and b'name="request"' in seen["body"]


def test_open_chat_uses_baseten():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["model"] = json.loads(req.content)["model"]
        return httpx.Response(200, json=CHAT_OK)

    c = OpenClient(api_key="bk", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.chat([{"role": "user", "content": "x"}]) == "{\"ok\": true}"
    assert seen["url"] == "https://inference.baseten.co/v1/chat/completions"
    assert seen["model"] == "meta-models/muse-spark-open"


def test_retry_then_model_error(monkeypatch):
    attempts = []
    monkeypatch.setattr("backend.clients.time.sleep", lambda s: None)

    def handler(req):
        attempts.append(1)
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    c = MetaClient(api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ModelError):
        c.chat([{"role": "user", "content": "x"}])
    assert len(attempts) == 4  # first try + 3 retries


def test_missing_key_is_model_error(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("META_API_KEY", raising=False)
    with pytest.raises(ModelError, match="MODEL_API_KEY"):
        MetaClient().chat([{"role": "user", "content": "x"}])


def test_fake_segment_finds_yellow_disk(tmp_path):
    path = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (200, 100))
    for i in range(5):
        img = np.full((100, 200, 3), 90, np.uint8)
        cv2.circle(img, (40 + 10 * i, 50), 10, (0, 230, 230), -1)  # BGR yellow
        vw.write(img)
    vw.release()
    objs = FakeClient().segment_video(path, "ball", {"clip": "t"})
    track = objs["0"]
    assert set(track) == set(range(5))
    m = track[3]
    ys, xs = np.nonzero(m.mask)
    assert abs(xs.mean() + m.left - 70) < 1.0 and abs(ys.mean() + m.top - 50) < 1.0
