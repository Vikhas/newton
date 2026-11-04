"""Meta Model API: Muse Spark (chat), SAM 3.1 (video segmentation), Muse Voice Transcribe.

Request shapes follow Meta's cookbook: https://github.com/meta-models/meta-model-cookbook
"""
import asyncio
import base64
import json
import os
from pathlib import Path

import httpx
import numpy as np
import openai
from meta_sam_parser import decode_mask_to_raster, parse_responses_stream, video_segmentation_format

from backend import config
from backend.clients import Instance, ModelError, call_with_retries
from backend.clients.cache import cached_json

TIMEOUT_S = 60


def _retryable(e: Exception) -> bool:
    if isinstance(e, openai.APIStatusError):
        return e.status_code == 429 or e.status_code >= 500
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code == 429 or e.response.status_code >= 500
    return isinstance(e, (openai.APIConnectionError, httpx.TransportError))


def _with_images(messages, images):
    """Attach JPEG images to the last user message (images in system/developer messages are rejected)."""
    if not images:
        return messages
    messages = [dict(m) for m in messages]
    last = messages[-1]
    content = last["content"] if isinstance(last["content"], list) else [{"type": "text", "text": last["content"]}]
    for img in images:
        url = "data:image/jpeg;base64," + base64.b64encode(img).decode()
        content = [*content, {"type": "image_url", "image_url": {"url": url}}]
    last["content"] = content
    return messages


def chat_completion(oa: openai.OpenAI, model: str, namespace: str, messages, images=(), json_schema=None) -> str:
    """Cached chat completion through an OpenAI-compatible client."""
    msgs = _with_images(messages, list(images))
    kwargs = {}
    if json_schema is not None:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": json_schema}}

    def call():
        r = oa.chat.completions.create(model=model, messages=msgs, **kwargs)
        return {"content": r.choices[0].message.content}

    key = {"model": model, "messages": msgs, "schema": json_schema}
    return cached_json(namespace, key, lambda: call_with_retries(call, _retryable))["content"]


def _close_lanes(events):
    """The API streams content_part.done but not output_text.done; the parser needs the latter."""
    out, done = [], set()
    for e in events:
        lane = (e.get("item_id"), e.get("output_index"), e.get("content_index"))
        if e.get("type") == "response.output_text.done":
            if lane not in done:
                done.add(lane)
                out.append(e)
            continue
        out.append(e)
        part = e.get("part")
        if (e.get("type") == "response.content_part.done" and lane not in done and isinstance(part, dict)
                and part.get("type") == "output_text" and isinstance(part.get("text"), str)):
            done.add(lane)
            out.append({"type": "response.output_text.done", "item_id": lane[0], "output_index": lane[1],
                        "content_index": lane[2], "text": part["text"]})
    return out


def parse_sam_events(events) -> dict[str, dict[int, Instance]]:
    """Decode cached SAM video events into {object_id: {frame: Instance}}."""
    async def gen():
        for e in _close_lanes(events):
            yield e

    async def run():
        return await parse_responses_stream(gen(), video_segmentation_format()).final_result()

    result = asyncio.run(run())
    if result.outcome.status != "completed":
        raise ModelError(f"SAM segmentation incomplete: {getattr(result.outcome, 'reason', 'unknown')}")
    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        raise ModelError("SAM parser rejected records: " + "; ".join(d.message for d in errors))
    objs: dict[str, dict[int, Instance]] = {}
    for r in result.records:
        if r.kind != "mask" or r.frame is None:
            continue
        raster = np.frombuffer(decode_mask_to_raster(r.mask), np.uint8).reshape(r.mask.height, r.mask.width)
        objs.setdefault(r.object_id, {})[r.frame.frame_index] = Instance(
            mask=raster.astype(bool), left=int(r.bounds.left), top=int(r.bounds.top), score=r.confidence)
    return objs


class MetaClient:
    def __init__(self, api_key=None, http_client=None):
        self._key = api_key or os.getenv("MODEL_API_KEY") or os.getenv("META_API_KEY")
        self._http = http_client
        self._oa = None

    def _client(self) -> openai.OpenAI:
        if not self._key:
            raise ModelError("MODEL_API_KEY (or META_API_KEY) is not set")
        if self._oa is None:
            self._oa = openai.OpenAI(base_url=config.META_BASE_URL, api_key=self._key, max_retries=0,
                                     timeout=TIMEOUT_S, http_client=self._http)
        return self._oa

    def chat(self, messages, images=(), json_schema=None) -> str:
        return chat_completion(self._client(), config.SPARK_MODEL, "spark", messages, images, json_schema)

    def segment_video(self, video_path: Path, concept: str, cache_key: dict) -> dict[str, dict[int, Instance]]:
        def call():
            oa = self._client()
            data = Path(video_path).read_bytes()
            fid = oa.files.create(file=(Path(video_path).name, data, "video/mp4"), purpose="user_data").id
            stream = oa.responses.create(model=config.SAM_MODEL, stream=True, input=[{
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": concept}, {"type": "input_video", "file_id": fid}]}])
            with stream:
                return [e.model_dump() if hasattr(e, "model_dump") else dict(e) for e in stream]

        key = {**cache_key, "model": config.SAM_MODEL, "concept": concept}
        events = cached_json("sam", key, lambda: call_with_retries(call, _retryable))
        return parse_sam_events(events)

    def transcribe(self, wav: bytes) -> str:
        def call():
            if not self._key:
                raise ModelError("MODEL_API_KEY (or META_API_KEY) is not set")
            req = {"mode": "PUSH_TO_TALK", "model": config.VOICE_MODEL, "audioEncoding": "WAV"}
            http = self._http or httpx.Client(timeout=TIMEOUT_S)
            r = http.post(config.META_ASR_URL, headers={"Authorization": f"Bearer {self._key}"},
                          files={"request": (None, json.dumps(req), "application/json"),
                                 "audio": ("whatif.wav", wav, "audio/wav")})
            r.raise_for_status()
            return {"transcript": r.json().get("transcript", "")}

        key = {"model": config.VOICE_MODEL, "audio": wav}
        return cached_json("voice", key, lambda: call_with_retries(call, _retryable))["transcript"]
