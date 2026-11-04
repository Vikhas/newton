# Meta Model API notes

These notes are copied from Meta's official cookbook, <https://github.com/meta-models/meta-model-cookbook>, read on Nov 4, 2026. The relevant recipes are `07_sam/01_api_basics`, `06_muse_voice/01_voice_api_fundamentals`, and `01_api_fundamentals/04_structured_output` and `07_vision_input`. The Baseten details come from the Baseten changelog entry "Muse Spark Open … available on Baseten".

## Common

- Base URL: `https://api.meta.ai/v1`. It is OpenAI-compatible, and the cookbook uses the official `openai` SDK.
- Auth: `Authorization: Bearer $MODEL_API_KEY`. The key looks like `LLM|{numeric_id}|{secret}`. The OpenAI SDK does not read `MODEL_API_KEY` by itself, so pass it explicitly. Newton also accepts `META_API_KEY`, which is the design doc's name.

## Muse Spark (planner, explainer, what-if parser)

- Model: `muse-spark-1.3`.
- Endpoint: `POST /chat/completions` (`client.chat.completions.create`).
- Images: a `user` message whose content is a list such as `[{"type": "text", "text": ...}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]`. Images placed in a `developer` or `system` message return HTTP 400.
- Structured output: `response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}`. The reply is in `choices[0].message.content`, as JSON text.

## SAM 3.1 (ball and hand tracking)

- Model: `sam-3.1`.
- Endpoint: the Responses API, `client.responses.create(model="sam-3.1", stream=True, input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<short noun phrase>"}, MEDIA]}])`.
  - Image: `MEDIA = {"type": "input_image", "image_url": "data:image/jpeg;base64,..."}`.
  - **Video:** first upload with `client.files.create(file=(name, bytes, "video/mp4"), purpose="user_data")`, which returns an id starting `file-`. Then `MEDIA = {"type": "input_video", "file_id": id}`.
- Each request takes **one noun phrase**. Ask for the ball and the hand in separate requests.
- Output is a streamed text grammar decoded by the official parser `meta_sam_parser`: `parse_responses_stream(events, video_segmentation_format())`, then `.final_result()`.
  - The API streams `response.content_part.done` but not `response.output_text.done`, so the client has to synthesise the latter. The cookbook's `stream_events()` does this, and Newton's `clients/meta.py` copies the approach.
  - Records have `kind` set to `"box"` or `"mask"`, an `object_id` that is **stable across frames (tracking)**, and `frame.frame_index`, which is a 0-based source-video frame.
  - A mask record carries `mask` (with `width` and `height` for the raster itself) and `bounds` (`left`, `top`, `right`, `bottom` in source pixels). **Masks are local to the box.** `decode_mask_to_raster(mask) -> bytes` gives `width*height` bytes, one per pixel, each 0 or 1.
  - Validate with `result.outcome.status == "completed"` and check for no `diagnostics` with severity `error`. A completed request with zero masks is a valid empty match.
- Newton caches the raw stream events as JSON, then replays them through the parser. Re-runs therefore make no API calls.
- Newton uploads an H.264 mp4 encoded from its own extracted frames at capture fps, so SAM's `frame_index` matches the frame numbers Newton uses.

## Muse Voice Transcribe (what-if)

- Model: `muse-voice-transcribe-1.0`.
- One-shot endpoint: `POST https://api.meta.ai/v1/asr/transcribe` with header `Authorization: Bearer $MODEL_API_KEY`. The body is multipart with two parts:
  - `request`: JSON (content-type `application/json`), `{"mode": "PUSH_TO_TALK", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}`
  - `audio`: the file, a **mono 16-bit WAV at 24 kHz or 16 kHz**. Newton converts the browser's webm/opus to WAV with ffmpeg: `-ac 1 -ar 24000 -sample_fmt s16`.
- Response: `{"sessionId", "transcript", "audioDurationMs", "turns": [...]}`.
- Status codes: 400 for a malformed request or audio over 10 minutes, 401 for a bad credential, 413 for over 32 MB, 429 for rate limiting, 500 when the server exceeds its processing budget.
- A streaming WebSocket also exists (`wss://api.meta.ai/v1/asr/realtime`), but Newton doesn't use it.

## Muse Spark Open (critic) on Baseten

- Base URL: `https://inference.baseten.co/v1`, OpenAI-compatible `POST /chat/completions`, with `Authorization: Bearer $BASETEN_API_KEY`.
- Model id: **`meta-models/muse-spark-open`**, confirmed live on Nov 4. The hackathon guide's `meta-models/muse-spark-open-1026` returns 404 on Baseten. `OPEN_MODEL` overrides it.

## Live check status (Nov 4, 2026, ~03:45 UTC)

- **Muse Spark** (`muse-spark-1.3`) works: chat with `json_schema` response_format returned `{"ok": true}`.
- **SAM 3.1** video mode works. The upload, the streamed response and the parser all succeed. On a synthetic clip, the concept `"yellow circle"` was tracked as one stable object across 60/60 frames. Centroid error averaged 0.27 px (0.37 px at worst), and the diameter read 36.7 px against about 37 px drawn.
  - A flat cartoon disk returns an **empty match** for `"ball"` and `"tennis ball"`. That's a valid completed response with no output text. Real footage should be checked with `"ball"` or `"tennis ball"`; `BALL_PROMPT` overrides the concept.
- **Muse Voice Transcribe** works: one second of silence returned `''`.
- **Muse Spark Open** on Baseten works. It answered as `meta-models/muse-spark-open`, and it critiqued fits live in the agent loop.
- The key format `LLM_<id>_<secret>` (underscores) is accepted as-is.
