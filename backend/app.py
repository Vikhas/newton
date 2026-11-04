"""HTTP API and static files. Run: .venv/bin/uvicorn backend.app:app --port 8000"""
import json
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import config, pipeline
from backend.clients import ModelError
from backend.voice import WhatIfError

app = FastAPI(title="Newton")
STATUS: dict[str, dict] = {}


class RunRequest(BaseModel):
    ball_xy: list[float]
    surface_y_px: float
    capture_fps: float | None = None
    release_frame: int | None = None
    use_agent: bool = True
    ball_diam_cm: float | None = None
    ball_prompt: str | None = None


def _set_status(clip_id, stage, progress, message=""):
    STATUS[clip_id] = {"stage": stage, "progress": progress, "message": message}
    d = pipeline.clip_dir(clip_id)
    if d.exists():
        (d / "status.json").write_text(json.dumps(STATUS[clip_id]))


@app.post("/api/clips")
async def upload(file: UploadFile = File(...), capture_fps: float | None = Form(None)):
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=suffix, dir=config.CACHE_DIR, delete=False) as tmp:
        tmp.write(await file.read())
    try:
        return await run_in_threadpool(pipeline.ingest, tmp.name, capture_fps)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"could not read the video: {e}") from e
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/api/clips/{clip_id}/run")
def run(clip_id: str, req: RunRequest):
    if not (pipeline.clip_dir(clip_id) / "meta.json").exists():
        raise HTTPException(404, "unknown clip")
    _set_status(clip_id, "queued", 0.0)

    def job():
        try:
            pipeline.run(clip_id, req.ball_xy, req.surface_y_px, capture_fps=req.capture_fps,
                         release_frame=req.release_frame, use_agent=req.use_agent,
                         ball_diam_m=req.ball_diam_cm / 100 if req.ball_diam_cm else None,
                         ball_prompt=(req.ball_prompt or "").strip() or None,
                         on_progress=lambda stage, frac: _set_status(clip_id, stage, frac))
        except Exception as e:  # noqa: BLE001 - surfaced to the UI
            _set_status(clip_id, "error", 1.0, str(e))

    threading.Thread(target=job, daemon=True).start()
    return {"status": "queued"}


@app.get("/api/clips/{clip_id}/status")
def status(clip_id: str):
    if clip_id in STATUS:
        return STATUS[clip_id]
    p = pipeline.clip_dir(clip_id) / "status.json"
    if p.exists():
        return json.loads(p.read_text())
    raise HTTPException(404, "unknown clip")


@app.get("/api/clips/{clip_id}/result")
def result(clip_id: str):
    p = pipeline.clip_dir(clip_id) / "result.json"
    if not p.exists():
        raise HTTPException(404, "no result yet")
    return json.loads(p.read_text())


@app.post("/api/clips/{clip_id}/whatif")
async def whatif(clip_id: str, request: Request):
    if not (pipeline.clip_dir(clip_id) / "result.json").exists():
        raise HTTPException(404, "run the prediction first")
    if request.headers.get("content-type", "").startswith("multipart/"):
        audio = (await request.form())["audio"]
        kwargs = {"audio": await audio.read(), "mime": audio.content_type}
    else:
        kwargs = {"text": str((await request.json()).get("text", ""))}
    try:
        return await run_in_threadpool(pipeline.whatif, clip_id, **kwargs)
    except WhatIfError as e:
        raise HTTPException(422, str(e)) from e
    except ModelError as e:
        raise HTTPException(502, f"the model call failed: {e}") from e


@app.get("/media/{clip_id}/{path:path}")
def media(clip_id: str, path: str):
    root = (config.CACHE_DIR / "clips").resolve()
    f = (root / clip_id / path).resolve()
    if not f.is_relative_to(root / clip_id) or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/")
def index():
    return FileResponse(config.ROOT / "web" / "index.html")
