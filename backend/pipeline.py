"""Clip in, prediction out: ingest -> track -> calibrate -> fit -> predict -> score."""
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from backend import config
from backend.agent import AgentError, AgentTools, run_agent
from backend.clients import ModelError, get_clients
from backend.physics import fit_projectile, predict_landing
from backend.track import encode_video, extract_frames, probe, track_ball, track_hands
from backend.trajectory import (
    actual_landing_px, clean_blur, compute_scale, cutoff_frame, detect_release, m_to_px, slice_to_cutoff,
)


def clip_dir(clip_id: str) -> Path:
    return config.CACHE_DIR / "clips" / clip_id


def _write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def ingest(upload_path, capture_fps=None) -> dict:
    upload_path = Path(upload_path)
    clip_id = hashlib.sha256(upload_path.read_bytes()).hexdigest()[:12]
    d = clip_dir(clip_id)
    meta_path = d / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        d.mkdir(parents=True, exist_ok=True)
        src = d / f"source{upload_path.suffix.lower() or '.mp4'}"
        shutil.copyfile(upload_path, src)
        n = extract_frames(src, d / "frames", max_frames=config.MAX_FRAMES)
        if n == 0:
            raise ValueError("could not read any frames from the video")
        h, w = cv2.imread(str(d / "frames" / "00000.jpg")).shape[:2]
        encode_video(d / "frames", d / "player.mp4", config.PLAYBACK_FPS)
        meta = {"clip_id": clip_id, "width": w, "height": h, "capture_fps": probe(src)["fps"], "n_frames": n,
                "truncated": n >= config.MAX_FRAMES,
                "first_frame_url": f"/media/{clip_id}/frames/00000.jpg", "player_url": f"/media/{clip_id}/player.mp4"}
        _write_json(meta_path, meta)
    if capture_fps:
        meta = {**meta, "capture_fps": float(capture_fps)}
    return meta


def _keyframes(frames_dir, tools):
    return [(Path(frames_dir) / f"{f:05d}.jpg").read_bytes() for f in tools.keyframe_ids()]


def _fit_stage(tools, frames_dir, spark, critic, use_agent):
    """Agent loop (planner + critic); any model failure falls back to the plain projectile fit."""
    note = "Standard projectile fit, g fixed at 9.81 m/s²."
    if use_agent:
        try:
            r = run_agent(tools, _keyframes(frames_dir, tools), spark, critic)
            return r.fit, r.reasoning, r.rounds
        except Exception as e:  # noqa: BLE001 - any agent failure must fall back, never kill the run
            note = f"Agent unavailable — standard projectile fit used ({type(e).__name__}: {e})."
    tools.dropped.clear()
    return tools.fit_projectile(False, False), [{"round": 1, "role": "system", "text": note}], 1


def run(clip_id, ball_xy, surface_y_px, capture_fps=None, release_frame=None, cutoff_s=config.CUTOFF_S,
        use_agent=True, on_progress=None, clients=None, ball_diam_m=None, ball_prompt=None) -> dict:
    progress = on_progress or (lambda stage, frac: None)
    d = clip_dir(clip_id)
    meta = json.loads((d / "meta.json").read_text())
    fps = float(capture_fps or meta["capture_fps"])
    w, h = meta["width"], meta["height"]
    spark, critic = clients or get_clients()
    player = d / "player.mp4"

    progress("tracking", 0.05)
    ball_diam_m = float(ball_diam_m or config.BALL_DIAM_M)
    points = track_ball(player, ball_xy, spark, clip_id, w, on_progress=lambda f: progress("tracking", 0.05 + 0.5 * f),
                        prompts=[ball_prompt] if ball_prompt else None)
    if len(points) < config.MIN_FIT_POINTS:
        raise ValueError("the ball was not found in the clip — click right on the ball in the first frame")
    if release_frame is None:
        progress("tracking", 0.6)
        hands = track_hands(player, spark, clip_id, (h, w), max_frame=meta["n_frames"] - 1)
        release_frame = detect_release(points, hands)

    progress("calibrating", 0.7)
    cut = cutoff_frame(release_frame, fps, cutoff_s)
    if cut > meta["n_frames"] - 1:
        raise ValueError(f"the clip ends less than {cutoff_s:g} s after release — keep filming until the ball lands")
    # Everything that feeds the prediction uses frames <= cutoff only.
    seen, dropped = clean_blur([p for p in points if p.frame <= cut])
    used = slice_to_cutoff(seen, release_frame, cut)
    if len(used) < config.MIN_FIT_POINTS:
        raise ValueError("not enough tracked points before cutoff")
    tools = AgentTools(seen, release_frame, cut, compute_scale(used, ball_diam_m), h, fps, surface_y_px, ball_diam_m)

    progress("fitting", 0.8)
    fit, reasoning, rounds = _fit_stage(tools, d / "frames", spark, critic, use_agent)
    # The agent may have dropped frames or re-measured the scale; everything below uses its final state.
    scale = tools.scale
    radius_px = ball_diam_m / 2 / scale
    landing_y_m = tools.landing_y_m()
    dropped = sorted(set(dropped) | tools.dropped)
    g_check = fit_projectile(*tools.world(), fit_g=True).g

    progress("predicting", 0.9)
    landing = predict_landing(fit, landing_y_m)
    ax, ay = m_to_px(landing.arc[:, 1], landing.arc[:, 2], scale, h)
    af = release_frame + landing.arc[:, 0] * fps
    lx, ly = m_to_px(landing.x_m, landing.y_m, scale, h)

    progress("scoring", 0.95)
    after, _ = clean_blur([p for p in points if p.frame > cut])
    measured = actual_landing_px(used[-1:] + after, surface_y_px, radius_px, cut)
    actual = error_cm = None
    if measured is not None:
        mx, my, mf = measured
        actual = {"landing_px": [mx, my], "landing_frame": mf, "landing_x_m": mx * scale}
        error_cm = abs(landing.x_m - mx * scale) * 100

    result = {
        "clip_id": clip_id, "width": w, "height": h, "capture_fps": fps, "playback_fps": config.PLAYBACK_FPS,
        "n_frames": meta["n_frames"], "player_url": meta["player_url"],
        "release_frame": int(release_frame), "cutoff_frame": int(cut), "fit_frames": [int(f) for f in fit.frames],
        "dropped_frames": [int(f) for f in dropped], "surface_y_px": float(surface_y_px), "ball_xy": list(ball_xy),
        "scale_m_per_px": scale, "radius_px": radius_px, "ball_diam_m": ball_diam_m, "landing_y_m": landing_y_m,
        "track": [{"frame": p.frame, "x": p.x_px, "y": p.y_px, "d": p.diam_px, "contour": p.contour} for p in points],
        "prediction": {"arc_px": np.column_stack([ax, ay, af]).tolist(), "landing_px": [float(lx), float(ly)],
                       "landing_frame": float(release_frame + landing.t_s * fps),
                       "ring_radius_px": float(max(2 * landing.sigma_m / scale, radius_px)),
                       "landing_x_m": landing.x_m, "sigma_cm": landing.sigma_m * 100},
        "actual": actual, "error_cm": error_cm,
        "fit": {"model": "projectile_drag" if fit.drag else "projectile", "speed_mps": fit.speed_mps,
                "angle_deg": fit.angle_deg, "g_used": fit.g, "g_check": g_check,
                "g_warning": not (config.G_SANE[0] <= g_check <= config.G_SANE[1]),
                "rms_cm": fit.rms_cm, "rounds": rounds},
        "fit_state": fit.to_dict(),
        "reasoning": reasoning,
    }
    _write_json(d / "result.json", result)
    with open(config.CACHE_DIR / "takes.jsonl", "a") as log:
        log.write(json.dumps({"clip_id": clip_id, "error_cm": error_cm, "rms_cm": fit.rms_cm, "g_check": g_check,
                              "release_frame": release_frame, "cutoff_frame": cut, "rounds": rounds}) + "\n")
    progress("done", 1.0)
    return result


def whatif(clip_id, text=None, audio=None, mime=None, clients=None) -> dict:
    """Re-simulate the fitted launch under a spoken or typed change; returns the ghost ball's path."""
    from backend.physics import LaunchFit
    from backend.voice import WhatIfError, parse_whatif, transcribe

    r = json.loads((clip_dir(clip_id) / "result.json").read_text())
    spark, _ = clients or get_clients()
    transcript = text if text is not None else transcribe(audio, mime, spark)
    if not transcript.strip():
        raise WhatIfError("I didn't catch that — hold the button while you speak")
    change = parse_whatif(transcript, spark)
    fit = LaunchFit.from_dict(r["fit_state"])
    try:
        landing = predict_landing(fit, r["landing_y_m"], g=change.g, speed_scale=change.speed_scale or 1.0, n_mc=0)
    except ValueError as e:
        raise WhatIfError("With that change the ball never comes back down") from e
    scale, h, fps, rel = r["scale_m_per_px"], r["height"], r["capture_fps"], r["release_frame"]
    ax, ay = m_to_px(landing.arc[:, 1], landing.arc[:, 2], scale, h)
    lx, ly = m_to_px(landing.x_m, landing.y_m, scale, h)
    real_x = (r["actual"] or r["prediction"])["landing_x_m"]
    return {
        "transcript": transcript,
        "change": {"g": change.g, "speed_scale": change.speed_scale},
        "label": change.label,
        "ghost": {
            "arc_px": np.column_stack([ax, ay, rel + landing.arc[:, 0] * fps]).tolist(),
            "landing_px": [float(lx), float(ly)],
            "landing_frame": float(rel + landing.t_s * fps),
            "offscreen": bool(not 0 <= lx <= r["width"] or ay.min() < 0),
            "landing_distance_m": abs(landing.x_m - fit.x0),
            "real_distance_m": abs(real_x - fit.x0),
        },
    }
