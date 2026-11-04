"""Frames out of the clip, SAM 3.1 masks in, a clean ball track out."""
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from backend import config
from backend.trajectory import TrackPoint

MAX_JUMP_FRAC = 0.25  # of frame width: farther than this from the last centroid is not our ball


def ffmpeg_exe() -> str:
    """The static ffmpeg bundled with imageio-ffmpeg (a known version), else the system one."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return shutil.which("ffmpeg") or "ffmpeg"


def _run(args):
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def probe(path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)), "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    finally:
        cap.release()
    return info


def extract_frames(src, out_dir, max_frames=1500) -> int:
    """Every source frame as out_dir/00000.jpg, 00001.jpg, ... (rotation applied). Returns the count."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(["-i", str(src), "-fps_mode", "passthrough", "-frames:v", str(max_frames), "-start_number", "0",
          "-q:v", "2", str(out_dir / "%05d.jpg")])
    return len(list(out_dir.glob("*.jpg")))


def encode_video(frames_dir, out_path, fps):
    """H.264 mp4 with one video frame per extracted frame, so frame i plays at i / fps."""
    _run(["-framerate", str(fps), "-start_number", "0", "-i", str(Path(frames_dir) / "%05d.jpg"),
          "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
          str(out_path)])


def _measure(inst, frame) -> TrackPoint:
    ys, xs = np.nonzero(inst.mask)
    area = xs.size
    contours, _ = cv2.findContours(inst.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    eps = 0.5
    poly = cv2.approxPolyDP(c, eps, True)
    while len(poly) > 24:
        eps *= 1.5
        poly = cv2.approxPolyDP(c, eps, True)
    contour = [[int(p[0][0]) + inst.left, int(p[0][1]) + inst.top] for p in poly]
    # Motion blur stretches the mask along the flight only; scale the area diameter by the ellipse's axis ratio
    # so a smeared ball still reads its true size (a round mask has ratio ~1 and is unchanged).
    diam = 2 * np.sqrt(area / np.pi)
    if len(c) >= 5:
        (_, _), (a, b), _ = cv2.fitEllipse(c)
        if max(a, b) > 0:
            diam *= np.sqrt(min(a, b) / max(a, b))
    return TrackPoint(frame=frame, x_px=float(xs.mean() + inst.left), y_px=float(ys.mean() + inst.top),
                      diam_px=float(diam), contour=contour)


def _centroid(inst):
    ys, xs = np.nonzero(inst.mask)
    return xs.mean() + inst.left, ys.mean() + inst.top


def track_ball(video_path, click_xy, client, clip_id, width, on_progress=None, prompts=None) -> list[TrackPoint]:
    """Segment the ball through the whole clip and follow the object nearest the user's click.

    SAM gives stable object ids, but ids can switch after a fast or blurred stretch, so each frame takes
    whichever instance is nearest the previous centroid. Each description in `prompts` is tried until one tracks.
    """
    prompts = prompts or config.BALL_PROMPTS
    points = []
    for i, prompt in enumerate(prompts):
        points = _follow(client.segment_video(video_path, prompt, {"clip": clip_id}), click_xy, width)
        if on_progress:
            on_progress((i + 1) / len(prompts))
        if len(points) >= config.MIN_FIT_POINTS:
            break
    return points


def _follow(objs, click_xy, width) -> list[TrackPoint]:
    by_frame: dict[int, list] = {}
    for track in objs.values():
        for f, inst in track.items():
            if inst.mask.any():
                by_frame.setdefault(f, []).append(inst)
    limit = MAX_JUMP_FRAC * width
    prev, points = None, []
    for f in sorted(by_frame):
        ref = prev if prev is not None else click_xy
        cands = [(np.hypot(*np.subtract(_centroid(i), ref)), i) for i in by_frame[f]]
        dist, best = min(cands, key=lambda c: c[0])
        if dist > limit:
            continue
        p = _measure(best, f)
        points.append(p)
        prev = (p.x_px, p.y_px)
    return points


class HandMasks(Mapping):
    """frame -> full-frame union of hand instances, built on access (a 240 fps clip has too many to hold)."""

    def __init__(self, by_frame: dict[int, list], shape):
        self._by_frame, self._shape = by_frame, shape

    def __getitem__(self, f):
        h, w = self._shape
        m = np.zeros((h, w), bool)
        for inst in self._by_frame[f]:
            mh, mw = inst.mask.shape
            y0, x0 = max(inst.top, 0), max(inst.left, 0)
            y1, x1 = min(inst.top + mh, h), min(inst.left + mw, w)
            m[y0:y1, x0:x1] |= inst.mask[y0 - inst.top:y1 - inst.top, x0 - inst.left:x1 - inst.left]
        return m

    def __iter__(self):
        return iter(self._by_frame)

    def __len__(self):
        return len(self._by_frame)


def track_hands(video_path, client, clip_id, shape, max_frame) -> HandMasks:
    """Hand masks for frames up to max_frame (used to find the release frame)."""
    objs = client.segment_video(video_path, config.HAND_PROMPT, {"clip": clip_id})
    by_frame: dict[int, list] = {}
    for track in objs.values():
        for f, inst in track.items():
            if f <= max_frame:
                by_frame.setdefault(f, []).append(inst)
    return HandMasks(by_frame, shape)
