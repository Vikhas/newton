"""Offline stand-in for every model (NEWTON_OFFLINE=1 and tests).

Segmentation is an HSV colour threshold: yellow for the ball, magenta for the "hand" in synthetic clips.
Chat returns canned valid JSON keyed on the system prompt's first word.
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np

from backend.clients import Instance

_HSV = {"ball": ((20, 120, 120), (40, 255, 255)), "hand": ((140, 80, 80), (170, 255, 255))}


def _largest_component(mask: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 4:
        return None
    x, y, w, h = stats[i, :4]
    return Instance(mask=labels[y:y + h, x:x + w] == i, left=int(x), top=int(y), score=1.0)


def _refine_on_luma(inst: Instance, gray: np.ndarray) -> Instance:
    """Chroma is subsampled in yuv420 video, so a colour mask bleeds ~1 px; re-cut the edge on full-res luma."""
    pad = 8
    y0, x0 = max(inst.top - pad, 0), max(inst.left - pad, 0)
    y1 = min(inst.top + inst.mask.shape[0] + pad, gray.shape[0])
    x1 = min(inst.left + inst.mask.shape[1] + pad, gray.shape[1])
    hue = np.zeros((y1 - y0, x1 - x0), np.uint8)
    hy, hx = inst.top - y0, inst.left - x0
    hue[hy:hy + inst.mask.shape[0], hx:hx + inst.mask.shape[1]] = inst.mask
    patch = gray[y0:y1, x0:x1].astype(float)
    inner = cv2.erode(hue, np.ones((5, 5), np.uint8)) > 0
    near = cv2.dilate(hue, np.ones((5, 5), np.uint8)) > 0
    ring = (cv2.dilate(hue, np.ones((11, 11), np.uint8)) > 0) & ~near
    if not inner.any() or not ring.any():
        return inst
    # Half-way between ball and surroundings is the unbiased edge of a blurred disk.
    thresh = (np.median(patch[inner]) + np.median(patch[ring])) / 2
    full = np.zeros(gray.shape, bool)
    full[y0:y1, x0:x1] = near & (patch > thresh)
    return _largest_component(full) or inst


class FakeClient:
    def segment_video(self, video_path: Path, concept: str, cache_key: dict):
        is_hand = "hand" in concept
        lo, hi = _HSV["hand" if is_hand else "ball"]
        cap = cv2.VideoCapture(str(video_path))
        track, frame = {}, 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            inst = _largest_component(cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lo, hi) > 0)
            if inst is not None and not is_hand:
                inst = _refine_on_luma(inst, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if inst is not None:
                track[frame] = inst
            frame += 1
        cap.release()
        return {"0": track} if track else {}

    def chat(self, messages, images=(), json_schema=None) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        role = (re.match(r"[A-Z]+", system) or [""])[0]
        user = json.dumps(messages[-1]["content"]).lower()
        if role == "PLANNER":
            return json.dumps({"model": "projectile", "fit_g": False, "drop_frames": [], "recheck_scale": False,
                               "note": "Clean parabola; standard projectile with g fixed at 9.81."})
        if role == "CRITIC":
            return json.dumps({"verdict": "accept", "frames": [], "reason": "Residuals are small and unstructured."})
        if role == "EXPLAIN":
            return json.dumps({"lines": ["Model: projectile, g fixed at 9.81 m/s².",
                                         "Launch speed and angle from the first quarter-second.",
                                         "Accepted on the first round."]})
        if role == "WHATIF":
            planets = {"moon": 1.62, "mars": 3.71, "jupiter": 24.79}
            g = next((v for k, v in planets.items() if k in user), None)
            speed = 2.0 if re.search(r"twice|double|2x", user) else None
            label = next((k.title() for k in planets if k in user), "Twice as hard" if speed else "")
            return json.dumps({"g": g, "speed_scale": speed, "label": label})
        return json.dumps({"ok": True})

    def transcribe(self, wav: bytes) -> str:
        return "what if this were on the moon"
