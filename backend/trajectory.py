"""Tracked ball points: cleaning, calibration, release/cutoff, measured landing."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.config import BALL_DIAM_M, BLUR_JUMP, CUTOFF_S


@dataclass
class TrackPoint:
    frame: int
    x_px: float
    y_px: float
    diam_px: float
    contour: list[list[int]] = field(default_factory=list)


@dataclass
class Track:
    points: list[TrackPoint]
    capture_fps: float
    width: int
    height: int


def clean_blur(points, max_jump=BLUR_JUMP):
    """Drop points whose mask area jumps more than max_jump from the median of nearby points (±2)."""
    areas = np.array([p.diam_px**2 for p in points], dtype=float)
    kept, dropped = [], []
    for i, p in enumerate(points):
        nb = np.concatenate([areas[max(0, i - 2):i], areas[i + 1:i + 3]])
        if nb.size and abs(areas[i] - np.median(nb)) / np.median(nb) > max_jump:
            dropped.append(p.frame)
        else:
            kept.append(p)
    return kept, dropped


def compute_scale(points, ball_diam_m=BALL_DIAM_M) -> float:
    """Metres per pixel, using the ball's own diameter as the ruler."""
    return ball_diam_m / float(np.median([p.diam_px for p in points]))


def to_world(points, scale, height, capture_fps, release_frame):
    frames = np.array([p.frame for p in points], dtype=int)
    t = (frames - release_frame) / capture_fps
    x = np.array([p.x_px for p in points]) * scale
    y = (height - np.array([p.y_px for p in points])) * scale
    return t, x, y, frames


def m_to_px(x_m, y_m, scale, height):
    return np.asarray(x_m) / scale, height - np.asarray(y_m) / scale


def detect_release(points, hand_masks, confirm=3) -> int:
    """First frame of `confirm` consecutive frames where the ball disk is clear of the (dilated) hand mask.

    A frame with no hand mask is unknown (SAM often misses a blurred arm), so it neither confirms nor breaks a run.
    If the hand leaves the frame before a run is confirmed, release is the first frame after the last contact.
    """
    if not hand_masks:
        return points[0].frame
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    run_start, run, last_contact = None, 0, None
    for p in points:
        hand = hand_masks.get(p.frame)
        if hand is None or not hand.any():
            continue
        hand = cv2.dilate(hand.astype(np.uint8), kernel) > 0
        disk = np.zeros(hand.shape, np.uint8)
        cv2.circle(disk, (round(p.x_px), round(p.y_px)), max(1, round(p.diam_px / 2)), 1, -1)
        if (hand & (disk > 0)).any():
            run, last_contact = 0, p.frame
            continue
        run_start = p.frame if run == 0 else run_start
        run += 1
        if run >= confirm:
            return run_start
    if run_start is not None:
        return run_start
    after = [p.frame for p in points if last_contact is not None and p.frame > last_contact]
    return after[0] if after else points[0].frame


def cutoff_frame(release_frame, capture_fps, cutoff_s=CUTOFF_S) -> int:
    return release_frame + round(cutoff_s * capture_fps)


def slice_to_cutoff(points, release_frame, cutoff):
    return [p for p in points if release_frame <= p.frame <= cutoff]


def actual_landing_px(points, surface_y_px, radius_px, after_frame):
    """(x_px, y_px, frame) where the ball centre first falls through the landing line, or None."""
    line = surface_y_px - radius_px
    tol = 0.2 * radius_px  # a resting ball's measured centre can sit a hair above the line
    pts = [p for p in points if p.frame >= after_frame]
    for a, b in zip(pts, pts[1:]):
        if a.y_px < line - tol and b.y_px >= line - tol and b.y_px > a.y_px:
            # interpolate to the line (extrapolates slightly when b stopped just short of it)
            f = (line - a.y_px) / (b.y_px - a.y_px)
            return a.x_px + f * (b.x_px - a.x_px), float(line), a.frame + f * (b.frame - a.frame)
    return None
