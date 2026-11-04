import numpy as np
import pytest

from backend.trajectory import (
    TrackPoint, actual_landing_px, clean_blur, compute_scale, cutoff_frame, detect_release,
    m_to_px, slice_to_cutoff, to_world,
)


def pt(frame, diam, x=0.0, y=0.0):
    return TrackPoint(frame=frame, x_px=x, y_px=y, diam_px=diam)


def test_clean_blur_drops_area_jump():
    kept, dropped = clean_blur([pt(i, d) for i, d in enumerate([40, 40, 52, 40, 40])])
    assert dropped == [2] and len(kept) == 4


def test_scale_from_ball_size():
    assert compute_scale([pt(i, 40) for i in range(5)]) == pytest.approx(0.067 / 40)


def test_to_world_uses_capture_fps():
    t, x, y, fr = to_world([pt(10, 40, y=100), pt(70, 40, y=100)], 0.01, 1080, 240, 10)
    assert t.tolist() == pytest.approx([0.0, 0.25]) and y[0] == pytest.approx(9.8)
    assert fr.tolist() == [10, 70]


def test_cutoff_frame_240fps():
    assert cutoff_frame(12, 240) == 72


def test_slice_excludes_future():
    pts = [pt(i, 40) for i in range(200)]
    s = slice_to_cutoff(pts, 12, 72)
    assert max(p.frame for p in s) == 72 and min(p.frame for p in s) == 12


def test_detect_release():
    ball_pts = [pt(i, 10, x=20 + 5 * i, y=50) for i in range(20)]
    hand_masks = {}
    for i in range(20):
        m = np.zeros((100, 200), bool)
        m[30:70, : 30 + 5 * min(i, 9)] = True  # hand follows the ball until frame 9, then stops
        hand_masks[i] = m
    # ball centre x=20+5i, radius 5; hand covers cols < 30+5*min(i,9), i.e. up to 74 from i=9,
    # dilated by 5 -> 79. Ball left edge 15+5i first exceeds 79 at i=13.
    assert detect_release(ball_pts, hand_masks) == 13


def test_detect_release_without_hands():
    assert detect_release([pt(3, 10), pt(4, 10)], {}) == 3


def test_actual_landing_interpolates():
    # surface at 500 px, radius 10 -> centre line 490; falling from 470 to 500 between frames 4 and 5
    pts = [pt(0, 20, x=0, y=300), pt(2, 20, x=20, y=250), pt(4, 20, x=40, y=470), pt(5, 20, x=50, y=500)]
    x, y, f = actual_landing_px(pts, 500, 10, 0)
    assert f == pytest.approx(4 + 20 / 30) and x == pytest.approx(40 + 10 * 20 / 30) and y == pytest.approx(490)


def test_actual_landing_none_when_never_crosses():
    rising = [pt(i, 20, x=i, y=400 - 10 * i) for i in range(10)]
    assert actual_landing_px(rising, 1000, 20, 0) is None


def test_m_to_px_inverts_to_world():
    pts = [pt(0, 40, x=123, y=456)]
    _, x, y, _ = to_world(pts, 0.002, 720, 240, 0)
    px, py = m_to_px(x, y, 0.002, 720)
    assert px[0] == pytest.approx(123) and py[0] == pytest.approx(456)


def test_actual_landing_when_ball_settles_just_short_of_line():
    # line = 500 - 10 = 490; the ball comes to rest at 489.8 (measurement noise), never strictly crossing
    pts = [pt(0, 20, x=0, y=300), pt(1, 20, x=10, y=470), pt(2, 20, x=20, y=489.8), pt(3, 20, x=20, y=489.8)]
    x, y, f = actual_landing_px(pts, 500, 10, 0)
    assert f == pytest.approx(1 + 20 / 19.8) and x == pytest.approx(10 + 10 * 20 / 19.8) and y == pytest.approx(490)


def _hands_until(n_frames, hold_until, missing=()):
    masks = {}
    for i in range(n_frames):
        if i in missing:
            continue
        m = np.zeros((100, 200), bool)
        m[30:70, : 30 + 5 * min(i, hold_until)] = True
        masks[i] = m
    return masks


def test_detect_release_ignores_one_missed_hand_mask():
    ball_pts = [pt(i, 10, x=20 + 5 * i, y=50) for i in range(20)]
    assert detect_release(ball_pts, _hands_until(20, 9, missing={5})) == 13


def test_detect_release_needs_three_separated_frames():
    ball_pts = [pt(i, 10, x=20 + 5 * i, y=50) for i in range(20)]
    hands = _hands_until(20, 9)
    hands[4] = np.zeros((100, 200), bool)  # one frame where the hand mask came back empty-ish
    assert detect_release(ball_pts, hands) == 13
