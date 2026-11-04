"""Synthetic throws for tests: exact physics plus noise."""
import numpy as np


def make_throw(vx=3.0, vy=4.0, x0=0.5, y0=1.0, g=9.81, k=0.0, fps=240, dur=1.2, noise_m=0.002, seed=0):
    t = np.arange(0, dur, 1 / fps)
    if k > 0:
        e = 1 - np.exp(-k * t)
        x = x0 + vx / k * e
        y = y0 + (vy + g / k) / k * e - g / k * t
    else:
        x = x0 + vx * t
        y = y0 + vy * t - 0.5 * g * t**2
    rng = np.random.default_rng(seed)
    return t, x + rng.normal(0, noise_m, t.size), y + rng.normal(0, noise_m, t.size)


def _draw_disk(img, x, y, r, color, ss=8):
    """Exact-area anti-aliased disk (cv2.circle's AA disk comes out ~0.5 px too large in radius)."""
    import cv2

    x0, y0 = int(np.floor(x - r)) - 2, int(np.floor(y - r)) - 2
    n = int(np.ceil(2 * r)) + 5
    big = np.zeros((n * ss, n * ss), np.uint8)
    cv2.circle(big, (int(round((x - x0) * ss * 16)), int(round((y - y0) * ss * 16))), int(round(r * ss * 16)),
               255, -1, cv2.LINE_8, shift=4)
    a = cv2.resize(big, (n, n), interpolation=cv2.INTER_AREA).astype(float)[..., None] / 255
    roi = img[y0:y0 + n, x0:x0 + n]
    roi[:] = (roi * (1 - a[:roi.shape[0], :roi.shape[1]]) + np.array(color) * a[:roi.shape[0], :roi.shape[1]]).astype(np.uint8)


def make_synthetic_clip(path, vx=2.5, vy=3.5, g=9.81, capture_fps=240, scale=0.0025, size=(1280, 720),
                        release_frame=10, dur=1.2, launch_px=(150, 450), surface_y_px=680, blank_frames=(), ball_diam_m=0.067):
    """Write a side-on slow-mo clip of a yellow 6.7 cm ball thrown from a magenta 'hand'. Returns the truth."""
    import cv2

    w, h = size
    r_px = ball_diam_m / 2 / scale
    line_px = surface_y_px - r_px
    n = int(round(dur * capture_fps))
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), capture_fps, (w, h))
    points, landing = [], None
    for i in range(n):
        t = max(0.0, (i - release_frame) / capture_fps)
        x = launch_px[0] + vx * t / scale
        y = launch_px[1] - (vy * t - 0.5 * g * t**2) / scale
        if y >= line_px:  # landed: rest on the floor (no bounce)
            if landing is None:
                # exact crossing time for the truth
                tl = (vy + np.sqrt(vy**2 + 2 * g * (line_px - launch_px[1]) * scale)) / g
                landing = (launch_px[0] + vx * tl / scale, release_frame + tl * capture_fps)
            x, y = landing[0], line_px
        img = np.full((h, w, 3), 110, np.uint8)
        cv2.rectangle(img, (0, surface_y_px), (w, h), (70, 70, 70), -1)
        hx = launch_px[0] if i >= release_frame else x
        cv2.rectangle(img, (int(hx - 45), int(launch_px[1] - 25)), (int(hx + 2), int(launch_px[1] + 25)),
                      (230, 0, 230), -1)
        if i not in blank_frames:
            _draw_disk(img, x, y, r_px, (0, 230, 230))
            points.append((i, x, y))
        vw.write(img)
    vw.release()
    return {"scale": scale, "surface_y_px": surface_y_px, "release_frame": release_frame, "radius_px": r_px,
            "landing_x_px": landing[0] if landing else None, "landing_frame": landing[1] if landing else None,
            "click_xy": [launch_px[0], launch_px[1]], "points": points, "n_frames": n, "capture_fps": capture_fps,
            "width": w, "height": h}
