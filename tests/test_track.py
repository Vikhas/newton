import shutil

import pytest

from backend.clients.fake import FakeClient
from backend.track import encode_video, extract_frames, ffmpeg_exe, probe, track_ball, track_hands
from tests.fixtures.synth import make_synthetic_clip


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("clip")
    truth = make_synthetic_clip(d / "src.mp4")
    n = extract_frames(d / "src.mp4", d / "frames")
    encode_video(d / "frames", d / "player.mp4", 30)
    return d, truth, n


def test_ffmpeg_available():
    assert ffmpeg_exe()


def test_extract_frames_count(clip):
    d, truth, n = clip
    assert n == truth["n_frames"] == len(list((d / "frames").glob("*.jpg")))
    assert (d / "frames" / "00000.jpg").exists()


def test_probe_reads_capture_fps(clip):
    d, truth, _ = clip
    p = probe(d / "src.mp4")
    assert p["fps"] == pytest.approx(240) and p["width"] == 1280 and p["height"] == 720


def test_player_video_is_30fps(clip):
    d, truth, n = clip
    p = probe(d / "player.mp4")
    assert p["fps"] == pytest.approx(30) and p["n_frames"] == n


def test_track_synthetic_centroids(clip):
    d, truth, _ = clip
    pts = {p.frame: p for p in track_ball(d / "player.mp4", truth["click_xy"], FakeClient(), "synth", truth["width"])}
    for frame, x, y in truth["points"]:
        p = pts[frame]
        assert abs(p.x_px - x) < 1.0 and abs(p.y_px - y) < 1.0, frame
        assert abs(p.diam_px - 2 * truth["radius_px"]) < 1.5
        assert 3 <= len(p.contour) <= 24


def test_track_skips_frames_without_ball(tmp_path):
    truth = make_synthetic_clip(tmp_path / "src.mp4", dur=0.3, blank_frames={40})
    extract_frames(tmp_path / "src.mp4", tmp_path / "frames")
    encode_video(tmp_path / "frames", tmp_path / "player.mp4", 30)
    frames = {p.frame for p in track_ball(tmp_path / "player.mp4", truth["click_xy"], FakeClient(), "b", 1280)}
    assert 40 not in frames and 39 in frames and 41 in frames


def test_track_hands_full_frame_masks(clip):
    d, truth, _ = clip
    hands = track_hands(d / "player.mp4", FakeClient(), "synth", (720, 1280), max_frame=20)
    assert set(hands) == set(range(21)) and hands[0].shape == (720, 1280) and hands[0].any()


def test_diameter_ignores_motion_blur():
    import cv2
    import numpy as np

    from backend.clients import Instance
    from backend.track import _measure

    m = np.zeros((60, 90), np.uint8)
    for dx in np.linspace(0, 7, 15):  # a 27 px ball smeared 7 px (26%) along x
        cv2.circle(m, (int(round(30 + dx)), 30), 13, 1, -1)
    p = _measure(Instance(mask=m.astype(bool), left=0, top=0), 0)
    assert abs(p.diam_px - 27) < 1.2


def test_prefers_bundled_ffmpeg(monkeypatch):
    import imageio_ffmpeg

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert ffmpeg_exe() == imageio_ffmpeg.get_ffmpeg_exe()


def test_track_tries_next_ball_description(clip):
    d, truth, _ = clip

    class OnlyWhiteBall(FakeClient):
        def segment_video(self, video_path, concept, cache_key):
            return super().segment_video(video_path, concept, cache_key) if concept == "white ball" else {}

    pts = track_ball(d / "player.mp4", truth["click_xy"], OnlyWhiteBall(), "synth", truth["width"],
                     prompts=["ball", "white ball"])
    assert len(pts) > 100
