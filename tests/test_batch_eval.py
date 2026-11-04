import csv
import json

from backend import config
from scripts.batch_eval import evaluate
from tests.fixtures.synth import make_synthetic_clip


def test_evaluate_writes_csv_chart_and_median(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OFFLINE", True)
    takes = []
    for i, (vx, vy) in enumerate([(2.5, 3.5), (2.0, 3.8)]):
        src = tmp_path / f"take{i}.mp4"
        truth = make_synthetic_clip(src, vx=vx, vy=vy)
        takes.append({"file": str(src), "ball_xy": truth["click_xy"], "surface_y_px": truth["surface_y_px"],
                      "capture_fps": 240, "release_frame": None})
    short = tmp_path / "short.mp4"
    t = make_synthetic_clip(short, dur=0.6)
    takes.append({"file": str(short), "ball_xy": t["click_xy"], "surface_y_px": t["surface_y_px"]})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(takes))

    out = evaluate(manifest, tmp_path / "results")
    rows = list(csv.DictReader(open(tmp_path / "results" / "takes.csv")))
    assert len(rows) == 3 and [r["landing_visible"] for r in rows] == ["True", "True", "False"]
    assert out["n"] == 2 and out["median_cm"] < 2.0
    assert (tmp_path / "results" / "error_hist.png").stat().st_size > 1000
