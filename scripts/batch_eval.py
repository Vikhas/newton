"""Run every take in a manifest and report the honest error spread.

    .venv/bin/python scripts/batch_eval.py [--manifest clips/manifest.json] [--out results] [--no-agent]

Manifest entries: {"file", "ball_xy", "surface_y_px", "capture_fps"?, "release_frame"?}. The UI logs the
clicks for an entry to the browser console when you press "Predict landing".
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from backend import config, pipeline  # noqa: E402

COLUMNS = ["file", "error_cm", "rms_cm", "g_check", "release_frame", "cutoff_frame", "rounds", "landing_visible"]


def evaluate(manifest_path, out_dir, use_agent=True) -> dict:
    manifest_path, out_dir = Path(manifest_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for take in json.loads(manifest_path.read_text()):
        src = Path(take["file"])
        if not src.is_absolute():
            src = config.ROOT / src
        meta = pipeline.ingest(src, take.get("capture_fps"))
        r = pipeline.run(meta["clip_id"], take["ball_xy"], take["surface_y_px"], capture_fps=take.get("capture_fps"),
                         release_frame=take.get("release_frame"), use_agent=use_agent)
        rows.append({"file": take["file"], "error_cm": r["error_cm"], "rms_cm": r["fit"]["rms_cm"],
                     "g_check": r["fit"]["g_check"], "release_frame": r["release_frame"],
                     "cutoff_frame": r["cutoff_frame"], "rounds": r["fit"]["rounds"],
                     "landing_visible": r["actual"] is not None})
        print(f"{take['file']}: " + (f"{r['error_cm']:.1f} cm" if r["error_cm"] is not None else "landing not visible"))

    with open(out_dir / "takes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    errors = [r["error_cm"] for r in rows if r["error_cm"] is not None]
    median = statistics.median(errors) if errors else None
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    ax.hist(errors, bins=max(5, len(errors) // 2), color="#5cc8ff", edgecolor="#18201e")
    if median is not None:
        ax.axvline(median, color="#ff7a6b", linewidth=2, label=f"median {median:.1f} cm")
        ax.legend(frameon=False)
    ax.set_xlabel("Landing error (cm)")
    ax.set_ylabel("Takes")
    ax.set_title(f"Newton landing error across {len(errors)} takes")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "error_hist.png")
    plt.close(fig)

    if median is not None:
        print(f"median error: {median:.1f} cm over {len(errors)} takes "
              f"({len(rows) - len(errors)} without a visible landing left out)")
    return {"median_cm": median, "n": len(errors), "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(config.ROOT / "clips" / "manifest.json"))
    ap.add_argument("--out", default=str(config.ROOT / "results"))
    ap.add_argument("--no-agent", action="store_true", help="plain projectile fit, no model calls for fitting")
    a = ap.parse_args()
    evaluate(a.manifest, a.out, use_agent=not a.no_agent)


if __name__ == "__main__":
    main()
