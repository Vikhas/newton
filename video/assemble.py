"""Demo video: timing plan and final assembly.

    python video/assemble.py plan    # -> video/out/plan.json (cue times the HTML scenes are recorded with)
    python video/assemble.py build   # -> video/out/newton-demo.mp4 (needs the recordings from record.mjs)
"""
import json
import subprocess
import sys
from pathlib import Path

from backend.track import ffmpeg_exe

HERE = Path(__file__).parent
OUT = HERE / "out"
VOICE = OUT / "voice"
ORDER = ["hero", "title", "pipeline", "reasoning", "second", "whatif", "outro"]
HTML = {"title": "title", "pipeline": "pipe", "reasoning": "agent", "outro": "outro"}  # VO scene -> scenes.html id
FPS, W, H = 30, 1920, 1080


def html_cues(durs, lead=0.5, gap=0.6):
    cues, t = [], lead
    for d in durs:
        cues.append(round(t, 3))
        t += d + gap
    return cues, round(t - gap, 3)  # cue per line, end of the last line


def plan():
    vo = json.loads((VOICE / "timings.json").read_text())
    marks = json.loads((OUT / "ui" / "marks.json").read_text())
    p = {}
    for name in ORDER:
        durs = [r["dur"] for r in vo[name]]
        if name in HTML:
            cues, end = html_cues(durs, lead=0.6 if name == "title" else 0.5)
            tail = 4.5 if name == "outro" else 0.9
            p[name] = {"kind": "html", "cues": cues, "end": end + 0.6, "dur": round(end + tail, 3)}
        elif name == "hero":
            m = marks["hero"]
            l0 = max(0.3, m["freeze1"] - 0.2)
            p[name] = {"kind": "ui", "cuts": [[0.0, m["end"]]],
                       "at": [l0, max(m["reveal1"] + 0.5, l0 + durs[0] + 0.6)]}
        elif name == "second":
            m = marks["second"]
            l1 = max(m["reveal"] + 0.3, 0.4 + durs[0] + 0.5)
            p[name] = {"kind": "ui", "cuts": [[0.0, max(m["end"], l1 + durs[1] + 0.8)]], "at": [0.4, l1]}
        elif name == "whatif":
            m = marks["whatif"]
            l0 = m["mic_down"] - durs[0] - 0.3
            l2 = m["mic_up"] + 0.3
            l3 = max(m["ghost1"] + 0.3, l2 + durs[2] + 0.3)
            cuts = [[l0 - 0.6, m["twice_click"] + 0.15], [m["ghost2"] - 0.45, m["end"]]]
            if m["ghost2"] - m["twice_click"] < 1.0:  # answered from cache, no wait to cut out
                cuts = [[l0 - 0.6, m["end"]]]
            p[name] = {"kind": "ui", "cuts": cuts,
                       "at": [l0, m["mic_down"] + 0.5, l2, l3, m["ghost2"] + 0.2]}
    (OUT / "plan.json").write_text(json.dumps(p, indent=1))
    return p


def run(args):
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def to_output_time(t, cuts):
    """Recording time -> time in the cut-together scene."""
    acc = 0.0
    for a, b in cuts:
        if a <= t <= b:
            return acc + t - a
        acc += b - a
    raise ValueError(f"{t} falls outside the kept parts {cuts}")


def build():
    p = json.loads((OUT / "plan.json").read_text())
    vo = json.loads((VOICE / "timings.json").read_text())
    marks = json.loads((OUT / "ui" / "marks.json").read_text())
    seg_dir = OUT / "segments"
    seg_dir.mkdir(exist_ok=True)
    segs = []
    for name in ORDER:
        s = p[name]
        if s["kind"] == "html":
            src, start = OUT / "ui" / f"{HTML[name]}.webm", marks[HTML[name]]["start"]
            cuts = [[start, start + s["dur"]]]
            at = [start + c for c in s["cues"]]
        else:
            src, cuts, at = OUT / "ui" / f"{name}.webm", s["cuts"], s["at"]
        rec_end = marks[HTML.get(name, name)]["end"]
        total = sum(b - a for a, b in cuts)
        # video: keep the cut ranges, freeze the last frame if the plan runs past the recording
        vf = []
        for i, (a, b) in enumerate(cuts):
            vf.append(f"[0:v]trim={a:.3f}:{min(b, rec_end):.3f},setpts=PTS-STARTPTS,"
                      f"tpad=stop_mode=clone:stop_duration={max(0, b - rec_end):.3f}[v{i}]")
        vf.append("".join(f"[v{i}]" for i in range(len(cuts))) + f"concat=n={len(cuts)}:v=1:a=0,"
                  f"fps={FPS},scale={W}:{H},format=yuv420p,"
                  f"fade=t=in:st=0:d=0.25,fade=t=out:st={total - 0.3:.3f}:d=0.3[v]")
        # audio: each narration line placed at its time
        inputs, af = ["-i", str(src)], []
        for j, (row, t) in enumerate(zip(vo[name], at)):
            inputs += ["-i", str(VOICE / row["file"])]
            ms = int(round(to_output_time(t, cuts) * 1000))
            af.append(f"[{j + 1}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={ms}:all=1[a{j}]")
        n = len(vo[name])
        af.append("".join(f"[a{j}]" for j in range(n)) +
                  f"amix=inputs={n}:normalize=0:duration=longest,apad,atrim=0:{total:.3f}[a]")
        out = seg_dir / f"{ORDER.index(name)}_{name}.mp4"
        run([*inputs, "-filter_complex", ";".join(vf + af), "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-r", str(FPS),
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.3f}", str(out)])
        segs.append(out)
        print(f"{name}: {total:.1f} s")
    lst = seg_dir / "list.txt"
    lst.write_text("".join(f"file '{s.resolve()}'\n" for s in segs))
    final = OUT / "newton-demo.mp4"
    run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-c:a", "aac", "-b:a", "192k", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-movflags", "+faststart", str(final)])
    print(final)


if __name__ == "__main__":
    {"plan": plan, "build": build}[sys.argv[1]]()
