"""The physicist agent: Muse Spark plans a fit, Python runs it, Muse Spark Open critiques, repeat.

The tools hold only the points up to the cutoff, so nothing the agent does can see the rest of the flight.
"""
import json
import re
from dataclasses import dataclass

import numpy as np

from backend import config, physics, prompts
from backend.clients import ModelError
from backend.trajectory import compute_scale, slice_to_cutoff, to_world


class AgentError(RuntimeError):
    """The agent loop could not produce a fit (bad model replies or API failure)."""


class AgentTools:
    def __init__(self, points, release_frame, cutoff, scale, height, capture_fps, surface_y_px,
                 ball_diam_m=config.BALL_DIAM_M):
        self.points = slice_to_cutoff(points, release_frame, cutoff)
        self.release_frame, self.cutoff = release_frame, cutoff
        self.scale, self.height, self.capture_fps = scale, height, capture_fps
        self.surface_y_px, self.ball_diam_m = surface_y_px, ball_diam_m
        self.dropped: set[int] = set()
        self.log: list[dict] = []

    def active(self):
        return [p for p in self.points if p.frame not in self.dropped]

    def world(self):
        return to_world(self.active(), self.scale, self.height, self.capture_fps, self.release_frame)

    def landing_y_m(self):
        return (self.height - self.surface_y_px) * self.scale + self.ball_diam_m / 2

    def keyframe_ids(self):
        return [self.release_frame, (self.release_frame + self.cutoff) // 2, self.cutoff]

    def fit_projectile(self, fit_g: bool, drag: bool) -> physics.LaunchFit:
        t, x, y, frames = self.world()
        fit = physics.fit_projectile(t, x, y, frames, fit_g=fit_g, drag=drag)
        self.log.append({"tool": "fit_projectile", "args": {"fit_g": fit_g, "drag": drag},
                         "result": {"rms_cm": fit.rms_cm, "g": fit.g}})
        return fit

    def drop_frames(self, ids) -> list[int]:
        known = {p.frame for p in self.points}
        ids = sorted({int(i) for i in ids if int(i) in known and int(i) not in self.dropped})
        if len(self.active()) - len(ids) < config.MIN_FIT_POINTS:
            ids = []
        self.dropped.update(ids)
        self.log.append({"tool": "drop_frames", "args": {"ids": ids}, "result": {"dropped": ids}})
        return ids

    def recheck_scale(self) -> dict:
        d = np.array([p.diam_px for p in self.active()])
        steady = [p for p in self.active() if abs(p.diam_px - np.median(d)) <= 0.1 * np.median(d)]
        old = self.scale
        self.scale = compute_scale(steady or self.active(), self.ball_diam_m)
        out = {"scale_old": old, "scale_new": self.scale}
        self.log.append({"tool": "recheck_scale", "args": {}, "result": out})
        return out

    def predict_landing(self, fit) -> physics.Landing:
        return physics.predict_landing(fit, self.landing_y_m())


@dataclass
class AgentResult:
    fit: physics.LaunchFit
    landing: physics.Landing
    reasoning: list[dict]
    rounds: int


def parse_json(text: str, required: set[str]) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1] if start >= 0 else text)
    if not isinstance(data, dict) or not required <= data.keys():
        raise ValueError(f"reply is missing {required - set(data) if isinstance(data, dict) else required}")
    return data


def ask(client, system, payload, schema, images=()):
    """One JSON exchange, retried once on a malformed reply."""
    user = json.dumps(payload)
    for attempt in range(2):
        content = user if attempt == 0 else user + "\nReturn only the JSON object."
        try:
            text = client.chat([{"role": "system", "content": system}, {"role": "user", "content": content}],
                               images=images, json_schema=schema)
            return parse_json(text, set(schema["required"]))
        except ModelError:
            raise
        except (ValueError, json.JSONDecodeError):
            continue
    raise AgentError("the model did not return valid JSON")


def _points_payload(tools):
    t, x, y, frames = tools.world()
    diam = {p.frame: p.diam_px for p in tools.active()}
    return [{"frame": int(f), "t": round(float(a), 4), "x_m": round(float(b), 4), "y_m": round(float(c), 4),
             "diam_px": round(diam[int(f)], 1)} for a, b, c, f in zip(t, x, y, frames)]


def _fit_summary(fit, tools):
    worst = np.argsort(fit.residuals_cm)[::-1][:5]
    return {"model": "projectile_drag" if fit.drag else "projectile", "fit_g": fit.fit_g,
            "g_fitted": round(fit.g, 3), "drag_k": round(fit.k, 4), "rms_cm": round(fit.rms_cm, 3),
            "speed_mps": round(fit.speed_mps, 3), "angle_deg": round(fit.angle_deg, 2), "n_points": len(fit.frames),
            "worst_residuals": [{"frame": int(fit.frames[i]), "residual_cm": round(float(fit.residuals_cm[i]), 3)}
                                for i in worst],
            "dropped_frames": sorted(int(f) for f in tools.dropped)}


def run_agent(tools, keyframes, spark, critic, max_rounds=config.MAX_ROUNDS,
              rms_target_cm=config.RMS_TARGET_CM) -> AgentResult:
    reasoning, history, rounds_fits, accepted = [], [], [], None
    rnd = 0
    for rnd in range(1, max_rounds + 1):
        payload = {"round": rnd, "scale_m_per_px": tools.scale, "capture_fps": tools.capture_fps,
                   "release_frame": tools.release_frame, "cutoff_frame": tools.cutoff,
                   "points": _points_payload(tools), "history": history}
        plan = ask(spark, prompts.PLANNER, payload, prompts.PLANNER_SCHEMA, images=keyframes if rnd == 1 else ())
        reasoning.append({"round": rnd, "role": "planner", "text": str(plan.get("note", ""))})

        if plan.get("drop_frames"):
            dropped = tools.drop_frames(plan["drop_frames"])
            if dropped:
                reasoning.append({"round": rnd, "role": "tool",
                                  "text": f"Dropped frame{'s' if len(dropped) > 1 else ''} "
                                          f"{', '.join(str(f) for f in dropped)}."})
        if plan.get("recheck_scale"):
            r = tools.recheck_scale()
            reasoning.append({"round": rnd, "role": "tool",
                              "text": f"Rechecked scale: {r['scale_old'] * 100:.3f} → {r['scale_new'] * 100:.3f} cm/px."})

        fit = tools.fit_projectile(bool(plan.get("fit_g")), plan.get("model") == "projectile_drag")
        summary = _fit_summary(fit, tools)
        g_note = f", g = {fit.g:.2f} m/s²" if fit.fit_g else ""
        reasoning.append({"round": rnd, "role": "tool", "text": f"Fit {summary['model'].replace('_', ' + ')}: "
                          f"{fit.rms_cm:.2f} cm RMS{g_note}."})

        review = ask(critic, prompts.CRITIC, summary, prompts.CRITIC_SCHEMA)
        reasoning.append({"round": rnd, "role": "critic", "text": str(review.get("reason", ""))})
        rounds_fits.append(fit)
        history.append({"round": rnd, "plan": plan, "fit": summary, "critique": review})
        if review.get("verdict") == "accept" and fit.rms_cm < rms_target_cm:
            accepted = fit
            break

    final = accepted or min(rounds_fits, key=lambda f: f.rms_cm)
    landing = tools.predict_landing(final)
    try:
        lines = ask(spark, prompts.EXPLAIN, {"history": history, "final": _fit_summary(final, tools)},
                    prompts.EXPLAIN_SCHEMA)["lines"]
        reasoning += [{"round": rnd, "role": "summary", "text": str(line)} for line in lines[:3]]
    except AgentError:
        pass  # the explanation is a nicety; the prediction stands without it
    return AgentResult(fit=final, landing=landing, reasoning=reasoning, rounds=rnd)
