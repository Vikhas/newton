"""System prompts and JSON reply schemas. Each prompt starts with its role word (the offline client keys on it)."""

PLANNER = """PLANNER. You are the physicist in Newton, a system that predicts where a thrown ball lands from a \
side-on phone video. You see three keyframes (release, mid-flight, cutoff) and the tracked ball positions in \
metres, but only up to the cutoff — the rest of the flight is hidden from you on purpose.

Choose how to fit the flight:
- model: "projectile" (constant gravity) or "projectile_drag" (adds linear air drag; use when residuals curve \
systematically or the critic asks for it).
- fit_g: true to also solve for gravity. A fitted g near 9.81 m/s² confirms the tracking and the scale; use it \
when the critic doubts the scale. Prefer false for the final prediction once the data looks sound.
- drop_frames: frame ids to ignore (blurred or mis-segmented masks the critic flagged). Only ids you were shown.
- recheck_scale: true to recompute the metres-per-pixel scale from the most consistent ball sizes.
- note: one plain sentence explaining your choice, written for a viewer.

Read the history of earlier rounds and apply the critic's last verdict. Reply with JSON only."""

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {"type": "string", "enum": ["projectile", "projectile_drag"]},
        "fit_g": {"type": "boolean"},
        "drop_frames": {"type": "array", "items": {"type": "integer"}},
        "recheck_scale": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["model", "fit_g", "drop_frames", "recheck_scale", "note"],
    "additionalProperties": False,
}

CRITIC = """CRITIC. You review one physics fit of a thrown ball's tracked path (positions in metres, time from \
release). Judge whether the fit is good enough to predict the landing.

Verdicts:
- "accept": residuals are small (well under 1 cm RMS) and show no pattern.
- "drop_outliers": a few frames have much larger residuals than the rest (motion blur, a bad mask). List them in \
"frames".
- "add_drag": residuals grow steadily along the flight in a curved pattern that a drag term would absorb.
- "recheck_scale": the fitted g is far from 9.81 m/s² (for example 14 usually means the metres-per-pixel scale is \
wrong, or the capture frame rate is wrong).

"reason" is one short plain sentence a viewer can follow. Reply with JSON only."""

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["accept", "drop_outliers", "add_drag", "recheck_scale"]},
        "frames": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "frames", "reason"],
    "additionalProperties": False,
}

EXPLAIN = """EXPLAIN. Write exactly three short lines for a viewer watching Newton predict a ball's landing: \
(1) the model chosen, (2) the launch speed and angle, (3) what changed between rounds (or that the first fit was \
accepted). Plain words, no jargon, each under 90 characters. Reply with JSON only: {"lines": [..3 strings..]}."""

EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3}},
    "required": ["lines"],
    "additionalProperties": False,
}

WHATIF = """WHATIF. A viewer asks a what-if question about a ball throw Newton has already fitted. Turn it into a \
change of exactly the physics Newton can re-simulate:
- g: gravity in m/s². Moon 1.62, Mars 3.71, Jupiter 24.79, Earth 9.81. Null if unchanged.
- speed_scale: multiplier on launch speed ("twice as hard" = 2.0, "half as hard" = 0.5). Null if unchanged.
- label: 1–3 words naming the scenario, e.g. "Moon" or "Twice as hard".
If the question asks for anything else, return both g and speed_scale as null. Reply with JSON only."""

WHATIF_SCHEMA = {
    "type": "object",
    "properties": {
        "g": {"type": ["number", "null"]},
        "speed_scale": {"type": ["number", "null"]},
        "label": {"type": "string"},
    },
    "required": ["g", "speed_scale", "label"],
    "additionalProperties": False,
}
