"""Projectile fitting, landing prediction and what-if re-simulation.

World units are metres and seconds, y up, t = 0 at release.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, least_squares

from backend.config import G_EARTH, MIN_FIT_POINTS

_K_EPS = 1e-6


def _xy(x0, y0, vx, vy, g, k, t):
    """Position at time t; linear drag with coefficient k (1/s), plain projectile when k ~ 0."""
    t = np.asarray(t, dtype=float)
    if k < _K_EPS:
        return x0 + vx * t, y0 + vy * t - 0.5 * g * t**2
    e = 1 - np.exp(-k * t)
    return x0 + vx / k * e, y0 + (vy + g / k) / k * e - g / k * t


@dataclass
class LaunchFit:
    x0: float
    y0: float
    vx: float
    vy: float
    g: float
    k: float
    fit_g: bool
    drag: bool
    frames: list[int]
    residuals_cm: np.ndarray
    rms_cm: float
    cov: np.ndarray
    free_params: list[str] = field(default_factory=list)

    @property
    def speed_mps(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.vy, abs(self.vx))))

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("x0", "y0", "vx", "vy", "g", "k", "fit_g", "drag", "rms_cm", "free_params")}
        d.update(frames=[int(f) for f in self.frames], residuals_cm=self.residuals_cm.tolist(), cov=self.cov.tolist())
        return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "LaunchFit":
        d = dict(d)
        d["residuals_cm"] = np.asarray(d["residuals_cm"], dtype=float)
        d["cov"] = np.asarray(d["cov"], dtype=float)
        return cls(**d)


@dataclass
class Landing:
    t_s: float
    x_m: float
    y_m: float
    sigma_m: float
    arc: np.ndarray  # N x 3: t, x_m, y_m


def fit_projectile(t, x, y, frames, fit_g=False, drag=False, g=G_EARTH) -> LaunchFit:
    t, x, y = (np.asarray(a, dtype=float) for a in (t, x, y))
    if t.size < MIN_FIT_POINTS:
        raise ValueError("not enough tracked points before cutoff")

    names = ["x0", "y0", "vx", "vy"] + (["g"] if fit_g else []) + (["k"] if drag else [])
    px = np.polyfit(t, x, 1)
    py = np.polyfit(t, y, 2)
    p0 = [px[1], py[2], px[0], py[1]] + ([g] if fit_g else []) + ([0.1] if drag else [])
    lo = [-np.inf] * 4 + ([0.0] if fit_g else []) + ([0.0] if drag else [])
    hi = [np.inf] * 4 + ([100.0] if fit_g else []) + ([5.0] if drag else [])

    def unpack(p):
        v = dict(zip(names, p))
        return v["x0"], v["y0"], v["vx"], v["vy"], v.get("g", g), v.get("k", 0.0)

    def resid(p):
        mx, my = _xy(*unpack(p), t)
        return np.concatenate([mx - x, my - y])

    sol = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac")
    n, p = t.size, len(names)
    dx, dy = sol.fun[:n], sol.fun[n:]
    per_point = np.hypot(dx, dy) * 100
    s2 = float(np.sum(sol.fun**2)) / max(2 * n - p, 1)
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * s2
    x0, y0, vx, vy, g_, k = unpack(sol.x)
    return LaunchFit(
        x0=float(x0), y0=float(y0), vx=float(vx), vy=float(vy), g=float(g_), k=float(k),
        fit_g=fit_g, drag=drag, frames=list(frames), residuals_cm=per_point,
        rms_cm=float(np.sqrt(np.mean(per_point**2))), cov=cov, free_params=names,
    )


def _state(fit: LaunchFit, g, speed_scale):
    return (fit.x0, fit.y0, fit.vx * speed_scale, fit.vy * speed_scale,
            fit.g if g is None else g, fit.k)


def simulate(fit: LaunchFit, t, g=None, speed_scale=1.0):
    return _xy(*_state(fit, g, speed_scale), t)


def _crossing(state, landing_y_m, t_grid, refine):
    """Time of the last downward crossing of landing_y_m, or None."""
    _, y = _xy(*state, t_grid)
    idx = np.nonzero((y[:-1] > landing_y_m) & (y[1:] <= landing_y_m))[0]
    if idx.size == 0:
        return None
    i = idx[-1]
    if not refine:
        f = (y[i] - landing_y_m) / (y[i] - y[i + 1])
        return t_grid[i] + f * (t_grid[i + 1] - t_grid[i])
    return brentq(lambda s: _xy(*state, s)[1] - landing_y_m, t_grid[i], t_grid[i + 1])


def predict_landing(fit: LaunchFit, landing_y_m, g=None, speed_scale=1.0, t_max=30.0, n_mc=300, seed=0) -> Landing:
    t_grid = np.linspace(0, t_max, int(t_max / 0.002) + 1)
    state = _state(fit, g, speed_scale)
    t_land = _crossing(state, landing_y_m, t_grid, refine=True)
    if t_land is None:
        raise ValueError("trajectory never reaches landing surface")
    x_land = float(_xy(*state, t_land)[0])

    # Landing spread from parameter uncertainty.
    rng = np.random.default_rng(seed)
    base = np.array([getattr(fit, n) for n in fit.free_params])
    xs = []
    for p in rng.multivariate_normal(base, fit.cov, size=n_mc, check_valid="ignore"):
        s = dict(zip(fit.free_params, p))
        sample = LaunchFit(**{**fit.__dict__, **s, "k": max(s.get("k", fit.k), 0.0)})
        st = _state(sample, g, speed_scale)
        tc = _crossing(st, landing_y_m, t_grid, refine=False)
        if tc is not None:
            xs.append(_xy(*st, tc)[0])
    sigma = float(np.std(xs)) if len(xs) > 1 else 0.0

    ts = np.linspace(0, t_land, 200)
    ax, ay = _xy(*state, ts)
    return Landing(t_s=float(t_land), x_m=x_land, y_m=float(landing_y_m), sigma_m=sigma,
                   arc=np.column_stack([ts, ax, ay]))
