import json

import numpy as np
import pytest

from backend.physics import LaunchFit, fit_projectile, predict_landing
from tests.fixtures.synth import make_throw


def _fit(t, x, y, cutoff=0.25, **kw):
    m = t <= cutoff
    return fit_projectile(t[m], x[m], y[m], list(range(int(m.sum()))), **kw)


def test_fixed_g_fit_recovers_launch():
    f = _fit(*make_throw())
    assert abs(f.vx - 3.0) < 0.05 and abs(f.vy - 4.0) < 0.05 and f.rms_cm < 0.5
    assert abs(f.angle_deg - 53.13) < 1.0 and abs(f.speed_mps - 5.0) < 0.05


def test_fit_g_recovers_gravity():
    assert abs(_fit(*make_throw(), fit_g=True).g - 9.81) < 0.3


def test_landing_within_1cm():
    f = _fit(*make_throw())
    land_y = 0.0335
    t_land = (4.0 + np.sqrt(16 + 2 * 9.81 * (1.0 - land_y))) / 9.81
    lp = predict_landing(f, land_y)
    assert abs(lp.x_m - (0.5 + 3.0 * t_land)) < 0.01 and lp.sigma_m > 0 and lp.arc.shape[1] == 3


def test_leftward_throw():
    assert predict_landing(_fit(*make_throw(vx=-3.0)), 0.0).x_m < 0.5


def test_drag_fit():
    f = _fit(*make_throw(k=0.3), cutoff=0.4, drag=True)
    assert f.rms_cm < 0.5 and abs(f.k - 0.3) < 0.15


def test_too_few_points_raises():
    t, x, y = make_throw()
    with pytest.raises(ValueError, match="not enough tracked points before cutoff"):
        fit_projectile(t[:5], x[:5], y[:5], list(range(5)))


def test_whatif_overrides():
    f = _fit(*make_throw(y0=0.0335))
    earth = predict_landing(f, 0.0335).x_m - 0.5
    assert predict_landing(f, 0.0335, g=1.62).x_m - 0.5 > 5 * earth
    assert abs((predict_landing(f, 0.0335, speed_scale=2.0).x_m - 0.5) / earth - 4.0) < 0.1


def test_never_reaches_surface_raises():
    with pytest.raises(ValueError, match="never reaches"):
        predict_landing(_fit(*make_throw()), 50.0)


def test_fit_roundtrip_dict():
    f = _fit(*make_throw())
    assert LaunchFit.from_dict(json.loads(json.dumps(f.to_dict()))).vx == pytest.approx(f.vx)
