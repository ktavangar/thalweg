"""Tests for `IsochroneCMD`'s luminosity function and selection-function
normalization (synthetic track; no external isochrone file needed)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stream_membership.distributions import (
    SALPETER_BREAKS,
    SALPETER_SLOPES,
    IsochroneCMD,
    MagLimSelectionFunction,
)

jax.config.update("jax_enable_x64", True)

# Synthetic main-sequence-like track: bright M = -1 (massive) to faint M = 9.
_M = np.linspace(-1.0, 9.0, 120)
_MASS = 0.8 * 10 ** (-0.1 * (_M + 1.0))  # 0.8 -> 0.08 Msun, monotone decreasing with M (dN/dM ~ m^(1-alpha): rises to faint)
_KW = dict(
    track_abs_mag=_M,
    track_color=0.35 + 0.08 * (_M - 4.0) ** 2 / 10.0,
    distmod_coeffs=np.array([15.0]),  # constant distance modulus
)
_DM = 15.0


def _integral(d, phi2=None):
    mag = np.linspace(_M.min() + _DM, _M.max() + _DM, 2500)
    col = np.linspace(-0.5, 1.5, 1200)
    g = jnp.stack(jnp.meshgrid(jnp.asarray(mag), jnp.asarray(col), indexing="ij"), -1)
    g = g.reshape(-1, 2)
    val = jnp.stack([g[:, 0], g[:, 0] - g[:, 1]], -1)
    kw = {} if phi2 is None else {"phi2": jnp.zeros(len(val))}
    lp = d.log_prob(val, x=jnp.zeros(len(val)), **kw)
    p = np.exp(np.asarray(lp)).reshape(len(mag), len(col))
    return np.trapezoid(np.trapezoid(p, col, axis=1), mag)


def _sf():
    return MagLimSelectionFunction(
        phi1_nodes=jnp.linspace(-10, 10, 5),
        phi2_nodes=jnp.linspace(-5, 5, 3),
        mlim_grid=jnp.full((5, 3), 21.0),
        width=0.2,
        mag_min=14.0,
        mag_max=25.0,
    )


@pytest.mark.parametrize("lf", ["flat", "kroupa", "salpeter"])
@pytest.mark.parametrize("with_sf", [False, True])
def test_density_integrates_to_one(lf, with_sf):
    extra = {}
    if lf != "flat":
        extra["track_mass"] = _MASS
    if lf == "salpeter":
        extra.update(imf_slopes=SALPETER_SLOPES, imf_breaks=SALPETER_BREAKS)
    if with_sf:
        extra.update(selection_function=_sf(), phi2=jnp.zeros(1))
    d = IsochroneCMD(**_KW, **extra, x=jnp.zeros(1))
    assert _integral(d, phi2=0.0 if with_sf else None) == pytest.approx(1.0, abs=2e-3)


def test_luminosity_function_shifts_mass_to_faint_end():
    flat = IsochroneCMD(**_KW, x=jnp.zeros(1))
    kro = IsochroneCMD(**_KW, track_mass=_MASS, x=jnp.zeros(1))
    key = jax.random.PRNGKey(0)
    m_flat = np.asarray(flat.sample(key, (50_000,)))[..., 0].ravel()
    m_kro = np.asarray(kro.sample(key, (50_000,)))[..., 0].ravel()
    assert m_kro.mean() > m_flat.mean() + 0.5  # fainter on average


def test_default_is_flat_and_independent_of_imf_args():
    a = IsochroneCMD(**_KW, x=jnp.zeros(1))
    b = IsochroneCMD(**_KW, x=jnp.zeros(1), imf_slopes=(1.0,), imf_breaks=())
    v = jnp.array([[19.0, 18.5], [21.0, 20.4]])
    assert np.allclose(a.log_prob(v), b.log_prob(v))


def test_gradient_wrt_dm_offset_is_finite():
    sf = _sf()

    def f(off):
        d = IsochroneCMD(
            **_KW,
            track_mass=_MASS,
            x=jnp.zeros(50),
            dm_offset=off,
            selection_function=sf,
            phi2=jnp.zeros(50),
        )
        return d.log_prob(jnp.tile(jnp.array([[19.0, 18.6]]), (50, 1))).sum()

    assert np.isfinite(float(jax.grad(f)(0.0)))


def test_bad_imf_args_raise():
    with pytest.raises(ValueError):
        IsochroneCMD(**_KW, track_mass=_MASS, x=jnp.zeros(1), imf_slopes=(1.0, 2.0), imf_breaks=())
