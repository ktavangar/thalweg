"""
Analytic convolution of (truncated) Normal densities with per-star Gaussian
measurement errors.

Why this exists
---------------
The model used to handle measurement errors on pm1/pm2/radial_velocity by
sampling one latent "true" value per star (the ``<coord>:modeldata`` sites)
and then scoring the observation with ``Normal(true, err)``. With an
``AutoNormal`` guide that adds two variational parameters per star per
coordinate (~6M for the full DES sample), and their sampling noise is what
forced several ELBO particles and many thousands of SVI steps.

For every Gaussian-family density in this package the latent can instead be
integrated out exactly:

    p(x_obs) = int p(t | theta) N(x_obs | t, err) dt

The fitted ``scale`` parameters are still the *intrinsic* (deconvolved)
widths; the per-star ``err`` only enters inside the likelihood. This is the
same "extreme deconvolution" idea as Bovy, Hogg & Roweis (2011).

Untruncated Normal:

    p(x) = N(x | mu, s),  s^2 = sigma^2 + err^2

Truncated Normal on [low, high] (exact, not an approximation):

    p(x) = N(x | mu, s) * [Phi(b') - Phi(a')] / [Phi(b) - Phi(a)]

with a = (low - mu)/sigma, b = (high - mu)/sigma, and a', b' the same
bounds standardized by the Gaussian posterior of the true value given x,

    m = (mu err^2 + x sigma^2) / s^2,   v = sigma err / s,
    a' = (low - m)/v,  b' = (high - m)/v.

Unlike the un-convolved truncated density, this is non-zero for observed
values outside [low, high] (an error can scatter a star across the bound).
"""

__all__ = [
    "convolved_log_prob",
    "log_normal_convolved",
    "log_truncated_normal_convolved",
]

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.special import log_ndtr
from jax.typing import ArrayLike


def _log1mexp(x: jax.Array) -> jax.Array:
    """log(1 - exp(x)) for x <= 0, computed stably (Maechler 2012)."""
    x = jnp.minimum(x, -jnp.finfo(jnp.result_type(x, float)).tiny)
    return jnp.where(
        x > -0.6931471805599453, jnp.log(-jnp.expm1(x)), jnp.log1p(-jnp.exp(x))
    )


def _log_ndtr_diff(a: Any, b: Any) -> jax.Array:
    """log(Phi(b) - Phi(a)) for a < b, stable in both tails.

    ``a`` or ``b`` may be ``None`` for an infinite bound (-inf / +inf); this is
    handled in Python so no infinities ever enter the autodiff graph.
    """
    if a is None and b is None:
        return jnp.zeros(())
    if a is None:
        return log_ndtr(b)
    if b is None:
        return log_ndtr(-a)
    # Work in the lower tail, where log_ndtr is accurate: if both bounds are
    # positive, use Phi(b) - Phi(a) = Phi(-a) - Phi(-b).
    flip = a > 0
    lo = jnp.where(flip, -b, a)
    hi = jnp.where(flip, -a, b)
    l_hi = log_ndtr(hi)
    return l_hi + _log1mexp(log_ndtr(lo) - l_hi)


_LOG_2PI = 1.8378770664093453  # log(2 pi)


def _log_normal_from_var(value: Any, loc: Any, var: Any) -> jax.Array:
    """log N(value | loc, sqrt(var)), written in terms of the variance so no
    sqrt is needed (one log + one divide per element; this runs once per
    (star, component) pair in the large mixtures)."""
    return -0.5 * (jnp.square(value - loc) / var + jnp.log(var) + _LOG_2PI)


def log_normal_convolved(
    value: ArrayLike, loc: ArrayLike, scale: ArrayLike, err: ArrayLike
) -> jax.Array:
    """log N(value | loc, sqrt(scale^2 + err^2))."""
    return _log_normal_from_var(
        jnp.asarray(value), jnp.asarray(loc), jnp.square(scale) + jnp.square(err)
    )


def log_truncated_normal_convolved(
    value: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
    err: ArrayLike,
    low: Any | None = None,
    high: Any | None = None,
    include_posterior_mass: bool = True,
) -> jax.Array:
    """Exact log density of TruncatedNormal(loc, scale, low, high) convolved
    with N(0, err). See the module docstring for the formula. ``low``/``high``
    may be ``None`` for an unbounded side; with both ``None`` this reduces to
    `log_normal_convolved`. ``err`` is assumed > 0.

    include_posterior_mass
        If False, drop the log[Phi(b') - Phi(a')] factor (the fraction of the
        star's true-value posterior that lies inside [low, high]). That factor
        is ~0 in log unless the star or the component sits within a few
        sigma of a bound, but it costs two log_ndtr evaluations per
        (star, component) pair, which dominates the run time for terms with
        many components (e.g. the offtrack TruncatedNormalGMMConditional with
        K ~ 400 nodes). The cheap normalizer log[Phi(b) - Phi(a)], which only
        depends on the component, is always kept.
    """
    if low is None and high is None:
        return log_normal_convolved(value, loc, scale, err)

    value = jnp.asarray(value)
    loc = jnp.asarray(loc)
    scale = jnp.asarray(scale)
    err = jnp.asarray(err)

    var_int = jnp.square(scale)
    var_err = jnp.square(err)
    var_tot = var_int + var_err

    def _std(bound, center, width):
        return None if bound is None else (jnp.asarray(bound) - center) / width

    if include_posterior_mass:
        s = jnp.sqrt(var_tot)
        # Gaussian posterior of the true value given this observation:
        m = (loc * var_err + value * var_int) / var_tot
        # Floor keeps the standardization finite if an error is exactly zero.
        v = jnp.maximum(scale * err / s, 1e-12 * scale)
        log_post_mass = _log_ndtr_diff(_std(low, m, v), _std(high, m, v))
    else:
        log_post_mass = 0.0
    # Depends on the component only (shape of loc/scale), not on the star.
    log_norm = _log_ndtr_diff(_std(low, loc, scale), _std(high, loc, scale))

    return _log_normal_from_var(value, loc, var_tot) + (log_post_mass - log_norm)


def convolved_log_prob(d: dist.Distribution, value: ArrayLike, err: ArrayLike) -> jax.Array:
    """log of ``d`` convolved with per-value Gaussian errors ``err``.

    Uses ``d.log_prob_convolved(value, err)`` when the distribution defines it
    (the spline / spline-mixture / GMM-conditional distributions in this
    package do), and handles plain numpyro (truncated) Normals directly.
    Anything else raises, rather than silently ignoring the errors.
    """
    if hasattr(d, "log_prob_convolved"):
        return d.log_prob_convolved(value, err)

    if isinstance(d, dist.Normal):
        return log_normal_convolved(value, d.loc, d.scale, err)

    truncated_types = (
        dist.TwoSidedTruncatedDistribution,
        dist.LeftTruncatedDistribution,
        dist.RightTruncatedDistribution,
    )
    if isinstance(d, truncated_types) and isinstance(d.base_dist, dist.Normal):
        return log_truncated_normal_convolved(
            value,
            d.base_dist.loc,
            d.base_dist.scale,
            err,
            low=getattr(d, "low", None),
            high=getattr(d, "high", None),
        )

    msg = (
        f"No analytic error convolution is implemented for {type(d).__name__}. "
        "Add a `log_prob_convolved(value, err)` method to it, or don't pass "
        "errors for this coordinate."
    )
    raise NotImplementedError(msg)
