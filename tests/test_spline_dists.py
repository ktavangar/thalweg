import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stream_membership.distributions import (
    DirichletSpline,
    NormalSpline,
    TruncatedNormalSpline,
)


def make_dists(seed=867):
    rng = np.random.default_rng(seed)

    knots = jnp.linspace(0, 10, 8)
    x = jnp.linspace(1, 8, 32)

    dists = []

    d = NormalSpline(
        knots=knots,
        loc_vals=rng.uniform(-5, 5, knots.size),
        scale_vals=np.ones(knots.size),
        x=x,
    )
    dists.append(d)

    d = NormalSpline(
        knots=knots,
        loc_vals=rng.uniform(-5, 5, knots.size),
        scale_vals=rng.uniform(0.2, 1, knots.size),
        x=x,
        clip_scales=(0.1, 10.0),
    )
    dists.append(d)

    d = NormalSpline(
        knots=knots,
        loc_vals=rng.uniform(-5, 5, knots.size),
        scale_vals=rng.uniform(0.2, 1, knots.size),
        x=x,
        clip_scales=(0.1, None),
    )
    dists.append(d)

    d = TruncatedNormalSpline(
        knots=knots,
        loc_vals=rng.uniform(-5, 5, knots.size),
        scale_vals=np.ones(knots.size),
        low=-4.0,
        high=4.0,
        x=x,
    )
    dists.append(d)

    d = DirichletSpline(
        knots=knots, concentration_vals=rng.uniform(0, 1, (3, knots.size)), x=x
    )
    dists.append(d)

    return dists


test_dists = make_dists()


@pytest.mark.parametrize("dist_", test_dists)
def test_methods(dist_):
    # New x grid to be used below:
    x2 = jnp.linspace(1, 8, 64)

    s = dist_.sample(jax.random.PRNGKey(0))
    assert s.shape == (dist_.x.size, *dist_.event_shape)
    s = dist_.sample(jax.random.PRNGKey(1), sample_shape=(10,))
    assert s.shape == (10, dist_.x.size, *dist_.event_shape)

    # Test sample with a new x grid:
    s = dist_.sample(jax.random.PRNGKey(2), x=x2)
    assert s.shape == (x2.size, *dist_.event_shape)
    s = dist_.sample(jax.random.PRNGKey(3), x=x2, sample_shape=(10,))
    assert s.shape == (10, x2.size, *dist_.event_shape)

    # Test evaluating log_prob with previously provided x grid:
    shape = dist_.x.shape + dist_.event_shape
    y = np.linspace(0, 1, np.prod(shape)).reshape(shape)
    lp = dist_.log_prob(y)
    assert lp.shape == dist_.x.shape

    # Test evaluating log_prob with a different x grid:
    shape = x2.shape + dist_.event_shape
    y2 = np.linspace(0, 1, np.prod(shape)).reshape(shape)
    lp = dist_.log_prob(y2, x=x2)
    assert lp.shape == x2.shape


# TODO: test that evaluating spline log_prob with an extra event_dim is ignored...?


def test_truncated_normal_convolved_matches_quadrature():
    """Exact error-convolved truncated Normal vs. numerical integration."""
    from scipy.integrate import quad
    from scipy.stats import norm, truncnorm

    from stream_membership.distributions import log_truncated_normal_convolved

    mu, sig, err, lo, hi = 0.3, 0.7, 0.4, -1.0, 2.0
    for x in [-2.5, -1.0, 0.0, 1.5, 3.0]:
        def f(t, x=x):
            a, b = (lo - mu) / sig, (hi - mu) / sig
            return truncnorm.pdf(t, a, b, mu, sig) * norm.pdf(x, t, err)

        ref = quad(f, lo, hi)[0]
        got = float(log_truncated_normal_convolved(x, mu, sig, err, lo, hi))
        np.testing.assert_allclose(got, np.log(ref), rtol=1e-8)
