import contextlib

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from stream_membership.distributions.gmm import IndependentGMM


@pytest.mark.parametrize(
    "kwargs",
    [
        {  # Two 1D distributions, truncated low/high:
            "probs": [1.0, 0.2],
            "locs": np.array([[1.0], [2.0]]).T,
            "scales": np.array([[0.25], [0.1]]).T,
            "low": 0.5,
            "high": 2.1,
        },
        {  # Three 2D distributions, truncated low:
            "probs": [1.0, 0.5, 0.2],
            "locs": np.array([[1.0, 2.0], [1.5, 0.5], [0.5, 1.0]]).T,
            "scales": 0.2,
            "low": np.array([0.2, 0.0])[:, None],
        },
        {  # Three 2D distributions, truncated low but only one dim:
            "probs": [1.0, 0.5, 0.2],
            "locs": np.array([[1.0, 2.0], [1.5, 0.5], [0.5, 1.0]]).T,
            "scales": 0.2,
            "low": np.array([0.2, -np.inf])[:, None],
        },
        {  # Two 3D distributions, truncated high:
            "probs": [1.0, 0.2],
            "locs": np.array([[1.0, 2.0, 0.0], [1.5, 0.5, -1]]).T,
            "scales": np.array([[1.0, 1.0, 2.0], [2, 1, 1]]).T,
            "high": np.array([2.2, 1.5, 3.5])[:, None],
        },
    ],
)
def test_gridgmm(kwargs):
    """
    A mixture of two 1D distributions
    """
    mix = dist.Categorical(probs=jnp.array(kwargs.pop("probs")))
    gmm = IndependentGMM(mix, **kwargs)
    gmm_notrunc = IndependentGMM(
        mix, **{k: v for k, v in kwargs.items() if k not in ["low", "high"]}
    )
    D = gmm._D

    N_samples = 10

    rng = np.random.default_rng(seed=42)
    vals = rng.uniform(-10, 10, size=(N_samples, D))
    # `support.check(...)` already reduces over the D (event) axis -- per
    # numpyro's constraint contract, a `check()` on an `event_dim=1`
    # constraint returns a mask shaped like the *batch* dims only (D is
    # consumed), not (N, D) -- so no further `np.all(..., axis=1)` is needed
    # (nor correct: it would over-reduce an already-(N,)-shaped result).
    check = np.asarray(gmm.support.check(vals))

    logprob_vals = gmm.log_prob(vals)
    assert np.all(np.isfinite(logprob_vals[check]))
    assert np.all(~np.isfinite(logprob_vals[~check]))
    assert np.all(logprob_vals[check] >= gmm_notrunc.log_prob(vals[check]))


def test_mixture_general_of_independent_gmms_preserves_shape_under_validation():
    """Regression test for the Stage-3 `ComponentMixtureModel` crash.

    `dist.MixtureGeneral([IndependentGMM(...), IndependentGMM(...)])` (e.g. a
    bkg/stream mixture over a 1D `phi1`-like coordinate) used to return a
    `log_prob` shaped `(1, N)` instead of `(N,)` whenever
    `numpyro.validation_enabled()` was active (as it always is inside
    `run_SVI`), because `IndependentGMM.support` forwarded the underlying
    `TruncatedNormal`'s (D, 1)-shaped `Interval` constraint directly, which
    broadcasts wrongly against an (N,)-shaped `value`. This should hold for
    both `validate_args` settings, and should match with/without a `high`/
    `low` bound (e.g. `MixtureSameFamily`'s own unbounded phi1 case).
    """
    knots = jnp.arange(5.0).reshape(1, -1)
    N = 37

    for validate in (False, True):
        ctx = numpyro.validation_enabled() if validate else contextlib.nullcontext()
        with ctx:
            bkg = IndependentGMM(
                dist.CategoricalProbs(jnp.ones(5) / 5),
                locs=knots,
                scales=jnp.ones((1, 5)),
                low=jnp.array([[-10.0]]),
                high=jnp.array([[10.0]]),
            )
            stream = IndependentGMM(
                dist.CategoricalProbs(jnp.ones(3) / 3),
                locs=knots[:, :3],
                scales=jnp.ones((1, 3)),
                low=jnp.array([[-10.0]]),
                high=jnp.array([[10.0]]),
            )
            mixture = dist.MixtureGeneral(
                dist.CategoricalProbs(jnp.array([0.4, 0.6])), [bkg, stream]
            )
            value = jnp.linspace(-5, 5, N)

            assert mixture.support.check(value).shape == (N,)
            assert mixture.log_prob(value).shape == (N,)
