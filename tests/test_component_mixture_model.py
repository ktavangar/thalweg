"""Tests for the per-star joint likelihood in `ComponentMixtureModel.__call__`.

The mixture must be ``sum_n logsumexp_k [ln pi_k + sum_c ln p_k(x_nc)]``:
one membership per star across all coordinates. It must also handle

- ragged coordinates (a subset of stars, with a ``<coord>_valid`` mask),
- coordinates missing from `data` entirely,
- joint (tuple) coordinates, including components that group the same
  coordinates differently (offtrack's joint ("phi1","phi2") grid next to
  bkg/stream's separate "phi1"/"phi2"),
- per-star measurement errors, handled by analytic convolution (no
  per-star ``:modeldata`` latent sites).
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax.scipy.special import logsumexp
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from stream_membership import ModelComponent
from stream_membership.distributions import IndependentGMM
from stream_membership.model import ComponentMixtureModel

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `field\\(init=False\\)`:UserWarning"
)

numpyro.enable_x64()


def _make_component(name, loc_phi1, loc_pm1, loc_mag):
    return ModelComponent(
        name=name,
        coord_distributions={
            "phi1": dist.Normal,
            "pm1": dist.Normal,
            "radial_velocity": dist.Normal,
            ("mag1", "mag2"): dist.MultivariateNormal,
        },
        coord_parameters={
            "phi1": {"loc": loc_phi1, "scale": 5.0},
            "pm1": {"loc": loc_pm1, "scale": 1.0},
            "radial_velocity": {"loc": 0.0, "scale": 50.0},
            ("mag1", "mag2"): {
                "loc": jnp.array([loc_mag, loc_mag + 1.0]),
                "covariance_matrix": jnp.eye(2) * 0.1,
            },
        },
    )


@pytest.fixture
def mixture_model():
    bkg = _make_component("bkg", loc_phi1=0.0, loc_pm1=0.0, loc_mag=15.0)
    stream = _make_component("stream", loc_phi1=0.0, loc_pm1=5.0, loc_mag=15.5)
    return ComponentMixtureModel(
        mixing_probs=dist.Dirichlet(jnp.array([1.0, 1.0])),
        components=[bkg, stream],
    )


def _ragged_data(key, n=50, n_pm=20):
    k1, k2, k3 = jax.random.split(key, 3)
    valid = jnp.zeros(n, dtype=bool).at[jax.random.choice(k3, n, (n_pm,), replace=False)].set(True)
    return {
        "phi1": dist.Normal(0.0, 5.0).sample(k1, (n,)),
        "pm1": dist.Normal(2.0, 2.0).sample(k2, (n_pm,)),
        "pm1_valid": valid,
        "mag1": jnp.linspace(14.5, 16.0, n),
        "mag2": jnp.linspace(15.5, 17.0, n),
    }


def _factor_value(tr):
    site = tr["mixture-ln_like"]
    return float(site["fn"].log_prob(site["value"]).sum())


def test_matches_manual_per_star_mixture(mixture_model):
    """Ragged pm1 (with mask), joint mags, rv missing entirely."""
    data = _ragged_data(jax.random.PRNGKey(0))
    probs = jnp.array([0.7, 0.3])
    tr = trace(substitute(seed(mixture_model, 0), data={"mixture-probs": probs})).get_trace(
        data=data, err=None
    )
    got = _factor_value(tr)

    n = data["phi1"].shape[0]
    lps = []
    for loc_pm1, loc_mag in [(0.0, 15.0), (5.0, 15.5)]:
        lp = dist.Normal(0.0, 5.0).log_prob(data["phi1"])
        lp_pm = jnp.zeros(n).at[jnp.flatnonzero(data["pm1_valid"])].set(
            dist.Normal(loc_pm1, 1.0).log_prob(data["pm1"])
        )
        lp_mag = dist.MultivariateNormal(
            jnp.array([loc_mag, loc_mag + 1.0]), jnp.eye(2) * 0.1
        ).log_prob(jnp.stack([data["mag1"], data["mag2"]], -1))
        lps.append(lp + lp_pm + lp_mag)
    expected = logsumexp(jnp.stack(lps) + jnp.log(probs)[:, None], axis=0).sum()
    np.testing.assert_allclose(got, float(expected), rtol=1e-10)

    # radial_velocity absent -> no latent site for it:
    assert "radial_velocity:modeldata" not in tr


def test_ragged_without_mask_raises(mixture_model):
    data = _ragged_data(jax.random.PRNGKey(1))
    del data["pm1_valid"]
    with pytest.raises(ValueError, match="pm1_valid"):
        trace(seed(mixture_model, 0)).get_trace(data=data, err=None)


def _spatial_mixture(offtrack_weights):
    """bkg/stream with separate phi1/phi2; offtrack with a joint 2x2 grid."""
    def sep(name, loc2, scale2):
        return ModelComponent(
            name=name,
            coord_distributions={"phi1": dist.Uniform, "phi2": dist.Normal},
            coord_parameters={
                "phi1": {"low": -40.0, "high": 0.0},
                "phi2": {"loc": loc2, "scale": scale2},
            },
        )

    offtrack = ModelComponent(
        name="offtrack",
        coord_distributions={("phi1", "phi2"): IndependentGMM},
        coord_parameters={
            ("phi1", "phi2"): {
                "mixing_distribution": (dist.Categorical, offtrack_weights),
                "locs": jnp.array([[-30.0, -30.0, -10.0, -10.0], [2.0, -2.0, 2.0, -2.0]]),
                "scales": jnp.array([[1.0] * 4, [0.3] * 4]),
            },
        },
    )
    return ComponentMixtureModel(
        mixing_probs=jnp.array([0.9, 0.07, 0.03]),
        components=[sep("bkg", 0.0, 3.0), sep("stream", 0.0, 0.2), offtrack],
    )


def test_offtrack_grid_is_evaluated_jointly():
    """Two weight maps with identical phi1/phi2 marginals must give
    different likelihoods: the spatial node weights are identifiable."""
    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    # Off-track stars at (-30, +2) and (-10, -2):
    phi1 = jnp.concatenate([dist.Normal(-30, 1).sample(k1, (100,)), dist.Normal(-10, 1).sample(k2, (100,))])
    phi2 = jnp.concatenate([jnp.full(100, 2.0), jnp.full(100, -2.0)])
    data = {"phi1": phi1, "phi2": phi2}

    true_w = jnp.array([0.5, 1e-6, 1e-6, 0.5])  # nodes (-30,+2), (-10,-2)
    mirror_w = jnp.array([1e-6, 0.5, 0.5, 1e-6])  # nodes (-30,-2), (-10,+2)
    vals = []
    for w in (true_w, mirror_w):
        tr = trace(seed(_spatial_mixture(w / w.sum()), 0)).get_trace(data=data)
        vals.append(_factor_value(tr))
    assert vals[0] > vals[1] + 100


def test_error_convolution_and_svi(mixture_model):
    """Errors are convolved analytically: no per-star latent sites, and the
    pm1 term for each component is Normal(loc, sqrt(scale**2 + err**2))."""
    data = _ragged_data(jax.random.PRNGKey(4), n=40, n_pm=15)
    e = jnp.linspace(0.05, 2.0, 15)
    err = {"pm1": e}
    probs = jnp.array([0.6, 0.4])

    tr = trace(substitute(seed(mixture_model, 0), data={"mixture-probs": probs})).get_trace(
        data=data, err=err
    )
    assert not any(k.endswith(":modeldata") for k in tr)

    n = data["phi1"].shape[0]
    lps = []
    for loc_pm1, loc_mag in [(0.0, 15.0), (5.0, 15.5)]:
        lp = dist.Normal(0.0, 5.0).log_prob(data["phi1"])
        lp_pm = jnp.zeros(n).at[jnp.flatnonzero(data["pm1_valid"])].set(
            dist.Normal(loc_pm1, jnp.sqrt(1.0 + e**2)).log_prob(data["pm1"])
        )
        lp_mag = dist.MultivariateNormal(
            jnp.array([loc_mag, loc_mag + 1.0]), jnp.eye(2) * 0.1
        ).log_prob(jnp.stack([data["mag1"], data["mag2"]], -1))
        lps.append(lp + lp_pm + lp_mag)
    expected = logsumexp(jnp.stack(lps) + jnp.log(probs)[:, None], axis=0).sum()
    np.testing.assert_allclose(_factor_value(tr), float(expected), rtol=1e-10)

    guide = AutoNormal(mixture_model)
    svi = SVI(mixture_model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(5), 50, data=data, err=err, progress_bar=False)
    assert jnp.all(jnp.isfinite(res.losses))
