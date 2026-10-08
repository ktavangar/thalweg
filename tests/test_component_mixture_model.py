"""Tests for the ragged, per-coordinate-plate `ComponentMixtureModel.__call__`.

These tests intentionally avoid `NormalSpline`/`TruncatedNormalSpline` (which
have an unrelated, pre-existing `ln_scale_vals` signature mismatch on this
branch -- see `test_model.py::test_subclass`/`test_conditional_data`) and
instead build components out of plain `dist.Normal`/`dist.MultivariateNormal`
coordinates, which is enough to exercise the new architecture:

- coordinates with different numbers of valid stars (ragged plates)
- a coordinate missing entirely from `data` (skipped, not error-filled)
- a joint/tuple coordinate (e.g. a CMD-like `(mag1, mag2)` pair)
- that `mixing-probs` is sampled exactly once and shared across coordinates
- that SVI can actually run end to end on the new model
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from stream_membership import ModelComponent
from stream_membership.model import ComponentMixtureModel

# `ComponentMixtureModel` uses `eqx.field(init=False)` for a few derived
# attributes (`coord_names`, `_tied_order`, `_components`), which triggers an
# unrelated, pre-existing equinox UserWarning about gradient behavior around
# `jax.grad` on every construction. This is a pre-existing property of the
# class (unrelated to the per-coordinate-plate refactor under test here), so
# we silence it at the module level rather than changing the class's field
# declarations as part of this change.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `field\\(init=False\\)`:UserWarning"
)


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


def test_ragged_plates_have_independent_sizes(mixture_model):
    """Coordinates with fewer valid stars should produce differently-sized
    plates/sample sites, and a coordinate missing entirely from `data`
    (here: radial_velocity) should be silently skipped rather than
    requiring a placeholder value + huge error for every star."""
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)

    n_phot = 50  # every star has photometry
    n_pm = 20  # only a subset has proper motions
    # NOTE: radial_velocity intentionally omitted from `data` entirely.

    data = {
        "phi1": dist.Normal(0.0, 5.0).sample(keys[0], (n_phot,)),
        "pm1": dist.Normal(0.0, 1.0).sample(keys[1], (n_pm,)),
        "mag1": jnp.full((n_phot,), 15.0),
        "mag2": jnp.full((n_phot,), 16.0),
    }
    err = {
        "phi1": jnp.full(n_phot, 0.1),
        "pm1": jnp.full(n_pm, 0.05),
    }

    tr = trace(seed(mixture_model, rng_seed=0)).get_trace(data=data, err=err)

    # mixing-probs sampled exactly once:
    assert "mixture-probs" in tr

    # radial_velocity was omitted from `data` entirely -> no sites for it at all:
    assert "radial_velocity-obs" not in tr
    assert "radial_velocity:modeldata" not in tr

    # phi1 and pm1 each get their own independently-sized plate/obs site:
    assert tr["phi1-obs"]["value"].shape == (n_phot,)
    assert tr["pm1-obs"]["value"].shape == (n_pm,)

    # joint (tuple) coordinate obs site uses the "-".join(...) naming and has
    # the full photometric sample size with 2 columns:
    assert tr["mag1-mag2-obs"]["value"].shape == (n_phot, 2)

    # plates are independently sized and named per-coordinate:
    assert tr["phi1-obs"]["cond_indep_stack"][0].name == "data-phi1"
    assert tr["phi1-obs"]["cond_indep_stack"][0].size == n_phot
    assert tr["pm1-obs"]["cond_indep_stack"][0].name == "data-pm1"
    assert tr["pm1-obs"]["cond_indep_stack"][0].size == n_pm


def test_missing_coordinate_no_error(mixture_model):
    """A coordinate that is entirely absent from `data` must not raise --
    this is the core computational-savings behavior being tested."""
    key = jax.random.PRNGKey(1)
    n = 10
    data = {
        "phi1": dist.Normal(0.0, 5.0).sample(key, (n,)),
    }
    # No pm1, radial_velocity, or mag1/mag2 at all.
    tr = trace(seed(mixture_model, rng_seed=1)).get_trace(data=data, err=None)
    assert "phi1-obs" in tr
    assert "pm1-obs" not in tr
    assert "radial_velocity-obs" not in tr
    assert "mag1-mag2-obs" not in tr


def test_svi_runs_end_to_end(mixture_model):
    """The new ragged-plate model should be usable in a normal SVI loop
    without shape errors, including with mismatched-N coordinates."""
    numpyro.enable_x64()
    key = jax.random.PRNGKey(2)
    keys = jax.random.split(key, 3)

    n_phot = 40
    n_pm = 15
    data = {
        "phi1": dist.Normal(0.0, 5.0).sample(keys[0], (n_phot,)),
        "pm1": dist.Normal(0.0, 1.0).sample(keys[1], (n_pm,)),
        "mag1": jnp.full((n_phot,), 15.0),
        "mag2": jnp.full((n_phot,), 16.0),
    }
    err = {
        "phi1": jnp.full(n_phot, 0.1),
        "pm1": jnp.full(n_pm, 0.05),
    }

    guide = AutoNormal(mixture_model)
    svi = SVI(mixture_model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    svi_result = svi.run(key, 25, data=data, err=err, progress_bar=False)

    assert jnp.isfinite(svi_result.losses[-1])
