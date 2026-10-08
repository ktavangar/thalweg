"""Tests for `TruncatedNormalGMMConditional` and the `FromDist` marker.

These guard the fix for the `KeyError: 'phi1'` crash that used to occur when
an "offtrack"-style component's joint `(phi1, phi2)` `IndependentGMM` grid
was combined, inside a `ComponentMixtureModel`, with sibling components
(background/stream) that instead used separate, string-keyed "phi1" and
"phi2" coordinates. The fix factorizes the joint grid into a 1D `phi1`
`IndependentGMM` marginal plus one or more `TruncatedNormalGMMConditional`
coordinates (e.g. "phi2", "pm1", "pm2", "radial_velocity") that each share
phi1's *exact* Bayes-rule responsibilities via a `FromDist("phi1")` marker in
`coord_parameters` -- this is the exact marginal/conditional factorization of
the old joint grid, generalized so each conditional coordinate can have its
own free per-node values instead of being forced to match a fixed grid.

Also covers the later generalization to a D>1 `conditioning_dist` (e.g. a
genuine joint ("phi1","phi2") 2D grid, restored so the offtrack component can
represent multi-modal phi2 structure at a given phi1 while still giving each
2D grid node its own fully independent pm1/pm2/radial_velocity, via
`FromDist(("phi1","phi2"))`): `TruncatedNormalGMMConditional` must handle both
a full joint `x` (shape (..., D), e.g. real per-star (phi1,phi2) pairs) and a
single-axis `x` (shape (...,), e.g. a 1D phi1 grid during plotting) against a
D>1 `conditioning_dist`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from stream_membership import ModelComponent
from stream_membership.distributions import IndependentGMM, TruncatedNormalGMMConditional
from stream_membership.model import ComponentMixtureModel, FromDist

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `field\\(init=False\\)`:UserWarning"
)


def test_conditional_matches_exact_joint_factorization():
    """`TruncatedNormalGMMConditional` built from a phi1-axis marginal (an
    `IndependentGMM` with D=1) and phi2-axis loc/scale values must
    reproduce a full joint (phi1, phi2) `IndependentGMM`'s exact
    `log_prob(phi1, phi2)` to numerical precision, since this is the exact
    marginal * conditional factorization of an `IndependentGMM` with no
    cross-axis correlation."""
    K = 6
    phi1_lim = (-40.0, 10.0)
    phi2_lim = (-5.0, 5.0)

    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    phi1_locs = jax.random.uniform(keys[0], (K,), minval=phi1_lim[0], maxval=phi1_lim[1])
    phi2_locs = jax.random.uniform(keys[1], (K,), minval=phi2_lim[0], maxval=phi2_lim[1])
    phi1_scales = jax.random.uniform(keys[2], (K,), minval=0.3, maxval=1.5)
    phi2_scales = jax.random.uniform(keys[3], (K,), minval=0.3, maxval=1.5)
    logits = jax.random.normal(jax.random.PRNGKey(1), (K,))
    mix = dist.CategoricalLogits(logits=logits)

    joint = IndependentGMM(
        mix,
        locs=jnp.stack([phi1_locs, phi2_locs], axis=0),
        scales=jnp.stack([phi1_scales, phi2_scales], axis=0),
        low=jnp.array([phi1_lim[0], phi2_lim[0]])[:, None],
        high=jnp.array([phi1_lim[1], phi2_lim[1]])[:, None],
    )

    # phi1 marginal: same weights/loc/scale as the joint's phi1 axis:
    phi1_marginal = IndependentGMM(
        mix,
        locs=phi1_locs[None, :],
        scales=phi1_scales[None, :],
        low=jnp.array([[phi1_lim[0]]]),
        high=jnp.array([[phi1_lim[1]]]),
    )
    conditional = TruncatedNormalGMMConditional(
        conditioning_dist=phi1_marginal,
        loc_vals=phi2_locs,
        scale_vals=phi2_scales,
        x=jnp.linspace(*phi1_lim, 200),
        low=phi2_lim[0],
        high=phi2_lim[1],
    )

    rng = np.random.default_rng(0)
    phi1_vals = jnp.asarray(rng.uniform(*phi1_lim, 200))
    phi2_vals = jnp.asarray(rng.uniform(*phi2_lim, 200))

    joint_lp = joint.log_prob(jnp.stack([phi1_vals, phi2_vals], axis=-1))
    marginal_lp = phi1_marginal.log_prob(phi1_vals[:, None])
    conditional_lp = conditional.log_prob(phi2_vals, x=phi1_vals)

    np.testing.assert_allclose(
        np.asarray(marginal_lp + conditional_lp), np.asarray(joint_lp), atol=1e-4
    )


def test_conditional_full_joint_matches_3d_factorization():
    """Same exact-factorization check as
    `test_conditional_matches_exact_joint_factorization`, but with a D=2
    `conditioning_dist` (a joint (phi1,phi2) grid, as used by the offtrack
    component's 2D spatial term) and a full joint `x` of shape (N, 2) --
    this is the real-data-fitting code path for coordinates like "pm1" that
    are conditioned on `FromDist(("phi1","phi2"))`."""
    K = 6
    phi1_lim = (-40.0, 10.0)
    phi2_lim = (-5.0, 5.0)
    pm1_lim = (-15.0, 5.0)

    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    phi1_locs = jax.random.uniform(keys[0], (K,), minval=phi1_lim[0], maxval=phi1_lim[1])
    phi2_locs = jax.random.uniform(keys[1], (K,), minval=phi2_lim[0], maxval=phi2_lim[1])
    pm1_locs = jax.random.uniform(keys[2], (K,), minval=pm1_lim[0], maxval=pm1_lim[1])
    phi1_scales = jax.random.uniform(keys[3], (K,), minval=0.3, maxval=1.5)
    phi2_scales = jax.random.uniform(keys[4], (K,), minval=0.3, maxval=1.5)
    pm1_scales = jax.random.uniform(keys[5], (K,), minval=0.3, maxval=1.5)
    logits = jax.random.normal(jax.random.PRNGKey(1), (K,))
    mix = dist.CategoricalLogits(logits=logits)

    joint3d = IndependentGMM(
        mix,
        locs=jnp.stack([phi1_locs, phi2_locs, pm1_locs], axis=0),
        scales=jnp.stack([phi1_scales, phi2_scales, pm1_scales], axis=0),
        low=jnp.array([phi1_lim[0], phi2_lim[0], pm1_lim[0]])[:, None],
        high=jnp.array([phi1_lim[1], phi2_lim[1], pm1_lim[1]])[:, None],
    )

    # (phi1, phi2) marginal: same weights/loc/scale as the joint's first two axes:
    phi12_marginal = IndependentGMM(
        mix,
        locs=jnp.stack([phi1_locs, phi2_locs], axis=0),
        scales=jnp.stack([phi1_scales, phi2_scales], axis=0),
        low=jnp.array([phi1_lim[0], phi2_lim[0]])[:, None],
        high=jnp.array([phi1_lim[1], phi2_lim[1]])[:, None],
    )
    conditional = TruncatedNormalGMMConditional(
        conditioning_dist=phi12_marginal,
        loc_vals=pm1_locs,
        scale_vals=pm1_scales,
        x=jnp.zeros((200, 2)),  # placeholder, overridden via log_prob's `x=`
        low=pm1_lim[0],
        high=pm1_lim[1],
    )

    rng = np.random.default_rng(0)
    phi1_vals = jnp.asarray(rng.uniform(*phi1_lim, 200))
    phi2_vals = jnp.asarray(rng.uniform(*phi2_lim, 200))
    pm1_vals = jnp.asarray(rng.uniform(*pm1_lim, 200))
    x_joint = jnp.stack([phi1_vals, phi2_vals], axis=-1)

    joint_lp = joint3d.log_prob(jnp.stack([phi1_vals, phi2_vals, pm1_vals], axis=-1))
    marginal_lp = phi12_marginal.log_prob(x_joint)
    conditional_lp = conditional.log_prob(pm1_vals, x=x_joint)

    np.testing.assert_allclose(
        np.asarray(marginal_lp + conditional_lp), np.asarray(joint_lp), atol=1e-4
    )


def test_conditional_marginal_fallback_matches_single_axis_conditioning_dist():
    """When `conditioning_dist` is D>1 (e.g. a joint (phi1,phi2) grid) but
    `x` only supplies a single axis' worth of values (shape (...,), as
    produced by grid-based plotting's `conditional_data[coord] = {"x":
    "phi1"}` resolution), `_log_responsibilities` must fall back to the
    exact single-axis marginal -- i.e. it must match a `TruncatedNormalGMMConditional`
    built directly from a D=1 `conditioning_dist` sharing the same weights
    and phi1-axis loc/scale/bounds."""
    K = 5
    phi1_lim = (-40.0, 10.0)
    phi2_lim = (-5.0, 5.0)
    pm1_lim = (-15.0, 5.0)

    keys = jax.random.split(jax.random.PRNGKey(2), 5)
    phi1_locs = jax.random.uniform(keys[0], (K,), minval=phi1_lim[0], maxval=phi1_lim[1])
    phi2_locs = jax.random.uniform(keys[1], (K,), minval=phi2_lim[0], maxval=phi2_lim[1])
    phi1_scales = jax.random.uniform(keys[2], (K,), minval=0.3, maxval=1.5)
    phi2_scales = jax.random.uniform(keys[3], (K,), minval=0.3, maxval=1.5)
    logits = jax.random.normal(jax.random.PRNGKey(3), (K,))
    mix = dist.CategoricalLogits(logits=logits)

    phi12_marginal = IndependentGMM(
        mix,
        locs=jnp.stack([phi1_locs, phi2_locs], axis=0),
        scales=jnp.stack([phi1_scales, phi2_scales], axis=0),
        low=jnp.array([phi1_lim[0], phi2_lim[0]])[:, None],
        high=jnp.array([phi1_lim[1], phi2_lim[1]])[:, None],
    )
    phi1_only_marginal = IndependentGMM(
        mix,
        locs=phi1_locs[None, :],
        scales=phi1_scales[None, :],
        low=jnp.array([[phi1_lim[0]]]),
        high=jnp.array([[phi1_lim[1]]]),
    )

    loc_vals = jax.random.uniform(keys[4], (K,), minval=pm1_lim[0], maxval=pm1_lim[1])
    scale_vals = jnp.full(K, 0.7)

    rng = np.random.default_rng(1)
    phi1_vals = jnp.asarray(rng.uniform(*phi1_lim, 150))
    pm1_vals = jnp.asarray(rng.uniform(*pm1_lim, 150))

    fallback = TruncatedNormalGMMConditional(
        conditioning_dist=phi12_marginal,
        loc_vals=loc_vals,
        scale_vals=scale_vals,
        x=phi1_vals,
        low=pm1_lim[0],
        high=pm1_lim[1],
    )
    direct = TruncatedNormalGMMConditional(
        conditioning_dist=phi1_only_marginal,
        loc_vals=loc_vals,
        scale_vals=scale_vals,
        x=phi1_vals,
        low=pm1_lim[0],
        high=pm1_lim[1],
    )

    np.testing.assert_allclose(
        np.asarray(fallback._log_responsibilities()),
        np.asarray(direct._log_responsibilities()),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fallback.log_prob(pm1_vals)),
        np.asarray(direct.log_prob(pm1_vals)),
        atol=1e-6,
    )


def test_conditional_gradients_finite_at_out_of_bounds():
    """Matches the NaN-gradient-safety pattern used by `IndependentGMM`/
    `TruncatedNormalSpline`: evaluating at an out-of-bounds `value` or `x`
    must give a finite (sentinel-masked) log_prob and a finite gradient,
    never a NaN gradient from clipping."""
    K = 4
    phi1_lim = (-10.0, 10.0)
    phi1_marginal = IndependentGMM(
        dist.CategoricalProbs(jnp.ones(K) / K),
        locs=jnp.linspace(*phi1_lim, K).reshape(1, -1),
        scales=jnp.full((1, K), 2.0),
        low=jnp.array([[phi1_lim[0]]]),
        high=jnp.array([[phi1_lim[1]]]),
    )

    def loss(loc_vals):
        conditional = TruncatedNormalGMMConditional(
            conditioning_dist=phi1_marginal,
            loc_vals=loc_vals,
            scale_vals=jnp.full(K, 0.5),
            x=jnp.array([-100.0, 0.0, 100.0]),  # includes out-of-bounds x
            low=-5.0,
            high=5.0,
        )
        value = jnp.array([-50.0, 0.0, 50.0])  # includes out-of-bounds value
        return jnp.sum(conditional.log_prob(value))

    loc_vals0 = jnp.linspace(-3.0, 3.0, K)
    val, grad = jax.value_and_grad(loss)(loc_vals0)
    assert jnp.isfinite(val)
    assert jnp.all(jnp.isfinite(grad))


def test_from_dist_resolves_sibling_and_requires_earlier_order():
    """`FromDist("phi1")` must reuse phi1's already-built distribution
    object when phi1 is built first, and must raise a clear `KeyError`
    (not e.g. a silent wrong value) when the referenced coordinate hasn't
    been built yet."""
    K = 4
    phi1_lim = (-10.0, 10.0)
    phi2_lim = (-3.0, 3.0)
    x = jnp.linspace(*phi1_lim, 20)

    def _make(coord_order):
        coord_distributions = {
            "phi1": IndependentGMM,
            "phi2": TruncatedNormalGMMConditional,
        }
        coord_parameters = {
            "phi1": {
                "mixing_distribution": (
                    dist.Categorical,
                    dist.Dirichlet(jnp.ones(K)),
                ),
                "locs": jnp.linspace(*phi1_lim, K).reshape(1, -1),
                "scales": dist.HalfNormal(2.0).expand([K]),
                "low": jnp.array([phi1_lim[0]])[:, None],
                "high": jnp.array([phi1_lim[1]])[:, None],
            },
            "phi2": {
                "conditioning_dist": FromDist("phi1"),
                "loc_vals": dist.Uniform(*phi2_lim).expand([K]),
                "scale_vals": dist.HalfNormal(1.0).expand([K]),
                "x": x,
                "low": phi2_lim[0],
                "high": phi2_lim[1],
            },
        }
        return ModelComponent(
            name="test",
            coord_distributions={k: coord_distributions[k] for k in coord_order},
            coord_parameters={k: coord_parameters[k] for k in coord_order},
            conditional_data={"phi2": {"x": "phi1"}},
        )

    # phi1 first (correct order) -> builds fine and samples successfully:
    model = _make(["phi1", "phi2"])
    samples = model.sample(jax.random.PRNGKey(0))
    assert "phi1" in samples and "phi2" in samples

    # phi2 first (misordered) -> FromDist("phi1") isn't built yet -> KeyError:
    bad_model = _make(["phi2", "phi1"])
    with pytest.raises(KeyError, match="FromDist"):
        bad_model.sample(jax.random.PRNGKey(0))


def test_offtrack_style_heterogeneous_mixture_runs_end_to_end():
    """Regression test for the original `KeyError: 'phi1'` bug: a
    `ComponentMixtureModel` combining a "bkg"/"stream"-like component (plain
    string-keyed "phi1"/"phi2"/"pm1" coordinates) with an "offtrack"-like
    component (a factorized `IndependentGMM` "phi1" marginal plus
    `TruncatedNormalGMMConditional` "phi2"/"pm1" coordinates sharing phi1's
    responsibilities via `FromDist`) must build and run SVI end to end --
    with no `KeyError` from mismatched coordinate structures, and no
    `ValueError: All component distributions must have the same support`
    from `dist.MixtureGeneral` combining differently-typed per-coordinate
    distributions."""
    numpyro.enable_x64()
    phi1_lim = (-40.0, 10.0)
    phi2_lim = (-5.0, 5.0)
    pm1_lim = (-15.0, 5.0)
    K = 5

    def _make_simple(name, phi1_offset):
        """A 'bkg'/'stream'-style component. Uses `IndependentGMM` for
        "phi1" and bounded `TruncatedNormal`s for "phi2"/"pm1" -- not
        `dist.Normal` -- specifically so each coordinate's support *type*
        matches offtrack's corresponding coordinate (`IndependentGMM`'s
        custom bounded support for "phi1", `constraints.interval(...)` for
        "phi2"/"pm1"), mirroring the real bkg/stream components closely
        enough to exercise `dist.MixtureGeneral`'s "all component
        distributions must have the same support" check for real."""
        K_simple = 4
        phi1_knots = jnp.linspace(*phi1_lim, K_simple)
        return ModelComponent(
            name=name,
            coord_distributions={
                "phi1": IndependentGMM,
                "phi2": dist.TruncatedNormal,
                "pm1": dist.TruncatedNormal,
            },
            coord_parameters={
                "phi1": {
                    "mixing_distribution": (
                        dist.Categorical,
                        dist.Dirichlet(jnp.ones(K_simple)),
                    ),
                    "locs": (phi1_knots + phi1_offset).reshape(1, -1),
                    "scales": dist.HalfNormal(5.0).expand([K_simple]),
                    "low": jnp.array([phi1_lim[0]])[:, None],
                    "high": jnp.array([phi1_lim[1]])[:, None],
                },
                "phi2": {"loc": 0.0, "scale": 1.0, "low": phi2_lim[0], "high": phi2_lim[1]},
                "pm1": {"loc": -5.0, "scale": 1.0, "low": pm1_lim[0], "high": pm1_lim[1]},
            },
        )

    def _make_offtrack(x):
        # `x` mirrors the real `make_offtrack_model_component`'s pattern of
        # baking each conditional coordinate's own phi1-conditioning array
        # into `coord_parameters["x"]` directly at construction time (e.g.
        # via `stream_model.coord_parameters["pm1"]["x"]`), rather than
        # relying on `conditional_data` to substitute it dynamically from
        # `ComponentMixtureModel.__call__`'s `data` argument -- so it must
        # already be the correctly-sized per-star array, not a placeholder.
        phi1_nodes = jnp.linspace(*phi1_lim, K)
        return ModelComponent(
            name="offtrack",
            coord_distributions={
                "phi1": IndependentGMM,
                "phi2": TruncatedNormalGMMConditional,
                "pm1": TruncatedNormalGMMConditional,
            },
            coord_parameters={
                "phi1": {
                    "mixing_distribution": (
                        dist.Categorical,
                        dist.Dirichlet(jnp.ones(K)),
                    ),
                    "locs": phi1_nodes.reshape(1, -1),
                    "scales": dist.HalfNormal(5.0).expand([K]),
                    "low": jnp.array([phi1_lim[0]])[:, None],
                    "high": jnp.array([phi1_lim[1]])[:, None],
                },
                "phi2": {
                    "conditioning_dist": FromDist("phi1"),
                    "loc_vals": dist.Uniform(*phi2_lim).expand([K]),
                    "scale_vals": dist.HalfNormal(1.0).expand([K]),
                    "x": x,
                    "low": phi2_lim[0],
                    "high": phi2_lim[1],
                },
                "pm1": {
                    "conditioning_dist": FromDist("phi1"),
                    "loc_vals": dist.Uniform(*pm1_lim).expand([K]),
                    "scale_vals": dist.HalfNormal(1.0).expand([K]),
                    "x": x,
                    "low": pm1_lim[0],
                    "high": pm1_lim[1],
                },
            },
            conditional_data={"phi2": {"x": "phi1"}, "pm1": {"x": "phi1"}},
        )

    rng = np.random.default_rng(0)
    N = 60
    data = {
        "phi1": jnp.asarray(rng.uniform(*phi1_lim, N)),
        "phi2": jnp.asarray(rng.normal(0, 1, N)),
        "pm1": jnp.asarray(rng.normal(-5, 1, N)),
    }

    bkg = _make_simple("bkg", phi1_offset=0.0)
    stream = _make_simple("stream", phi1_offset=-15.0)
    offtrack = _make_offtrack(x=data["phi1"])

    mm = ComponentMixtureModel(
        dist.Dirichlet(jnp.array([1.0, 1.0, 1.0])),
        components=[bkg, stream, offtrack],
    )

    guide = AutoNormal(mm)
    svi = SVI(mm, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    svi_result = svi.run(jax.random.PRNGKey(3), 15, data=data, progress_bar=False)

    assert jnp.isfinite(svi_result.losses[-1])


def test_offtrack_style_2d_grid_with_independent_kinematics_runs_end_to_end():
    """Regression test for the "option 3" offtrack redesign: a genuine 2D
    joint ("phi1","phi2") `IndependentGMM` grid (so multi-modal phi2
    structure at a given phi1 is representable again), combined with
    "pm1"/"pm2" coordinates that are each their own free, per-2D-node
    `TruncatedNormalGMMConditional` sharing the grid's responsibilities via
    `FromDist(("phi1","phi2"))` -- i.e. every 2D node gets its own
    independent kinematics, fixing the *original* pre-redesign bug (nodes at
    the same phi1 but different phi2 forced to share identical pm1/pm2)
    without reintroducing the 1D-chain redesign's loss of phi2 multimodality.

    Must build and run SVI end to end, combined in a `ComponentMixtureModel`
    with "bkg"/"stream"-like siblings that use separate string-keyed "phi1"
    and "phi2" coordinates instead of a joint tuple key (mirroring
    `ComponentMixtureModel.__post_init__`'s strict `coord_names` equality
    check across components, which requires the joint key to flatten to the
    same ordered coordinate list)."""
    numpyro.enable_x64()
    phi1_lim = (-40.0, 10.0)
    phi2_lim = (-5.0, 5.0)
    pm1_lim = (-15.0, 5.0)
    pm2_lim = (-10.0, 10.0)

    def _make_simple(name, phi1_offset):
        K_simple = 4
        phi1_knots = jnp.linspace(*phi1_lim, K_simple)
        return ModelComponent(
            name=name,
            coord_distributions={
                "phi1": IndependentGMM,
                "phi2": dist.TruncatedNormal,
                "pm1": dist.TruncatedNormal,
                "pm2": dist.TruncatedNormal,
            },
            coord_parameters={
                "phi1": {
                    "mixing_distribution": (
                        dist.Categorical,
                        dist.Dirichlet(jnp.ones(K_simple)),
                    ),
                    "locs": (phi1_knots + phi1_offset).reshape(1, -1),
                    "scales": dist.HalfNormal(5.0).expand([K_simple]),
                    "low": jnp.array([phi1_lim[0]])[:, None],
                    "high": jnp.array([phi1_lim[1]])[:, None],
                },
                "phi2": {"loc": 0.0, "scale": 1.0, "low": phi2_lim[0], "high": phi2_lim[1]},
                "pm1": {"loc": -5.0, "scale": 1.0, "low": pm1_lim[0], "high": pm1_lim[1]},
                "pm2": {"loc": 0.0, "scale": 1.0, "low": pm2_lim[0], "high": pm2_lim[1]},
            },
        )

    def _make_offtrack(x_joint):
        # A true 2D grid: K1 x K2 nodes over (phi1, phi2).
        K1, K2 = 4, 3
        K = K1 * K2
        phi1_nodes = jnp.linspace(*phi1_lim, K1)
        phi2_nodes = jnp.linspace(*phi2_lim, K2)
        phi1_grid, phi2_grid = jnp.meshgrid(phi1_nodes, phi2_nodes, indexing="ij")
        locs = jnp.stack([phi1_grid.ravel(), phi2_grid.ravel()], axis=0)  # (2, K)

        return ModelComponent(
            name="offtrack",
            coord_distributions={
                ("phi1", "phi2"): IndependentGMM,
                "pm1": TruncatedNormalGMMConditional,
                "pm2": TruncatedNormalGMMConditional,
            },
            coord_parameters={
                ("phi1", "phi2"): {
                    "mixing_distribution": (
                        dist.Categorical,
                        dist.Dirichlet(jnp.ones(K)),
                    ),
                    "locs": locs,
                    "scales": dist.HalfNormal(5.0).expand([2, K]),
                    "low": jnp.array([phi1_lim[0], phi2_lim[0]])[:, None],
                    "high": jnp.array([phi1_lim[1], phi2_lim[1]])[:, None],
                },
                "pm1": {
                    "conditioning_dist": FromDist(("phi1", "phi2")),
                    "loc_vals": dist.Uniform(*pm1_lim).expand([K]),
                    "scale_vals": dist.HalfNormal(1.0).expand([K]),
                    "x": x_joint,
                    "low": pm1_lim[0],
                    "high": pm1_lim[1],
                },
                "pm2": {
                    "conditioning_dist": FromDist(("phi1", "phi2")),
                    "loc_vals": dist.Uniform(*pm2_lim).expand([K]),
                    "scale_vals": dist.HalfNormal(1.0).expand([K]),
                    "x": x_joint,
                    "low": pm2_lim[0],
                    "high": pm2_lim[1],
                },
            },
            # Grid-plotting fallback: still resolved as a single "phi1" string,
            # even though `conditioning_dist` is D=2 -- exercises the
            # single-axis marginal-fallback path in `_log_responsibilities`.
            conditional_data={"pm1": {"x": "phi1"}, "pm2": {"x": "phi1"}},
        )

    rng = np.random.default_rng(4)
    N = 60
    phi1_data = jnp.asarray(rng.uniform(*phi1_lim, N))
    phi2_data = jnp.asarray(rng.normal(0, 1, N))
    data = {
        "phi1": phi1_data,
        "phi2": phi2_data,
        "pm1": jnp.asarray(rng.normal(-5, 1, N)),
        "pm2": jnp.asarray(rng.normal(0, 1, N)),
    }

    bkg = _make_simple("bkg", phi1_offset=0.0)
    stream = _make_simple("stream", phi1_offset=-15.0)
    offtrack = _make_offtrack(x_joint=jnp.stack([phi1_data, phi2_data], axis=-1))

    # Both siblings flatten to ["phi1", "phi2", "pm1", "pm2"], matching the
    # joint ("phi1","phi2") offtrack coordinate's flattened names -- required
    # by `ComponentMixtureModel.__post_init__`'s strict equality check.
    assert tuple(bkg.coord_names) == tuple(offtrack.coord_names)

    mm = ComponentMixtureModel(
        dist.Dirichlet(jnp.array([1.0, 1.0, 1.0])),
        components=[bkg, stream, offtrack],
    )

    guide = AutoNormal(mm)
    svi = SVI(mm, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    svi_result = svi.run(jax.random.PRNGKey(5), 15, data=data, progress_bar=False)

    assert jnp.isfinite(svi_result.losses[-1])
