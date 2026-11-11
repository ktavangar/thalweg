__all__ = ["Normal1DSplineMixture", "TruncatedNormal1DSplineMixture"]

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from stream_membership.distributions import NormalSpline, TruncatedNormalSpline
from stream_membership.distributions.normal_spline import _clip_preserve_gradients



class Normal1DSplineMixture(dist.MixtureGeneral):
    def __init__(
        self,
        mixing_distribution: dist.CategoricalProbs | dist.CategoricalLogits | None = None,
        loc_vals: ArrayLike = None,
        scale_vals: ArrayLike = None,
        knots: ArrayLike = None,
        x: ArrayLike = None,
        mixing_vals: ArrayLike | None = None,
        mixing_knots: ArrayLike | None = None,
        spline_k: int | dict[str, int] = 3,
        clip_locs: tuple[float | None, float | None] = (None, None),
        clip_scales: tuple[float | None, float | None] = (None, None),
        clip_mixing_logits: tuple[float | None, float | None] = (None, None),
        ordered_scales: bool = True,
        validate_args=None,
    ) -> None:
        """
        Represents a mixture of Normal distributions where the parameters are controlled
        by splines that are evaluated at some other parameter values x.

        Parameters
        ----------

        x
            Array of x values at which to evaluate the splines.
        mixing_distribution (optional)
            A fixed (not x-dependent) mixing distribution over components, e.g.
            ``dist.Categorical(probs=...)``. Exactly one of ``mixing_distribution`` or
            ``mixing_vals`` must be provided.
        mixing_vals (optional)
            Array of shape ``(n_components, n_mixing_knots)`` giving per-knot mixing
            logits for each component. If provided, the mixing weights are
            deterministically spline-interpolated (and softmaxed) at ``x``, exactly the
            way ``loc_vals``/``scale_vals`` are turned into per-x ``loc``/``scale``
            values -- i.e. this makes the mixing weights vary smoothly with ``x`` (e.g.
            phi1) instead of being fixed for the whole dataset. Exactly one of
            ``mixing_distribution`` or ``mixing_vals`` must be provided.
        mixing_knots (optional)
            Array of spline knot locations for the mixing-weight spline. Defaults to
            ``knots`` (the same knots used for loc/scale) if not given.
        spline_k
            Degree of the spline. Can be a single integer, applied to loc, scale, and
            mixing splines, or a dict with keys "loc", "scale", "mixing".
        clip_mixing_logits (optional)
            Bounds to clip the interpolated mixing logits into, using the same
            straight-through-gradient trick as ``clip_locs``/``clip_scales``. Only used
            when ``mixing_vals`` is provided.
        """
        # Should have shape (n_knots, )
        self.knots = jnp.array(knots)
        self._n_knots = len(self.knots)
        self.clip_locs = tuple(clip_locs)
        self.clip_scales = tuple(clip_scales)
        self.clip_mixing_logits = tuple(clip_mixing_logits)

        # The pre-specified grid to evaluate on
        self.x = jnp.array(x)

        # Spline order:
        if not isinstance(spline_k, dict):
            spline_k = {"loc": spline_k, "scale": spline_k, "mixing": spline_k}
        spline_k.setdefault("mixing", spline_k.get("loc", 3))
        self.spline_k = spline_k

        # Should have shape (n_components, n_knots)
        combined_shape = jax.lax.broadcast_shapes(
            jnp.shape(loc_vals), jnp.shape(scale_vals)
        )
        if validate_args and (
            len(combined_shape) != 2 or combined_shape[-1] != self._n_knots
        ):
            msg = (
                "locs, scales, and concentrations must have 2 axes, but got "
                f"{len(combined_shape)}. The shape must be broadcastable to: "
                "(n_components, n_knots), where n_components is the number of mixture "
                "components and n_knots is the number of spline knots"
            )
            raise ValueError(msg)
        self._n_components = combined_shape[0]

        self.loc_vals = jnp.array(loc_vals)
        self.scale_vals = jnp.array(scale_vals)

        # Broadcasted arrays:
        self._loc_vals = jnp.broadcast_to(
            self.loc_vals, (self._n_components, self._n_knots)
        )
        self._scale_vals = jnp.broadcast_to(
            self.scale_vals, (self._n_components, self._n_knots)
        )
        if ordered_scales:
            # If specified, treat the scales as cumulatively summed variances
            self._scale_vals = jnp.stack(
                [
                    jnp.sqrt(jnp.sum(self._scale_vals[:i]**2, axis=0)) for i in range(1, self._scale_vals.shape[0] + 1)
                ],
                # [
                #     0.5
                #     * jax.scipy.special.logsumexp(2 * self._ln_scale_vals[:i], axis=0)
                #     for i in range(1, self._scale_vals.shape[0] + 1)
                # ],
                axis=0,
            )

        # If mixing_vals is provided, the mixing weights are a deterministic
        # spline function of x (per-component logits interpolated from
        # mixing_vals at the knots, then softmaxed) instead of a single fixed
        # mixing_distribution shared across all x. Exactly one of
        # mixing_distribution / mixing_vals must be provided.
        if (mixing_distribution is None) == (mixing_vals is None):
            msg = (
                "Exactly one of `mixing_distribution` or `mixing_vals` must be "
                "provided to Normal1DSplineMixture/TruncatedNormal1DSplineMixture."
            )
            raise ValueError(msg)

        self.mixing_vals = None
        self.mixing_knots = None
        self._mixing_spls = None
        if mixing_vals is not None:
            self.mixing_vals = jnp.array(mixing_vals)
            self.mixing_knots = (
                jnp.array(mixing_knots) if mixing_knots is not None else self.knots
            )
            if validate_args and self.mixing_vals.shape[0] != self._n_components:
                msg = (
                    "mixing_vals must have shape (n_components, n_mixing_knots), "
                    f"but got shape {self.mixing_vals.shape} for "
                    f"{self._n_components} components."
                )
                raise ValueError(msg)
            self._mixing_spls = [
                InterpolatedUnivariateSpline(
                    self.mixing_knots,
                    self.mixing_vals[i],
                    k=self.spline_k["mixing"],
                )
                for i in range(self._n_components)
            ]
            init_mixing_distribution = self._make_mixing_distribution(self.x)
        else:
            init_mixing_distribution = mixing_distribution

        super().__init__(
            init_mixing_distribution,
            self._make_components(),
            validate_args=validate_args,
        )

        if self.mixing_vals is not None:
            expected = (self.x.size, self._n_components)
            if validate_args and self.mixing_distribution.probs.shape != expected:
                msg = (
                    "The shape of the mixing distribution probabilities must be "
                    "broadcastable to (len(x), n_components)"
                )
                raise ValueError(msg)

    def _make_mixing_distribution(
        self, x: ArrayLike | None = None
    ) -> dist.CategoricalProbs | dist.CategoricalLogits:
        """
        Returns the mixing distribution to use for a given x. If mixing_vals was not
        provided at construction, this just returns the fixed mixing_distribution
        (independent of x). Otherwise, the per-component mixing splines are evaluated
        at x and turned into a (batched) CategoricalLogits distribution -- this is what
        makes the mixing weights vary with x (e.g. phi1).
        """
        x = self.x if x is None else x
        if self.mixing_vals is None:
            return self.mixing_distribution

        logits = jnp.stack([spl(x) for spl in self._mixing_spls], axis=-1)
        logits = _clip_preserve_gradients(logits, *self.clip_mixing_logits)
        return dist.CategoricalLogits(logits=logits, validate_args=False)

    def _make_components(self, x: ArrayLike | None = None) -> list[NormalSpline]:
        x = self.x if x is None else x
        return [
            NormalSpline(self._loc_vals[i], self._scale_vals[i], self.knots, x,
                         spline_k=self.spline_k, clip_locs=self.clip_locs, clip_scales=self.clip_scales)
            for i in range(self._n_components)
        ]

    def component_sample(
        self,
        key: jax.Array,
        sample_shape: tuple = (),
        x: ArrayLike | None = None,
    ) -> jax.Array:
        components = self._make_components(x)
        keys = jax.random.split(key, self._n_components)
        samples = [
            d.sample(keys[i], sample_shape, x=x) for i, d in enumerate(components)
        ]
        return jnp.stack(samples, axis=-1)

    def component_log_probs(
        self,
        value: ArrayLike,
        x: ArrayLike | None = None,
        err: ArrayLike | None = None,
    ) -> jax.Array:
        value = jnp.array(value)
        mixing_distribution = self._make_mixing_distribution(x)
        if err is not None:
            # Analytic error convolution (replaces the per-star :modeldata
            # latents): each spline component is convolved with N(0, err)
            # via its own `log_prob_convolved`, then the log mixing weights
            # are added, matching what MixtureGeneral.component_log_probs
            # returns in the err=None branch below.
            lps = jnp.stack(
                [c.log_prob_convolved(value, err, x=x) for c in self._make_components(x)],
                axis=-1,
            )
            return jax.nn.log_softmax(mixing_distribution.logits, axis=-1) + lps
        try:
            helper = dist.MixtureSameFamily(
            mixing_distribution, self._make_components(x), validate_args=False
        )
        except:
            helper = dist.MixtureGeneral(
                mixing_distribution, self._make_components(x), validate_args=False
            )
        return helper.component_log_probs(value)

    def log_prob(self,
                 value: ArrayLike,
                 x: ArrayLike | None = None,
            ) -> jax.Array | Any:
        log_prob = jax.scipy.special.logsumexp(self.component_log_probs(value=value, x=x), axis=-1)
        return log_prob

    def log_prob_convolved(
        self, value: ArrayLike, err: ArrayLike, x: ArrayLike | None = None
    ) -> jax.Array:
        """Log density convolved with per-value Gaussian errors ``err``; the
        fitted component scales stay intrinsic widths. See
        ``stream_membership.distributions.convolved``."""
        return jax.scipy.special.logsumexp(
            self.component_log_probs(value=value, x=x, err=err), axis=-1
        )


class TruncatedNormal1DSplineMixture(Normal1DSplineMixture):
    def __init__(
        self,
        mixing_distribution: dist.CategoricalProbs | dist.CategoricalLogits | None = None,
        loc_vals: ArrayLike = None,
        scale_vals: ArrayLike = None,
        knots: ArrayLike = None,
        x: ArrayLike = None,
        mixing_vals: ArrayLike | None = None,
        mixing_knots: ArrayLike | None = None,
        low: Any | None = None,
        high: Any | None = None,
        spline_k: int = 3,
        clip_locs: tuple[float | None, float | None] = (None, None),
        clip_scales: tuple[float | None, float | None] = (None, None),
        clip_mixing_logits: tuple[float | None, float | None] = (None, None),
        ordered_scales: bool = True,
        validate_args=None,
    ) -> None:
        """
        Represents a mixture of Normal distributions where the parameters are controlled
        by splines that are evaluated at some other parameter values x.

        Parameters
        ----------

        x
            Array of x values at which to evaluate the splines.
        mixing_vals (optional)
            See ``Normal1DSplineMixture`` -- if provided (instead of
            ``mixing_distribution``), the mixing weights are a smooth spline function of
            x rather than fixed.
        spline_k
            Degree of the spline.
        ordered_scales
            See ``Normal1DSplineMixture`` -- if True (the default), scale_vals is
            treated as quadrature increments so the reconstructed scale is forced
            non-decreasing in component index at every knot. Set to False to use
            scale_vals directly as the actual per-component scale, with components
            identified by loc instead.
        """
        self.low = low
        self.high = high
        if validate_args and (self.low is None and self.high is None):
            msg = "Use Normal1DSplineMixture if no truncation is needed"
            raise ValueError(msg)

        super().__init__(
            mixing_distribution=mixing_distribution,
            loc_vals=loc_vals,
            scale_vals=scale_vals,
            knots=knots,
            x=x,
            mixing_vals=mixing_vals,
            mixing_knots=mixing_knots,
            spline_k=spline_k,
            clip_locs=clip_locs,
            clip_scales=clip_scales,
            clip_mixing_logits=clip_mixing_logits,
            ordered_scales=ordered_scales,
            validate_args=validate_args,
        )

    @property
    def support(self):
        if self.low is None and self.high is None:
            return dist.constraints.real
        elif self.low is None:
            return dist.constraints.less_than(self.high)
        elif self.high is None:
            return dist.constraints.greater_than(self.low)
        else:
            return dist.constraints.interval(self.low, self.high)

    def _make_components(self, x: ArrayLike | None = None) -> list[NormalSpline]:
        x = self.x if x is None else x
        return [
            TruncatedNormalSpline(
                self._loc_vals[i],
                self._scale_vals[i],
                self.knots,
                x,
                low=self.low,
                high=self.high,
                spline_k=self.spline_k,
                clip_locs=self.clip_locs,
                clip_scales=self.clip_scales,
            )
            for i in range(self._n_components)
        ]

    def component_log_probs(
        self, value: ArrayLike, x: ArrayLike | None = None, err: ArrayLike | None = None
    ) -> jax.Array:
        x = x if x is not None else self.x
        value = jnp.asarray(value)

        if err is not None:
            # Error-convolved path: the convolved density is non-zero outside
            # [low, high] (an error can scatter a star across a bound), so the
            # support mask below must NOT be applied here.
            return super().component_log_probs(value, x, err=err)

        # NOTE: we tried a "clip value into [low, high] before computing
        # component_log_probs" fix here, on the theory that a `value` far
        # outside [low, high] could produce a NaN gradient that then leaks
        # through the `jnp.where` mask below. Direct testing (both of
        # numpyro's raw `TruncatedNormal.log_prob` and of this class) showed
        # that an out-of-bounds `value` alone -- with `loc` inside
        # [low, high] -- never actually produces a NaN gradient; numpyro
        # already handles that case safely. So that fix was reverted as
        # unnecessary/not the real root cause. The actual bug (loc drifting
        # outside [low, high]) is fixed at its source in
        # `normal_spline.py`'s `TruncatedNormalSpline._make_helper_dist`,
        # see the comment there for the verified mechanism.
        component_log_probs = super().component_log_probs(value, x)

        value = jnp.expand_dims(value, axis=-1)
        return jnp.where(
            self.support.check(value),
            component_log_probs,
            -jnp.inf,
        )
