__all__ = ["NormalSpline", "TruncatedNormalSpline"]

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import lax
from jax.typing import ArrayLike
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from .convolved import log_normal_convolved, log_truncated_normal_convolved

def _clip_preserve_gradients(x, min_, max_):
    return x + lax.stop_gradient(jnp.clip(x, min_, max_) - x)


class NormalSpline(dist.Distribution):
    support = dist.constraints.real

    def __init__(
        self,
        loc_vals: ArrayLike,
        scale_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        spline_k: int | dict[str, int] = 3,
        clip_locs: tuple[float | None, float | None] = (None, None),
        clip_scales: tuple[float | None, float | None] = (None, None),
    ) -> None:
        """
        Represents a Normal distribution where the loc and (log)scale parameters are
        controlled by splines that are evaluated at some other parameter values x. In
        other words, this distribution is conditional on x.

        Parameters
        ----------
        loc_vals
            Array of loc values at the knot locations.
        scale_vals
            Array of scale values at the knot locations.
        knots
            Array of spline knot locations.
        x
            Array of x values at which to evaluate the splines.
        spline_k (optional)
            Degree of the spline. Can be a single integer or a dictionary with keys
            "loc" and "scale" specifying the degrees of the loc and scale splines
            respectively.
        """
        x = jnp.asarray(x)
        super().__init__(batch_shape=x.shape, event_shape=())

        self.knots = jnp.array(knots)
        self.clip_locs = tuple(clip_locs)
        self.clip_scales = tuple(clip_scales)

        if not isinstance(spline_k, dict):
            spline_k = {"loc": spline_k, "scale": spline_k}
        self.spline_k = spline_k
        self.x = x
        self.loc_vals = jnp.array(loc_vals)
        self.scale_vals = jnp.array(scale_vals)

        # TODO: probably a way to vmap here instead...
        if self.loc_vals.ndim == 0:
            self._loc_spl = lambda _: self.loc_vals
        else:
            self._loc_spl = InterpolatedUnivariateSpline(
                self.knots,
                self.loc_vals,
                k=self.spline_k["loc"],
                endpoints="not-a-knot",  # TODO: make this customizable?
            )

        if self.scale_vals.ndim == 0:
            self._scale_spl = lambda _: self.scale_vals
        else:
            self._scale_spl = InterpolatedUnivariateSpline(
                self.knots,
                self.scale_vals,
                k=self.spline_k["scale"],
                #endpoints="not-a-knot",  # TODO: make this customizable?
            )

    def _make_helper_dist(self, x: ArrayLike | None = None) -> dist.Normal:
        x = self.x if x is None else x
        return dist.Normal(
            loc=_clip_preserve_gradients(self._loc_spl(x), *self.clip_locs),
            scale=_clip_preserve_gradients(self._scale_spl(x), *self.clip_scales),
        )

    def sample(
        self,
        key: jax.Array,
        sample_shape: Any = (),
        x: ArrayLike | None = None,
    ) -> jax.Array | Any:
        """
        Draws samples from the distribution.

        Parameters
        ----------
        key
            JAX random number generator key.
        sample_shape
            Shape of the sample.
        x
            Array of x values at which to evaluate the splines. If not provided, the
            x values provided at initialization will be used.
        """
        helper = self._make_helper_dist(x)
        return helper.sample(key=key, sample_shape=sample_shape)

    def log_prob(self, value: ArrayLike, x: ArrayLike | None = None) -> jax.Array | Any:
        """
        Evaluates the log probability density for a batch of samples given by value.

        Parameters
        ----------
        value
            Array of samples to evaluate the log probability for.
        x
            Array of x values at which to evaluate the splines. If not provided, the
            x values provided at initialization will be used.
        """
        helper = self._make_helper_dist(x)
        return helper.log_prob(value)

    def log_prob_convolved(
        self, value: ArrayLike, err: ArrayLike, x: ArrayLike | None = None
    ) -> jax.Array:
        """
        Log density of this distribution convolved with per-value Gaussian
        measurement errors ``err``, i.e. ``Normal(loc(x), sqrt(scale(x)**2 +
        err**2))``. The spline ``scale`` stays the intrinsic width. Used by the
        model in place of per-star ``:modeldata`` latents; see
        ``stream_membership.distributions.convolved``.
        """
        helper = self._make_helper_dist(x)
        return log_normal_convolved(value, helper.loc, helper.scale, err)


class TruncatedNormalSpline(NormalSpline):
    def __init__(
        self,
        loc_vals: ArrayLike,
        scale_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        low: Any | None = None,
        high: Any | None = None,
        spline_k: int | dict[str, int] = 3,
        clip_locs: tuple[float | None, float | None] = (None, None),
        clip_scales: tuple[float | None, float | None] = (None, None),
    ) -> None:
        """
        Represents a truncated Normal distribution where the loc and (log)scale
        parameters are controlled by splines that are evaluated at some other parameter
        values x. In other words, this distribution is conditional on x.

        Parameters
        ----------
        loc_vals
            Array of loc values at the knot locations.
        scale_vals
            Array of log scale values at the knot locations.
        knots
            Array of spline knot locations.
        x
            Array of x values at which to evaluate the splines.
        low (optional)
            Lower bound of the distribution.
        high (optional)
            Upper bound of the distribution.
        spline_k (optional)
            Degree of the spline. Can be a single integer or a dictionary with keys
            "loc" and "scale" specifying the degrees of the loc and scale splines
            respectively.
        """
        super().__init__(
            loc_vals=loc_vals,
            scale_vals=scale_vals,
            knots=knots,
            x=x,
            spline_k=spline_k,
        )

        self.low = low
        self.high = high

        self.clip_locs = tuple(clip_locs)
        self.clip_scales = tuple(clip_scales)

    def _make_helper_dist(self, x: ArrayLike | None = None) -> dist.Normal:
        x = self.x if x is None else x
        loc = _clip_preserve_gradients(self._loc_spl(x), *self.clip_locs)
        scale = _clip_preserve_gradients(self._scale_spl(x), *self.clip_scales)

        # --- fix for "FloatingPointError: invalid value (nan) encountered
        # in mul" during SVI, verified against real production data ---
        # `clip_locs` (set by the caller, e.g. (-10, 10) for a Uniform(-10,
        # 10) prior on loc_vals) and `low`/`high` (the truncation window,
        # e.g. a narrow data/polygon-derived pm range) are independent and
        # can be very different sizes. During SVI, `loc` (this spline's
        # interpolated mean) is only constrained by `clip_locs`, so nothing
        # stops it from being pushed -- via an Adam step, especially after a
        # huge early loss -- to a value many `scale`s outside [low, high]
        # even though `clip_locs` itself was respected.
        #
        # When that happens, `dist.TruncatedNormal(loc, scale, low, high)`'s
        # normalizing constant Z = CDF(high) - CDF(low) (evaluated in the
        # *standardized*, i.e. (x - loc) / scale, frame) underflows to
        # exactly 0.0 in float arithmetic once loc is far enough outside
        # [low, high] relative to scale. That makes log_prob evaluate to
        # +inf (not -inf!) instead of a large-but-finite negative number,
        # and autodiff resolves the resulting 0/0 in d(log Z)/d(loc) as NaN.
        # That NaN then poisons every parameter sharing this mixture
        # component on the very next Adam step -- this is exactly the
        # observed bug: step 1 loss=-inf (since Trace_ELBO's loss = -ELBO,
        # and ELBO -> +inf when log_prob -> +inf), then step 2 loss=nan,
        # with every `pm2` guide parameter (loc_vals, scale_vals,
        # mixing_distribution, and even the unrelated per-datapoint
        # `:modeldata` sites, since they share the same guide param
        # gradient step) going NaN simultaneously.
        #
        # Reproduced directly: `TruncatedNormal(loc=-10, scale=0.1, low=-5,
        # high=5).log_prob(0.0)` gives `log_prob=inf`, `grad_loc=nan`,
        # `grad_scale=nan`; the same call with `loc` clipped into [-5, 5]
        # first gives a finite (if large, since it's a genuinely unlikely
        # configuration) loss and finite gradients.
        #
        # NOTE: this used to hard-clip `loc` into exactly [low, high]. That
        # over-corrected: a `loc` sitting just outside [low, high] is a
        # completely legitimate fit (e.g. a boundary-pinned/tail-like
        # component, which is exactly what fit_truncated_mixture's
        # unconstrained per-knot optimization can and does converge to --
        # nothing keeps its `loc` inside the truncation window). Verified by
        # direct comparison against a raw scipy.stats.truncnorm evaluation
        # using the same loc/scale/weights: hard-clipping `loc` into
        # [low, high] measurably distorts the density (diff ~0.08 at some
        # grid points in a realistic test case) relative to what was
        # actually fit, which was showing up as "the model doesn't look
        # right at the knots" even though the per-knot fit itself was fine.
        # The actual failure mode above only triggers when `loc` is *many*
        # scales outside [low, high] (the original reproduction used 50
        # scales) -- so clip into a wide, scale-relative margin around
        # [low, high] instead of the window itself. This still makes
        # catastrophic excursions impossible (preventing the NaN), while no
        # longer deforming legitimate fits that are only modestly outside
        # the window.
        #
        # Margin calibration: directly scanned where Z actually underflows
        # to 0 at the float32 precision JAX runs at by default here (not
        # float64) -- log_prob/grad are finite out to ~5.4 scales past the
        # boundary, and blow up (log_prob=inf, grad=nan) at ~5.5 scales and
        # beyond. (The "50 scales" in the original reproduction was just
        # what the production bug happened to hit, not the actual minimal
        # threshold -- it's roughly 10x more headroom than actually needed.)
        # 3 scales of margin stays safely clear of that ~5.5-scale cliff
        # while comfortably covering realistic per-knot fits, which are
        # rarely pushed more than a couple of scales past the window.
        if self.low is not None and self.high is not None:
            margin = 3.0 * scale
            loc = _clip_preserve_gradients(loc, self.low - margin, self.high + margin)

        return dist.TruncatedNormal(
            loc=loc,
            scale=scale,
            low=self.low,
            high=self.high,
        )

    def log_prob_convolved(
        self, value: ArrayLike, err: ArrayLike, x: ArrayLike | None = None
    ) -> jax.Array:
        """
        Exact log density of this truncated Normal convolved with per-value
        Gaussian measurement errors ``err`` (see
        ``stream_membership.distributions.convolved``). Uses the same clipped
        loc/scale as ``log_prob`` (including the NaN-guard margin above).
        """
        helper = self._make_helper_dist(x)
        base = getattr(helper, "base_dist", helper)  # plain Normal if untruncated
        return log_truncated_normal_convolved(
            value, base.loc, base.scale, err, low=self.low, high=self.high
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
