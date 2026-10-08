__all__ = [
    "LinearGradient1D",
    "LinearGradientSpline",
    "LinearGradientSplineWithGap",
    "QuadraticGradientSplineWithGap",
]

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jax.typing import ArrayLike
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline


def _clip_preserve_gradients(x, min_, max_):
    return x + lax.stop_gradient(jnp.clip(x, min_, max_) - x)


def _eval_poly(coeffs: ArrayLike, x: ArrayLike) -> jax.Array:
    """Horner evaluation of polynomial coefficients (highest power first,
    matching ``numpy.poly1d``'s convention). Duplicated from
    ``isochrone_cmd.py`` (small enough that it's kept file-local, matching
    this module's existing convention of not cross-importing private
    helpers -- see ``_clip_preserve_gradients`` above)."""
    coeffs = jnp.asarray(coeffs)

    def body(acc, c):
        return acc * x + c, None

    result, _ = lax.scan(body, jnp.zeros_like(x, dtype=coeffs.dtype), coeffs)
    return result


class LinearGradient1D(dist.Distribution):
    """
    A distribution on the interval [low, high] with a linear density gradient:

        pdf(x) = (1 + a * u(x)) / (high - low)

    where ``u(x) = 2 * (x - low) / (high - low) - 1`` is ``x`` rescaled to
    ``[-1, 1]``, and ``a`` (in ``(-1, 1)``) sets the fractional density contrast
    between the two ends of the interval: the density at ``x=high`` is
    ``(1 + a)`` times the mean density (``1 / (high - low)``), and the density at
    ``x=low`` is ``(1 - a)`` times the mean density. ``a=0`` recovers the Uniform
    distribution on ``[low, high]`` exactly, so, unlike a heavily truncated
    Normal, this distribution does not require an extreme, poorly-conditioned
    parameter regime to represent a shallow or near-uniform density gradient.
    """

    arg_constraints = {"a": dist.constraints.interval(-1, 1)}

    def __init__(
        self,
        a: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        validate_args=None,
    ) -> None:
        """
        Parameters
        ----------
        a
            Tilt parameter that must be in ``(-1, 1)``. ``a=0`` is equivalent to a
            Uniform distribution on ``[low, high]``.
        low
            Lower bound of the distribution.
        high
            Upper bound of the distribution.
        """
        self.a = jnp.asarray(a)
        self.low = jnp.asarray(low)
        self.high = jnp.asarray(high)

        batch_shape = lax.broadcast_shapes(
            jnp.shape(self.a), jnp.shape(self.low), jnp.shape(self.high)
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def support(self):
        return dist.constraints.interval(self.low, self.high)

    def _u(self, value: ArrayLike) -> jax.Array:
        return 2 * (value - self.low) / (self.high - self.low) - 1

    def log_prob(self, value: ArrayLike) -> jax.Array:
        value = jnp.asarray(value)
        u = self._u(value)
        log_prob = jnp.log1p(self.a * u) - jnp.log(self.high - self.low)
        return jnp.where(self.support.check(value), log_prob, -jnp.inf)

    def icdf(self, p: ArrayLike) -> jax.Array:
        """
        Inverts the CDF (in the rescaled coordinate ``s = (x - low) / (high -
        low) in [0, 1]``), ``F(s) = a * s**2 + (1 - a) * s``, via the quadratic
        formula. The ``a -> 0`` limit (where the quadratic formula would divide
        by zero) is special-cased to the linear solution ``s = p``.
        """
        a = self.a
        p = jnp.asarray(p)

        is_linear = jnp.abs(a) < 1e-8
        # Avoid division by (near-)zero in the unselected branch -- `jnp.where`
        # evaluates both branches, and a literal 0 in the denominator would
        # otherwise poison gradients even though this branch is discarded.
        safe_denom = jnp.where(is_linear, 1.0, 2 * a)

        disc = (1 - a) ** 2 + 4 * a * p
        disc = jnp.clip(disc, 0.0, None)  # guard tiny negative values from roundoff
        s_quad = (-(1 - a) + jnp.sqrt(disc)) / safe_denom
        s_lin = p

        s = jnp.where(is_linear, s_lin, s_quad)
        return self.low + s * (self.high - self.low)

    def sample(self, key: jax.Array, sample_shape: Any = ()) -> jax.Array:
        shape = tuple(sample_shape) + self.batch_shape
        p = jax.random.uniform(key, shape=shape)
        return self.icdf(p)


class LinearGradientSpline(dist.Distribution):
    def __init__(
        self,
        a_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        spline_k: int = 3,
        clip_a: tuple[float | None, float | None] = (None, None),
    ) -> None:
        """
        Represents a `LinearGradient1D` distribution where the tilt parameter
        `a` is controlled by a spline that is evaluated at some other parameter
        values x. In other words, this distribution is conditional on x.

        Parameters
        ----------
        a_vals
            Array of tilt parameter (`a`) values at the knot locations.
        knots
            Array of spline knot locations.
        x
            Array of x values at which to evaluate the spline.
        low
            Lower bound of the distribution.
        high
            Upper bound of the distribution.
        spline_k (optional)
            Degree of the spline.
        clip_a (optional)
            If specified, clips the spline-interpolated tilt parameter `a` into
            this range (using a straight-through-gradient trick, so forward
            values are clipped but gradients still flow as if unclipped). Since
            `a` must stay in `(-1, 1)` for the density to be valid, and nothing
            otherwise stops an optimizer (e.g. during SVI) from pushing the
            spline-interpolated `a` outside that range, it is recommended to set
            this to something like `(-0.999, 0.999)` in practice.
        """
        x = jnp.asarray(x)
        super().__init__(batch_shape=x.shape, event_shape=())

        self.knots = jnp.array(knots)
        self.low = low
        self.high = high
        self.clip_a = tuple(clip_a)

        self.spline_k = int(spline_k)
        self.x = x
        self.a_vals = jnp.array(a_vals)

        if self.a_vals.ndim == 0:
            self._a_spl = lambda _: self.a_vals
        else:
            self._a_spl = InterpolatedUnivariateSpline(
                self.knots,
                self.a_vals,
                k=self.spline_k,
                endpoints="not-a-knot",
            )

    def _make_helper_dist(self, x: ArrayLike | None = None) -> LinearGradient1D:
        x = self.x if x is None else x
        a = _clip_preserve_gradients(self._a_spl(x), *self.clip_a)
        return LinearGradient1D(a=a, low=self.low, high=self.high)

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
            Array of x values at which to evaluate the spline. If not provided,
            the x values provided at initialization will be used.
        """
        helper = self._make_helper_dist(x)
        return helper.sample(key=key, sample_shape=sample_shape)

    def log_prob(self, value: ArrayLike, x: ArrayLike | None = None) -> jax.Array | Any:
        """
        Evaluates the log probability density for a batch of samples given by
        value.

        Parameters
        ----------
        value
            Array of samples to evaluate the log probability for.
        x
            Array of x values at which to evaluate the spline. If not provided,
            the x values provided at initialization will be used.
        """
        helper = self._make_helper_dist(x)
        return helper.log_prob(value)

    @property
    def support(self):
        return dist.constraints.interval(self.low, self.high)


class LinearGradientSplineWithGap(dist.Distribution):
    """
    A `LinearGradientSpline` variant for a coordinate (e.g. phi2) whose
    training data has an interior band censored out ("gap") -- e.g. the
    background's phi2 density, fit on data with the on-stream footprint
    masked out. The gap's edges can themselves vary with the conditioning
    variable ``x`` (e.g. following a phi1-dependent stream track/pawprint
    footprint).

    This explicitly renormalizes the likelihood by the model's own predicted
    probability mass inside the excised gap, ``P_gap(a)``, i.e. use
    ``log p(x) - log(1 - P_gap(a))`` instead of the naive ``log p(x)``.
    This is exact for *any* gap position or degree of asymmetry -- no
    symmetry assumption required -- and, because `LinearGradient1D`'s CDF
    is already closed-form (the same quadratic used by its ``icdf``),
    ``P_gap(a)`` is cheap, closed-form, and fully differentiable: no
    numerical integration needed.

    Parameters
    ----------
    a_vals
        Array of tilt parameter (`a`) values at the knot locations (same as
        `LinearGradientSpline`).
    knots
        Array of spline knot locations for the `a` spline (same as
        `LinearGradientSpline`).
    x
        Array of x values at which to evaluate the `a` spline and the
        (possibly x-dependent) gap edges.
    low
        Lower bound of the (fixed, not x-dependent) distribution domain.
    high
        Upper bound of the (fixed, not x-dependent) distribution domain.
    gap_center_spl
        Spline for the gap's center as a function of ``x``. This must be
        precomputed and passed in, e.g. by fitting a spline to the on-sky 
        footprint's centerline.
    gap_half_width
        Half-width of the gap which can be fixed or an x-dependent spline.
        If x-dependent, this must be precomputed and passed in.
    spline_k (optional)
        Degree of the `a` spline. Default 3.
    clip_a (optional)
        Same as `LinearGradientSpline`: if specified, clips the
        spline-interpolated tilt parameter `a` into this range (using a
        straight-through-gradient trick). Recommended, e.g. ``(-0.999,
        0.999)``, since nothing else stops an optimizer from pushing `a`
        outside the valid ``(-1, 1)`` range during SVI.
    gap_renormalize (optional)
        Default True (original behavior): ``log_prob`` is divided by
        ``1 - P_gap`` so the density integrates to 1 OVER THE NON-GAP
        REGION. That is correct when *fitting* to data with the corridor
        excised (e.g. the background-only sample). It is WRONG when
        *evaluating* the component on a sample that still contains the
        corridor (e.g. ``run_data`` in the bkg+stream mixture): there the
        background genuinely has stars inside the corridor, and the extra
        ``1/(1 - P_gap)`` factor inflates the background density at every
        star by 10-30%, which biases the mixture fraction of any narrow
        component (e.g. the stream's phi2 term) toward zero. Set False for a
        component built to be evaluated on the full sample: the density is
        then the plain (un-excised) polynomial, normalized over the whole
        ``[low, high]`` (using the ``a_vals`` fit with the gap excised).
    """

    def __init__(
        self,
        a_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        gap_center_spl: InterpolatedUnivariateSpline,
        gap_half_width: ArrayLike,
        spline_k: int = 3,
        clip_a: tuple[float | None, float | None] = (None, None),
        gap_renormalize: bool = True,
    ) -> None:
        x = jnp.asarray(x)
        super().__init__(batch_shape=x.shape, event_shape=())

        self.knots = jnp.array(knots)
        self.low = low
        self.high = high
        self.gap_renormalize = bool(gap_renormalize)
        self.gap_center_spl = gap_center_spl
        if isinstance(gap_half_width, InterpolatedUnivariateSpline):
            self.gap_half_width_spl = gap_half_width
        else:
            self.gap_half_width = jnp.asarray(gap_half_width)
        self.clip_a = tuple(clip_a)

        self.spline_k = int(spline_k)
        self.x = x
        self.a_vals = jnp.array(a_vals)

        if self.a_vals.ndim == 0:
            self._a_spl = lambda _: self.a_vals
        else:
            self._a_spl = InterpolatedUnivariateSpline(
                self.knots,
                self.a_vals,
                k=self.spline_k,
                endpoints="not-a-knot",
            )

    def _gap_bounds(self, x: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """Evaluate the fixed (not-fit) gap edges at `x`."""
        if hasattr(self, "gap_half_width_spl"):
            g1 = self.gap_center_spl(x) - self.gap_half_width_spl(x)
            g2 = self.gap_center_spl(x) + self.gap_half_width_spl(x)
        else:
            g1 = self.gap_center_spl(x) - self.gap_half_width
            g2 = self.gap_center_spl(x) + self.gap_half_width
        return g1, g2

    @staticmethod
    def _cdf_s(a: ArrayLike, s: ArrayLike) -> jax.Array:
        """CDF of `LinearGradient1D` in the rescaled coordinate ``s = (x -
        low) / (high - low) in [0, 1]`` -- the same ``F(s) = a*s**2 +
        (1-a)*s`` used by `LinearGradient1D.icdf`."""
        return a * s**2 + (1 - a) * s

    def _gap_log_mass(self, a: ArrayLike, x: ArrayLike) -> jax.Array:
        """``log(1 - P_gap(a))``, the log-normalization correction for the
        probability mass the (uncensored) model would assign to the gap at
        this `x`."""
        g1, g2 = self._gap_bounds(x)
        s1 = (g1 - self.low) / (self.high - self.low)
        s2 = (g2 - self.low) / (self.high - self.low)
        p_gap = self._cdf_s(a, s2) - self._cdf_s(a, s1)
        # Guard against p_gap creeping to/past 1 during early SVI steps
        # (e.g. if a is transiently extreme), which would send log(1-p_gap)
        # to -inf/NaN. Straight-through-gradient clip, as used elsewhere in
        # this codebase (see IsochroneCMD, IndependentGMM).
        p_gap = _clip_preserve_gradients(p_gap, 0.0, 1 - 1e-6)
        return jnp.log1p(-p_gap)

    def _tilt(self, x: ArrayLike) -> jax.Array:
        return _clip_preserve_gradients(self._a_spl(x), *self.clip_a)

    def _make_helper_dist(self, x: ArrayLike | None = None) -> LinearGradient1D:
        x = self.x if x is None else jnp.asarray(x)
        a = self._tilt(x)
        return LinearGradient1D(a=a, low=self.low, high=self.high)

    def sample(
        self,
        key: jax.Array,
        sample_shape: Any = (),
        x: ArrayLike | None = None,
    ) -> jax.Array | Any:
        """
        Draws samples from the (gap-excised, renormalized) distribution, by
        inverse-CDF sampling with the gap's CDF interval collapsed out.

        Parameters
        ----------
        key
            JAX random number generator key.
        sample_shape
            Shape of the sample.
        x
            Array of x values at which to evaluate the spline and the gap
            edges. If not provided, the x values provided at initialization
            will be used.
        """
        x = self.x if x is None else jnp.asarray(x)
        a = self._tilt(x)
        if not self.gap_renormalize:
            return LinearGradient1D(a=a, low=self.low, high=self.high).sample(
                key, sample_shape
            )
        g1, g2 = self._gap_bounds(x)
        s1 = (g1 - self.low) / (self.high - self.low)
        s2 = (g2 - self.low) / (self.high - self.low)
        p_gap = _clip_preserve_gradients(
            self._cdf_s(a, s2) - self._cdf_s(a, s1), 0.0, 1 - 1e-6
        )
        f_g1 = self._cdf_s(a, s1)

        shape = tuple(sample_shape) + x.shape
        v = jax.random.uniform(key, shape=shape)
        # Map v in [0, 1] (the *observed*, gap-excised CDF) to p in [0, 1]
        # (the base, un-excised CDF), by inserting the gap's CDF interval
        # [f_g1, f_g1 + p_gap] wherever v crosses f_g1 / (1 - p_gap).
        below_gap = v <= (f_g1 / (1 - p_gap))
        p = jnp.where(below_gap, v * (1 - p_gap), v * (1 - p_gap) + p_gap)

        helper = LinearGradient1D(a=a, low=self.low, high=self.high)
        return helper.icdf(p)

    def log_prob(self, value: ArrayLike, x: ArrayLike | None = None) -> jax.Array | Any:
        """
        Evaluates the log probability density (renormalized for the excised
        gap) for a batch of samples given by value.

        Parameters
        ----------
        value
            Array of samples to evaluate the log probability for. Assumed
            to already exclude the gap (e.g. real background data, which by
            construction has nothing in the on-sky-masked corridor) --
            evaluating this at a value that falls inside the gap will
            silently return the (renormalized) density there rather than
            flagging an error, since nothing here checks `value` against
            `_gap_bounds`.
        x
            Array of x values at which to evaluate the spline and the gap
            edges. If not provided, the x values provided at initialization
            will be used.
        """
        x = self.x if x is None else jnp.asarray(x)
        a = self._tilt(x)
        helper = LinearGradient1D(a=a, low=self.low, high=self.high)
        if not self.gap_renormalize:
            return helper.log_prob(value)
        return helper.log_prob(value) - self._gap_log_mass(a, x)

    @property
    def support(self):
        return dist.constraints.interval(self.low, self.high)


class QuadraticGradientSplineWithGap(LinearGradientSplineWithGap):
    """
    A degree-2 generalization of `LinearGradientSplineWithGap`: the density on
    ``[low, high]`` is a quadratic polynomial in the rescaled coordinate
    ``u = 2*(x - low)/(high - low) - 1 in [-1, 1]``, written in the Legendre
    basis,

        pdf(x) = [1 + a * P1(u) + b * P2(u)] / (high - low),
        P1(u) = u,  P2(u) = (3*u**2 - 1) / 2,

    so ``a`` is the same tilt parameter as in `LinearGradient1D` and ``b`` is
    a new curvature parameter (``b > 0``: density piles up at both ends /
    dips in the middle; ``b < 0``: peaked in the middle). P1 and P2 each
    integrate to zero over ``[-1, 1]``, so the density stays exactly
    normalized for ANY ``(a, b)`` -- no renormalization needed -- and
    ``b = 0`` recovers `LinearGradientSplineWithGap` exactly (``a=b=0`` is
    Uniform). Both ``a`` and ``b`` are splines in the conditioning variable
    ``x`` (sharing the same ``knots``), and the gap renormalization is
    identical to the parent class, using the closed-form cubic CDF below
    (still no numerical quadrature; fully differentiable).

    Positivity: the density is only valid where
    ``1 + a*u + b*(3u^2 - 1)/2 >= 0`` on ``[-1, 1]`` (always true when
    ``|a| + |b| <= 1/2``, e.g. the +/-0.2 priors used for the linear
    model's ``a``, with a similar prior on ``b``). Nothing else stops an
    optimizer from leaving that region, so the polynomial is clipped to a
    small positive floor inside ``log_prob`` (straight-through gradient, as
    elsewhere in this codebase); ``clip_a``/``clip_b`` can additionally clamp
    the spline-interpolated parameters. Stay inside the valid region via the
    priors on ``a_vals``/``b_vals``.

    Parameters
    ----------
    b_vals
        Curvature parameter values at the knot locations (same shape as
        ``a_vals``, or a scalar). All other parameters are as in
        `LinearGradientSplineWithGap`.
    clip_b (optional)
        Straight-through clip range for the spline-interpolated ``b``.
    """

    def __init__(
        self,
        a_vals: ArrayLike,
        b_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        gap_center_spl: InterpolatedUnivariateSpline,
        gap_half_width: ArrayLike,
        spline_k: int = 3,
        clip_a: tuple[float | None, float | None] = (None, None),
        clip_b: tuple[float | None, float | None] = (None, None),
        gap_renormalize: bool = True,
    ) -> None:
        super().__init__(
            a_vals=a_vals,
            knots=knots,
            x=x,
            low=low,
            high=high,
            gap_center_spl=gap_center_spl,
            gap_half_width=gap_half_width,
            spline_k=spline_k,
            clip_a=clip_a,
            gap_renormalize=gap_renormalize,
        )
        self.clip_b = tuple(clip_b)
        self.b_vals = jnp.array(b_vals)
        if self.b_vals.ndim == 0:
            self._b_spl = lambda _: self.b_vals
        else:
            self._b_spl = InterpolatedUnivariateSpline(
                self.knots, self.b_vals, k=self.spline_k, endpoints="not-a-knot"
            )

    def _curv(self, x: ArrayLike) -> jax.Array:
        return _clip_preserve_gradients(self._b_spl(x), *self.clip_b)

    @staticmethod
    def _poly(a: ArrayLike, b: ArrayLike, s: ArrayLike) -> jax.Array:
        """Density (w.r.t. ``s = (x - low)/(high - low)``) before clipping."""
        u = 2 * s - 1
        return 1 + a * u + b * (3 * u**2 - 1) / 2

    @staticmethod
    def _cdf_s2(a: ArrayLike, b: ArrayLike, s: ArrayLike) -> jax.Array:
        """Closed-form CDF in ``s``: integral of the density from 0 to s,
        ``F(s) = s + a*(s^2 - s) + b*((u^3 + 1)/4 - s/2)``, ``u = 2s - 1``
        (reduces to the linear model's ``a*s^2 + (1-a)*s`` at b=0)."""
        u = 2 * s - 1
        return s + a * (s**2 - s) + b * ((u**3 + 1) / 4 - s / 2)

    def _icdf2(self, a, b, p, n_iter: int = 50):
        """Inverts the cubic CDF by bisection (the CDF is monotone wherever
        the density is non-negative). Only used for sampling, never in the
        likelihood path."""
        p = jnp.asarray(p)
        lo = jnp.zeros_like(p)
        hi = jnp.ones_like(p)

        def body(_, carry):
            lo, hi = carry
            mid = 0.5 * (lo + hi)
            go_up = self._cdf_s2(a, b, mid) < p
            return jnp.where(go_up, mid, lo), jnp.where(go_up, hi, mid)

        lo, hi = lax.fori_loop(0, n_iter, body, (lo, hi))
        return self.low + 0.5 * (lo + hi) * (self.high - self.low)

    def _gap_mass2(self, a, b, x):
        g1, g2 = self._gap_bounds(x)
        s1 = (g1 - self.low) / (self.high - self.low)
        s2 = (g2 - self.low) / (self.high - self.low)
        p_gap = self._cdf_s2(a, b, s2) - self._cdf_s2(a, b, s1)
        return _clip_preserve_gradients(p_gap, 0.0, 1 - 1e-6), self._cdf_s2(a, b, s1)

    def sample(
        self,
        key: jax.Array,
        sample_shape: Any = (),
        x: ArrayLike | None = None,
    ) -> jax.Array | Any:
        """Inverse-CDF sampling with the gap's CDF interval collapsed out
        (same scheme as the parent class, with the cubic CDF)."""
        x = self.x if x is None else jnp.asarray(x)
        a, b = self._tilt(x), self._curv(x)
        shape = tuple(sample_shape) + x.shape
        if not self.gap_renormalize:
            p = jax.random.uniform(key, shape=shape)
            return self._icdf2(
                jnp.broadcast_to(a, shape), jnp.broadcast_to(b, shape), p
            )
        p_gap, f_g1 = self._gap_mass2(a, b, x)
        v = jax.random.uniform(key, shape=shape)
        below_gap = v <= (f_g1 / (1 - p_gap))
        p = jnp.where(below_gap, v * (1 - p_gap), v * (1 - p_gap) + p_gap)
        return self._icdf2(jnp.broadcast_to(a, shape), jnp.broadcast_to(b, shape), p)

    def log_prob(self, value: ArrayLike, x: ArrayLike | None = None) -> jax.Array | Any:
        x = self.x if x is None else jnp.asarray(x)
        value = jnp.asarray(value)
        a, b = self._tilt(x), self._curv(x)
        s = (value - self.low) / (self.high - self.low)
        dens = _clip_preserve_gradients(self._poly(a, b, s), 1e-6, None)
        lp = jnp.log(dens) - jnp.log(self.high - self.low)
        if self.gap_renormalize:
            p_gap, _ = self._gap_mass2(a, b, x)
            lp = lp - jnp.log1p(-p_gap)
        return jnp.where(self.support.check(value), lp, -jnp.inf)
