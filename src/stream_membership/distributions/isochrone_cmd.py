__all__ = [
    "IsochroneCMD",
    "KROUPA_SLOPES",
    "KROUPA_BREAKS",
    "SALPETER_SLOPES",
    "SALPETER_BREAKS",
]

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jax.typing import ArrayLike
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from .selection_function import _bilinear_interp

# Broken power-law initial mass functions, dN/dm ~ m**(-alpha_i) in each mass
# segment (masses in Msun; slopes has one more entry than breaks).
KROUPA_SLOPES = (0.3, 1.3, 2.3)  # Kroupa (2001)
KROUPA_BREAKS = (0.08, 0.5)
SALPETER_SLOPES = (2.35,)  # Salpeter (1955)
SALPETER_BREAKS = ()


def _clip_preserve_gradients(x, min_, max_):
    return x + lax.stop_gradient(jnp.clip(x, min_, max_) - x)


def _eval_poly(coeffs: ArrayLike, x: ArrayLike) -> jax.Array:
    """Horner evaluation of polynomial coefficients (highest power first,
    matching ``numpy.poly1d``'s convention)."""
    coeffs = jnp.asarray(coeffs)

    def body(acc, c):
        return acc * x + c, None

    result, _ = lax.scan(body, jnp.zeros_like(x, dtype=coeffs.dtype), coeffs)
    return result


def _log_imf(mass, slopes, breaks):
    """Log of a *continuous* broken power-law IMF (up to a constant):
    ``xi(m) ~ m**(-slopes[i])`` between ``breaks[i-1]`` and ``breaks[i]``."""
    slopes = tuple(float(a) for a in slopes)
    breaks = tuple(float(b) for b in breaks)
    if len(slopes) != len(breaks) + 1:
        msg = "imf_slopes must have exactly one more entry than imf_breaks."
        raise ValueError(msg)
    logm = jnp.log(mass)
    out = -slopes[0] * logm
    for i, b in enumerate(breaks):
        out = out + jnp.where(
            mass > b, -(slopes[i + 1] - slopes[i]) * (logm - jnp.log(b)), 0.0
        )
    return out


class IsochroneCMD(dist.Distribution):
    """
    Joint (magnitude1, magnitude2) density for a single-stellar-population
    (SSP) isochrone track, evaluated at a phi1-dependent apparent distance
    modulus -- following the stream CMD modeling approach of Starkman et al.
    2025 ("Stream Members Only"). Rather than a flexible density estimator
    (as used for the more data-rich background CMD; see ``FlowDensity`` /
    ``CalibratedFlowDensity``), the stream's photometry is modeled as a
    single old, metal-poor isochrone track (fixed shape -- generated once
    offline, see ``scripts/generate_stream_isochrone.py`` in ``gd1-dr3``)
    shifted along phi1 by a distance-modulus track, with intrinsic Gaussian
    scatter in color around the ridge line and a uniform marginal density in
    absolute magnitude over the isochrone's valid range.

    Observed data is passed as two raw apparent magnitudes,
    ``(magnitude1, magnitude2)`` -- e.g. ``(PS_g, PS_r)`` -- matching the
    same two-raw-band convention used elsewhere in this codebase (e.g. the
    background's ``CalibratedFlowDensity`` term), rather than a precomputed
    ``(color, magnitude)`` pair. Internally, ``color = magnitude1 -
    magnitude2`` and ``magnitude1`` is treated as the apparent magnitude,
    consistent with ``track_abs_mag``/``track_color`` below (see
    ``generate_stream_isochrone.py``, which builds ``track_abs_mag =
    abs_mag1`` and an implicit ``track_color = abs_mag1 - abs_mag2``). Both
    components must therefore share the same shared coordinate ordering as
    the isochrone track was generated with (``mag_param_name1``,
    ``mag_param_name2``).

    Follows the same conditioning convention as the rest of this codebase's
    distributions (``NormalSpline``, ``FlowDensity``, ...): ``x`` (here,
    phi1) is fixed at construction time but can be overridden by passing a
    new ``x`` to ``.log_prob()`` / ``.sample()``.

    Parameters
    ----------
    track_abs_mag
        Array of absolute-magnitude values (e.g. absolute Gaia G) along the
        isochrone, strictly monotonic (increasing OR decreasing), used as
        the spline "knots" for the ridge-line color(abs_mag) relation. See
        ``generate_stream_isochrone.restrict_to_monotonic_branch``.
    track_color
        Array of color values (e.g. BP-RP) at each ``track_abs_mag``.
    distmod_coeffs
        Coefficients (highest power first, as in ``numpy.poly1d``) of a
        polynomial giving the apparent distance modulus as a function of
        phi1 (degrees), e.g. the Valluri et al. 2024 GD-1 track:
        ``np.poly1d([1/64**2, 100/64**2, (50/64)**2 + 18.82 - 4.45])``.
    x
        Array of phi1 values (degrees) at which to evaluate the distance
        modulus.
    dm_offset (optional)
        Additive offset (mag) applied to the distance modulus, on top of
        ``distmod_coeffs(x)``. Defaults to 0 (i.e. trust the fixed
        Valluri+24 track exactly). Can be passed a numpyro prior (via
        ``coord_parameters``) to let the joint fit calibrate for any small,
        phi1-independent mismatch between the fixed literature distance
        track and this isochrone/photometric system.
    color_offset (optional)
        Additive offset (mag) applied to the isochrone's color, to absorb
        small systematics (e.g. reddening, metallicity, or model-isochrone
        mismatches) relative to the real data. Defaults to 0.
    color_scale
        Intrinsic Gaussian scatter (mag) in color around the (offset)
        isochrone ridge line, at fixed absolute magnitude. Captures
        photometric errors, unresolved binaries, and any real population
        width not captured by the single-age/single-metallicity track.
    spline_k (optional)
        Degree of the color(abs_mag) interpolating spline. Default 1
        (piecewise-linear), since isochrone tracks are typically sampled
        densely enough that higher-order interpolation isn't needed and a
        linear spline is guaranteed not to introduce spurious wiggles.
    track_mass (optional)
        Array of initial stellar masses (Msun) at each track point (same
        length/order as ``track_abs_mag``; saved as ``initial_mass`` by
        ``generate_stream_isochrone.py``). If given, the marginal density in
        absolute magnitude is a real luminosity function instead of uniform:
        ``dN/dM = xi(m) |dm/dM|`` for an initial mass function ``xi`` (see
        ``imf_slopes``/``imf_breaks``), evaluated robustly as the IMF-weighted
        number of stars per ``lf_smooth``-mag bin along the track (no noisy
        finite derivatives). This matters a lot for the mixture model: a flat
        luminosity function puts most of the stream's mass at sparse bright
        magnitudes (giants, subgiants) and underweights the faint main
        sequence where nearly all observed stream stars live, so the CMD
        coordinate's likelihood actively disfavors a non-zero stream
        fraction. Defaults to ``None`` (uniform; exact old behavior).
    imf_slopes, imf_breaks (optional)
        Slopes ``alpha_i`` (``dN/dm ~ m**-alpha_i``) and mass breaks (Msun) of
        the broken power-law IMF used when ``track_mass`` is given. Default
        Kroupa (2001): slopes ``(0.3, 1.3, 2.3)``, breaks ``(0.08, 0.5)``.
        For Salpeter use ``imf_slopes=(2.35,)``, ``imf_breaks=()`` (see
        ``SALPETER_SLOPES``/``SALPETER_BREAKS``). Static (not fitted).
    lf_smooth (optional)
        Absolute-magnitude bin width (mag) over which the luminosity function
        is averaged, default 0.1.
    off_track_scale (optional)
        How the density behaves for a star whose implied absolute magnitude
        falls outside the isochrone track's range. ``None`` (default; old
        behavior): the density is held CONSTANT beyond the track ends (with
        straight-through gradients), which avoids ``-inf`` losses but has two
        pathologies: that density is unnormalized (it is not counted in
        ``Z``), and a large ``|dm_offset|`` can slide the whole track away so
        that every star sits in this free constant region -- a degenerate
        "outlier" solution (observed: ``dm_offset`` running to -16 while the
        loss climbs, because the straight-through gradient no longer
        descends the real loss). A float (mag, e.g. 0.3) instead multiplies
        the density by ``exp(-0.5 (excess / off_track_scale)^2)``, where
        ``excess`` is how far outside the track range the star's absolute
        magnitude is, and uses ordinary (true-gradient) clipping: still
        finite and smooth, but the penalty is steep, genuine, and
        normalizable, so the track cannot be slid away from the data.
    selection_function (optional)
        A fixed (non-learned) :class:`~stream_membership.distributions.selection_function.MagLimSelectionFunction`
        (or any object exposing ``log_S(mag, phi1, phi2)`` plus
        ``phi1_nodes``/``phi2_nodes`` attributes), used to correct this
        term's density for survey incompleteness at faint magnitudes. This is
        needed here -- unlike for the background's ``CalibratedFlowDensity``
        term -- because this distribution is a fixed *physical* model of the
        stream's true, pre-selection population, not a density estimator fit
        directly to already-selection-thinned data. Renormalization is
        ``p_obs(M) = p_true(M) S(M + dm) / Z`` with
        ``Z(phi1, phi2) = \int p_true(M) S(M + dm(phi1)) dM`` computed by
        quadrature over the track's absolute-magnitude range with the
        luminosity-function weights (on a coarse phi1 x phi2 grid, then
        bilinearly interpolated to each star), so the phi1-dependent
        distance modulus (and the fitted ``dm_offset``) is accounted for and
        the result integrates to 1 over observed (mag, color). Defaults to
        ``None``: no selection correction, no ``phi2`` required.
    phi2 (optional)
        Array of phi2 values (degrees), needed only if ``selection_function``
        is provided. Follows the same fixed-at-construction-but-overridable-
        per-call convention as ``x``. Ignored if ``selection_function`` is
        ``None``.
    """

    support = dist.constraints.real_vector

    def __init__(
        self,
        track_abs_mag: ArrayLike,
        track_color: ArrayLike,
        distmod_coeffs: ArrayLike,
        x: ArrayLike,
        dm_offset: ArrayLike = 0.0,
        color_offset: ArrayLike = 0.0,
        color_scale: ArrayLike = 0.05,
        spline_k: int = 1,
        selection_function: Any | None = None,
        phi2: ArrayLike | None = None,
        track_mass: ArrayLike | None = None,
        imf_slopes: tuple[float, ...] = KROUPA_SLOPES,
        imf_breaks: tuple[float, ...] = KROUPA_BREAKS,
        lf_smooth: float = 0.1,
        off_track_scale: float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        x = jnp.asarray(x)
        super().__init__(
            batch_shape=x.shape, event_shape=(2,), validate_args=validate_args
        )

        self.track_abs_mag = jnp.asarray(track_abs_mag)
        self.track_color = jnp.asarray(track_color)
        self.distmod_coeffs = jnp.asarray(distmod_coeffs)
        self.x = x
        self.dm_offset = dm_offset
        self.color_offset = color_offset
        self.color_scale = color_scale
        self.spline_k = spline_k
        self.selection_function = selection_function
        self.phi2 = None if phi2 is None else jnp.asarray(phi2)

        # InterpolatedUnivariateSpline requires strictly increasing knots;
        # isochrone tracks are typically ordered so abs_mag is *decreasing*
        # with increasing mass, so flip if needed. NOTE: this must be
        # implemented with `jnp.where` rather than a Python-level `if`, even
        # though `track_abs_mag`/`track_color` are conceptually "fixed"
        # (non-numpyro-sampled) data: `ModelComponent.make_dists` (and
        # therefore this constructor) gets called from inside numpyro/SVI
        # machinery such as `find_valid_initial_params`'s `lax.while_loop`,
        # which abstractly traces *everything* reachable inside it,
        # including closed-over "constant" arrays -- not just numpyro sample
        # sites. A Python `if` on a traced value raises
        # `TracerBoolConversionError` in that context (verified directly:
        # see this fix's commit message / test script for the reproduction).
        abs_mag = self.track_abs_mag
        color = self.track_color
        is_decreasing = abs_mag[-1] < abs_mag[0]
        self._abs_mag_sorted = jnp.where(is_decreasing, abs_mag[::-1], abs_mag)
        self._color_sorted = jnp.where(is_decreasing, color[::-1], color)

        self._color_spl = InterpolatedUnivariateSpline(
            self._abs_mag_sorted, self._color_sorted, k=self.spline_k
        )
        self._abs_mag_min = jnp.min(self.track_abs_mag)
        self._abs_mag_max = jnp.max(self.track_abs_mag)

        # --- luminosity function (absolute-magnitude marginal) ---------------
        # Fixed (track-only) quadrature grid in absolute magnitude, used both
        # for the LF density and for Z(phi1, phi2) when a selection function
        # is attached. Everything here is built with jnp ops only (no Python
        # control flow on values), for the same tracing reason as above.
        self.off_track_scale = None if off_track_scale is None else float(off_track_scale)
        self.imf_slopes = tuple(float(a) for a in imf_slopes)
        self.imf_breaks = tuple(float(b) for b in imf_breaks)
        self._has_lf = track_mass is not None
        n_q = 400
        self._M_q = self._abs_mag_min + (self._abs_mag_max - self._abs_mag_min) * jnp.linspace(0.0, 1.0, n_q)
        if self._has_lf:
            mass = jnp.asarray(track_mass)
            mass_sorted = jnp.where(is_decreasing, mass[::-1], mass)  # aligned with _abs_mag_sorted
            xi = jnp.exp(_log_imf(mass_sorted, self.imf_slopes, self.imf_breaks))
            dn = 0.5 * (xi[1:] + xi[:-1]) * jnp.abs(mass_sorted[1:] - mass_sorted[:-1])
            n_cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dn)])  # monotone in M
            half = 0.5 * lf_smooth
            lo = jnp.maximum(self._M_q - half, self._abs_mag_min)
            hi = jnp.minimum(self._M_q + half, self._abs_mag_max)
            w = (jnp.interp(hi, self._abs_mag_sorted, n_cum) - jnp.interp(lo, self._abs_mag_sorted, n_cum)) / (hi - lo)
            self._w_q = jnp.maximum(w, 1e-12 * jnp.max(w))
        else:
            self._w_q = jnp.ones_like(self._M_q)
        self._w_norm = jnp.trapezoid(self._w_q, self._M_q)

    def _distmod(self, x: ArrayLike) -> jax.Array:
        return _eval_poly(self.distmod_coeffs, jnp.asarray(x)) + self.dm_offset

    def log_prob(
        self,
        value: ArrayLike,
        x: ArrayLike | None = None,
        phi2: ArrayLike | None = None,
    ) -> jax.Array | Any:
        """
        Evaluates the log probability density for a batch of (magnitude1,
        magnitude2) samples, e.g. ``(PS_g, PS_r)``.

        Parameters
        ----------
        value
            Array of shape ``(..., 2)`` with columns ``(magnitude1,
            magnitude2)`` -- two raw apparent magnitudes, NOT a precomputed
            ``(color, magnitude)`` pair. The color is computed internally
            as ``magnitude1 - magnitude2``, and ``magnitude1`` is used as
            the apparent magnitude.
        x
            Array of phi1 values at which to evaluate the distance modulus.
            If not provided, the ``x`` values provided at initialization
            will be used.
        phi2
            Array of phi2 values, only used (and only required, from either
            here or construction time) if ``self.selection_function`` is not
            ``None``. Ignored otherwise.
        """
        x = self.x if x is None else jnp.asarray(x)
        value = jnp.asarray(value)
        mag1_obs, mag2_obs = value[..., 0], value[..., 1]
        color_obs = mag1_obs - mag2_obs
        mag_obs = mag1_obs

        dm = self._distmod(x)
        abs_mag = mag_obs - dm

        # --- fix for SVI diverging to +inf loss ---
        # This used to be `jnp.where(in_range, color_lp + mag_lp, -jnp.inf)`,
        # i.e. an exact zero density (`-inf` log_prob) for any star whose
        # implied abs_mag falls outside the fixed isochrone track's
        # [abs_mag_min, abs_mag_max] range. Verified directly on real GD-1
        # data that during SVI (free `dm_offset` + fitted phi1-dependent
        # splines), at least one star's implied abs_mag routinely gets
        # pushed just past this boundary -- one star already sits at the
        # track's bright-end edge even at dm_offset=0. The instant that
        # happens, that single star's `-inf` makes the whole particle's
        # log-joint `-inf`, so the SVI loss jumps to `+inf`; and because a
        # hard cutoff contributes exactly zero gradient, the optimizer gets
        # no signal to correct course and the fit never recovers.
        #
        # Fix: evaluate the predicted color using abs_mag *clipped* into the
        # track's valid range via the same straight-through-gradient trick
        # used elsewhere in this codebase (`_clip_preserve_gradients`) --
        # the forward value is clamped to the boundary (so `_color_spl` is
        # never evaluated out of its domain and `mag_lp` is always the same
        # finite constant), while the gradient still flows as if unclipped.
        # A star sitting off the track now pulls the fit back via the
        # (smooth, always-finite) color-mismatch term instead of a hard
        # -inf cliff.
        if self.off_track_scale is None:
            clipped_abs_mag = _clip_preserve_gradients(
                abs_mag, self._abs_mag_min, self._abs_mag_max
            )
        else:
            clipped_abs_mag = jnp.clip(abs_mag, self._abs_mag_min, self._abs_mag_max)
        pred_color = self._color_spl(clipped_abs_mag) + self.color_offset

        color_lp = dist.Normal(loc=pred_color, scale=self.color_scale).log_prob(
            color_obs
        )
        if self._has_lf:
            # Real luminosity function (IMF-weighted), normalized over the
            # track's absolute-magnitude range.
            w_obs = jnp.interp(clipped_abs_mag, self._M_q, self._w_q)
            mag_lp = jnp.log(w_obs) - jnp.log(self._w_norm)
        else:
            # Uniform marginal density in absolute magnitude over the track's
            # valid range (old behavior; pass `track_mass` for a real LF).
            mag_lp = -jnp.log(self._abs_mag_max - self._abs_mag_min)

        if self.off_track_scale is not None:
            excess = abs_mag - clipped_abs_mag
            mag_lp = mag_lp - 0.5 * (excess / self.off_track_scale) ** 2

        if self.selection_function is None:
            return color_lp + mag_lp

        # Selection-function correction:
        #   p_obs = p_true * S / Z,   Z(phi1, phi2) = int p_true(M) S(M + dm) dM
        # (renormalizing this component's density before it's mixed with any
        # other components -- see IsochroneCMD's docstring). S depends on the
        # observed apparent magnitude (not absolute), and on sky position.
        phi2_ = self.phi2 if phi2 is None else jnp.asarray(phi2)
        if phi2_ is None:
            msg = (
                "self.selection_function is not None, so a `phi2` array is "
                "required -- either pass it at construction or as a "
                "log_prob(..., phi2=...) argument."
            )
            raise ValueError(msg)

        log_s = self.selection_function.log_S(mag_obs, x, phi2_)
        return color_lp + mag_lp + log_s - self._log_Z(x, phi2_)

    def _log_Z(self, x: ArrayLike, phi2: ArrayLike) -> jax.Array:
        """log Z(phi1, phi2): the selection-weighted mass of the true
        (pre-selection) population, by quadrature on a coarse phi1 x phi2
        grid (using the current distance-modulus offset), bilinearly
        interpolated to the requested positions."""
        sf = self.selection_function
        p1 = jnp.linspace(sf.phi1_nodes[0], sf.phi1_nodes[-1], 64)
        p2 = jnp.asarray(sf.phi2_nodes)
        dm_g = self._distmod(p1)  # (64,)
        mag = self._M_q[None, None, :] + dm_g[:, None, None]
        s_grid = jnp.exp(sf.log_S(mag, p1[:, None, None], p2[None, :, None]))  # (64, n2, Q)
        z = jnp.trapezoid(s_grid * self._w_q, self._M_q, axis=-1) / self._w_norm
        return _bilinear_interp(p1, p2, jnp.log(jnp.maximum(z, 1e-300)), x, phi2)

    def sample(
        self,
        key: jax.Array,
        sample_shape: Any = (),
        x: ArrayLike | None = None,
    ) -> jax.Array | Any:
        """
        Draws (magnitude1, magnitude2) samples: absolute magnitude uniform
        over the track's valid range, color drawn from a Gaussian around
        the (offset) isochrone ridge line at that absolute magnitude, then
        both shifted to apparent magnitude via the phi1-dependent distance
        modulus and converted back to the raw (magnitude1, magnitude2)
        pair (``magnitude1`` = apparent magnitude, ``magnitude2`` =
        ``magnitude1 - color``).

        NOTE: intentionally does NOT draw selection-thinned samples even if
        ``self.selection_function`` is set -- it always draws from the
        "true", pre-selection track density. This method isn't used inside
        the SVI likelihood path (only ``log_prob`` is), so this doesn't
        affect fits; it only matters if you use ``.sample()`` directly for
        mock-catalog generation or predictive checks, in which case you'd
        currently need to apply the thinning (e.g. rejection sampling
        against ``selection_function.log_S``) yourself. Left as future work.
        """
        # NOTE: a per-call `x` must *fully* override the batch shape (matching
        # `log_prob`'s convention, and the sibling `NormalSpline.sample`),
        # not broadcast against `self.batch_shape` (the shape of whatever `x`
        # was passed at construction time). Mixing the two in with
        # `jnp.broadcast_shapes` was a bug: e.g. sampling one star at a time
        # via `jax.vmap(lambda k, x: dist.sample(k, x=x))(keys, phi1_array)`
        # -- exactly the pattern used to sanity-check model samples against
        # real data elsewhere in this pipeline -- would broadcast each
        # per-star scalar `x` back up against the *original*, full-length
        # construction-time `x`, silently producing a wrongly-shaped
        # `(len(original_x), 2)` output per vmap iteration instead of `(2,)`.
        x = self.x if x is None else jnp.asarray(x)
        shape = tuple(sample_shape) + x.shape

        key_mag, key_color = jax.random.split(key)
        if self._has_lf:
            cdf = jnp.concatenate(
                [jnp.zeros(1), jnp.cumsum(0.5 * (self._w_q[1:] + self._w_q[:-1]) * jnp.diff(self._M_q))]
            )
            cdf = cdf / cdf[-1]
            abs_mag = jnp.interp(jax.random.uniform(key_mag, shape), cdf, self._M_q)
        else:
            abs_mag = jax.random.uniform(
                key_mag, shape, minval=self._abs_mag_min, maxval=self._abs_mag_max
            )
        pred_color = self._color_spl(abs_mag) + self.color_offset
        color = pred_color + self.color_scale * jax.random.normal(key_color, shape)

        dm = self._distmod(jnp.broadcast_to(x, shape))
        mag1 = abs_mag + dm
        mag2 = mag1 - color

        return jnp.stack([mag1, mag2], axis=-1)
