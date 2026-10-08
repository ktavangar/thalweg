__all__ = ["MagLimSelectionFunction"]

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def _clip_preserve_gradients(x, min_, max_):
    return x + jax.lax.stop_gradient(jnp.clip(x, min_, max_) - x)


def _bilinear_interp(
    x_nodes: ArrayLike, y_nodes: ArrayLike, values: ArrayLike, x: ArrayLike, y: ArrayLike
) -> jax.Array:
    """
    Differentiable bilinear interpolation of a 2D array ``values`` (shape
    ``(len(x_nodes), len(y_nodes))``) on a rectangular grid, evaluated at
    query points ``(x, y)``.

    ``x_nodes``/``y_nodes`` must each be strictly increasing. Query points
    outside the grid are clamped to the grid edges (with gradients preserved
    via the same straight-through trick used elsewhere in this codebase,
    e.g. ``IsochroneCMD``'s ``_clip_preserve_gradients``) rather than
    extrapolated -- appropriate here since a selection-function calibration
    grid is only ever expected to be trustworthy within the footprint it was
    built from.
    """
    x_nodes = jnp.asarray(x_nodes)
    y_nodes = jnp.asarray(y_nodes)
    values = jnp.asarray(values)

    x = _clip_preserve_gradients(jnp.asarray(x), x_nodes[0], x_nodes[-1])
    y = _clip_preserve_gradients(jnp.asarray(y), y_nodes[0], y_nodes[-1])

    ix = jnp.clip(jnp.searchsorted(x_nodes, x, side="right") - 1, 0, len(x_nodes) - 2)
    iy = jnp.clip(jnp.searchsorted(y_nodes, y, side="right") - 1, 0, len(y_nodes) - 2)

    x0, x1 = x_nodes[ix], x_nodes[ix + 1]
    y0, y1 = y_nodes[iy], y_nodes[iy + 1]
    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)

    v00 = values[ix, iy]
    v10 = values[ix + 1, iy]
    v01 = values[ix, iy + 1]
    v11 = values[ix + 1, iy + 1]

    v0 = v00 * (1 - tx) + v10 * tx
    v1 = v01 * (1 - tx) + v11 * tx
    return v0 * (1 - ty) + v1 * ty


class MagLimSelectionFunction:
    """
    A fixed (non-learned) survey selection/completeness function, giving the
    probability that a star of a given *observed* apparent magnitude at a
    given sky position (phi1, phi2) is detected/passes quality cuts.

    This is intentionally NOT a numpyro ``Distribution`` -- it's a small,
    fixed calibration object (analogous in spirit to a pretrained
    ``FlowDensity``/``CalibratedFlowDensity`` or a fixed isochrone track):
    built once, offline, from external survey completeness information
    (e.g. artificial-star-test recovery fractions, or a depth/exposure map),
    and then passed into a coordinate distribution (e.g. ``IsochroneCMD``)
    to correct that term's density for incompleteness. See the module-level
    discussion in ``IsochroneCMD`` for why this correction is needed there
    but NOT for the background's ``CalibratedFlowDensity`` term (which is
    fit directly to already-selection-thinned real data, so it has no
    "true", pre-selection density to correct).

    The completeness curve is modeled as a logistic function of magnitude
    relative to a *local* limiting magnitude that varies smoothly over the
    sky (``m_lim(phi1, phi2)``, e.g. driven by exposure depth, crowding, or
    sky background) --

        S(mag, phi1, phi2) = sigmoid((m_lim(phi1, phi2) - mag) / width)

    -- i.e. the completeness curve's *shape* (set by ``width``) is assumed
    universal across the footprint, and all of the spatial dependence is
    absorbed into the single scalar field ``m_lim(phi1, phi2)``. This is a
    standard approximation for sparse, high-Galactic-latitude fields (the
    regime stellar-stream fields like GD-1 live in); it would need to be
    relaxed (e.g. by also letting ``width`` vary with position) for heavily
    crowded fields where the completeness curve's shape itself changes
    across the footprint, not just its depth.

    Because this functional form is a closed-form logistic, both ``log_S``
    and its normalizing constant ``log_Z`` (the integral of ``S`` over a
    fixed magnitude range, needed to renormalize a selection-corrected
    density -- see ``IsochroneCMD.log_prob``) are cheap, exact, and JAX
    differentiable everywhere, with no numerical quadrature required. The
    only interpolation involved is the (differentiable, bilinear) lookup of
    ``m_lim(phi1, phi2)`` on the calibration grid below.

    Parameters
    ----------
    phi1_nodes, phi2_nodes
        1D, strictly increasing arrays giving the phi1/phi2 grid nodes (in
        degrees) that ``mlim_grid`` is tabulated on. This grid is expected
        to be coarse -- completeness/depth is assumed to vary smoothly over
        the survey footprint, so a modest number of nodes (e.g. O(10-20)
        per axis) covering the data footprint should suffice; it does not
        need to resolve individual stars.
    mlim_grid
        2D array of shape ``(len(phi1_nodes), len(phi2_nodes))`` giving the
        local limiting (50%-completeness) apparent magnitude at each grid
        node. Build this once, offline, from your survey's own
        artificial-star-test or depth-map data -- analogous to how
        ``generate_stream_isochrone.py`` builds the fixed isochrone track
        used by ``IsochroneCMD``.
    width
        Scale (mag) controlling how sharply completeness turns over around
        ``m_lim`` -- e.g. from a logistic fit to artificial-star recovery
        fractions vs. magnitude. Larger ``width`` means a more gradual
        (less step-function-like) completeness dropoff.
    mag_min, mag_max
        Fixed apparent-magnitude integration bounds used to compute
        ``log_Z`` -- should match the valid apparent-magnitude range implied
        by the coordinate distribution this is attached to (e.g.
        ``IsochroneCMD``'s isochrone-track-implied apparent magnitude range).
    """

    def __init__(
        self,
        phi1_nodes: ArrayLike,
        phi2_nodes: ArrayLike,
        mlim_grid: ArrayLike,
        width: ArrayLike,
        mag_min: ArrayLike,
        mag_max: ArrayLike,
    ) -> None:
        self.phi1_nodes = jnp.asarray(phi1_nodes)
        self.phi2_nodes = jnp.asarray(phi2_nodes)
        self.mlim_grid = jnp.asarray(mlim_grid)
        self.width = jnp.asarray(width)
        self.mag_min = jnp.asarray(mag_min)
        self.mag_max = jnp.asarray(mag_max)

    def mlim(self, phi1: ArrayLike, phi2: ArrayLike) -> jax.Array:
        """Locally interpolated limiting (50%-completeness) magnitude."""
        return _bilinear_interp(
            self.phi1_nodes, self.phi2_nodes, self.mlim_grid, phi1, phi2
        )

    def log_S(self, mag: ArrayLike, phi1: ArrayLike, phi2: ArrayLike) -> jax.Array:
        """Log completeness, ``log S(mag, phi1, phi2)``."""
        u = (self.mlim(phi1, phi2) - jnp.asarray(mag)) / self.width
        return jax.nn.log_sigmoid(u)

    def log_Z(
        self,
        phi1: ArrayLike,
        phi2: ArrayLike,
        mag_min: ArrayLike | None = None,
        mag_max: ArrayLike | None = None,
    ) -> jax.Array:
        """
        Log of ``Z(phi1, phi2) = \\int S(mag, phi1, phi2) dmag`` over
        ``[mag_min, mag_max]`` (defaulting to the bounds set at
        construction). Closed form via the antiderivative of the logistic
        function, ``\\int sigmoid(u) du = softplus(u)``:

            Z = width * (softplus(u_min) - softplus(u_max))

        with ``u_min = (m_lim - mag_min) / width`` and
        ``u_max = (m_lim - mag_max) / width``.
        """
        mlim = self.mlim(phi1, phi2)
        mag_min = self.mag_min if mag_min is None else jnp.asarray(mag_min)
        mag_max = self.mag_max if mag_max is None else jnp.asarray(mag_max)

        u_min = (mlim - mag_min) / self.width
        u_max = (mlim - mag_max) / self.width
        return jnp.log(self.width) + jnp.log(
            jax.nn.softplus(u_min) - jax.nn.softplus(u_max)
        )
