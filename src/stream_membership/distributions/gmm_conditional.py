__all__ = ["TruncatedNormalGMMConditional"]

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike

from .gmm import IndependentGMM


def _x_is_joint_value(x: jax.Array, D: int) -> bool:
    """Whether `x` already carries a full joint value along `conditioning_dist`'s D axes.

    Shared between `__init__` (to get `batch_shape` right) and
    `_log_responsibilities` (to decide whether to use `x` as-is or fall back
    to the single-axis marginal path) so the two stay consistent. See
    `_log_responsibilities`'s docstring for the two supported shapes of `x`.
    """
    return D > 1 and x.ndim >= 1 and x.shape[-1] == D


class TruncatedNormalGMMConditional(dist.Distribution):
    """A 1D truncated-Normal mixture conditioned on another GMM's marginal.

    Represents::

        p(value | x) = sum_k r_k(x) * TruncatedNormal(loc_vals[k], scale_vals[k], low, high)

    where ``r_k(x)`` are the *exact* Bayes-rule responsibilities of
    ``conditioning_dist``'s K components evaluated at ``x``::

        r_k(x) = softmax_k(log(w_k) + log N_k(x))

    with the weights ``w_k`` and each component's own loc/scale pulled
    directly from ``conditioning_dist`` (an already-built `IndependentGMM`,
    possibly D>1, e.g. a joint ("phi1","phi2") grid) -- *not* resampled
    independently. In practice ``conditioning_dist`` is a sibling
    coordinate's (joint) marginal (e.g. "phi1", or ("phi1","phi2")), shared
    via `stream_membership.model.FromDist` in a `ModelComponent`'s
    `coord_parameters`, so that this coordinate (e.g. "pm1") and any other
    conditional coordinates built the same way (e.g. "pm2",
    "radial_velocity") all agree on which of the K underlying "nodes" a
    given star most likely belongs to, while each still has its own,
    independent per-node loc/scale along its own axis.

    This is the mathematically exact marginal/conditional factorization of
    a joint (x, value) `IndependentGMM` (since `IndependentGMM`'s components
    have no cross-axis correlation, the x-marginal is just the x-axis(es)
    GMM, and the conditional is this responsibility-weighted mixture) --
    except that ``loc_vals``/``scale_vals`` here are free to be *any*
    per-node values (e.g. fit independently, or given an informative prior),
    not literally another axis of a fixed grid.

    Parameters
    ----------
    conditioning_dist
        An already-constructed `IndependentGMM` (e.g. a phi1 marginal, or a
        joint ("phi1","phi2") grid) whose K components define the
        responsibilities ``r_k(x)``.
    loc_vals, scale_vals
        This coordinate's own per-node loc/scale, shape ``(K,)``. Unlike
        ``conditioning_dist``'s own loc (which is often a fixed grid), these
        are typically sampled from their own prior.
    x
        This coordinate's own conditioning array. Either shape ``(N,)`` (this
        coordinate's own ragged-subsample of ``conditioning_dist``'s axis-0
        values, e.g. phi1) or, when ``conditioning_dist`` is D>1, shape
        ``(N, D)`` (the full joint value along all of `conditioning_dist`'s
        axes, e.g. this coordinate's own ragged-subsample of (phi1,phi2)
        pairs) -- see `_log_responsibilities` for how each is handled.
    low, high (optional)
        Truncation bounds for this coordinate's own axis.
    """

    def __init__(
        self,
        conditioning_dist: IndependentGMM,
        loc_vals: ArrayLike,
        scale_vals: ArrayLike,
        x: ArrayLike,
        low: ArrayLike | None = None,
        high: ArrayLike | None = None,
        *,
        validate_args=None,
    ) -> None:
        x = jnp.asarray(x)
        # `x` may carry a trailing D axis (a full joint conditioning value,
        # e.g. (N, 2) phi1/phi2 pairs) -- in that case the batch of
        # independent data points is still just N, not (N, D).
        batch_shape = (
            x.shape[:-1]
            if _x_is_joint_value(x, conditioning_dist._D)
            else x.shape
        )
        super().__init__(
            batch_shape=batch_shape, event_shape=(), validate_args=validate_args
        )

        self.conditioning_dist = conditioning_dist
        self.loc_vals = jnp.asarray(loc_vals)
        self.scale_vals = jnp.asarray(scale_vals)
        self.x = x
        self.low = low
        self.high = high

        self._K = self.loc_vals.shape[-1]
        if self._K != conditioning_dist._K:
            msg = (
                "conditioning_dist and loc_vals/scale_vals must all describe "
                f"the same number of components K, but got K={conditioning_dist._K} "
                f"for conditioning_dist and K={self._K} for loc_vals/scale_vals."
            )
            raise ValueError(msg)

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

    def _log_responsibilities(self, x: ArrayLike | None = None) -> jax.Array:
        """log r_k(x), shape (..., K).

        NOTE: this deliberately reuses `IndependentGMM.component_log_probs`
        (a "private"-ish but already NaN-safe method, see its long comment
        about clipping out-of-bounds points before calling `log_prob`)
        instead of re-deriving the same per-component log density from
        `conditioning_dist`'s loc/scale/low/high by hand, so the
        responsibilities are guaranteed to be computed exactly consistently
        with `conditioning_dist`'s own marginal density (and inherit the same
        NaN-safety guarantees).

        `conditioning_dist` may be D>1 (e.g. a joint ("phi1","phi2") GMM for
        the offtrack component's 2D spatial grid). Two shapes of `x` are
        supported in that case:

        - Full joint value, shape (..., D): e.g. real per-star (phi1, phi2)
          pairs at fitting time. Used as-is -- every axis contributes to the
          responsibility.
        - Single-axis value, shape (...,): e.g. a 1D grid of just this
          coordinate's own conditioning axis (always axis 0 by convention,
          "phi1" in ("phi1", "phi2")), as produced by grid-based plotting's
          `conditional_data[coord] = {"x": "phi1"}` resolution. Since
          `IndependentGMM`'s components are diagonal (no cross-axis
          covariance), the *exact* marginal responsibility along a single
          axis only depends on that axis's own per-component log-density, so
          this is computed by evaluating `component_log_probs` with harmless
          dummy placeholder values (0.0) in the other D-1 axes and then
          discarding their (irrelevant) contribution, keeping only axis 0's.
        """
        x = self.x if x is None else jnp.asarray(x)
        cd = self.conditioning_dist
        D = cd._D

        if D == 1:
            # cd expects value shape (..., D) with D=1 here; component_log_probs
            # returns shape (..., D, K), so sum away the (trivial, size-1) D axis.
            anchor_log_probs = cd.component_log_probs(x[..., None]).sum(
                axis=cd._dim_dim
            )
        elif _x_is_joint_value(x, D):
            # Full joint value already provided (e.g. real (phi1, phi2) data).
            anchor_log_probs = cd.component_log_probs(x).sum(axis=cd._dim_dim)
        else:
            # Single-axis fallback: pad with dummy values in the other D-1
            # axes, then keep only axis 0's (this coordinate's own) log-probs.
            dummy = jnp.zeros(x.shape + (D - 1,), dtype=x.dtype)
            value = jnp.concatenate([x[..., None], dummy], axis=-1)
            full_log_probs = cd.component_log_probs(value)  # (..., D, K)
            anchor_log_probs = jnp.take(full_log_probs, 0, axis=cd._dim_dim)

        log_w = jax.nn.log_softmax(cd.mixing_distribution.logits)
        return jax.nn.log_softmax(log_w + anchor_log_probs, axis=-1)

    def _own_component_distribution(self) -> dist.TruncatedNormal:
        return dist.TruncatedNormal(
            loc=self.loc_vals, scale=self.scale_vals, low=self.low, high=self.high
        )

    def log_prob(self, value: ArrayLike, x: ArrayLike | None = None) -> jax.Array:
        value = jnp.asarray(value)
        log_resp = self._log_responsibilities(x)  # (..., K)

        own = self._own_component_distribution()
        value_expanded = jnp.expand_dims(value, axis=-1)  # (..., 1)

        # Same defensive clip-then-mask pattern as IndependentGMM.component_log_probs
        # / TruncatedNormalSpline: evaluating TruncatedNormal.log_prob directly at an
        # out-of-bounds point gives the right forward value (-inf) but a NaN gradient,
        # so clip the input into the valid support first and mask the *output*
        # afterwards with a large-but-finite sentinel (not literal -inf, to keep the
        # logsumexp below gradient-safe even if every component is masked).
        low = jnp.asarray(-jnp.inf) if self.low is None else jnp.asarray(self.low)
        high = jnp.asarray(jnp.inf) if self.high is None else jnp.asarray(self.high)
        safe_value = jnp.clip(value_expanded, low, high)

        neg_inf_sentinel = jnp.asarray(-1e10, dtype=log_resp.dtype)
        own_log_probs = jnp.where(
            own.support.check(value_expanded),
            own.log_prob(safe_value),
            neg_inf_sentinel,
        )  # (..., K)

        return jax.scipy.special.logsumexp(log_resp + own_log_probs, axis=-1)

    def sample(
        self, key: jax.Array, sample_shape: tuple = (), x: ArrayLike | None = None
    ) -> jax.Array:
        x = self.x if x is None else jnp.asarray(x)
        log_resp = self._log_responsibilities(x)

        key_z, key_val = jax.random.split(key)
        z = dist.CategoricalLogits(logits=log_resp).sample(key_z, sample_shape)
        loc = jnp.take_along_axis(
            jnp.broadcast_to(self.loc_vals, z.shape + (self._K,)),
            z[..., None],
            axis=-1,
        )[..., 0]
        scale = jnp.take_along_axis(
            jnp.broadcast_to(self.scale_vals, z.shape + (self._K,)),
            z[..., None],
            axis=-1,
        )[..., 0]
        return dist.TruncatedNormal(loc=loc, scale=scale, low=self.low, high=self.high).sample(
            key_val
        )
