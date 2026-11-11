__all__ = ["PrecomputedAnchors", "TruncatedNormalGMMConditional"]

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike

from .convolved import log_truncated_normal_convolved
from .gmm import IndependentGMM


def _x_is_joint_value(x: jax.Array, D: int) -> bool:
    """Whether `x` already carries a full joint value along `conditioning_dist`'s D axes.

    Shared between `__init__` (to get `batch_shape` right) and
    `_log_responsibilities` (to decide whether to use `x` as-is or fall back
    to the single-axis marginal path) so the two stay consistent. See
    `_log_responsibilities`'s docstring for the two supported shapes of `x`.
    """
    return D > 1 and x.ndim >= 1 and x.shape[-1] == D


class PrecomputedAnchors:
    """Precomputed spatial log-densities ``log N_k(x)`` of a FIXED grid.

    When the conditioning `IndependentGMM`'s locs/scales are fixed (only its
    weights are fitted, e.g. the offtrack grid with ``fixed_scales=True``),
    ``log N_k(x)`` for a coordinate's own baked-in ``x`` never changes, so
    computing it once instead of on every SVI step removes several full
    (N, K) passes from each step. Pass an instance as
    ``TruncatedNormalGMMConditional(..., anchor_log_probs=...)``; share one
    instance between coordinates with the same ``x`` (e.g. pm1 and pm2) so
    XLA can also reuse the per-star normaliser between them.

    The array is computed lazily (eagerly, outside any jit trace) on first
    use and is NOT pickled, so saved SVI results/guides don't grow by
    N x K numbers; it is recomputed after unpickling.

    Each row is shifted by its own max over K. Responsibilities only depend
    on differences within a row, so this changes nothing, but keeps the
    values O(1) so float32 ``dtype`` is safe.
    """

    def __init__(self, locs, scales, x, low=None, high=None, dtype=None):
        self.locs = jnp.asarray(locs)
        self.scales = jnp.broadcast_to(jnp.asarray(scales), self.locs.shape)
        self.low = low
        self.high = high
        self.x = jnp.asarray(x)
        self.dtype = dtype
        self._cache = None

    def get(self) -> jax.Array:
        if self._cache is None:
            with jax.ensure_compile_time_eval():
                K = self.locs.shape[-1]
                gmm = IndependentGMM(
                    dist.CategoricalLogits(jnp.zeros(K)),
                    locs=self.locs, scales=self.scales,
                    low=self.low, high=self.high, validate_args=False,
                )
                x = self.x if self.x.ndim > 1 else self.x[:, None]
                a = gmm.component_log_probs(x).sum(axis=gmm._dim_dim)  # (N, K)
                a = a - jnp.max(a, axis=-1, keepdims=True)
                if self.dtype is not None:
                    a = a.astype(self.dtype)
                self._cache = jax.block_until_ready(a)
        return self._cache

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state


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
    exact_truncation (optional)
        See `log_prob_convolved`.
    anchor_log_probs (optional)
        A `PrecomputedAnchors` (or an (N, K) array) holding ``log N_k(x)``
        for this distribution's own ``x``. Only valid when
        ``conditioning_dist``'s locs/scales are fixed; used whenever no
        explicit ``x`` is passed to ``log_prob``/``log_prob_convolved``/
        ``sample``. Rows may be shifted by any per-row constant.
    compute_dtype (optional)
        dtype for the (N, K) arithmetic in ``log_prob_convolved`` (e.g.
        ``jnp.float32``, ~2x faster on CPU for large K). The result is cast
        back to ``value``'s dtype. Default None: use the inputs' dtype.
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
        exact_truncation: bool = False,
        anchor_log_probs=None,
        compute_dtype=None,
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
        self.exact_truncation = exact_truncation
        self.anchor_log_probs = anchor_log_probs
        self.compute_dtype = compute_dtype

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
        return jax.nn.log_softmax(self._log_weighted_anchors(x), axis=-1)

    def _anchors(self) -> jax.Array | None:
        a = self.anchor_log_probs
        return a.get() if hasattr(a, "get") else a

    def _log_weighted_anchors(self, x: ArrayLike | None = None, dtype=None) -> jax.Array:
        """log w_k + log N_k(x), shape (..., K) (unnormalised over k)."""
        cd = self.conditioning_dist
        log_w = jax.nn.log_softmax(cd.mixing_distribution.logits)
        if x is None and self.anchor_log_probs is not None:
            anchors = self._anchors()
            dtype = anchors.dtype if dtype is None else dtype
            return log_w.astype(dtype) + anchors.astype(dtype)

        x = self.x if x is None else jnp.asarray(x)
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

        out = log_w + anchor_log_probs
        return out if dtype is None else out.astype(dtype)

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

    def log_prob_convolved(
        self, value: ArrayLike, err: ArrayLike, x: ArrayLike | None = None
    ) -> jax.Array:
        """
        Log density convolved with per-value Gaussian errors ``err``: each of
        the K truncated-Normal components is convolved with N(0, err) (see
        ``stream_membership.distributions.convolved``), with the same
        responsibilities as ``log_prob``. No support mask/sentinel is needed:
        the convolved density is finite everywhere.

        SPEED NOTE: by default (``exact_truncation=False``) the per-(star,
        node) posterior-mass factor log[Phi(b') - Phi(a')] is dropped. It is
        ~0 unless a star or node sits within a few sigma of the [low, high]
        window edge (where the offtrack density is negligible anyway), but
        computing it exactly for N stars x K ~ 400 nodes made each SVI step
        ~4.5x slower (measured: 4.5 s vs 1.0 s per step at 40k stars x 400
        nodes). Pass ``exact_truncation=True`` in the coordinate's
        parameters to restore the exact version.
        """
        value = jnp.asarray(value)
        err = jnp.asarray(err)
        dt = self.compute_dtype

        def _c(a):
            return None if a is None else (jnp.asarray(a) if dt is None else jnp.asarray(a, dtype=dt))

        # LA = log w_k + log N_k(x): (..., K), not normalised over k.
        LA = self._log_weighted_anchors(x, dtype=dt)
        own_log_probs = log_truncated_normal_convolved(
            _c(value)[..., None],
            _c(self.loc_vals),
            _c(self.scale_vals),
            _c(err)[..., None],
            low=_c(self.low),
            high=_c(self.high),
            include_posterior_mass=self.exact_truncation,
        )  # (..., K)
        # sum_k r_k C_k with r = softmax(LA), in log space:
        #   lse(log_softmax(LA) + C) = lse(LA + C) - lse(LA)
        # This skips materialising the normalised responsibilities (one fewer
        # (N, K) pass forward and backward), and lse(LA) is identical for
        # coordinates sharing the same anchors (pm1/pm2), so XLA computes it once.
        lse = jax.scipy.special.logsumexp
        out = lse(LA + own_log_probs, axis=-1) - lse(LA, axis=-1)
        return out.astype(value.dtype)

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
