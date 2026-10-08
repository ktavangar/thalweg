import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

__all__ = ["IndependentGMM", "ScalarTruncatedNormalGMM"]


def _drop_trailing_singleton_axis(x: ArrayLike) -> jax.Array:
    """Undo the "append a (D, 1)-broadcasting axis" convention.

    `IndependentGMM._low`/`_high` are stored in whatever shape they need for
    broadcasting against `locs`/`scales`'s (D, K) shape when constructing the
    component `TruncatedNormal` -- typically (D, 1) (see the note in
    `component_log_probs`). For checking a `value` of shape (..., D) against
    them (as `_IndependentGMMSupport` needs to), that trailing, size-1 K-axis
    must be dropped first so `low`/`high` end up shaped (D,) (or scalar) and
    broadcast against `value`'s *last* (D) axis instead of reappearing as a
    spurious leading axis.
    """
    x = jnp.asarray(x)
    if x.ndim >= 1 and x.shape[-1] == 1:
        return jnp.squeeze(x, axis=-1)
    return x


class _IndependentGMMSupport(dist.constraints.Constraint):
    """The support of an `IndependentGMM`: per-dimension bounds, reduced over D.

    This can't just be `constraints.independent(constraints.interval(...), 1)`
    because that requires `value` to already carry an explicit trailing D
    axis, which -- for the D=1 case this class is exercised with everywhere
    in this codebase -- it never does (`value` arrives shaped `(N,)`, not
    `(N, 1)`). Instead this mirrors the same defensive `atleast_2d(value.T).T`
    normalization `component_log_probs` already does, so `check(value)`
    always returns a shape matching `value`'s batch axes only (no leftover D
    axis), regardless of whether that trailing D axis was present or not.
    """

    event_dim = 1

    def __init__(self, low: ArrayLike, high: ArrayLike):
        self._low = _drop_trailing_singleton_axis(low)
        self._high = _drop_trailing_singleton_axis(high)
        super().__init__()

    def __call__(self, value: ArrayLike) -> jax.Array:
        value = jnp.atleast_2d(jnp.asarray(value).T).T
        mask = (value >= self._low) & (value <= self._high)
        return mask.all(axis=-1)

    def __repr__(self) -> str:
        return f"IndependentGMMSupport(low={self._low}, high={self._high})"

    def feasible_like(self, prototype: ArrayLike) -> jax.Array:
        mid = (self._low + self._high) / 2
        return jnp.broadcast_to(mid, jnp.shape(prototype))

    def tree_flatten(self):
        return (self._low, self._high), (("_low", "_high"), {})


class ScalarTruncatedNormalGMM(dist.Distribution):
    """Vectorized scalar-event (``event_shape=()``) 1D truncated-Normal mixture.

    Exactly equivalent to ``dist.MixtureGeneral(mixing, [TruncatedNormal(loc[k],
    scale[k], low, high) for k in range(K)])``, but evaluated with a single
    vectorized (K-axis) computation. `MixtureGeneral` instead evaluates each of
    its component distributions' ``log_prob`` separately in a Python loop, so
    the traced/compiled graph grows linearly with K -- a major compile-time
    cost for large K (e.g. a fine off-track lattice). Its `.support` is a plain
    `constraints.interval`, the same type `dist.TruncatedNormal` exposes, so it
    can be combined with sibling scalar distributions in an outer
    `dist.MixtureGeneral`.
    """

    def __init__(self, mixing_distribution, locs, scales, low=None, high=None,
                 *, validate_args=None):
        self._mixing_distribution = mixing_distribution
        self.locs = jnp.asarray(locs)  # (K,)
        self.scales = jnp.asarray(scales)  # (K,)
        # Scalar bounds (a size-1 (1,) / (1, 1) array is squeezed to a scalar).
        self.low = None if low is None else jnp.reshape(jnp.asarray(low), ())
        self.high = None if high is None else jnp.reshape(jnp.asarray(high), ())
        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)

    @property
    def mixing_distribution(self):
        return self._mixing_distribution

    @property
    def support(self):
        if self.low is None and self.high is None:
            return dist.constraints.real
        low = -jnp.inf if self.low is None else self.low
        high = jnp.inf if self.high is None else self.high
        return dist.constraints.interval(low, high)

    def _component(self):
        kwargs = {}
        if self.low is not None:
            kwargs["low"] = self.low
        if self.high is not None:
            kwargs["high"] = self.high
        return dist.TruncatedNormal(self.locs, self.scales, **kwargs)

    def log_prob(self, value):
        value = jnp.asarray(value)
        v = value[..., None]  # (..., 1) against K components
        low = -jnp.inf if self.low is None else jnp.asarray(self.low)
        high = jnp.inf if self.high is None else jnp.asarray(self.high)
        # Same clip-then-mask NaN-gradient guard as IndependentGMM.component_log_probs.
        safe_v = jnp.clip(v, low, high)
        comp_lp = self._component().log_prob(safe_v)
        comp_lp = jnp.where(
            (v >= low) & (v <= high), comp_lp, jnp.asarray(-1e10, dtype=comp_lp.dtype)
        )
        return logsumexp(
            jax.nn.log_softmax(self.mixing_distribution.logits) + comp_lp, axis=-1
        )

    def sample(self, key, sample_shape=()):
        key_z, key_x = jax.random.split(key)
        z = self.mixing_distribution.sample(key_z, sample_shape)
        kwargs = {}
        if self.low is not None:
            kwargs["low"] = self.low
        if self.high is not None:
            kwargs["high"] = self.high
        return dist.TruncatedNormal(self.locs[z], self.scales[z], **kwargs).sample(key_x)


class IndependentGMM(dist.MixtureSameFamily):
    def __init__(
        self,
        mixing_distribution: dist.CategoricalLogits | dist.CategoricalProbs,
        locs: ArrayLike = 0.0,
        scales: ArrayLike = 1.0,
        low: ArrayLike | None = None,
        high: ArrayLike | None = None,
        *,
        validate_args=True,
    ):
        """
        A Gaussian Mixture Model where the components are fixed to their input locations
        and there are no covariances (but each dimension can have different scales /
        standard deviations).

        Parameters
        ----------
        mixing_distribution
            Distribution over the mixture components.
        locs
            Array of means for each component. This should have shape (D, K) where D is
            the dimensionality of the data and K is the number of mixture components.
        scales
            Array of standard deviations for each component. This should have shape (D,
            K) where D is the dimensionality of the data and K is the number of mixture
            components.
        low
            Lower bounds for each dimension. This should either be a scalar or have
            shape (D,) where D is the dimensionality of the data.
        high
            Upper bounds for each dimension. This should either be a scalar or have
            shape (D,) where D is the dimensionality of the data.
        """
        # K = mixture components, D = dimensions
        # - event_shape is the dimensionality of the data - number of dependent
        #   coordinates, i.e., "D" in the below
        # - batch_shape is the number of independent dimensions - here "K"
        combined_shape = jax.lax.broadcast_shapes(jnp.shape(locs), jnp.shape(scales))
        if len(combined_shape) != 2:
            msg = (
                f"locs and scales must have 2 axes, but got {len(combined_shape)}. The "
                "shape must be: (D, K) where D is the dimensionality of the data and K "
                "is the number of mixture components."
            )
            raise ValueError(msg)
        self._D, self._K = combined_shape

        # Kept (broadcast to the full (D, K) shape) so `marginal` can slice
        # out a single axis's own locs/scales later, without having to dig
        # them back out of `component_distribution`'s internals.
        self._locs = jnp.broadcast_to(jnp.asarray(locs), combined_shape)
        self._scales = jnp.broadcast_to(jnp.asarray(scales), combined_shape)

        component_kwargs = {"loc": locs, "scale": scales}
        if low is not None:
            component_kwargs["low"] = low
        if high is not None:
            component_kwargs["high"] = high

        component = dist.TruncatedNormal(**component_kwargs)
        component._batch_shape = (self._D, self._K)
        self._low = low
        self._high = high

        # NOTE: we deliberately do *not* call `super().__init__()`
        # (`MixtureSameFamily.__init__`) here. As of numpyro>=0.20,
        # `MixtureSameFamily.__init__` asserts that `component_distribution.support`
        # is a `ParameterFreeConstraint`, which a bounded `TruncatedNormal` never
        # satisfies (its support is a parameterized `Interval(low, high)`). That
        # check exists so the base class's generic `log_prob`/`cdf`/etc. can assume
        # a fixed support across components, but `IndependentGMM` overrides all of
        # those (see `log_prob`, `component_log_probs`, `support` below) and
        # explicitly masks out-of-bounds values itself, so the restriction doesn't
        # apply to us. Instead, we replicate the (small) subset of
        # `MixtureSameFamily.__init__`'s bookkeeping that we actually rely on,
        # skipping straight to `Distribution.__init__`. This mirrors the numpyro
        # 0.19 and 0.20 implementations identically apart from the added assert.
        n_components = getattr(mixing_distribution, "probs", None)
        n_components = (
            n_components.shape[-1]
            if n_components is not None
            else mixing_distribution.logits.shape[-1]
        )
        if component.batch_shape[-1] != n_components:
            msg = (
                "Component distribution batch shape last dimension "
                f"(size={component.batch_shape[-1]}) needs to correspond to the "
                f"mixture_size={n_components}!"
            )
            raise ValueError(msg)

        dist.Distribution.__init__(
            self, batch_shape=(), event_shape=(self._D,), validate_args=validate_args
        )
        self._mixing_distribution = mixing_distribution
        self._component_distribution = component
        self._mixture_size = n_components
        self._dim_dim = -2

    @property
    def mixture_dim(self):
        return -1

    @property
    def support(self):
        # NOTE: this used to just forward `self.component_distribution.support`
        # (the underlying `TruncatedNormal`'s `Interval` constraint), but that
        # constraint's `low`/`high` are shaped (D, 1) -- broadcastable against
        # `locs`/`scales`'s (D, K) shape (see the long comment in
        # `component_log_probs` below) -- which is the wrong shape to check a
        # `value` against here: this distribution's own `event_shape` is (D,),
        # with no K axis at all. Checking e.g. a (400,) `value` (D=1) against
        # a (1, 1)-shaped `low`/`high` broadcasts to (1, 400) instead of the
        # required (400,) (the two size-1 axes both left-pad against the
        # missing dims instead of disappearing), which silently corrupts every
        # downstream shape once `numpyro.validate_args`/`validation_enabled()`
        # actually calls `support.check(value)` -- e.g. from within
        # `MixtureGeneral.log_prob`, whose `@validate_sample` decorator masks
        # `log_prob` with exactly this wrongly-shaped array, turning a
        # `(400,)` result into a `(1, 400)` one (this is the root cause of the
        # "Cannot broadcast to shape with fewer dimensions" crash in
        # `ComponentMixtureModel`). Use `_IndependentGMMSupport` instead,
        # which normalizes `low`/`high` and `value` the same way
        # `component_log_probs` does and reduces over the D axis, so
        # `check(value)` always returns a shape matching `value`'s batch axes
        # (no leftover D or K axes).
        #
        # NOTE: this stays `_IndependentGMMSupport` even for D=1 -- do *not*
        # special-case D=1 to return a plain `constraints.interval`-family
        # object to try to type-match sibling `dist.TruncatedNormal`/
        # `TruncatedNormalGMMConditional` distributions in
        # `ComponentMixtureModel` (a `dist.MixtureGeneral` there requires
        # every component to expose the same `.support` *type*, see its
        # `__init__`). Doing so broke `test_gridgmm` in `test_gmm.py`: that
        # test (and, more importantly, `_IndependentGMMSupport`'s own
        # `event_dim=1` contract) calls `.support.check(value)` directly with
        # an explicit (N, D) `value` and expects a *reduced*, (N,)-shaped
        # mask back -- which only `_IndependentGMMSupport` provides (a plain
        # `constraints.interval` has `event_dim=0` and returns an unreduced
        # (N, D) mask instead). Callers that need a D=1 `IndependentGMM` to
        # look like a plain scalar (event_shape=(), interval-typed) sibling
        # distribution -- i.e. `ComponentMixtureModel._coord_eval_units`,
        # when a coordinate decomposed out of a joint `IndependentGMM` must
        # combine with such siblings -- should instead call
        # `to_scalar_mixture()`, which rebuilds an exactly equivalent
        # distribution out of genuine `dist.TruncatedNormal` components
        # (whose own native support already has the right type), rather than
        # relying on this property to change meaning based on D.
        low = -jnp.inf if self._low is None else self._low
        high = jnp.inf if self._high is None else self._high
        return _IndependentGMMSupport(low, high)

    def component_log_probs(self, value: ArrayLike) -> jax.Array:
        value = jnp.array(value)
        value = jnp.atleast_2d(value.T).T

        if value.shape[-1] != self._D:
            msg = (
                "The input array must have the same number of coordinate dimensions "
                f"as the distribution. Expected {self._D}, got {value.shape}."
            )
            raise ValueError(msg)

        tmp = jnp.expand_dims(value, self.mixture_dim)

        # Two distinct NaN-gradient hazards are handled here, both stemming from
        # the same root cause: real data can easily contain a point outside the
        # (still-converging, during early SVI steps) truncation bounds, and
        # naively computing log_prob there and masking the *output* with
        # `jnp.where` is not gradient-safe.
        #
        # (1) Evaluating TruncatedNormal.log_prob() directly at an out-of-bounds
        #     point gives the mathematically correct forward value (-inf), but
        #     its *gradient* w.r.t. loc/scale is NaN, and `jnp.where` does not
        #     protect against NaN gradients flowing through the branch it
        #     doesn't select. Fix: clip the input into the valid support (a
        #     "safe" placeholder that can never produce a NaN gradient) before
        #     calling log_prob, so it's never evaluated at an invalid point.
        # (2) `low`/`high` are shared across all K mixture components, so a
        #     point outside bounds in even one dimension is out-of-support for
        #     *every* component simultaneously. If we mask with a literal
        #     `-jnp.inf`, `log_prob`'s `logsumexp` below is then taken over a
        #     vector that is entirely -inf, which is a genuine 0/0 in
        #     logsumexp's softmax-gradient formula (NaN), independent of fix
        #     (1) above. Fix: mask with a large-but-finite sentinel instead of
        #     literal -inf, so it's numerically indistinguishable from zero
        #     probability but keeps logsumexp's gradient well-defined.
        # See: https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where
        low = jnp.asarray(-jnp.inf) if self._low is None else jnp.asarray(self._low)
        high = jnp.asarray(jnp.inf) if self._high is None else jnp.asarray(self._high)
        # `tmp` has shape (..., D, K) (K = number of mixture components,
        # broadcast via `mixture_dim=-1`). `low`/`high` describe a per-
        # dimension (D) bound that doesn't depend on K, so they need a
        # trailing size-1 axis to broadcast against `tmp`'s last axis. But
        # `self._low`/`self._high` may *already* carry that trailing axis:
        # `__init__` requires shape (D, 1) (not bare (D,)) for `low`/`high`
        # to broadcast correctly against `loc`/`scale`'s (D, K) shape when
        # constructing the underlying `TruncatedNormal` (a raw (D,) array
        # would wrongly align against the *K* axis there instead of *D*, and
        # fail unless D happened to equal K). So only append a new axis here
        # if `low`/`high` are still in bare 1-D (D,) form (e.g. if a caller
        # passes a plain list per the docstring) -- appending one
        # unconditionally double-adds an axis for the (D, 1) case already
        # required by construction, producing an unbroadcastable (D, 1, 1)
        # array (only surfaces once this GMM is evaluated with D > 1, e.g.
        # inside a `ComponentMixtureModel`, since a D=1 mismatch is masked by
        # broadcasting's leading-1 rule).
        low = low.reshape(low.shape + (1,)) if low.ndim == 1 else low
        high = high.reshape(high.shape + (1,)) if high.ndim == 1 else high
        safe_tmp = jnp.clip(tmp, low, high)
        component_log_probs = self.component_distribution.log_prob(safe_tmp)

        value = jnp.expand_dims(value, axis=-1)
        neg_inf_sentinel = jnp.asarray(-1e10, dtype=component_log_probs.dtype)
        return jnp.where(
            self.component_distribution.support.check(value),
            component_log_probs,
            neg_inf_sentinel,
        )

    def marginal(self, axis: int) -> "IndependentGMM":
        """The exact 1D marginal `IndependentGMM` along one axis.

        This is mathematically exact -- not an approximation -- because
        `IndependentGMM`'s components are diagonal (no cross-axis
        covariance): marginalizing any subset of axes out of a joint
        `IndependentGMM` gives another `IndependentGMM` over the remaining
        axis/axes, with the *same* mixing weights (the discrete "which
        component" latent is untouched by integrating out other axes) and
        each component's own loc/scale/bounds restricted to that axis.

        Used by `ComponentMixtureModel.__call__` when one component
        represents a set of coordinates as a single joint `IndependentGMM`
        (e.g. offtrack's joint `("phi1","phi2")` 2D grid) while a sibling
        component in the same mixture represents them as separate,
        string-keyed coordinates (e.g. bkg/stream's separate "phi1"/"phi2")
        -- each coordinate's own per-component mixture likelihood term still
        needs a single-axis distribution to combine with its siblings, so
        the joint component's contribution is this exact marginal instead.
        """
        low = None if self._low is None else jnp.asarray(self._low)[axis : axis + 1]
        high = None if self._high is None else jnp.asarray(self._high)[axis : axis + 1]
        return IndependentGMM(
            self.mixing_distribution,
            locs=self._locs[axis : axis + 1],
            scales=self._scales[axis : axis + 1],
            low=low,
            high=high,
            validate_args=False,
        )

    def to_scalar_mixture(self, vectorized: bool = True):
        """A scalar-event (`event_shape=()`) distribution equal to this D=1 `IndependentGMM`.

        `IndependentGMM` always has `event_shape=(self._D,)`, so even a D=1
        instance (e.g. one returned by `marginal(axis)`) has `event_shape=(1,)`,
        never a bare `()`. `numpyro.distributions.MixtureGeneral` requires
        every component distribution combined into it to agree exactly on
        `event_shape` *and* on `.support`'s type (see
        `ComponentMixtureModel._coord_eval_units`/`_harmonize_event_shapes` in
        `model.py`, which calls this method when a coordinate decomposed out
        of a joint `IndependentGMM` needs to be combined with sibling
        components' plain, scalar-event `dist.TruncatedNormal` or
        `TruncatedNormalGMMConditional` distributions for that same
        coordinate).

        This rebuilds an exactly equivalent distribution -- same mixing
        weights, same per-component loc/scale/bounds -- as a
        `dist.MixtureGeneral` over `K` individual scalar `dist.TruncatedNormal`
        components instead, which is genuinely scalar-event and whose
        `.support` is a plain `constraints.interval`-family constraint (the
        same type plain `TruncatedNormal` itself exposes), so it can be
        combined with those siblings.

        Only valid for a D=1 instance -- raises `ValueError` otherwise.

        With ``vectorized=True`` (default) this returns a
        `ScalarTruncatedNormalGMM`, which evaluates all K components in one
        vectorized computation (graph size independent of K). With
        ``vectorized=False`` it returns the original `dist.MixtureGeneral` of K
        separate `TruncatedNormal` objects (compile time grows with K).
        """
        if self._D != 1:
            msg = (
                "to_scalar_mixture() only applies to a D=1 IndependentGMM "
                f"(e.g. one axis of a marginal), but this instance has D={self._D}. "
                "Call `.marginal(axis)` first to reduce to a single axis."
            )
            raise ValueError(msg)

        loc = self._locs[0]  # (K,)
        scale = self._scales[0]  # (K,)
        low = (
            None
            if self._low is None
            else _drop_trailing_singleton_axis(jnp.asarray(self._low))
        )
        high = (
            None
            if self._high is None
            else _drop_trailing_singleton_axis(jnp.asarray(self._high))
        )
        bounds_kwargs = {}
        if low is not None:
            bounds_kwargs["low"] = low
        if high is not None:
            bounds_kwargs["high"] = high

        if vectorized:
            return ScalarTruncatedNormalGMM(
                self.mixing_distribution, loc, scale,
                low=bounds_kwargs.get("low"), high=bounds_kwargs.get("high"),
            )

        component_distributions = [
            dist.TruncatedNormal(loc=loc[k], scale=scale[k], **bounds_kwargs)
            for k in range(self._K)
        ]
        return dist.MixtureGeneral(self.mixing_distribution, component_distributions)

    def log_prob(self, value: ArrayLike) -> jax.Array:
        comp_lp = self.component_log_probs(value)
        return logsumexp(
            jax.nn.log_softmax(self.mixing_distribution.logits)
            + comp_lp.sum(axis=self._dim_dim),
            axis=self.mixture_dim,
        )

    def component_sample(
        self, key: jax.Array, sample_shape: tuple = ()
    ) -> jax.Array:
        return self.component_distribution.sample(
            key,
            sample_shape=sample_shape,  # + self.event_shape
        )

    # def sample_with_intermediates(
    #     self, key: jax.random.PRNGKey, sample_shape: tuple = ()
    # ) -> tuple:
    #     """
    #     A version of ``sample`` that also returns the sampled component indices

    #     Parameters
    #     ----------
    #     key
    #         The rng_key key to be used for the distribution.
    #     sample_shape
    #         The sample shape for the distribution.

    #     Returns
    #     -------
    #     samples
    #         The samples from the distribution.
    #     indices
    #         The indices of the sampled components.
    #     """
    #     key_comp, key_ind = jax.random.split(key)
    #     samples = self.component_sample(key_comp, sample_shape=sample_shape)

    #     # Sample selection indices from the categorical (shape will be sample_shape)
    #     indices = self.mixing_distribution.expand(
    #         sample_shape + self.batch_shape
    #     ).sample(key_ind)
    #     indices_expanded = indices.reshape(indices.shape + (1,))

    #     # Select samples according to indices samples from categorical
    #     samples_selected = jnp.take_along_axis(
    #         samples, indices=indices_expanded, axis=-2
    #     )
    #     samples_selected = jnp.squeeze(samples_selected, axis=-1)

    #     return samples_selected, indices

    # def sample(self, key, sample_shape=()):
    #     return self.sample_with_intermediates(key=key, sample_shape=sample_shape)[0]
