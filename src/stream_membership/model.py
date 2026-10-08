__all__ = ["ModelMixin", "ModelComponent", "ComponentMixtureModel", "FromDist"]

import copy
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.axes as mpl_axes
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jax_ext.integrate import ln_simpson
from numpyro.handlers import seed

from ._typing import CoordinateName
from .distributions.gmm import IndependentGMM
from .plot import _plot_projections
from .utils import get_coord_from_data_dict


@dataclass(frozen=True)
class FromDist:
    """A marker for use as a value in ``coord_parameters[coord][arg]``.

    Instead of being sampled fresh (like a raw value, a `dist.Distribution`,
    or a `numpyro.sample(**val)` dict would be), this tells `make_dists` to
    reuse the *already-constructed* distribution object for another
    coordinate of the same component, built earlier in the same
    `make_dists()` call.

    This is intra-component, cross-coordinate parameter sharing -- e.g. a
    "pm1|phi1" conditional coordinate reusing the exact sampled mixture
    weights/locs/scales of a "phi1" marginal coordinate's distribution
    object, rather than resampling an independent copy of them. (This is
    different from `ComponentMixtureModel`'s `tied_coordinates`, which
    shares a whole coordinate's distribution *across components*.)

    Parameters
    ----------
    coord_name
        The name of the sibling coordinate to pull the distribution object
        from. That coordinate must appear *earlier* than the coordinate
        using this marker in `coord_distributions` (dict insertion order
        controls the build order in `make_dists`), or a `KeyError` is
        raised.
    """

    coord_name: CoordinateName


def _extract_coord_marginal(
    dist_obj: dist.Distribution, container_key: CoordinateName, target_name: str
) -> dist.Distribution:
    """Pull `target_name`'s own exact marginal distribution out of `dist_obj`.

    Used by `ComponentMixtureModel._coord_eval_units` when sibling
    components disagree on how a coordinate is grouped -- e.g. offtrack
    represents ("phi1","phi2") as a single joint `IndependentGMM`, while
    bkg/stream instead have separate "phi1"/"phi2" distributions. In that
    case, offtrack's contribution to the (decomposed, per-coordinate)
    "phi1" evaluation unit must be *just* the phi1-axis marginal of its
    joint (phi1,phi2) distribution, not the whole joint object.

    If `container_key == target_name` (a plain string key), `dist_obj`
    already *is* that coordinate's own distribution, so it's returned
    unchanged -- no decomposition needed.

    Otherwise `container_key` must be a tuple containing `target_name`, and
    `dist_obj` must implement `.marginal(axis)` (currently only
    `IndependentGMM` does) returning the exact analytical marginal along
    that axis -- this is mathematically exact (not an approximation)
    because `IndependentGMM`'s components have no cross-axis covariance.
    """
    if container_key == target_name:
        return dist_obj

    if not isinstance(container_key, tuple) or target_name not in container_key:
        msg = (
            f"Coordinate {target_name!r} not found in container key "
            f"{container_key!r}."
        )
        raise ValueError(msg)

    if not hasattr(dist_obj, "marginal"):
        msg = (
            f"A sibling component represents {target_name!r} as its own "
            f"coordinate, but this component only has it bundled into the "
            f"joint distribution for {container_key!r} (type "
            f"{type(dist_obj).__name__}), which does not implement "
            f"`.marginal(axis)` to extract {target_name!r}'s own exact "
            f"marginal. Either give this component its own separate "
            f"{target_name!r} coordinate, or implement `.marginal()` on "
            f"{type(dist_obj).__name__}."
        )
        raise TypeError(msg)

    axis = container_key.index(target_name)
    return dist_obj.marginal(axis)


# Whether decomposed D=1 `IndependentGMM`s are converted to the vectorized
# `ScalarTruncatedNormalGMM` (compile time independent of the number of GMM
# components K) or the original `dist.MixtureGeneral` of K separate
# `TruncatedNormal`s (graph size grows with K). Set to False to restore the old
# behavior, e.g. ``stream_membership.model.VECTORIZED_SCALAR_MIXTURE = False``
# before building/running the model.
VECTORIZED_SCALAR_MIXTURE = True


def _harmonize_event_shapes(
    dists_by_component: dict[str, dist.Distribution],
) -> dict[str, dist.Distribution]:
    """Make every component's distribution for one decomposed coordinate agree.

    `dist.MixtureGeneral` (used in `ComponentMixtureModel.__call__` to
    combine one coordinate's per-component distributions) requires every
    component distribution to share the exact same `event_shape` (in
    addition to the same `.support` type, already handled by
    `IndependentGMM.support`'s D=1 special-case). A decomposed coordinate
    (see `_coord_eval_units`) can end up with a mix of:

    - D=1 `IndependentGMM`s, `event_shape=(1,)` -- either a component's own
      literal distribution for this coordinate (e.g. bkg/stream's own
      "phi1"), or one produced by `_extract_coord_marginal`'s
      `.marginal(axis)` call on a sibling's joint distribution.
    - Plain scalar distributions, `event_shape=()` -- e.g. a component's own
      `dist.TruncatedNormal` or `TruncatedNormalGMMConditional` for this
      coordinate (which deliberately mimics `TruncatedNormal`'s support
      type, but is still scalar-event).

    If *every* component's distribution for this coordinate is a D=1
    `IndependentGMM`, they already agree (all `event_shape=(1,)`) and
    nothing needs to change -- this is the existing, already-working case
    (e.g. "phi1" decomposed only because a sibling groups it into a joint
    tuple, while every component's *own* phi1 representation is still an
    `IndependentGMM`). Otherwise, every `IndependentGMM` among them is
    converted to its scalar-event equivalent
    (`IndependentGMM.to_scalar_mixture`) so all components agree on plain
    scalar semantics.
    """
    if all(isinstance(d, IndependentGMM) for d in dists_by_component.values()):
        return dists_by_component

    return {
        component_name: (
            d.to_scalar_mixture(vectorized=VECTORIZED_SCALAR_MIXTURE)
            if isinstance(d, IndependentGMM)
            else d
        )
        for component_name, d in dists_by_component.items()
    }


class ModelMixin:
    """
    Generic functionality for model component and mixture model, like evaluating on
    grids and plotting
    """

    def _get_grids_2d(
        self,
        grids_1d: dict[str, ArrayLike],
        grid_coord_names: list[tuple[str, str]],
    ) -> dict[tuple[str, str], list[jax.Array]]:
        """
        Takes a dictionary of 1D grids and returns a dictionary of 2D grids for each
        pair of coordinates in grid_coord_names, which will be used to evaluate and plot
        the model in 2D projections.
        """
        grids_2d = {}
        for name_pair in grid_coord_names:
            for name in name_pair:
                if name not in grids_1d:
                    msg = (
                        f"You must specify a 1D grid for the component '{name}' via "
                        "the grids_1d argument"
                    )
                    raise ValueError(msg)
            grids_2d[name_pair] = jnp.meshgrid(*[grids_1d[name] for name in name_pair])

        return grids_2d

    @abstractmethod
    def evaluate_on_2d_grids(
        self, pars, grids, grid_coord_names, x_coord_name
    ) -> tuple[
        dict[tuple[str, str], list[jax.Array]], dict[tuple[str, str], jax.Array]
    ]:
        pass

    def plot_model_projections(
        self,
        pars: dict[str, Any],
        grids: dict[str, ArrayLike],
        ndata: dict[str, Any] = None, #for scaling to data when making data+model+residual plots
        grid_coord_names: list[tuple[str, str]] | None = None,
        x_coord_name: str | None = None,
        axes: mpl_axes.Axes | None = None,
        label: bool = True,
        pcolormesh_kwargs: dict | None = None,
    ):
        """
        Plot the model evaluated on 2D grids.

        Parameters
        ----------
        data
            A dictionary of data arrays, where the keys are the names of the coordinates
            in the model component.
        pars
            A dictionary of parameter values for the model component.
        grids
            A dictionary of 1D grids for each coordinate in the model component. The
            keys should be the names of the coordinates you want to evaluate the model
            on, and must always contain the x coordinate.
        grid_coord_names
            A list of tuples of coordinate names to evaluate the model on. The default
            is to pair the x coordinate with each other coordinate in the model
            component. For example, if the model component has coordinates "phi1",
            "phi2", and "pm1", the default grid_coord_names would be [("phi1", "phi2"),
            ("phi1", "pm1")].
        x_coord_name
            The name of the x coordinate to use for evaluating the model. If None, the
            default x coordinate will be used, which is taken to be the 0th coordinate
            name in the specified "coord_distributions".
        axes
            A matplotlib axes object to plot the residuals on. If None, a new figure and
            axes will be created.
        label
            Whether to add labels to the axes.
        pcolormesh_kwargs
            Keyword arguments to pass to the matplotlib.pcolormesh() function.
        ndata
            Either a single scalar (used to scale every coordinate pair's model
            density into expected counts), or a dict to look up a per-pair/per-
            coordinate scalar from -- keyed either by the full `name_pair` tuple
            (e.g. `("phi1", "pm1")`) or by just the non-x coordinate name (e.g.
            `"pm1"`). A dict is required when calling this with multiple
            `grid_coord_names` at once and the coordinates don't all have the
            same number of valid stars (e.g. under a "ragged" per-coordinate
            `data` dict, where some coordinates have fewer valid measurements
            than others).
        """
        grids, ln_ps = self.evaluate_on_2d_grids(
            pars=pars,
            grids=grids,
            grid_coord_names=grid_coord_names,
            x_coord_name=x_coord_name,
        )

        if ndata is None:
            ims = {k: np.exp(v) for k, v in ln_ps.items()}

        else:
            # Compute the bin area for each 2D grid cell for a cheap integral...
            bin_area = {
                k: np.abs(np.diff(grid1[0])[None] * np.diff(grid2[:, 0])[:, None])
                for k, (grid1, grid2) in grids.items()
            }

            def _ndata_for(name_pair):
                if not isinstance(ndata, Mapping):
                    return ndata
                if name_pair in ndata:
                    return ndata[name_pair]
                if name_pair[1] in ndata:
                    return ndata[name_pair[1]]
                msg = (
                    f"No ndata entry found for coordinate pair {name_pair!r} (or "
                    f"its non-x coordinate {name_pair[1]!r}). Pass either a single "
                    "scalar ndata to use for every pair, or a dict keyed by the "
                    "pair or by the non-x coordinate name."
                )
                raise KeyError(msg)

            ln_ns = {
                k: ln_p + np.log(_ndata_for(k)) + np.log(bin_area[k])
                for k, ln_p in ln_ps.items()
            }
            ims = {k: np.exp(v) for k, v in ln_ns.items()}

        return _plot_projections(
            grids=grids,
            ims=ims,
            axes=axes,
            label=label,
            pcolormesh_kwargs=pcolormesh_kwargs,
        )

    def plot_residual_projections(
        self,
        data: dict[str, Any],
        pars: dict[str, Any],
        grids: dict[str, ArrayLike],
        grid_coord_names: list[tuple[str, str]] | None = None,
        x_coord_name: str | None = None,
        axes: mpl_axes.Axes | None = None,
        label: bool = True,
        pcolormesh_kwargs: dict | None = None,
        smooth: int | float | None = 1.0,
        x_data: dict[str, Any] | None = None,
    ):
        """
        Plot the residuals of the model evaluated on 2D grids compared to the input
        data, binned into the same 2D grids.

        Parameters
        ----------
        data
            A dictionary of data arrays, where the keys are the names of the coordinates
            in the model component. Coordinates are allowed to be "ragged" (i.e. not
            all the same length) -- for example, if some stars are missing a
            radial_velocity measurement and were dropped for that coordinate rather
            than sentinel-filled.
        pars
            A dictionary of parameter values for the model component.
        grids
            A dictionary of 1D grids for each coordinate in the model component. The
            keys should be the names of the coordinates you want to evaluate the model
            on, and must always contain the x coordinate.
        grid_coord_names
            A list of tuples of coordinate names to evaluate the model on. The default
            is to pair the x coordinate with each other coordinate in the model
            component. For example, if the model component has coordinates "phi1",
            "phi2", and "pm1", the default grid_coord_names would be [("phi1", "phi2"),
            ("phi1", "pm1")].
        x_coord_name
            The name of the x coordinate to use for evaluating the model. If None, the
            default x coordinate will be used, which is taken to be the 0th coordinate
            name in the specified "coord_distributions".
        axes
            A matplotlib axes object to plot the residuals on. If None, a new figure and
            axes will be created.
        label
            Whether to add labels to the axes.
        pcolormesh_kwargs
            Keyword arguments to pass to the matplotlib.pcolormesh() function.
        smooth
            The standard deviation of the Gaussian kernel to use for smoothing the
            residuals. If None, no smoothing is applied.
        x_data
            Optional per-coordinate override for the x-axis array used both for
            binning the data histogram and for normalizing the model counts (the
            "N_data" used to convert the model's probability density into expected
            counts). Callers with "ragged" `data` (see above) should pass a dict
            mapping each non-x coordinate name -> the x-array that is actually
            aligned, element-for-element, with `data[coord_name]` (e.g. that
            coordinate's own valid-star phi1 subsample). If a coordinate is missing
            from `x_data` (or `x_data` is None), this falls back to `data[x_name]`
            directly, which is only correct when every coordinate's array shares a
            common length/ordering.

        """
        from scipy.ndimage import gaussian_filter

        grids_2d, ln_ps = self.evaluate_on_2d_grids(
            pars=pars,
            grids=grids,
            grid_coord_names=grid_coord_names,
            x_coord_name=x_coord_name,
        )

        # Compute the bin area for each 2D grid cell for a cheap integral...
        bin_area = {
            k: np.abs(np.diff(grid1[0])[None] * np.diff(grid2[:, 0])[:, None])
            for k, (grid1, grid2) in grids_2d.items()
        }

        resid_ims = {}
        resid = None  # kept as the last pair's *pre-smoothing* residual, matching
        # the original (pre-ragged) implementation's percentile-based vmin/vmax
        for name_pair, ln_p in ln_ps.items():
            x_vals = (
                x_data[name_pair[1]]
                if x_data is not None and name_pair[1] in x_data
                else data[name_pair[0]]
            )
            y_vals = data[name_pair[1]]

            # Each coordinate pair gets its own N_data (rather than one shared
            # N_data for every pair) since, under a ragged `data` dict, different
            # coordinates can have different numbers of valid stars.
            N_data = len(y_vals)

            ln_n = ln_p + np.log(N_data) + np.log(bin_area[name_pair])
            model_im = np.exp(ln_n)

            # get the number density: density=True is the prob density, so need to
            # multiply back in the total number of data points
            H_data, *_ = np.histogram2d(
                x_vals,
                y_vals,
                bins=(grids[name_pair[0]], grids[name_pair[1]]),
            )
            data_im = H_data.T

            resid = model_im - data_im
            resid_ims[name_pair] = resid

            if smooth is not None:
                resid_ims[name_pair] = gaussian_filter(resid_ims[name_pair], smooth)

        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {}
        pcolormesh_kwargs.setdefault("cmap", "coolwarm_r")
        # TODO: based on residuals of last coordinate pair, but should use all residuals
        v = np.abs(np.nanpercentile(resid, [1, 99])).max()
        pcolormesh_kwargs.setdefault("vmin", -v)
        pcolormesh_kwargs.setdefault("vmax", v)

        return _plot_projections(
            grids=grids_2d,
            ims=resid_ims,
            axes=axes,
            label=label,
            pcolormesh_kwargs=pcolormesh_kwargs,
        )


@jax.tree_util.register_pytree_node_class
class CoordinateMapping(dict):
    """Note: This is needed because we use a mix of tuples and strings as keys in the
    field dictionaries in ModelComponent below.
    See: https://github.com/jax-ml/jax/issues/15358

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tree_flatten(self):
        sorted_keys = sorted(
            self.keys(), key=lambda k: k[0] if isinstance(k, tuple) else k
        )
        return (tuple(self[k] for k in sorted_keys), sorted_keys)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tmp = cls()
        for k, v in zip(aux_data, children, strict=True):
            tmp[k] = v
        return tmp


class ModelComponent(eqx.Module, ModelMixin):
    """
    Creating evaluating the different components of the density model.

    :param name:
        the name of the model component (usually either 'background', 'stream', or 'offtrack')
    :type name: str

    :param coord_distributions:
        a dictionary of the distributions of the component parameters.
        The keys are the names of the component parameters
        (e.g. `'phi1'`, `'phi2'`, `'mu_phi1'`, `'mu_phi2'`, `('phi1', 'phi2')`, etc.)
    :type coord_distributions: dict[str | tuple, Any]

    :param coord_parameters:
        a dictionary of the parameters of the distributions in `coord_distributions`.
        The keys are the names of the component parameters (the keys in `coord_distributions`)
        and the values are dictionaries containing the parameters for the distributions.
        For example, a truncated normal distribution (`dist.TruncatedNormal` in numpyro)
        might have the parameters loc, scale, low, and high.
    :type coord_parameters: dict[str | tuple, dict[str, dist.Distribution | tuple | ArrayLike | dict]]

    :param default_x_coord:
        (optional) the default x-coordinate for the model component. default=None
    :type default_x_coord: str | None

    :param conditional_data:
        (optional) a dictionary of any additional data that is required for evaluating the
        log-probability of a coordinate's probability distribution. For example, a
        spline-enabled distribution might require the phi1 data to evaluate the spline
        at the phi1 values.
        The keys are the names of the component parameters
        (e.g. `'phi1'`, `'phi2'`, `'mu_phi1'`, `'mu_phi2'`, `('phi1', 'phi2')`, etc.)
        and the values are dictionaries of the conditional data for those parameters. default=None
    :type conditional_data: dict[CoordinateName, dict[str, str]]

    :attr:`_coord_names`
        the names of the component parameters (the keys in `coord_distributions` and `coord_parameters`)
    :type _coord_names: list[str]

    :attr:`_sample_order`
        the order in which the component parameters should be sampled
    :type _sample_order: list[CoordinateName]
    """

    name: str
    coord_distributions: dict[CoordinateName, Any]
    coord_parameters: dict[
        CoordinateName, dict[str, dist.Distribution | tuple | ArrayLike | dict]
    ]
    default_x_coord: str | None = None
    conditional_data: dict[CoordinateName, dict[str, str]] = eqx.field(default=None)
    _coord_names: list[str] = eqx.field(init=False)
    _sample_order: list[CoordinateName] = eqx.field(init=False)

    def __init__(
        self,
        name,
        coord_distributions,
        coord_parameters,
        default_x_coord=None,
        conditional_data=None,
    ):
        self.name = name
        self.coord_distributions = CoordinateMapping(coord_distributions)
        self.coord_parameters = CoordinateMapping(coord_parameters)
        self.default_x_coord = default_x_coord
        # NOTE: conditional_data can also have coordinate names (str or tuple) as
        # keys (e.g. a joint coordinate like ("bp_rp", "phot_g_mean_mag") may itself
        # require conditional data). It must be wrapped in the same pytree-safe
        # CoordinateMapping as coord_distributions/coord_parameters above --
        # otherwise, as soon as any tuple keys are mixed in with string keys, JAX's
        # default dict pytree flattening (which sorts keys with the builtin `<`)
        # raises a TypeError when this Module is flattened (e.g. whenever a bound
        # method of this instance, such as self._make_sample_order, is accessed,
        # since equinox flattens `self` to construct the BoundMethod pytree). See
        # CoordinateMapping's docstring / https://github.com/jax-ml/jax/issues/15358
        self.conditional_data = CoordinateMapping(
            conditional_data if conditional_data is not None else {}
        )

        # def __post_init__(self):
        # Validate that the keys (i.e. coordinate names) in coord_distributions and
        # coord_parameters are the same
        if set(self.coord_distributions.keys()) != set(self.coord_parameters.keys()):
            msg = "Keys in coord_distributions and coord_parameters must match"
            raise ValueError(msg)

        if self.default_x_coord is None:
            self.default_x_coord = next(iter(self.coord_distributions.keys()))
            if not isinstance(self.default_x_coord, str):
                self.default_x_coord = self.default_x_coord[0]

        self._coord_names = []
        for coord_name in self.coord_distributions:
            if isinstance(coord_name, tuple):
                self._coord_names.extend(coord_name)
            else:
                self._coord_names.append(coord_name)

        # This is used to specify any extra data that is required for evaluating the
        # log-probability of a coordinate's probability distribution. For example, a
        # spline-enabled distribution might require the phi1 data to evaluate the spline
        # at the phi1 values
        if self.conditional_data is None:
            self.conditional_data = {}

        # Validate that there are no circular dependencies:
        _pairs = []
        for coord_name in self.coord_distributions:
            for val in self.conditional_data.get(coord_name, {}).values():
                _pairs.append((coord_name, val))
        for _pair in _pairs:
            if _pair[::-1] in _pairs:
                msg = f"Circular dependency: {_pair}"
                raise ValueError(msg)
        self._sample_order = self._make_sample_order()

        # Validate that coordinate names can't have "-" and component, coordinate, and
        # parameter names can't have ":"
        if any("-" in name for name in self._coord_names):
            msg = "Coordinate names can't contain '-'"
            raise ValueError(msg)

        if any(":" in name for name in self._coord_names) or ":" in self.name:
            msg = "Coordinate names and component names can't contain ':'"
            raise ValueError(msg)

    @property
    def coord_names(self):
        return self._coord_names

    def _make_numpyro_name(
        self, coord_name: CoordinateName, arg_name: str | None = None
    ) -> str:
        """
        Convert a nested set of component name (this class name), coordinate name, and
        parameter name into a single string for naming a parameter with
        numpyro.sample().

        Parameters
        ----------
        coord_name
            The name of the coordinate in the component. If a coordinate can only be
            modeled as a joint, pass a tuple of strings.
        arg_name
            The name of the parameter used in the model component for the coordinate.
        """
        if isinstance(coord_name, tuple):
            coord_name = "-".join(coord_name)

        name = f"{self.name}:{coord_name}"
        if arg_name is None:
            return name
        return f"{name}:{arg_name}"

    def _expand_numpyro_name(self, numpyro_name: str) -> tuple[str, str | tuple, str]:
        """
        Convert a numpyro name into a tuple of component name, coordinate name, and
        parameter name.

        Parameters
        ----------
        numpyro_name
            The name of the parameter in the numpyro model, i.e. the name of a parameter
            specified with numpyro.sample(). In the context of this model, this should
            be something like "background:phi2:loc", where the model is named
            "background", the coordinate is named "phi2", and the parameter is named
            "loc".
        """
        bits = numpyro_name.split(":")
        return (
            bits[0],
            tuple(bits[1].split("-")) if "-" in bits[1] else bits[1],
            bits[2],
        )

    def pack_params(self, pars: dict[CoordinateName, Any]) -> dict[str, Any]:
        """
        Convert a dictionary of parameters as a nested dictionary into a flat dictionary
        with packed numpyro-compatible names.

        Parameters
        ----------
        pars
            A dictionary of parameters where the keys are the names of the coordinates
            in the model component and the values are dictionaries of the parameters for
            each coordinate.
        """
        packed_pars: dict[str, Any] = {}

        for coord_name, sub_pars in pars.items():
            for arg_name, val in sub_pars.items():
                numpyro_name = self._make_numpyro_name(coord_name, arg_name)
                packed_pars[numpyro_name] = val

        return packed_pars

    def expand_numpyro_params(
        self, pars: dict[str, Any], skip_invalid: bool = False
    ) -> dict[str | tuple, Any]:
        """
        Convert a dictionary of numpyro parameters into a nested dictionary where the
        keys are the coordinate names and parameter name.

        Parameters
        ----------
        pars
            A dictionary of numpyro parameters where the keys are the names of the
            parameters created with numpyro.sample().
        """
        expanded_pars: dict[str, dict] = {}
        for k, v in pars.items():
            try:
                name, coord_name, arg_name = self._expand_numpyro_name(k)
            except Exception as e:
                if not skip_invalid:
                    raise e
                continue

            if name not in expanded_pars:
                expanded_pars[name] = {}
            if coord_name not in expanded_pars[name]:
                expanded_pars[name][coord_name] = {}
            expanded_pars[name][coord_name][arg_name] = v

        return expanded_pars[name]

    def make_dists(
        self,
        pars: dict[CoordinateName, Any] | None = None,
        dists: dict[CoordinateName, dist.Distribution] | None = None,
        ) -> dict[str | tuple, Any]:

        """
        Make a dictionary of distributions for each coordinate in the component.

        Parameters
        ----------
        pars
            A dictionary of parameters to pass to the numpyro.sample() calls that
            create the distributions. The dictionary should be structured as follows:
            {
                "coord_name": {
                    "arg_name": value
                }
            }
            where "coord_name" is the name of the coordinate and "arg_name" is the name
            of the argument to pass to the numpyro.sample() call that creates the
            distribution. "value" is the value to pass to the numpyro.sample() call.
        """

        pars = pars if pars is not None else {}
        dists = dists if dists is not None else {}

        for coord_name, Distribution in self.coord_distributions.items():
            kwargs = {}

            if coord_name in dists:
                continue

            for arg, val in self.coord_parameters.get(coord_name, {}).items():
                numpyro_name = self._make_numpyro_name(coord_name, arg)

                # Note: passing in a tuple as a value is a way to wrap the value in a
                # function or outer distribution, for example for a mixture model
                if isinstance(val, tuple) and callable(val[0]):
                    wrapper, val = val  # noqa: PLW2901
                else:
                    wrapper = lambda x: x  # noqa: E731

                if isinstance(val, FromDist):
                    # Reuse an already-built sibling coordinate's distribution
                    # object (see `FromDist`'s docstring) instead of sampling
                    # anything -- this never creates a numpyro sample site, so
                    # `val.coord_name` must have already been built earlier in
                    # this same `make_dists()` call (i.e. it must come earlier
                    # in `coord_distributions`' insertion order).
                    if val.coord_name not in dists:
                        msg = (
                            f"Coordinate {coord_name!r} references "
                            f"{val.coord_name!r} via FromDist, but "
                            f"{val.coord_name!r} hasn't been built yet in this "
                            "make_dists() call -- it must appear earlier than "
                            f"{coord_name!r} in coord_distributions."
                        )
                        raise KeyError(msg)
                    par = dists[val.coord_name]
                elif arg in pars.get(coord_name, {}):
                    # If an argument is passed in the pars dictionary, use that value.
                    # This is useful, for example, for constructing the coordinate
                    # distributions once a model is optimized or sampled, so you can
                    # pass in parameter values to evaluate the model.
                    par = pars[coord_name][arg]
                elif isinstance(val, dict):
                    par = numpyro.sample(numpyro_name, **val)
                elif isinstance(val, dist.Distribution):
                    par = numpyro.sample(numpyro_name, val)
                else:
                    par = val
                kwargs[arg] = wrapper(par)

            dists[coord_name] = Distribution(**kwargs)

        return {k: dists[k] for k in self.coord_distributions}

    def _make_conditional_data(
        self, data: dict[str, ArrayLike]
    ) -> dict[CoordinateName, dict]:
        conditional_data: dict[CoordinateName, dict] = {}
        for coord_name in self.coord_distributions:
            data_map = self.conditional_data.get(coord_name, {})

            conditional_data[coord_name] = {}
            for key, val in data_map.items():
                # NOTE: behavior - if key is missing from data, we pass None
                conditional_data[coord_name][key] = get_coord_from_data_dict(val, data)

        return conditional_data

    def _make_sample_order(self) -> list[CoordinateName]:
        sample_order = []

        conditional_data = {
            k: list(set(v.values())) for k, v in self.conditional_data.items()
        }

        # First, any coord or coord pair not in conditional_data can be done first:
        for coord_name in self.coord_distributions:
            if coord_name not in self.conditional_data:
                sample_order.append(coord_name)
                conditional_data.pop(coord_name, None)

        for _ in range(128):  # NOTE: max 128 iterations
            flat_sample_order = list(
                chain(*[(s,) if isinstance(s, str) else s for s in sample_order])
            )
            for coord_name, dependencies in conditional_data.items():
                if all(dep in flat_sample_order for dep in dependencies):
                    sample_order.append(coord_name)
                    conditional_data.pop(coord_name)
                    break

            if len(conditional_data) == 0:
                break

        else:
            msg = "Circular dependency likely in conditional_data"
            raise ValueError(msg)

        return sample_order

    def __call__(
        self, data: dict[str, ArrayLike], err: dict[str, ArrayLike] | None = None
    ) -> None:
        """
        This sets up the model component in numpyro.
        """
        if err is None:
            err = {}

        dists = self.make_dists()
        for coord_name, dist_ in dists.items():
            if isinstance(coord_name, tuple):
                _data = jnp.stack([data[k] for k in coord_name], axis=-1)
                _data_err = None  # TODO: we don't support errors for joint coordinates
            else:
                _data = jnp.asarray(data[coord_name])
                _data_err = err.get(coord_name, None)
            
            numpyro_name = self._make_numpyro_name(coord_name)
            if _data_err is not None:
                sample_shape = (_data.shape[0],) if dist_.batch_shape == () else ()
                model_val = numpyro.sample(
                    f"{numpyro_name}:modeldata", dist_, sample_shape=sample_shape
                )

                # NOTE: the plate name must be unique per coordinate (not a
                # shared literal "data"), because this loop can execute the
                # `with numpyro.plate(...)` block multiple times per model
                # call -- once for each ragged coordinate that has an error
                # term (e.g. pm1, pm2, radial_velocity), each with its own,
                # generally different, N. Reusing the same plate name across
                # those separate blocks causes numpyro's `initialize_model`/
                # `AutoNormal._setup_prototype` machinery to broadcast a
                # later coordinate's `obs=_data` against an earlier
                # coordinate's (differently-sized) plate frame, raising
                # `ValueError: Incompatible shapes for broadcasting`. This
                # was verified with a minimal repro: two ragged coordinates
                # (sizes 100 and 50) sharing a plate named "data" reproduce
                # exactly this error in `svi.init`; giving each coordinate
                # its own plate name (matching the
                # `f"data-{numpyro_name}"` convention already used in
                # `ComponentMixtureModel.__call__` below) fixes it.
                with numpyro.plate(f"data-{numpyro_name}", _data.shape[0]):
                    numpyro.sample(
                        f"{numpyro_name}-obs",
                        dist.Normal(model_val, _data_err),
                        obs=_data,
                    )
            else:
                numpyro.sample(f"{numpyro_name}-obs", dist_, obs=_data)

            # TODO: what to do if user wants to model number density?
            # Compute the log of the effective volume integral, used in the poisson
            # process likelihood
            # ln_n = obj.ln_number_density(data)
            # numpyro.factor(f"{cls.name}-factor-V", -obj.get_N())
            # numpyro.factor(f"{cls.name}-factor-ln_n", ln_n.sum())
            # numpyro.factor(f"{cls.name}-factor-extra_prior", obj.extra_ln_prior(pars))

    def sample(
        self,
        key: jax.Array,
        sample_shape: Any = (),
        pars: dict[CoordinateName, Any] | None = None,
        dists: dict[CoordinateName, dist.Distribution] | None = None,
    ) -> dict[CoordinateName, jax.Array]:
        """
        Sample from the model component. If no parameters `pars` are passed, this will
        sample from the prior. All of the coordinate distributions must be sample-able
        in order for this to work.

        Parameters
        ----------
        key
            A JAX random key.
        sample_shape (optional)
            The shape of the samples to draw.
        pars (optional)
            A dictionary of parameters for the model component.
        """
        if pars is None:
            dists_ = seed(self.make_dists, key)(dists=dists)
        else:
            dists_ = self.make_dists(pars=pars, dists=dists)

        keys = jax.random.split(key, len(self.coord_distributions))

        samples: dict[CoordinateName, jax.Array] = {}
        for coord_name, key_ in zip(self._sample_order, keys, strict=True):
            extra_data = self._make_conditional_data(samples)
            shape = sample_shape if len(extra_data[coord_name]) == 0 else ()
            samples[coord_name] = dists_[coord_name].sample(
                key_, shape, **extra_data[coord_name]
            )

        return {k: samples[k] for k in self.coord_distributions}

    def evaluate_on_2d_grids(
        self,
        pars: dict[str, Any],
        grids: dict[str, ArrayLike],
        grid_coord_names: list[tuple[str, str]] | None = None,
        x_coord_name: str | None = None,
        dists: dict[CoordinateName, dist.Distribution] | None = None,
    ):
        """
        Evaluate the log-density of the model on 2D grids of coordinates paired with the
        same x coordinate. For example, in the context of a stream model, the x
        coordinate would likely be the "phi1" coordinate.

        Parameters
        ----------
        pars
            A dictionary of parameter values for the model component.
        grids
            A dictionary of 1D grids for each coordinate in the model component. The
            keys should be the names of the coordinates you want to evaluate the model
            on, and must always contain the x coordinate.
        grid_coord_names
            A list of tuples of coordinate names to evaluate the model on. The default
            is to pair the x coordinate with each other coordinate in the model
            component. For example, if the model component has coordinates "phi1",
            "phi2", and "pm1", the default grid_coord_names would be [("phi1", "phi2"),
            ("phi1", "pm1")].
        x_coord_name
            The name of the x coordinate to use for evaluating the model. If None, the
            default x coordinate will be used, which is taken to be the 0th coordinate
            name in the specified "coord_distributions".
        """
        x_coord_name = self.default_x_coord if x_coord_name is None else x_coord_name

        if x_coord_name not in self._coord_names:
            msg = f"{x_coord_name} is not a valid coordinate name"
            raise ValueError(msg)

        if grid_coord_names is None:
            # Pair the x coordinate with each other coordinate in the model component
            grid_coord_names = [
                (x_coord_name, coord_name) for coord_name in self._coord_names[1:]
            ]

        for name_pair in grid_coord_names:
            if name_pair[0] != x_coord_name:
                # TODO: we could make this more general, but then some logic below needs
                # to become more general
                msg = (
                    "We currently only support evaluating on 2D grids with the same x "
                    "coordinate axis for all grids"
                )
                raise ValueError(msg)

        # validate grid_coord_names
        for name_pair in grid_coord_names:
            for name in name_pair:
                if name not in self._coord_names:
                    msg = f"{name} is not a valid coordinate name"
                    raise ValueError(msg)

        # First we have to check if the model component for the x coordinate is in a
        # joint distribution with another coordinate. If it is, we need to evaluate the
        # joint distribution on the grid and compute the marginal distribution for x --
        # which requires that joint pair's own 2D grid below, *regardless* of whether
        # the caller's `grid_coord_names` happens to include it. For example, calling
        # this with `grid_coord_names=[("phi1", "pm1")]` on a component where "phi1" is
        # only defined jointly as ("phi1", "phi2") (e.g. the offtrack component's
        # spatial term) still needs the ("phi1", "phi2") grid to compute the phi1
        # marginal.
        x_joint_name_pair = None
        if x_coord_name not in self.coord_distributions:
            # At this point, x_coord_name is definitely a valid coord name, but it
            # doesn't exist as a string key in coord_distributions - it must be in a
            # joint:
            for x_joint_name_pair in self.coord_distributions:
                if x_coord_name in x_joint_name_pair:
                    break

        # NOTE: `grids_2d` (below) is returned to the caller and is what downstream
        # plotting code (`plot.py::_plot_projections`, via `plot_model_projections`)
        # iterates over via `list(grids.keys())` to decide what to actually plot -- so
        # it must contain *only* the caller's originally-requested `grid_coord_names`,
        # never the extra `x_joint_name_pair` grid computed just above for internal use.
        # Merging it in here caused an `IndexError` in `_plot_projections` (an extra,
        # unrequested pair showed up in `grids.keys()`, and it tried to index a
        # single-Axes `axes` object at position 1). So we build that pair's grid
        # separately, below, rather than folding it into `grids_2d`.
        grids_2d = self._get_grids_2d(grids, grid_coord_names)

        # Extra data to pass to log_prob() for each coordinate:
        grid_cs = {k: 0.5 * (grids[k][:-1] + grids[k][1:]) for k in grids}
        conditional_data = self._make_conditional_data(grid_cs)

        # Make the distributions for each coordinate:
        dists = self.make_dists(pars=pars, dists=dists)

        if x_joint_name_pair is not None:
            # Evaluate the joint distribution on the grids -- reuse the grid already
            # in `grids_2d` if the caller happened to request this exact pair too,
            # otherwise compute it separately (see NOTE above for why this must not
            # be merged into the returned `grids_2d` dict).
            if x_joint_name_pair in grids_2d:
                x_joint_grid1, x_joint_grid2 = grids_2d[x_joint_name_pair]
            else:
                ((x_joint_grid1, x_joint_grid2),) = self._get_grids_2d(
                    grids, [x_joint_name_pair]
                ).values()
            grid1_c = 0.5 * (x_joint_grid1[:-1, :-1] + x_joint_grid1[1:, 1:])
            grid2_c = 0.5 * (x_joint_grid2[:-1, :-1] + x_joint_grid2[1:, 1:])

            ln_p = dists[x_joint_name_pair].log_prob(
                jnp.stack((grid1_c, grid2_c), axis=-1),
                **conditional_data[x_joint_name_pair],
            )

            # Integrates over the other coordinate to get the marginal distribution for
            # x:
            ln_p_x = ln_simpson(ln_p, grid2_c, axis=0)
            x_grid = grid1_c

        else:
            # Otherwise, we can just evaluate the model on the x coordinate grid:
            grid = grids[x_coord_name]
            x_grid = 0.5 * (grid[:-1] + grid[1:])
            ln_p_x = dists[x_coord_name].log_prob(
                x_grid, **conditional_data[x_coord_name]
            )

        evals = {}
        for name_pair in grid_coord_names:
            grid1, grid2 = grids_2d[name_pair]

            # grid edges passed in, but we evaluate at grid centers:
            grid1_c = 0.5 * (grid1[:-1, :-1] + grid1[1:, 1:])
            grid2_c = 0.5 * (grid2[:-1, :-1] + grid2[1:, 1:])

            # Evaluate the model on the grid
            if name_pair in self.coord_distributions:
                # It's a joint distribution:
                evals[name_pair] = dists[name_pair].log_prob(
                    jnp.stack((grid1_c, grid2_c), axis=-1),
                    **conditional_data[name_pair],
                )
            else:
                # It's an independent distribution from the x_coord_name:
                ln_p_y = dists[name_pair[1]].log_prob(
                    grid2_c, **conditional_data[name_pair[1]]
                )
                evals[name_pair] = ln_p_x + ln_p_y

        return grids_2d, evals

    ###################################################################################
    # Methods that can be overridden in subclasses:
    #
    def extra_ln_prior(self, pars: dict[str, Any]):
        """
        A log-prior to add to the total log-probability. This is useful for adding
        custom priors or regularizations that are not part of the model components.
        """
        return 0.0


class ComponentMixtureModel(eqx.Module, ModelMixin):
    """
    Creating a mixture model from multiple ModelComponent objects.

    :param mixing_probs:
        the distribution of the mixing probabilities for the components in the mixture model.
        Can be any numpyro ``Distribution`` with ``event_shape == (n_components,)`` and
        ``support == constraints.simplex`` (e.g. ``dist.Dirichlet``, or a
        ``dist.TransformedDistribution`` built from ``StickBreakingTransform``), or a plain
        fixed array of shape ``(n_components,)``.
    :type mixing_probs: dist.Distribution | ArrayLike

    :param components:
        a list of the model components that make up the mixture model
    :type components: list[ModelComponent]

    :param tied_coordinates:
        (optional) A dictionary of tied coordinates, where a key should be the name of a model
        component in the mixture, and the value should be a dictionary with keys as
        the names of the coordinates in the model component and values as the names
        of the other model component to tie that coordinate to. For example,
        tied_coordinates={"offtrack": {"pm1": "stream"}} means that for the model
        component named "offtrack", use the "pm1" coordinate from the "stream" model
        component. default=None
    :type tied_coordinates: dict[str, dict[str, str]]

    :attr:`coord_names`
        the names of the component parameters
        (the keys in `coord_distributions` and `coord_parameters` of each individual component).
        Every component must have the same coordinate names so they can be combined
    :type coord_names: tuple[str]

    :attr:`_tied_order`:
        Based on which coordinates are tied, the order in which the component distributions should be modeled.
        First, components with no dependencies are modeled, then components with dependencies are modeled.
    :type _tied_order: list[str]

    :attr:`_components`:
        a dictionary of the model components, where the keys are the names of the components and the values are the components themselves.
        This is just a restructuring of the input list of components into a dictionary for easier access.
    :type _components: dict[str, ModelComponent]
    """
    mixing_probs: dist.Distribution | ArrayLike
    components: list[ModelComponent]
    tied_coordinates: dict[str, dict[str, str]] = eqx.field(default=None)

    coord_names: tuple[str] = eqx.field(init=False)
    _tied_order: list[str] = eqx.field(init=False)
    _components: dict[str, ModelComponent] = eqx.field(init=False)

    def __post_init__(self):
        # Some validation of the input bits:
        coord_names = None
        for component in self.components:
            if not isinstance(component, ModelComponent):
                msg = "All components must be instances of ModelComponent"
                raise ValueError(msg)

            if coord_names is None:
                coord_names = tuple(component.coord_names)
            elif tuple(component.coord_names) != coord_names:
                msg = "All components must have the same coordinate names"
                raise ValueError(msg)
        self.coord_names = coord_names

        if len({component.name for component in self.components}) != len(
            self.components
        ):
            msg = "All components must have unique names"
            raise ValueError(msg)
        self._components = {component.name: component for component in self.components}

        mix_shape = (
            self.mixing_probs.event_shape[0]
            if isinstance(self.mixing_probs, dist.Distribution)
            else self.mixing_probs.shape[0]
        )

        if mix_shape != len(self.components):
            msg = (
                "The mixing distribution must have the same number of components as "
                "the model."
            )
            raise ValueError(msg)

        # Validate tied coordinates:
        self.tied_coordinates = (
            self.tied_coordinates if self.tied_coordinates is not None else {}
        )
        # Tied coordinates may be plain names ("pm1") or joint coordinates given
        # as a tuple of names (e.g. ("PS_g", "PS_r")), matching the keys of
        # the components' `coord_distributions`.
        for component_name, coords in self.tied_coordinates.items():
            if component_name not in self._components:
                msg = (
                    f"Component '{component_name}' passed in to tied_coordinates not "
                    "found in the mixture model"
                )
                raise ValueError(msg)

            for coord_name in coords:
                is_name = isinstance(coord_name, str)
                is_joint = isinstance(coord_name, tuple) and all(
                    isinstance(c, str) for c in coord_name
                )
                if not (is_name or is_joint):
                    msg = (
                        "Tied coordinates must be a coordinate name or a tuple of "
                        "coordinate names (joint coordinate)"
                    )
                    raise TypeError(msg)

        # Check for circular dependencies and set up order of components to create dists
        # for:
        self._tied_order = self._make_tied_order(self.tied_coordinates)

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(self._components.keys())

    def _make_tied_order(
        self, tied_coordinates: dict[str, dict[str, str]]
    ) -> list[str]:
        """
        Parameters
        ----------
        tied_coordinates
            A dictionary of tied coordinates, where a key should be the name of a model
            component in the mixture, and the value should be a dictionary with keys as
            the names of the coordinates in the model component and values as the names
            of the other model component to tie that coordinate to. For example,
            tied_coordinates={"offtrack": {"pm1": "stream}} means that for the model
            component named "offtrack", use the "pm1" coordinate from the "stream" model
            component.
        """
        tied_coordinates = copy.deepcopy(tied_coordinates)
        dependencies = {k: list(set(v.values())) for k, v in tied_coordinates.items()}

        tied_order = []

        # First, any component with no dependencies can be done first:
        for name in self.component_names:
            if name not in dependencies:
                tied_order.append(name)

        for _ in range(128):  # NOTE: max 128 iterations, arbitrary
            for name in tied_coordinates:
                if all(dep in tied_order for dep in dependencies[name]):
                    tied_order.append(name)
                    tied_coordinates.pop(name, None)
                    break

            if len(tied_coordinates) == 0:
                break

        else:
            msg = "Circular dependency likely in tied_coordinates"
            raise ValueError(msg)

        return tied_order

    def _make_component_dists(
        self,
    ) -> dict[str, dict[CoordinateName, dist.Distribution]]:
        """Build the per-coordinate distributions for every component.

        Unlike the old `_make_concatenated`, this does *not* glue each
        component's per-coordinate distributions together into a single
        joint `ConcatenatedDistributions` object -- it just returns the
        per-coordinate distributions directly, keyed by component name and
        then by coordinate name (preserving tuple keys for joint
        coordinates, e.g. `CMD_COORD`). This is what lets `__call__` below
        build one `MixtureGeneral` *per coordinate*, each with its own
        independently-sized `numpyro.plate`, instead of one plate shared
        across every coordinate.
        """
        # Deal with tied coordinates here across components:
        all_dists: dict[str, dict[CoordinateName, dist.Distribution]] = {}
        for component_name in self._tied_order:
            component = self._components[component_name]
            tied_map = self.tied_coordinates.get(component_name, {})

            override_dists = {
                override_coord: all_dists[dep][override_coord]
                for override_coord, dep in tied_map.items()
            }

            ## Having forced the offtrack proper motions (for example) to be identical
            ##  to the stream proper motions, make_dists will create the distributions
            ##  including numpyro.sample. This should be equivalent to creating an if
            ##  statement where for tied coordinates (or fixed coordinates),
            ##  we can get the numpyro_name here and do numpyro.deterministic for that parameter.
            ##  This adds the flexibility of being able to have "loosely tied" coordinates
            ##  i.e. where the tied coordinates are not identical, but are related in some way.

            all_dists[component_name] = component.make_dists(dists=override_dists)

        return all_dists

    def _coord_eval_units(
        self, all_dists: dict[str, dict[CoordinateName, dist.Distribution]]
    ) -> list[tuple[CoordinateName, dict[str, dist.Distribution]]]:
        """Group `all_dists` into per-likelihood-term evaluation units.

        Returns a list of `(coord_key, {component_name: dist})` pairs, one
        per coordinate (or joint group of coordinates) to be evaluated as its
        own `MixtureGeneral` + `numpyro.plate` in `__call__`.

        Each component's own `coord_distributions` may group coordinate
        names into joint/tuple keys differently from its siblings (e.g.
        offtrack's spatial term may be a single joint `("phi1","phi2")`
        `IndependentGMM` grid while bkg/stream instead use separate,
        string-keyed "phi1"/"phi2" coordinates) -- `__post_init__` only
        requires the *flattened* coordinate names to match across
        components, not this grouping. So: a group of names is evaluated
        jointly only when every component represents it with the exact same
        key (preserving existing behavior for e.g. CMD_COORD); otherwise the
        group is decomposed down to individual coordinate names, and any
        component that only has a given name as part of a larger joint
        distribution contributes its exact axis-marginal instead (see
        `_extract_coord_marginal`).
        """
        eval_units: list[tuple[CoordinateName, dict[str, dist.Distribution]]] = []
        seen_names: set[str] = set()

        for name in self.coord_names:
            if name in seen_names:
                continue

            # For each component, find the literal key (string or tuple) in
            # its own `all_dists[component_name]` that contains `name`.
            containing_keys: dict[str, CoordinateName] = {}
            for component_name in self.component_names:
                for key in all_dists[component_name]:
                    key_names = key if isinstance(key, tuple) else (key,)
                    if name in key_names:
                        containing_keys[component_name] = key
                        break

            unique_keys = set(containing_keys.values())
            if len(unique_keys) == 1:
                # Every component agrees on the same (possibly joint) key:
                # evaluate it jointly, exactly as before.
                joint_key = unique_keys.pop()
                joint_names = (
                    joint_key if isinstance(joint_key, tuple) else (joint_key,)
                )
                seen_names.update(joint_names)
                eval_units.append(
                    (
                        joint_key,
                        {
                            component_name: all_dists[component_name][joint_key]
                            for component_name in self.component_names
                        },
                    )
                )
            else:
                # Components disagree on how `name` is grouped: decompose
                # down to this single coordinate, extracting an exact
                # axis-marginal from any component that only has it bundled
                # into a larger joint distribution.
                seen_names.add(name)
                per_component = {
                    component_name: _extract_coord_marginal(
                        all_dists[component_name][key], key, name
                    )
                    for component_name, key in containing_keys.items()
                }
                eval_units.append((name, _harmonize_event_shapes(per_component)))

        return eval_units

    def __getitem__(self, key: str) -> ModelComponent:
        return self._components[key]

    def __call__(
        self, data: dict[str, ArrayLike], err: dict[str, ArrayLike] | None = None
    ) -> None:
        """
        This sets up the mixture model in numpyro.

        Each coordinate is modeled in its own `numpyro.plate`, sized
        independently based on however many stars have data for that
        coordinate. This means callers no longer need to fabricate a
        placeholder value plus a huge error (e.g. `radial_velocity=0`,
        `radial_velocity_err=1e4`) for stars missing a given coordinate --
        a coordinate can simply be omitted (or restricted to a smaller
        subsample) in `data`/`err`, and the log-likelihood contribution for
        that coordinate is only ever computed over the stars that actually
        have it.

        `mixing_probs` is still sampled exactly once, and the resulting
        `probs` are reused to build the per-coordinate `MixtureGeneral`s --
        the membership probabilities have to stay shared and self-
        consistent across coordinates, even though each coordinate's
        likelihood plate can have a different size.

        Parameters
        ----------
        data
            A dictionary mapping coordinate name -> array of observed
            values for that coordinate. A coordinate may be omitted
            entirely if no stars have data for it; when present, the
            array only needs to cover the stars that actually have valid
            data for that coordinate (it no longer needs to match the
            length of every other coordinate's array).
        err
            A dictionary mapping coordinate name -> array of measurement
            errors, aligned with the corresponding array in `data`. A
            coordinate present in `data` but missing from `err` (or missing
            entirely when `err=None`) is treated as being observed exactly,
            i.e. its likelihood is evaluated directly against the mixture
            density with no added per-star Gaussian error convolution.
        """
        probs = numpyro.sample("mixture-probs", self.mixing_probs)
        categorical = dist.Categorical(probs)

        all_dists = self._make_component_dists()

        # `__post_init__` only validates that every component's *flattened*
        # coordinate names match (see `ModelComponent._coord_names`) -- it
        # does not require every component to group those names into
        # `all_dists[name]` the same way. E.g. offtrack's spatial term may be
        # a single joint `("phi1","phi2")` `IndependentGMM` grid while
        # bkg/stream instead keep separate, string-keyed "phi1"/"phi2"
        # coordinates. `_coord_eval_units` resolves this: coordinate groups
        # that every component represents identically (e.g. CMD_COORD, when
        # present) are still evaluated jointly (unchanged, existing
        # behavior); groups where components disagree are decomposed down to
        # individual coordinate names, pulling the exact axis-marginal out of
        # whichever component(s) only have it as part of a larger joint
        # distribution (see `_extract_coord_marginal`).
        eval_units = self._coord_eval_units(all_dists)

        err = err if err is not None else {}

        for coord_key, component_dists_by_name in eval_units:
            if isinstance(coord_key, tuple):
                # Joint (e.g. CMD) coordinate: require every sub-coordinate
                # to be present for a star to be included.
                if not all(k in data for k in coord_key):
                    continue
                _data = jnp.stack([data[k] for k in coord_key], axis=-1)
                _data_err = None  # TODO: joint coordinates don't support errors yet
                numpyro_name = "-".join(coord_key)
            else:
                if coord_key not in data:
                    continue
                _data = jnp.asarray(data[coord_key])
                _data_err = err.get(coord_key)
                numpyro_name = coord_key

            component_dists = [
                component_dists_by_name[name] for name in self.component_names
            ]
            mixture = dist.MixtureGeneral(categorical, component_dists)

            if _data_err is None:
                with numpyro.plate(f"data-{numpyro_name}", _data.shape[0]):
                    numpyro.sample(f"{numpyro_name}-obs", mixture, obs=_data)
            else:
                sample_shape = (
                    (_data.shape[0],) if mixture.batch_shape == () else ()
                )
                model_val = numpyro.sample(
                    f"{numpyro_name}:modeldata", mixture, sample_shape=sample_shape
                )
                with numpyro.plate(f"data-{numpyro_name}", _data.shape[0]):
                    numpyro.sample(
                        f"{numpyro_name}-obs",
                        dist.Normal(model_val, _data_err),
                        obs=_data,
                    )

    def pack_params(
        self, pars: dict[str, dict[CoordinateName, dict]]
    ) -> dict[str, Any]:
        """
        Convert a dictionary of parameters as a nested dictionary into a flat dictionary
        with packed numpyro-compatible names.

        Parameters
        ----------
        pars
            A dictionary of parameters where the keys are the names of the components in
            the model and the values are dictionaries of the parameters for each
            coordinate in the component.
        """
        packed_pars = {}
        for component_name, component_pars in pars.items():
            packed_pars.update(
                self._components[component_name].pack_params(component_pars)
            )
        return packed_pars

    def expand_numpyro_params(
        self, pars: dict[str, Any], skip_invalid: bool = False
    ) -> dict[str, dict[CoordinateName, Any]]:
        """
        Convert a dictionary of numpyro parameters into a nested dictionary where the
        keys are the coordinate names and parameter name.

        Parameters
        ----------
        pars
            A dictionary of numpyro parameters where the keys are the names of the
            parameters created with numpyro.sample().
        """
        pars = copy.deepcopy(pars)

        expanded_pars: dict[str, dict] = {}
        for component_name in self._tied_order:
            component = self._components[component_name]
            tied_map = self.tied_coordinates.get(component.name, {})
            component_pars = {}
            for key in list(pars.keys()):  # convert to list because we change dict keys
                if key.startswith(f"{component.name}:"):
                    component_pars[key] = pars.pop(key)
            expanded_pars[component.name] = component.expand_numpyro_params(
                component_pars, skip_invalid=skip_invalid
            )

            # TODO(adrn): duplicated code
            for override_coord, dep in tied_map.items():
                # First, we set with the parameters from the coord_parameters, then
                # update values from the tied component
                expanded_pars[component.name][override_coord] = copy.deepcopy(
                    self._components[dep].coord_parameters[override_coord]
                )
                expanded_pars[component.name][override_coord].update(
                    expanded_pars[dep][override_coord]
                )

        expanded_pars.update(pars)
        return expanded_pars

    def evaluate_on_2d_grids(
        self,
        pars: dict[str, Any],
        grids: dict[str, ArrayLike],
        grid_coord_names: list[tuple[str, str]] | None = None,
        x_coord_name: str | None = None,
    ):
        # Crude way of detecting if pars are already expanded or not:
        if all(name in pars for name in self.component_names):
            expanded_pars = pars
        else:
            expanded_pars = self.expand_numpyro_params(pars)

        # Deal with tied coordinates here across components:
        # TODO(adrn): duplicated code
        component_dists: dict[str, dict[CoordinateName, dist.Distribution]] = {}
        dists: dict[str, dict] = {}
        for component_name in self._tied_order:
            tied_map = self.tied_coordinates.get(component_name, {})

            dists[component_name] = {
                override_coord: component_dists[dep][override_coord]
                for override_coord, dep in tied_map.items()
            }

            component_dists[component_name] = self._components[
                component_name
            ].make_dists(
                pars=expanded_pars[component_name],
                dists=dists.get(component_name, {}),
            )

        terms: dict[str, list[jax.Array]] = {}
        for component in self.components:
            # TODO: need to override dists here too, but damn that interface sucks
            all_grids, component_terms = component.evaluate_on_2d_grids(
                pars=expanded_pars[component.name],
                grids=grids,
                grid_coord_names=grid_coord_names,
                x_coord_name=x_coord_name,
                dists=dists.get(component.name, {}),
            )
            for k, v in component_terms.items():
                if k not in terms:
                    terms[k] = []
                terms[k].append(v)

        # use "mixture-probs" to weight the component terms
        terms = {
            k: jax.scipy.special.logsumexp(
                jnp.array(v).T, axis=-1, b=pars["mixture-probs"]
            ).T
            for k, v in terms.items()
        }
        return all_grids, terms
