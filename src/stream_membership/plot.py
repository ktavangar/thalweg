import numpy as np


def _plot_projections(grids, ims, axes=None, label=True, pcolormesh_kwargs=None):
    import matplotlib as mpl

    grid_coord_names = list(grids.keys())
    if len(grid_coord_names) < 1:
        msg = "You must pass in some grids"
        raise ValueError(msg)
    # TODO: Check that grids.keys() and ims.keys() are the same

    _default_labels = {
        "phi1": r"$\phi_1$ [deg]",
        "phi2": r"$\phi_2$ [deg]",
        "pm1": r"$\mu_{\phi_1}$ [mas/yr]",
        "pm2": r"$\mu_{\phi_2}$ [mas/yr]",
        "rv": r"$v_r$ [km/s]",
    }

    if pcolormesh_kwargs is None:
        pcolormesh_kwargs = {}

    # Check that all x coord names are the same
    sharex = all(
        name_pair[0] == grid_coord_names[0][0] for name_pair in grid_coord_names
    )

    if axes is None:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(
            len(grid_coord_names),
            1,
            figsize=(10, 2 + 2 * len(grid_coord_names)),
            sharex=sharex,
            sharey="row",
            constrained_layout=True,
        )

    if isinstance(axes, mpl.axes.Axes):
        axes = [axes]
    axes = np.array(axes)
    pcms = []

    for i, name_pair in enumerate(grid_coord_names):
        grid1, grid2 = grids[name_pair]
        pcm = axes[i].pcolormesh(grid1, grid2, ims[name_pair], **pcolormesh_kwargs)
        pcms.append(pcm)
        axes[i].set_ylim(grid2.min(), grid2.max())

        if label:
            axes[i].set_ylabel(_default_labels[name_pair[1]], fontsize=16)

        if not sharex:
            axes[i].set_xlabel(_default_labels[name_pair[0]], fontsize=16)

    if sharex:
        axes[-1].set_xlabel(_default_labels[name_pair[0]], fontsize=16)
    axes[0].set_xlim(grid1.min(), grid1.max())

    return axes.flat[0].figure, axes, pcms


def plot_data_projections(
    data,
    grids,
    coord_names=None,
    axes=None,
    label=True,
    smooth=1.0,
    pcolormesh_kwargs=None,
    x_data=None,
):
    """
    TODO:
    - coord_names should be a list of tuples like [('phi1', 'phi2')]

    Parameters
    ----------
    ...
    x_data : dict[str, array_like] | None
        Optional per-coordinate override for the x-axis array used when
        histogramming a `name_pair`. Callers whose `data` dict is "ragged"
        (i.e. `data[coord_name]` arrays are not all the same length, because
        rows missing a given coordinate were dropped rather than sentinel-
        filled) should pass a dict mapping `name_pair[1]` (the non-x
        coordinate) -> the x-array that is actually aligned, element-for-
        element, with `data[name_pair[1]]`. If a coordinate is missing from
        `x_data` (or `x_data` is None), this falls back to the old behavior
        of using `data[name_pair[0]]` directly, which is only correct when
        every coordinate's array shares a common length/ordering.
    """
    from scipy.ndimage import gaussian_filter

    if coord_names is None:
        tmp = list(grids.keys())
        coord_names = [(tmp[0], name) for name in tmp[1:]]

    ims = {}
    im_grids = {}
    for name_pair in coord_names:
        x_vals = (
            x_data[name_pair[1]]
            if x_data is not None and name_pair[1] in x_data
            else data[name_pair[0]]
        )
        H_data, xe, ye = np.histogram2d(
            x_vals,
            data[name_pair[1]],
            bins=(grids[name_pair[0]], grids[name_pair[1]]),
        )
        if smooth is not None:
            H_data = gaussian_filter(H_data, smooth)
        im_grids[name_pair] = (xe, ye)
        ims[name_pair] = H_data.T

    return _plot_projections(
        grids=im_grids,
        ims=ims,
        axes=axes,
        label=label,
        pcolormesh_kwargs=pcolormesh_kwargs,
    )
