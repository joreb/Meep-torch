# Shared plot builders for the mpb examples in this directory.
"""Band-diagram and epsilon figures, built but NOT saved.

Upstream's mpb examples print numbers and draw nothing, so every figure
here is a deviation from the script being ported (rule 8). Keeping the
drawing in one place means the deviation is one reviewable file instead of
ten near-copies.

The builders RETURN ``(fig, table, header)`` and never touch the disk;
:func:`save` writes that triple out. The split is deliberate: the example
still owns its ``OUT`` and its file names -- it passes both in -- so what
each script produces is still readable from the script itself, which is
what rule 4 checks for. What ``save`` removes is only the three identical
lines (savefig, close, savetxt) that every figure otherwise repeats.
"""
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["band_figure", "epsilon_figure", "eps_profile",
           "epsilon_3d_figure", "field_figure", "field_grid_figure",
           "save"]


def band_figure(ms, title, corner_labels=None, gap_floor=1.0,
                x=None, xlabel="k-point index"):
    """Frequency against k-point INDEX, gaps shaded.

    The index axis is MPB's own convention and the honest one: the path is
    a piecewise walk between symmetry points and the segments are not the
    same length in k, so plotting against |k| would stretch some segments
    and squash others.
    """
    a = np.asarray(ms.all_freqs, dtype=float)
    # The default index axis is MPB's own and the honest one for a
    # symmetry-point PATH, whose segments are not equal in |k|. Pass ``x``
    # when k IS a physical axis -- a waveguide dispersion sweeps one
    # propagation constant, and plotting that against an index would hide
    # the light line, which is a straight line only in beta.
    x = np.arange(a.shape[0]) if x is None else np.asarray(x, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    ax.plot(x, a, "-", color="tab:blue", lw=1.1)
    for pct, lo, hi in ms.gap_list:
        if pct > gap_floor:      # thinner than the line below this
            ax.fill_between(x, lo, hi, color="tab:orange", alpha=0.35, lw=0)
            ax.text(x[-1], 0.5 * (lo + hi), f" {pct:.1f}% ", va="center",
                    ha="right", fontsize=8, color="darkorange")
    if corner_labels:
        ticks = np.linspace(0, a.shape[0] - 1, len(corner_labels))
        ax.set_xticks(ticks)
        ax.set_xticklabels(corner_labels)
        for t in ticks[1:-1]:
            ax.axvline(t, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frequency  (c/a)")
    ax.set_title(title)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    header = ("k_index" if xlabel == "k-point index" else "k") + "," + \
        ",".join(f"band{b + 1}" for b in range(a.shape[1]))
    return fig, np.column_stack((x, a)), header


def save(fig, table, header, out, name):
    """Write one figure and the numbers behind it, then close the figure.

    Rule 4 of ``examples/example_template.py``: a PNG is not reviewable and
    not diffable, so every figure gets a CSV beside it. Closing matters as
    much as writing -- a multi-section script that leaves figures open runs
    matplotlib out of memory long before it runs out of sections.
    """
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    np.savetxt(out / f"{name}.csv", np.atleast_2d(table), delimiter=",",
               header=header, comments="" if "," in header else "# ")


def epsilon_3d_figure(ms, title, level=None, max_dim=48, elev=22, azim=-58):
    """The high-index region of a 3D cell as an isosurface.

    A 3D structure has no faithful 2D picture: a slice through a hole slab
    shows a slab with a gap in it, and a slice through the diamond lattice
    catches whichever spheres the plane happens to cross. The surface
    between high and low index is the thing that is actually periodic, so
    that is what gets drawn.

    ``marching_cubes`` from scikit-image, which is already a dependency.
    Matplotlib's ``voxels`` would need no dependency but renders every cell
    face -- a 32x32x65 cell is 66k voxels and minutes of drawing, against a
    triangle mesh that takes a second.

    The array is in LATTICE coordinates, so the vertices are mapped through
    the lattice vectors before plotting: on an oblique cell (the hole slab
    is triangular) the raw index space would draw the rhombic cell as a
    box and the round holes as ellipses -- the same shear ``rectify``
    exists to undo in 2D.

    ``level`` defaults to the midpoint of the epsilon range, which is the
    material boundary. ``max_dim`` downsamples so a fine grid stays
    plottable; the surface is a picture, not a measurement.
    """
    from skimage import measure

    eps = np.squeeze(np.asarray(ms.get_epsilon(), dtype=float))
    if eps.ndim != 3:
        raise ValueError(
            f"epsilon_3d_figure needs a 3D cell; got shape {eps.shape}. "
            f"Use epsilon_figure() for 1D and 2D.")
    step = [max(1, n // max_dim) for n in eps.shape]
    small = eps[::step[0], ::step[1], ::step[2]]
    lo, hi = float(small.min()), float(small.max())
    if hi - lo < 1e-12:
        raise ValueError("epsilon is uniform; there is no surface to draw")
    lvl = 0.5 * (lo + hi) if level is None else level

    verts, faces, _, _ = measure.marching_cubes(small, level=lvl)
    # index space -> fractional -> cartesian, through the lattice vectors
    frac = verts / np.array(small.shape, dtype=float)
    A = np.asarray(ms.get_lattice(), dtype=float)          # rows are a_i
    xyz = frac @ A

    fig = plt.figure(figsize=(5.5, 5), dpi=150)
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(xyz[:, 0], xyz[:, 1], faces, xyz[:, 2],
                    color="tab:blue", alpha=0.85, linewidth=0, shade=True)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"{title}\n$\\varepsilon$ = {lvl:.2f} isosurface",
                 fontsize=10)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect(np.ptp(xyz, axis=0))
    except Exception:                                # older matplotlib
        pass
    fig.tight_layout()
    table = np.column_stack((xyz, np.zeros(len(xyz))))
    return (fig, table[:, :3],
            "x,y,z")


def eps_profile(ax, ms, title, reference=None, mode="step"):
    """Draw a 1D epsilon profile on ``ax``; return ``(table, header)``.

    Takes an axes rather than making its own figure because the two
    callers lay out differently -- mpb_bragg puts the profile beside its
    band list in one 1x2 figure, mpb_bragg_sine gives it a figure of its
    own. A builder returning a whole figure would serve one and not the
    other.

    :func:`epsilon_figure` is deliberately NOT the right tool for a 1D
    cell: it imshows the grid, and a (32, 1) cell renders as a one-pixel
    strip.

    ``mode="step"`` for a piecewise-constant stack, ``"points"`` for a
    sampled profile against a smooth ``reference`` curve.

    The x axis is the SOLVER's cell centres, ``(i + 0.5)/n - 1/2``, which
    is why it lives here rather than being recomputed per example: that
    convention is not universal in this package (the 3D path samples at
    ``i/n`` instead), and two copies of it would drift apart silently the
    next time one changes.
    """
    eps = np.asarray(ms.get_epsilon())
    eps = eps[:, 0] if eps.ndim > 1 else eps
    x = (np.arange(eps.size) + 0.5) / eps.size - 0.5
    if reference is not None:
        ax.plot(x, reference, "-", color="0.7", lw=2.5, label=r"$n(x)^2$")
    if mode == "step":
        ax.step(x, eps, where="mid", color="tab:blue")
    else:
        ax.plot(x, eps, ".", color="tab:blue", ms=5, label="sampled grid")
    ax.set_xlabel("x / a")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if reference is not None:
        ax.legend()
        return (np.column_stack((x, eps, reference)),
                "x_over_a,epsilon,epsilon_analytic")
    return np.column_stack((x, eps)), "x_over_a,epsilon"


def field_figure(ms, band, title, component=2):
    """Re(E) of ONE band over the permittivity, meep's imshow convention.

    The grid twin of :func:`field_grid_figure`, for a script that wants a
    single band rather than a panel per band. Epsilon underneath in
    greyscale, the field over it on a symmetric diverging scale so that
    zero is the neutral colour and the sign is readable.
    """
    eps = np.asarray(ms.get_epsilon(), dtype=float)
    field = np.real(np.asarray(ms.get_efield(band))[:, :, 0, component])
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=150)
    ax.imshow(eps.T, interpolation="spline36", cmap="binary", origin="lower")
    lim = float(np.abs(field).max()) or 1.0
    im = ax.imshow(field.T, interpolation="spline36", cmap="RdBu", alpha=0.85,
                   origin="lower", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="Re(E)")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig, field, f"Re(E_{'xyz'[component]}) of band {band} on the lattice grid"


def epsilon_figure(ms, title, rectify=False, periods=1,
                   slice_axis=None, slice_index=None):
    """The dielectric the solver runs on, one panel per distinct component.

    ``get_epsilon()`` is the AXIAL component, which is what TM sees; TE
    runs on the in-plane pair. They are rarely the same array -- Kottke
    subpixel smoothing is directional, so even a scalar material gives
    three different boundary treatments -- and for an anisotropic material
    they are different structures entirely.

    The panels are the solver's grid, which is in LATTICE coordinates. On
    an oblique lattice that shears the picture: round rods come out as
    ellipses and the cell as a parallelogram drawn square. That is the
    grid being shown honestly, not a rasterization bug. ``rectify=True``
    maps it back to cartesian through ``MPBData``, which is what that
    class is for; mpb_data_analysis.py shows the two side by side.
    """
    # A cross-section cell (size=(0, y, z)) comes back (1, ny, nz): squeeze
    # the collapsed axis or imshow gets a 3D array. Squeezing only size-1
    # axes leaves every other case untouched.
    comps = [np.squeeze(np.asarray(ms.get_epsilon_tensor(c, c)))
             for c in range(3)]
    if slice_axis is not None:
        # A 3D cell has no single picture. Which plane is not a detail the
        # helper can guess: for a slab supercell the informative cut is
        # VERTICAL (through the slab and a hole), while for a bulk crystal
        # it is usually a mid-plane of the third axis. The caller says.
        idx = (comps[0].shape[slice_axis] // 2 if slice_index is None
               else slice_index)
        comps = [np.take(c, idx, axis=slice_axis) for c in comps]
    xx, yy, zz = comps
    eq = lambda p, q: np.allclose(p, q, rtol=0, atol=1e-12)
    shown = [(r"$\varepsilon_{zz}$  (TM)", zz)]
    if not eq(xx, zz):
        shown.append((r"$\varepsilon_{xx}$  (TE)", xx))
        if not eq(yy, xx):
            shown.append((r"$\varepsilon_{yy}$  (TE)", yy))
    if rectify:
        from meep_torch import mpb
        md = mpb.MPBData(rectify=True, periods=periods,
                         resolution=ms.resolution[0], lattice=ms.get_lattice())
        shown = [(lab, np.asarray(md.convert(mpb.MPBArray(a, ms.get_lattice()))))
                 for lab, a in shown]

    fig, axes = plt.subplots(1, len(shown), figsize=(4.2 * len(shown), 4.0),
                             dpi=150, squeeze=False)
    vmin = min(a.min() for _, a in shown)
    vmax = max(a.max() for _, a in shown)
    for ax, (lab, a) in zip(axes[0], shown):
        im = ax.imshow(np.atleast_2d(a).T, interpolation="nearest",
                       cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(lab)
        ax.axis("off")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.035, pad=0.02,
                 label="relative permittivity")
    fig.suptitle(title)
    # the LAST panel is the in-plane component when there is one, which is
    # what TE sees; ezz alone would misreport an anisotropic cell
    lab, arr = shown[-1]
    where = "rectified grid" if rectify else "lattice grid"
    if slice_axis is not None:
        where += f", slice {slice_axis}={idx}"
    return fig, np.atleast_2d(arr), f"{lab.strip()} on the {where}"


def field_grid_figure(fields, eps, title, ncol=3):
    """One panel per band: Re(field) under a contour of the same epsilon.

    ``fields`` and ``eps`` must both be MPBData-CONVERTED arrays, so they
    share a shape and the rods land under the field they belong to. The
    raw lattice arrays would not line up: for an oblique lattice the
    solver's grid is in the lattice basis, and rectifying it is what
    MPBData is for.

    Two deviations from the upstream tutorial's plotting, both so the
    panels can be read against each other:

    - one symmetric colour scale across all bands, not per-panel
      autoscaling, which would make a weak band look as strong as band 1;
    - ``origin="lower"`` on the imshow. matplotlib defaults imshow to
      ``origin="upper"`` and contour to y-up, so the upstream overlay of
      the two has the rods mirrored against the field. A triangular
      lattice is symmetric enough to hide it.

    The contour is drawn at ONE level, the midpoint of the epsilon range,
    which is the rod boundary. Upstream's default level set stacks every
    level inside the transition and fills some rods with stripes -- the
    array itself is a clean two-valued 1/12, so that pattern is the
    contour's, not the structure's.
    """
    n = len(fields)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.1 * nrow),
                             dpi=150, squeeze=False)
    vmax = max(float(np.abs(np.real(f)).max()) for f in fields)
    eps_t = np.asarray(eps).T
    rows = []
    im = None
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        re = np.real(np.asarray(fields[i]))
        im = ax.imshow(re.T, interpolation="spline36", cmap="RdBu",
                       origin="lower", vmin=-vmax, vmax=vmax, alpha=0.9)
        ax.contour(eps_t, levels=[0.5 * (eps_t.min() + eps_t.max())],
                   colors="0.25", linewidths=0.7, origin="lower")
        ax.set_title(f"band {i + 1}", fontsize=9)
        ax.axis("off")
        rows.append((i + 1, float(np.abs(fields[i]).max()),
                     float(re.max()), float(re.min())))
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                 label="Re(field)")
    fig.suptitle(title)
    return (fig, np.array(rows, dtype=float),
            "band,max_abs,max_real,min_real")
