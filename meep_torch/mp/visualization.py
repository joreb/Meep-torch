# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep visualization surface: plot2D / Animate2D / JS_Animation.

Ported from meep/python/visualization.py (same function/class names,
signatures, and default parameter dicts). Adapter notes vs real meep:

- eps comes from the shim's rasterized grid (``sim._rm.eps_tm`` /
  ``sim._eps3d``) sampled at the plot tics -- no ``get_epsilon_grid``, so
  ``eps_parameters['frequency']`` is accepted but does not change the plot
  (no material dispersion in the display).
- fields come from ``sim.get_array(component=...)``: the instantaneous state
  the engine keeps in ``sim._state`` (available after/while ``run()``);
  calling with ``fields=`` before any run raises the shim's RuntimeError.
- the shim Simulation stores no ``boundary_layers``/``dft_objects``/
  ``geometry`` attributes; boundaries are reconstructed from the PML/Absorber
  cell counts (``_npml``/``_nabs``, drawn on all sides) and monitors from
  ``sim._monitors`` line metadata. ``label_geometry`` is accepted but a no-op.
- no MPI: ``am_master()`` is always True. 2D cells (and 3D with an explicit
  ``output_plane``) are supported; cylindrical/1D branches are ported but
  untested here. ``plot3D``/``visualize_chunks`` (vispy/chunk-owner based)
  are not ported.

Animate2D follows meep's step-function protocol exactly: it is called as
``animate(sim, 'step')`` / ``animate(sim, 'finish')`` and carries the
``__code__.co_argcount == 2`` marker meep's run loop uses to detect
two-argument step functions, so it slots into
``sim.run(mp.at_every(dt, animate), until=...)``.
"""
import copy
import math
import warnings
from collections import namedtuple
from time import sleep
from typing import Callable, List, Optional, Tuple

import numpy as np

from .run_control import Volume
from .sources import Dielectric
from .vec import Vector3

# Direction / side / dimension constants (meep SWIG enum values;
# boundary_side is {High, Low} so High=0, Low=1, matching mp/material.py).
X, Y, Z, R, P = 0, 1, 2, 3, 4
High, Low = 0, 1
ALL = -1
CYLINDRICAL = -2


def am_master() -> bool:
    """meep mp.am_master(); single-process here, always True."""
    return True


# ------------------------------------------------------- #
# Functions used to define the default plotting parameters
# for the different plotting routines.

default_source_parameters = {
    "color": "r",
    "edgecolor": "r",
    "facecolor": "none",
    "hatch": "/",
    "linewidth": 2,
}

default_monitor_parameters = {
    "color": "b",
    "edgecolor": "b",
    "facecolor": "none",
    "hatch": "/",
    "linewidth": 2,
}

default_field_parameters = {
    "interpolation": "spline36",
    "cmap": "RdBu",
    "alpha": 0.8,
    "post_process": np.real,
    "colorbar": False,
}

default_eps_parameters = {
    "interpolation": "spline36",
    "cmap": "binary",
    "alpha": 1.0,
    "contour": False,
    "contour_linewidth": 1,
    "frequency": None,
    "resolution": None,
    "colorbar": False,
}

default_colorbar_parameters = {
    "label": None,
    "orientation": "vertical",
    "extend": None,
    "position": "right",
    "size": "5%",
    "pad": "2%",
}

default_boundary_parameters = {
    "color": "g",
    "edgecolor": "g",
    "facecolor": "none",
    "hatch": "/",
}

default_volume_parameters = {
    "alpha": 1.0,
    "color": "k",
    "linestyle": "-",
    "linewidth": 1,
    "marker": ".",
    "edgecolor": "k",
    "facecolor": "none",
    "hatch": "/",
}

default_label_parameters = {"label_color": "r", "offset": 20, "label_alpha": 0.3}

default_label_geometry_parameters = {
    "label_color": "w",
    "arrow_color": "r",
    "offset": 20,
    "label_alpha": 0.8,
}


def filter_dict(dict_to_filter: dict, func_with_kwargs: Callable) -> dict:
    """Keep only the keys of dict_to_filter that are parameters of the
    function (meep helper, used to pass parameter dicts to imshow etc.)."""
    import inspect

    sig = inspect.signature(func_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values()]
    return {k: dict_to_filter[k] for k in filter_keys if k in dict_to_filter}


# ------------------------------------------------------- #
# Shim accessors: meep Simulation attributes the adapter Simulation
# does not store, derived from the shim's GridSpec / PML cell counts.

def _cell_size(sim) -> Vector3:
    cs = getattr(sim, "cell_size", None)
    if cs is not None:
        return cs
    gs = sim.gs
    return Vector3(gs.sx, gs.sy, gs.sz if getattr(gs, "nz", 0) else 0.0)


def _geometry_center(sim) -> Vector3:
    return getattr(sim, "geometry_center", None) or Vector3()


def _is_cylindrical(sim) -> bool:
    return bool(getattr(sim, "is_cylindrical", False))


def _resolution(sim) -> float:
    res = getattr(sim, "resolution", None)
    return float(res) if res else 1.0 / sim.gs.dx


def _sim_sources(sim) -> list:
    src = getattr(sim, "sources", None)
    return src if src is not None else getattr(sim, "_sources", [])


def _boundary_layers(sim) -> list:
    bls = getattr(sim, "boundary_layers", None)
    if bls is not None:
        return bls
    from .material import PML, Absorber

    out = []
    dx = sim.gs.dx
    if getattr(sim, "_npml", 0):
        out.append(PML(thickness=sim._npml * dx))
    if getattr(sim, "_nabs", 0):
        out.append(Absorber(thickness=sim._nabs * dx))
    return out


def _finite_size(size: Optional[Vector3], cell: Vector3) -> Vector3:
    """Clamp mp.inf extents to the cell so volume vertex math stays finite."""
    s = size or Vector3()
    return Vector3(cell.x if math.isinf(s.x) else s.x,
                   cell.y if math.isinf(s.y) else s.y,
                   cell.z if math.isinf(s.z) else s.z)


def _pt_in_volume(volume: Volume, pt: Vector3, tol: float = 1e-9) -> bool:
    xmin, xmax, ymin, ymax, zmin, zmax = box_vertices(volume.center, volume.size)
    return (xmin - tol <= pt.x <= xmax + tol
            and ymin - tol <= pt.y <= ymax + tol
            and zmin - tol <= pt.z <= zmax + tol)


def _volume_from_vertices(vertices: List[Vector3]) -> Volume:
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    zs = [v.z for v in vertices]
    center = Vector3((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2,
                     (max(zs) + min(zs)) / 2)
    size = Vector3(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    return Volume(center=center, size=size)


# ------------------------------------------------------- #
# Routines to add legends to plot


def place_label(ax, label_text: str, x: float, y: float, centerx: float,
                centery: float, label_parameters: Optional[dict] = None):
    if label_parameters is None:
        label_parameters = default_label_parameters
    else:
        label_parameters = dict(default_label_parameters, **label_parameters)

    offset = label_parameters["offset"]
    alpha = label_parameters["label_alpha"]
    color = label_parameters["label_color"]
    arrow_color = label_parameters.get("arrow_color", color)

    xtext = -offset if x > centerx else offset
    ytext = -offset if y > centery else offset

    ax.annotate(
        label_text,
        xy=(x, y),
        xytext=(xtext, ytext),
        textcoords="offset points",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=alpha),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.5",
                        color=arrow_color),
    )
    return ax


# ------------------------------------------------------- #
# Helper functions used to plot volumes on a 2D plane

def intersect_volume_volume(volume1: Volume, volume2: Volume) -> List[Vector3]:
    """Intersection vertices of two axis-aligned Volumes (meep helper)."""
    U1 = [volume1.center.x + volume1.size.x / 2,
          volume1.center.y + volume1.size.y / 2,
          volume1.center.z + volume1.size.z / 2]
    L1 = [volume1.center.x - volume1.size.x / 2,
          volume1.center.y - volume1.size.y / 2,
          volume1.center.z - volume1.size.z / 2]
    U2 = [volume2.center.x + volume2.size.x / 2,
          volume2.center.y + volume2.size.y / 2,
          volume2.center.z + volume2.size.z / 2]
    L2 = [volume2.center.x - volume2.size.x / 2,
          volume2.center.y - volume2.size.y / 2,
          volume2.center.z - volume2.size.z / 2]

    U = np.min([U1, U2], axis=0)
    L = np.max([L1, L2], axis=0)

    # For single points we have to check manually
    if np.all(U - L == 0):
        if (not _pt_in_volume(volume1, Vector3(*U))) or (
                not _pt_in_volume(volume2, Vector3(*U))):
            return []

    # Check for two volumes that don't intersect
    if np.any(U - L < 0):
        return []

    # Pull all possible vertices
    vertices = []
    for x_vals in [L[0], U[0]]:
        for y_vals in [L[1], U[1]]:
            for z_vals in [L[2], U[2]]:
                vertices.append(Vector3(x_vals, y_vals, z_vals))

    # Remove any duplicate points caused by coplanar lines
    vertices = [vertices[i] for i, x in enumerate(vertices)
                if x not in vertices[i + 1:]]
    return vertices


def get_2D_dimensions(sim, output_plane: Optional[Volume]) -> Tuple[Vector3, Vector3]:
    """Resolve the plot plane: output_plane, sim.output_volume, or the full
    cell -- truncated to the simulation domain (meep helper)."""
    if output_plane:
        plane_center, plane_size = (output_plane.center, output_plane.size)
    elif getattr(sim, "output_volume", None):
        ov = sim.output_volume
        plane_center, plane_size = (ov.center, ov.size)
    else:
        if (sim.dimensions == CYLINDRICAL) or _is_cylindrical(sim):
            plane_center = _geometry_center(sim) + Vector3(_cell_size(sim).x / 2)
            plane_size = _cell_size(sim)
        else:
            plane_center, plane_size = (_geometry_center(sim), _cell_size(sim))
    plane_volume = Volume(center=plane_center, size=plane_size)

    if plane_size.x != 0 and plane_size.y != 0 and plane_size.z != 0:
        raise ValueError("Plane volume must be 2D (a plane).")
    if (sim.dimensions == CYLINDRICAL) or _is_cylindrical(sim):
        center = _geometry_center(sim) + Vector3(_cell_size(sim).x / 2)
        check_volume = Volume(center=center, size=_cell_size(sim))
    else:
        check_volume = Volume(center=_geometry_center(sim), size=_cell_size(sim))
    vertices = intersect_volume_volume(check_volume, plane_volume)

    if len(vertices) == 0:
        raise ValueError(
            "The specified user volume is completely outside of the simulation domain.")

    intersection_vol = _volume_from_vertices(vertices)

    if (intersection_vol.size != plane_volume.size) or (
            intersection_vol.center != plane_volume.center):
        warnings.warn(
            "The specified user volume is larger than the simulation domain "
            "and has been truncated.")

    return intersection_vol.center, intersection_vol.size


def box_vertices(box_center: Vector3, box_size: Vector3,
                 is_cylindrical: bool = False
                 ) -> Tuple[float, float, float, float, float, float]:
    # in cylindrical coordinates the radial axis is (0, R), not (-R/2, R/2)
    if is_cylindrical:
        xmin = 0
        xmax = box_size.x
    else:
        xmin = box_center.x - 0.5 * box_size.x
        xmax = box_center.x + 0.5 * box_size.x
    ymin = box_center.y - 0.5 * box_size.y
    ymax = box_center.y + 0.5 * box_size.y
    zmin = box_center.z - 0.5 * box_size.z
    zmax = box_center.z + 0.5 * box_size.z
    return xmin, xmax, ymin, ymax, zmin, zmax


# ------------------------------------------------------- #
# actual plotting routines

def plot_volume(sim, ax, volume: Volume, output_plane: Optional[Volume] = None,
                plotting_parameters: Optional[dict] = None,
                label: Optional[str] = None):
    import matplotlib.patches as patches

    if plotting_parameters is None:
        plotting_parameters = default_volume_parameters
    else:
        plotting_parameters = dict(default_volume_parameters, **plotting_parameters)

    sim_center, sim_size = get_2D_dimensions(sim, output_plane)
    plane = Volume(center=sim_center, size=sim_size)

    size = volume.size
    center = volume.center

    if label is not None and am_master():
        if sim_size.x == 0:
            ax = place_label(ax, label, center.y, center.z, sim_center.y,
                             sim_center.z, label_parameters=plotting_parameters)
        elif sim_size.y == 0:
            ax = place_label(ax, label, center.x, center.z, sim_center.x,
                             sim_center.z, label_parameters=plotting_parameters)
        elif sim_size.z == 0:
            ax = place_label(ax, label, center.x, center.y, sim_center.x,
                             sim_center.y, label_parameters=plotting_parameters)

    intersection = intersect_volume_volume(volume, plane)

    def sort_points(xy):
        # counter-clockwise sort so a convex polygon is formed
        xy = np.squeeze(xy)
        xy_mean = np.mean(xy, axis=0)
        theta = np.arctan2(xy[:, 1] - xy_mean[1], xy[:, 0] - xy_mean[0])
        return xy[np.argsort(theta, axis=0), :]

    if am_master():
        # Point volume
        if len(intersection) == 1:
            point_args = {k: v for k, v in plotting_parameters.items()
                          if k in ["color", "marker", "alpha", "linewidth"]}
            if sim_size.y == 0:
                ax.scatter(center.x, center.z, **point_args)
            elif sim_size.x == 0:
                ax.scatter(center.y, center.z, **point_args)
            elif sim_size.z == 0:
                ax.scatter(center.x, center.y, **point_args)
            return ax
        # Line volume
        elif len(intersection) == 2:
            line_args = {k: v for k, v in plotting_parameters.items()
                         if k in ["color", "linestyle", "linewidth", "alpha"]}
            if sim_size.x == 0:
                ax.plot([a.y for a in intersection], [a.z for a in intersection],
                        **line_args)
            elif sim_size.y == 0:
                ax.plot([a.x for a in intersection], [a.z for a in intersection],
                        **line_args)
            elif sim_size.z == 0:
                ax.plot([a.x for a in intersection], [a.y for a in intersection],
                        **line_args)
            return ax
        # Planar volume
        elif len(intersection) > 2:
            planar_args = {k: v for k, v in plotting_parameters.items()
                           if k in ["edgecolor", "linewidth", "facecolor",
                                    "hatch", "alpha"]}
            if sim_size.x == 0:
                ax.add_patch(patches.Polygon(
                    sort_points([[a.y, a.z] for a in intersection]), **planar_args))
            elif sim_size.y == 0:
                ax.add_patch(patches.Polygon(
                    sort_points([[a.x, a.z] for a in intersection]), **planar_args))
            elif sim_size.z == 0:
                ax.add_patch(patches.Polygon(
                    sort_points([[a.x, a.y] for a in intersection]), **planar_args))
            return ax
    return ax


def _add_colorbar(ax, cmap: str, vmin: float, vmax: float,
                  default_label: Optional[str] = None,
                  colorbar_parameters: Optional[dict] = None) -> None:
    """Add a colorbar to the parent Figure of 'ax' by creating an additional Axes."""
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if colorbar_parameters is None:
        colorbar_parameters = copy.deepcopy(default_colorbar_parameters)
    else:
        colorbar_parameters = dict(default_colorbar_parameters, **colorbar_parameters)

    if colorbar_parameters["label"] is None:
        colorbar_parameters["label"] = default_label

    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax),
                               cmap=mpl.colormaps[cmap])

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes(
        pad=colorbar_parameters.pop("pad"),
        size=colorbar_parameters.pop("size"),
        position=colorbar_parameters.pop("position"),
    )
    fig = ax.get_figure()
    fig.colorbar(mappable=sm, cax=cax, **colorbar_parameters)


def _sample_eps(sim, xtics, ytics, ztics) -> np.ndarray:
    """Sample the shim's rasterized eps at the requested tics (nearest cell).

    Stands in for meep's get_epsilon_grid: frequency-independent, drawn from
    sim._rm.eps_tm / eps_te_xx (2D) or sim._eps3d (3D).
    """
    gs = sim.gs

    def idx(tics, span, n):
        return np.clip(np.round(np.asarray(tics) / gs.dx
                                + span / (2 * gs.dx) - 0.5).astype(int), 0, n - 1)

    if sim.dimensions == 3:
        eps = sim._eps3d.detach().cpu().numpy()
        ii = idx(xtics, gs.sx, eps.shape[0])
        jj = idx(ytics, gs.sy, eps.shape[1])
        kk = idx(ztics, gs.sz, eps.shape[2])
        sub = eps[np.ix_(ii, jj, kk)]
        for axis in reversed(range(3)):
            if sub.shape[axis] == 1 and sub.ndim > 2:
                sub = np.squeeze(sub, axis=axis)
        return sub
    rm = getattr(sim, "_rm", None)
    if rm is not None:
        t = rm.eps_tm if getattr(sim, "_pol", "tm") == "tm" else rm.eps_te_xx
        eps = t.detach().cpu().numpy()
    else:
        eps = np.asarray(sim.get_array(component=Dielectric))
    ii = idx(xtics, gs.sx, eps.shape[0])
    jj = idx(ytics, gs.sy, eps.shape[1])
    return eps[np.ix_(ii, jj)]


def plot_eps(sim, ax=None, output_plane: Optional[Volume] = None,
             eps_parameters: Optional[dict] = None,
             colorbar_parameters: Optional[dict] = None,
             frequency: Optional[float] = None,
             label_geometry: bool = False):
    if eps_parameters is None:
        eps_parameters = default_eps_parameters
    else:
        eps_parameters = dict(default_eps_parameters, **eps_parameters)

    # Determine a frequency to plot all epsilon (meep chain; the shim's eps
    # display is frequency-independent, the value is informational only)
    if frequency is not None:
        warnings.warn(
            "The frequency parameter of plot2D has been deprecated. "
            "Use the frequency key of the eps_parameters dictionary instead.")
        eps_parameters["frequency"] = frequency
    if eps_parameters["frequency"] is None:
        sources = _sim_sources(sim)
        try:
            eps_parameters["frequency"] = sources[0].frequency       # continuous
        except (AttributeError, IndexError):
            try:
                eps_parameters["frequency"] = sources[0].src.frequency   # gaussian
            except (AttributeError, IndexError):
                try:
                    eps_parameters["frequency"] = sources[0].src.center_frequency
                except (AttributeError, IndexError):
                    eps_parameters["frequency"] = 0

    sim_center, sim_size = get_2D_dimensions(sim, output_plane)
    xmin, xmax, ymin, ymax, zmin, zmax = box_vertices(
        sim_center, sim_size, _is_cylindrical(sim))

    grid_resolution = eps_parameters["resolution"] or _resolution(sim)
    Nx = int((xmax - xmin) * grid_resolution + 1)
    Ny = int((ymax - ymin) * grid_resolution + 1)
    Nz = int((zmax - zmin) * grid_resolution + 1)

    if sim_size.x == 0:
        # Plot y on x axis, z on y axis (YZ plane)
        extent = [ymin, ymax, zmin, zmax]
        xlabel, ylabel = "Y", "Z"
        xtics = np.array([sim_center.x])
        ytics = np.linspace(ymin, ymax, Ny)
        ztics = np.linspace(zmin, zmax, Nz)
    elif sim_size.y == 0:
        # Plot x on x axis, z on y axis (XZ plane)
        extent = [xmin, xmax, zmin, zmax]
        xlabel = "R" if (sim.dimensions == CYLINDRICAL) or _is_cylindrical(sim) else "X"
        ylabel = "Z"
        xtics = np.linspace(xmin, xmax, Nx)
        ytics = np.array([sim_center.y])
        ztics = np.linspace(zmin, zmax, Nz)
    elif sim_size.z == 0:
        # Plot x on x axis, y on y axis (XY plane)
        extent = [xmin, xmax, ymin, ymax]
        xlabel, ylabel = "X", "Y"
        xtics = np.linspace(xmin, xmax, Nx)
        ytics = np.linspace(ymin, ymax, Ny)
        ztics = np.array([sim_center.z])
    else:
        raise ValueError("A 2D plane has not been specified...")

    if hasattr(sim, "get_epsilon_grid"):
        eps_data = np.rot90(np.real(sim.get_epsilon_grid(
            xtics, ytics, ztics, eps_parameters["frequency"])))
    else:
        eps_data = np.rot90(np.real(_sample_eps(sim, xtics, ytics, ztics)))

    if am_master():
        # If Axes was not provided, just return the eps_data, otherwise plot
        if not ax:
            return eps_data

        if eps_parameters["contour"]:
            ax.contour(eps_data, 0, levels=np.unique(eps_data), colors="black",
                       origin="upper", extent=extent,
                       linewidths=eps_parameters["contour_linewidth"])
        else:
            ax.imshow(eps_data, extent=extent, **filter_dict(eps_parameters, ax.imshow))

        if eps_parameters["colorbar"]:
            _add_colorbar(ax=ax, cmap=eps_parameters["cmap"],
                          vmin=np.amin(eps_data), vmax=np.amax(eps_data),
                          default_label=r"$\epsilon_r$",
                          colorbar_parameters=colorbar_parameters)

        if label_geometry:
            # shim Simulation does not retain the geometry list; label each
            # element only when a `geometry` attribute (with .label) exists
            for el in getattr(sim, "geometry", []) or []:
                if getattr(el, "label", None) is None:
                    continue
                if sim_size.x == 0:
                    firsts = (el.center.y, el.center.z, sim_center.y, sim_center.z)
                elif sim_size.y == 0:
                    firsts = (el.center.x, el.center.z, sim_center.x, sim_center.z)
                else:
                    firsts = (el.center.x, el.center.y, sim_center.x, sim_center.y)
                ax = place_label(ax, el.label, *firsts,
                                 label_parameters=default_label_geometry_parameters)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax


def get_boundary_volumes(sim, thickness: float, direction: float, side) -> Volume:
    cell_size = _cell_size(sim)
    geometry_center = _geometry_center(sim)
    xmin, xmax, ymin, ymax, zmin, zmax = box_vertices(
        geometry_center, cell_size, _is_cylindrical(sim))

    if direction == X and side == Low:
        return Volume(
            center=Vector3(xmin + 0.5 * thickness, geometry_center.y, geometry_center.z),
            size=Vector3(thickness, cell_size.y, cell_size.z))
    elif (direction == X and side == High) or direction == R:
        return Volume(
            center=Vector3(xmax - 0.5 * thickness, geometry_center.y, geometry_center.z),
            size=Vector3(thickness, cell_size.y, cell_size.z))
    elif direction == Y and side == Low:
        return Volume(
            center=Vector3(geometry_center.x, ymin + 0.5 * thickness, geometry_center.z),
            size=Vector3(cell_size.x, thickness, cell_size.z))
    elif direction == Y and side == High:
        return Volume(
            center=Vector3(geometry_center.x, ymax - 0.5 * thickness, geometry_center.z),
            size=Vector3(cell_size.x, thickness, cell_size.z))
    elif direction == Z and side == Low:
        xcen = geometry_center.x
        if _is_cylindrical(sim):
            xcen += 0.5 * cell_size.x
        return Volume(center=Vector3(xcen, geometry_center.y, zmin + 0.5 * thickness),
                      size=Vector3(cell_size.x, cell_size.y, thickness))
    elif direction == Z and side == High:
        xcen = geometry_center.x
        if _is_cylindrical(sim):
            xcen += 0.5 * cell_size.x
        return Volume(center=Vector3(xcen, geometry_center.y, zmax - 0.5 * thickness),
                      size=Vector3(cell_size.x, cell_size.y, thickness))
    else:
        raise ValueError("Invalid boundary type")


def plot_boundaries(sim, ax, output_plane: Optional[Volume] = None,
                    boundary_parameters: Optional[dict] = None):
    if boundary_parameters is None:
        boundary_parameters = default_boundary_parameters
    else:
        boundary_parameters = dict(default_boundary_parameters, **boundary_parameters)

    import itertools

    for boundary in _boundary_layers(sim):
        # shim PML/Absorber carry no direction/side -> meep's ALL/ALL default
        direction = getattr(boundary, "direction", None)
        direction = ALL if direction is None else direction
        side = getattr(boundary, "side", None)
        side = ALL if side is None else side

        # boundary on all four sides
        if direction == ALL and side == ALL:
            if sim.dimensions == 1:
                dims = [X]
            elif sim.dimensions == CYLINDRICAL or _is_cylindrical(sim):
                dims = [X, Z]
            elif sim.dimensions == 2:
                dims = [X] if getattr(sim, "_y_pmc", False) else [X, Y]
            elif sim.dimensions == 3:
                dims = [X, Y, Z]
            else:
                raise ValueError("Invalid simulation dimensions")
            for permutation in itertools.product(dims, [Low, High]):
                if ((permutation[0] == X) and (permutation[1] == Low)) and (
                        sim.dimensions == CYLINDRICAL or _is_cylindrical(sim)):
                    continue
                vol = get_boundary_volumes(sim, boundary.thickness, *permutation)
                ax = plot_volume(sim, ax, vol, output_plane,
                                 plotting_parameters=boundary_parameters)
        # boundary on only two of four sides
        elif side == ALL:
            for s in [Low, High]:
                if ((direction == X) and (s == Low)) and (
                        sim.dimensions == CYLINDRICAL or _is_cylindrical(sim)):
                    continue
                vol = get_boundary_volumes(sim, boundary.thickness, direction, s)
                ax = plot_volume(sim, ax, vol, output_plane,
                                 plotting_parameters=boundary_parameters)
        # boundary on just one side
        else:
            if ((direction == X) and (side == Low)) and (
                    sim.dimensions == CYLINDRICAL or _is_cylindrical(sim)):
                continue
            vol = get_boundary_volumes(sim, boundary.thickness, direction, side)
            ax = plot_volume(sim, ax, vol, output_plane,
                             plotting_parameters=boundary_parameters)
    return ax


def plot_sources(sim, ax, output_plane: Optional[Volume] = None,
                 labels: bool = False, source_parameters: Optional[dict] = None):
    if source_parameters is None:
        source_parameters = default_source_parameters
    else:
        source_parameters = dict(default_source_parameters, **source_parameters)

    label = "source" if labels else None
    cell = _cell_size(sim)

    for src in _sim_sources(sim):
        vol = Volume(center=src.center or Vector3(),
                     size=_finite_size(src.size, cell))
        ax = plot_volume(sim, ax, vol, output_plane,
                         plotting_parameters=source_parameters, label=label)
    return ax


def _monitor_volumes(sim) -> List[Volume]:
    """Monitor regions as Volumes: sim.dft_objects[..].regions when present
    (meep layout), else reconstructed from the shim's line-monitor metadata
    ({'idx','lo','hi'} grid indices; lines normal to x)."""
    vols = []
    for mon in getattr(sim, "dft_objects", None) or []:
        for reg in getattr(mon, "regions", None) or []:
            vols.append(Volume(center=reg.center, size=reg.size))
    if vols:
        return vols
    gs = sim.gs
    for (_name, _kind, _mon, meta) in getattr(sim, "_monitors", []):
        if not isinstance(meta, dict) or not {"idx", "lo", "hi"} <= meta.keys():
            continue
        x = meta["idx"] * gs.dx - gs.sx / 2
        ycen = (meta["lo"] + meta["hi"]) / 2 * gs.dx - gs.sy / 2
        vols.append(Volume(center=Vector3(x, ycen),
                           size=Vector3(0, (meta["hi"] - meta["lo"]) * gs.dx)))
    return vols


def plot_monitors(sim, ax, output_plane: Optional[Volume] = None,
                  labels: bool = False, monitor_parameters: Optional[dict] = None):
    if monitor_parameters is None:
        monitor_parameters = default_monitor_parameters
    else:
        monitor_parameters = dict(default_monitor_parameters, **monitor_parameters)

    label = "monitor" if labels else None

    for vol in _monitor_volumes(sim):
        ax = plot_volume(sim, ax, vol, output_plane,
                         plotting_parameters=monitor_parameters, label=label)
    return ax


def plot_fields(sim, ax=None, fields=None, output_plane: Optional[Volume] = None,
                field_parameters: Optional[dict] = None,
                colorbar_parameters: Optional[dict] = None):
    components = {
        "Ex", "Ey", "Ez", "Er", "Ep",
        "Dx", "Dy", "Dz", "Dr", "Dp",
        "Hx", "Hy", "Hz", "Hr", "Hp",
        "Bx", "By", "Bz", "Br", "Bp",
        "Sx", "Sy", "Sz", "Sr", "Sp",
    }

    if fields is None:
        return ax

    if field_parameters is None:
        field_parameters = default_field_parameters
    else:
        field_parameters = dict(default_field_parameters, **field_parameters)

    if fields not in components:
        raise ValueError("Please specify a valid field component (mp.Ex, mp.Ey, ...")

    sim_center, sim_size = get_2D_dimensions(sim, output_plane)
    xmin, xmax, ymin, ymax, zmin, zmax = box_vertices(
        sim_center, sim_size, _is_cylindrical(sim))

    if sim_size.x == 0:
        extent = [ymin, ymax, zmin, zmax]
        xlabel, ylabel = "Y", "Z"
    elif sim_size.y == 0:
        extent = [xmin, xmax, zmin, zmax]
        xlabel = "R" if (sim.dimensions == CYLINDRICAL) or _is_cylindrical(sim) else "X"
        ylabel = "Z"
    elif sim_size.z == 0:
        extent = [xmin, xmax, ymin, ymax]
        xlabel, ylabel = "X", "Y"

    # shim get_array returns the full cell; crop to the output plane here
    field_data = np.asarray(sim.get_array(center=sim_center, size=sim_size,
                                          component=fields))
    if field_data.ndim == 2:
        gs = sim.gs
        i0 = max(0, int(round((xmin + gs.sx / 2) / gs.dx)))
        i1 = min(field_data.shape[0], int(round((xmax + gs.sx / 2) / gs.dx)))
        j0 = max(0, int(round((ymin + gs.sy / 2) / gs.dx)))
        j1 = min(field_data.shape[1], int(round((ymax + gs.sy / 2) / gs.dx)))
        field_data = field_data[i0:max(i1, i0 + 1), j0:max(j1, j0 + 1)]

    field_data = field_parameters["post_process"](field_data)

    if (sim.dimensions == CYLINDRICAL) or _is_cylindrical(sim):
        field_data = np.flipud(field_data)
    else:
        field_data = np.rot90(field_data)

    # Either plot the field, or return the array
    if not ax:
        return field_data

    if am_master():
        ax.imshow(field_data, extent=extent,
                  **filter_dict(field_parameters, ax.imshow))
        if field_parameters["colorbar"]:
            _add_colorbar(ax=ax, cmap=field_parameters["cmap"],
                          vmin=np.amin(field_data), vmax=np.amax(field_data),
                          default_label="field value",
                          colorbar_parameters=colorbar_parameters)
    return ax


def plot2D(sim, ax=None, output_plane: Optional[Volume] = None, fields=None,
           labels: bool = False, label_geometry: bool = False,
           eps_parameters: Optional[dict] = None,
           boundary_parameters: Optional[dict] = None,
           source_parameters: Optional[dict] = None,
           monitor_parameters: Optional[dict] = None,
           field_parameters: Optional[dict] = None,
           colorbar_parameters: Optional[dict] = None,
           frequency: Optional[float] = None,
           show_epsilon: bool = True,
           show_sources: bool = True,
           show_monitors: bool = True,
           show_boundary_layers: bool = True,
           nb: bool = False,
           **kwargs):

    plot_eps_flag = kwargs.get("plot_eps_flag")
    if plot_eps_flag:
        warnings.warn("plot_eps_flag is deprecated. Use show_epsilon instead.")
        show_epsilon = plot_eps_flag

    plot_sources_flag = kwargs.get("plot_sources_flag")
    if plot_sources_flag:
        warnings.warn("plot_sources_flag is deprecated. Use show_sources instead.")
        show_sources = plot_sources_flag

    plot_monitors_flag = kwargs.get("plot_monitors_flag")
    if plot_monitors_flag:
        warnings.warn("plot_monitors_flag is deprecated. Use show_monitors instead.")
        show_monitors = plot_monitors_flag

    plot_boundaries_flag = kwargs.get("plot_boundaries_flag")
    if plot_boundaries_flag:
        warnings.warn(
            "plot_boundaries_flag is deprecated. Use show_boundary_layers instead.")
        show_boundary_layers = plot_monitors_flag

    # Ensure a figure axis exists
    if ax is None and am_master():
        from matplotlib import pyplot as plt

        ax = plt.gca()

    # validate the output plane to ensure proper 2D coordinates
    sim_center, sim_size = get_2D_dimensions(sim, output_plane)
    output_plane = Volume(center=sim_center, size=sim_size)

    if eps_parameters is not None and field_parameters is not None:
        if field_parameters.get("colorbar", False) and eps_parameters.get(
                "colorbar", False):
            raise ValueError(
                "'colorbar' parameter can only be specified for epsilon or "
                "fields, but not both.")

    if show_epsilon:
        ax = plot_eps(sim, ax, output_plane=output_plane,
                      eps_parameters=eps_parameters,
                      colorbar_parameters=colorbar_parameters,
                      frequency=frequency, label_geometry=label_geometry)
    if show_boundary_layers:
        ax = plot_boundaries(sim, ax, output_plane=output_plane,
                             boundary_parameters=boundary_parameters)
    if show_sources:
        ax = plot_sources(sim, ax, output_plane=output_plane, labels=labels,
                          source_parameters=source_parameters)
    if show_monitors:
        ax = plot_monitors(sim, ax, output_plane=output_plane, labels=labels,
                           monitor_parameters=monitor_parameters)
    if fields is not None:
        ax = plot_fields(sim, ax, fields, output_plane=output_plane,
                         field_parameters=field_parameters,
                         colorbar_parameters=colorbar_parameters)
    if am_master() and nb:
        display_figure_immediately(ax.figure)
        sleep(0.05)
    return ax


def display_figure_immediately(fig) -> None:
    """Force an ipympl figure to display immediately (meep helper; only used
    with plot2D(nb=True) inside Jupyter)."""
    from IPython.display import display

    canvas = fig.canvas
    display(canvas)
    canvas._handle_message(canvas, {"type": "send_image_mode"}, [])
    canvas._handle_message(canvas, {"type": "refresh"}, [])
    canvas._handle_message(canvas, {"type": "initialized"}, [])
    canvas._handle_message(canvas, {"type": "draw"}, [])


# ------------------------------------------------------- #
# JS_Animation
# ------------------------------------------------------- #
class JS_Animation:
    """Helper making jshtml animations embed seamlessly in Jupyter (meep)."""

    def __init__(self, jshtml: str):
        self.jshtml = jshtml

    def _repr_html_(self) -> str:
        return self.jshtml

    def get_jshtml(self) -> str:
        return self.jshtml


# ------------------------------------------------------- #
# Animate2D
# ------------------------------------------------------- #
class Animate2D:
    """Records plot2D frames during timestepping (meep Animate2D).

    Step-function protocol (meep): the object is called as
    ``animate(sim, 'step')`` on each firing and ``animate(sim, 'finish')`` at
    the end of the run; ``__code__.co_argcount == 2`` marks it as a
    two-argument step function for the run loop. Pass it to
    ``sim.run(mp.at_every(dt, animate), until=...)``.

    Frames are stored in memory as PNGs (or as raw field arrays when
    ``normalize=True``, normalized at 'finish'). Export with ``to_mp4`` /
    ``to_gif`` (both pipe the PNGs through ffmpeg) or ``to_jshtml``.

    Shim note: fields come from the engine state ``sim.get_array`` exposes,
    so frames rendered mid-run show the state as of the last completed engine
    chunk; a sim that has never stepped raises the shim's RuntimeError.
    """

    def __init__(self, sim=None, fields=None, f=None, realtime: bool = False,
                 normalize: bool = False, plot_modifiers: Optional[list] = None,
                 update_epsilon: bool = False, nb: bool = False,
                 **customization_args):
        if sim is not None:
            warnings.warn(
                "Warning: The 'sim' argument in Animate2D is deprecated and has "
                "no effect. It will be removed in a future release.")

        self.fields = fields
        self.update_epsilon = update_epsilon
        self.nb = nb

        if f:
            self.f = f
            self.ax = self.f.gca()
        elif am_master():
            from matplotlib import pyplot as plt

            if self.nb:
                plt.ioff()
            self.f = plt.figure()
            if self.nb:
                plt.ion()
            self.ax = self.f.gca()
        else:
            self.f = None
            self.ax = None

        self.realtime = realtime
        self.normalize = normalize
        self.plot_modifiers = plot_modifiers
        self.customization_args = customization_args

        self.cumulative_fields = []
        self._saved_frames = []

        self.frame_format = "png"   # format in which each frame is saved in memory
        self.codec = "h264"         # encoding of mp4 video
        self.default_mode = "loop"  # html5 video control mode

        self.init = False

        # Needed for step functions (meep's two-argument detection hack)
        self.__code__ = namedtuple("gna_hack", ["co_argcount"])
        self.__code__.co_argcount = 2

    def __call__(self, sim, todo: str) -> None:
        from matplotlib import pyplot as plt

        if todo == "step":
            if not self.init:
                filtered_plot2D = filter_dict(self.customization_args, plot2D)
                ax = plot2D(sim, ax=self.ax, fields=self.fields, nb=self.nb,
                            **filtered_plot2D)
                if self.plot_modifiers:
                    for k in range(len(self.plot_modifiers)):
                        ax = self.plot_modifiers[k](self.ax)
                if am_master():
                    fields = ax.images[-1].get_array() if ax.images else None
                    self.ax = ax
                    self.w, self.h = self.f.get_size_inches()
                self.init = True
            else:
                if self.update_epsilon:
                    filtered_plot_eps = filter_dict(self.customization_args, plot_eps)
                    # when calling with no 'ax', returns array of epsilon data
                    eps = plot_eps(sim=sim, **filtered_plot_eps)
                    if am_master():
                        eps_idx = -1 if not self.fields else -2
                        self.ax.images[eps_idx].set_data(eps)
                # Need to check if None because mp.Ex == 0
                if self.fields is not None:
                    filtered_plot_fields = filter_dict(self.customization_args,
                                                       plot_fields)
                    # when calling with no 'ax', returns array of fields data
                    fields = plot_fields(sim, fields=self.fields,
                                         **filtered_plot_fields)
                    if am_master():
                        self.ax.images[-1].set_data(fields)
                        self.ax.images[-1].set_clim(vmin=0.8 * np.min(fields),
                                                    vmax=0.8 * np.max(fields))
                if self.nb and am_master():
                    self.f.canvas.draw()

            if self.realtime and am_master():
                plt.pause(0.05) if not self.nb else sleep(0.05)

            if self.normalize and am_master():
                # Save fields as a numpy array to be normalized and saved later
                self.cumulative_fields.append(fields)
            elif am_master():
                # Capture figure as a png kept in memory (no disk writes)
                self.grab_frame()
            return
        elif todo == "finish":
            # Normalize the frames, if requested, and export
            if self.normalize and am_master():
                fields = np.array(self.cumulative_fields) / np.max(
                    np.abs(self.cumulative_fields), axis=(0, 1, 2))
                for k in range(len(self.cumulative_fields)):
                    self.ax.images[-1].set_data(fields[k, :, :])
                    self.ax.images[-1].set_clim(vmin=-0.8, vmax=0.8)
                    self.grab_frame()
            return

    @property
    def frame_size(self) -> Tuple[int, int]:
        # (width, height) in pixels of a movie frame (from matplotlib)
        w, h = self.f.get_size_inches()
        return int(w * self.f.dpi), int(h * self.f.dpi)

    def grab_frame(self) -> None:
        # Saves the figure's frame to memory (from matplotlib)
        from io import BytesIO

        bin_data = BytesIO()
        self.f.savefig(bin_data, format=self.frame_format)
        self._saved_frames.append(bin_data.getvalue())

    def _embedded_frames(self, frame_list: list, frame_format: str) -> str:
        # frame data in memory -> html5-friendly base64 (from matplotlib)
        import base64

        template = '  frames[{0}] = "data:image/{1};base64,{2}"\n'
        return "\n" + "".join(
            template.format(
                i, frame_format,
                base64.encodebytes(frame_data).decode("ascii").replace("\n", "\\\n"))
            for i, frame_data in enumerate(frame_list))

    def to_jshtml(self, fps: int) -> JS_Animation:
        """Interactive JSHTML animation object embeddable in Jupyter, with
        playback controls, at frame rate `fps`."""
        if am_master():
            from uuid import uuid4

            from matplotlib._animation_data import (DISPLAY_TEMPLATE, JS_INCLUDE,
                                                    STYLE_INCLUDE)

            fill_frames = self._embedded_frames(self._saved_frames, self.frame_format)
            Nframes = len(self._saved_frames)
            mode_dict = dict(once_checked="", loop_checked="", reflect_checked="")
            mode_dict[self.default_mode + "_checked"] = "checked"

            interval = 1000 // fps

            html_string = ""
            html_string += JS_INCLUDE
            html_string += STYLE_INCLUDE
            html_string += DISPLAY_TEMPLATE.format(
                id=uuid4().hex, Nframes=Nframes, fill_frames=fill_frames,
                interval=interval, **mode_dict)
            return JS_Animation(html_string)

    def to_gif(self, fps: int, filename: str) -> None:
        """Write the animation as a GIF at frame rate `fps`. Requires ffmpeg."""
        if am_master():
            from subprocess import PIPE, Popen

            FFMPEG_BIN = "ffmpeg"
            command = [
                FFMPEG_BIN,
                "-f", "image2pipe",             # force piping of rawvideo
                "-vcodec", self.frame_format,   # raw input codec
                "-s", "%dx%d" % (self.frame_size),
                "-r", str(fps),                 # frame rate in frames per second
                "-i", "pipe:",                  # the input comes from a pipe
                "-vcodec", "gif",               # output gif format
                "-r", str(fps),
                "-y",
                "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
                "-an", filename,                # output filename
            ]
            proc = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            for i in range(len(self._saved_frames)):
                proc.stdin.write(self._saved_frames[i])
            out, err = proc.communicate()       # pipe in images
            proc.stdin.close()
            proc.wait()
        return

    def to_mp4(self, fps: int, filename: str) -> None:
        """Write the animation as an mp4 (h264/yuv420p) at frame rate `fps`.
        Requires ffmpeg."""
        if am_master():
            from subprocess import PIPE, Popen

            FFMPEG_BIN = "ffmpeg"
            command = [
                FFMPEG_BIN,
                "-f", "image2pipe",             # force piping of rawvideo
                "-vcodec", self.frame_format,   # raw input codec
                "-s", "%dx%d" % (self.frame_size),
                "-r", str(fps),                 # frame rate in frames per second
                "-i", "pipe:",                  # the input comes from a pipe
                "-vcodec", self.codec,          # output mp4 format
                "-pix_fmt", "yuv420p",
                "-r", str(fps),
                "-y",
                "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
                "-an", filename,                # output filename
            ]
            proc = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            for i in range(len(self._saved_frames)):
                proc.stdin.write(self._saved_frames[i])
            out, err = proc.communicate()       # pipe in images
            proc.stdin.close()
            proc.wait()
        return

    def reset(self) -> None:
        self.cumulative_fields = []
        self.ax = None
        self.f = None

    def set_figure(self, f) -> None:
        self.f = f
