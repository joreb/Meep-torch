# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Shared GDS-import + port-detection helpers for the meep-vs-torch
comparison (imported by both the gmeep script, meep-oracle env, and the
meep-torch script, uv venv -- so: gdsfactory/numpy only, NO meep imports)."""
import numpy as np

LAYER = (4, 0)
WL = 0.674
RES = 30
EXTEND = 4.0        # port straight extension, um
TPML = 1.0
# small: the gmeep cell is tight (bbox + tpml), a wide monitor span reaches
# into the PML band and pollutes the mode overlap
PORT_MARGIN = 0.5
CASES = ["c_y_splitter", "c_edge_coupler"]
DEVICE_DIRS = {"c_y_splitter": "y_splitter", "c_edge_coupler": "edge_coupler"}


def design_gds(ex2_root, name):
    """Exported design GDS inside its device folder (see ../export_gds.py)."""
    return ex2_root / DEVICE_DIRS[name] / "output" / f"{name}.gds"


def clamped_margin(c):
    """port_margin that keeps source/monitor planes inside the cell."""
    maxw = max(p.width for p in c.ports)
    return min(PORT_MARGIN, max(0.2, (c.ysize + 2 * TPML - maxw) / 2 - 0.3))


def port_intervals(comp, x_edge):
    """y-intervals where core crosses the vertical line x=x_edge."""
    from matplotlib.path import Path
    ys = np.arange(comp.ymin, comp.ymax, 0.01)
    pts = np.stack([np.full_like(ys, x_edge), ys], 1)
    inside = np.zeros(len(ys), bool)
    # dict is keyed by klayout layer index; single-layer file, take all
    for plist in comp.get_polygons_points(merge=False).values():
        for p in plist:
            inside |= Path(p).contains_points(pts)
    ivals, start = [], None
    for i, v in enumerate(inside):
        if v and start is None:
            start = ys[i]
        if not v and start is not None:
            ivals.append((start, ys[i - 1])); start = None
    if start is not None:
        ivals.append((start, ys[-1]))
    return [iv for iv in ivals if iv[1] - iv[0] > 0.1]


def sanitize_polys(comp):
    """Split keyhole/self-touching polygons into simple parts (shapely).

    export_gds.py writes holes as GDS keyhole cuts; meep prisms can render
    the zero-width cut line as a slit through the core, so both engines get
    the repaired simple parts instead.
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.validation import make_valid
    out = []
    for plist in comp.get_polygons_points(merge=False).values():
        for p in plist:
            sp = ShapelyPolygon(p)
            geoms = [sp] if sp.is_valid else [
                g for g in getattr(make_valid(sp), "geoms", [make_valid(sp)])
                if g.geom_type == "Polygon"]
            out += [np.asarray(g.exterior.coords)[:-1]
                    for g in geoms if g.area > 1e-3]
    return out


def with_ports(gds_path, name):
    """gf.import_gds + sanitized polygons + auto-detected o1 (left, input) /
    o2.. (right) ports."""
    import gdsfactory as gf
    imported = gf.import_gds(gds_path)
    comp = gf.Component(f"{name}_clean")
    for v in sanitize_polys(imported):
        comp.add_polygon(v, layer=LAYER)
    c = gf.Component(f"{name}_ported")
    c << comp
    n = 0
    for x_edge, orientation in [(comp.xmin + 0.03, 180), (comp.xmax - 0.03, 0)]:
        for y0, y1 in port_intervals(comp, x_edge):
            pname = "o1" if orientation == 180 else f"o{n + 2}"
            c.add_port(name=pname, center=(comp.xmin if orientation == 180
                       else comp.xmax, (y0 + y1) / 2),
                       width=round(y1 - y0, 3), orientation=orientation,
                       layer=LAYER)
            if orientation == 0:
                n += 1
    return c
