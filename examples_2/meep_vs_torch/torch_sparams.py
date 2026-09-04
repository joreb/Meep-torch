# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch side of the meep-vs-torch parity check: the SAME ported +
extended component the gmeep script simulates (ports_util.with_ports +
gf.c.extend_ports) is rasterized with component_to_eps and run through
meep-torch FDTD with a TE0 mode source on o1 and mode monitors on every
output port. |S|^2 at 674 nm is compared against the gmeep npz written by
gmeep_fdtd.py, and the propagation-field correlations from
compare_propagation.py are folded into one parity verdict.

Run: uv run python examples_2/meep_vs_torch/torch_sparams.py
(after gmeep_fdtd.py has produced the *_sp.npz files)
"""
import pathlib
import sys

import gdsfactory as gf
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from ports_util import (CASES, EXTEND, TPML, WL, RES, clamped_margin,
                        design_gds, with_ports)

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_tm import init_tm_state
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port, validate_ports

gf.gpdk.PDK.activate()

HERE = pathlib.Path(__file__).parent
GDS_DIR = HERE.parent
FCEN = 1.0 / WL
N_SIN = Si3N4.n(FCEN).real
N_SIO2 = SiO2.n(FCEN).real
NPML = RES            # 1 um of PML, mirrors tpml=1.0 on the gmeep side
T_RUN = 200.0
SP_ATOL = 0.05        # |S|^2 parity gate vs gmeep


def build_sim(name):
    """Shared setup: ported+extended geometry -> grid, eps, coeffs, ports.

    Returns dict(grid, eps, coeffs, p_src, ports, outs, extent) where ports
    = [input monitor] + output monitors, planes mirroring gmeep defaults.
    """
    c = with_ports(design_gds(GDS_DIR, name), name)
    margin = clamped_margin(c)
    ext = gf.c.extend_ports(c, length=EXTEND).copy()
    ext.flatten()
    # keyhole polygons (holes cut by export_gds) trip the strict meep-torch
    # Polygon validator; shapely make_valid splits them into simple parts
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.validation import make_valid
    verts = []
    for plist in ext.get_polygons_points(merge=False).values():
        for p in plist:
            sp = ShapelyPolygon(p)
            geoms = [sp] if sp.is_valid else [
                g for g in getattr(make_valid(sp), "geoms", [make_valid(sp)])
                if g.geom_type == "Polygon"]
            verts += [np.asarray(g.exterior.coords)[:-1]
                      for g in geoms if g.area > 1e-3]
    # extra 1 um clad in y so wide-facet port lines end in cladding (the
    # ModeSourceTM guided-mode check needs clad at the line ends)
    pad_y = TPML + 1.0
    x0, y0 = ext.xmin - TPML, ext.ymin - pad_y
    grid = Grid(nx=int(round((ext.xsize + 2 * TPML) * RES)),
                ny=int(round((ext.ysize + 2 * pad_y) * RES)), dx=1.0 / RES)
    eps = rasterize_scene([(Polygon(v - [x0, y0]), N_SIN ** 2) for v in verts],
                          grid, background=N_SIO2 ** 2).eps_tm

    def make_port(p, direction, inward):
        # plane `inward` um inside the device from the port edge, mirroring
        # gmeep defaults: source o1+0.1, o1 monitor +0.3, outputs -0.1
        x = p.center[0] + (-inward if p.orientation == 0 else inward)
        half = p.width / 2 + margin
        lo = max(NPML + 2, int(round((p.center[1] - half - y0) * RES)))
        hi = min(grid.ny - NPML - 2, int(round((p.center[1] + half - y0) * RES)))
        return Port(axis="x", index=int(round((x - x0) * RES)), lo=lo, hi=hi,
                    direction=direction)

    outs = sorted((p for p in c.ports if p.name != "o1"), key=lambda p: p.name)
    p_src = make_port(c.ports["o1"], +1, 0.1)
    ports = [make_port(c.ports["o1"], +1, 0.3)] + [make_port(p, -1, 0.1)
                                                   for p in outs]
    validate_ports([p_src] + ports, grid, NPML)
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)
    extent = [x0, x0 + grid.nx / RES, y0, y0 + grid.ny / RES]
    return dict(grid=grid, eps=eps, coeffs=coeffs, p_src=p_src, ports=ports,
                outs=outs, extent=extent, x0=x0, y0=y0, margin=margin)


def torch_sparams(name):
    s = build_sim(name)
    grid, eps, coeffs = s["grid"], s["eps"], s["coeffs"]
    ports, outs = s["ports"], s["outs"]
    src = ModeSourceTM(grid, coeffs, s["p_src"], GaussianPulse(FCEN, 0.2), FCEN)
    mons = [FluxMonitorTM(grid, [FCEN], x_normal=True, idx=p.index,
                          lo=p.lo, hi=p.hi) for p in ports]
    _, accs = run_tm(init_tm_state(grid), coeffs, int(round(T_RUN / grid.dt)),
                     source=src, monitors=mons, checkpoint_every=128)
    dec, profiles = [], {}
    for p, gp, acc in zip([None] + outs, ports, accs):
        el = eps.detach().cpu()[gp.index, gp.lo:gp.hi]
        dec.append(decompose(acc.cpu(), el, grid.dx, [FCEN], "tm", 1,
                             axis_sign=-1.0))
        re_e, im_e = acc.cpu().unbind(1)[0], acc.cpu().unbind(1)[1]
        name_key = "o1" if p is None else p.name
        profiles[name_key] = dict(
            ez=torch.sqrt(re_e[0] ** 2 + im_e[0] ** 2).cpu().numpy(),
            ys=(np.arange(gp.lo, gp.hi) + 0.5) / RES + s["y0"])
    a_in = dec[0][0][0, 0]
    sp = {p.name: float(abs(dec[i + 1][0][0, 0] / a_in) ** 2)
          for i, p in enumerate(outs)}
    return sp, profiles


def gmeep_sparams(name):
    sp = np.load(HERE / f"{name}_sp.npz")
    mid = len(sp["wavelengths"]) // 2
    return {k.split("@")[0]: float(abs(sp[k][mid]) ** 2)
            for k in sp.files if k.endswith("o1@0") and not k.startswith("o1@")}


def main():
    all_ok = True
    for name in CASES:
        (ours, _), theirs = torch_sparams(name), gmeep_sparams(name)
        print(f"\n{name}:")
        for port in sorted(theirs):
            d = abs(ours[port] - theirs[port])
            ok = d <= SP_ATOL
            all_ok &= ok
            print(f"  |S({port}<-o1)|^2  meep-torch={ours[port]:.4f}  "
                  f"gmeep={theirs[port]:.4f}  |diff|={d:.4f}  "
                  f"{'OK' if ok else 'FAIL'} (atol {SP_ATOL})")
        cnpz = HERE / f"{name}_compare.npz"
        if cnpz.exists():
            corr = np.load(cnpz)["corr"]
            print(f"  propagation-field correlation (compare_propagation): "
                  f"min={corr.min():.4f}")
            all_ok &= corr.min() > 0.95
    print(f"\nPARITY {'HOLDS' if all_ok else 'BROKEN'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
