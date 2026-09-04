# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""GDS round-trip parity: the design GDS files exported by export_gds.py are
loaded back via meep_torch.geometry.gds.polygons_from_gds and simulated with
meep-torch FDTD; the SAME files are loaded natively in real pymeep
(mp.get_GDSII_prisms, conda env meep-oracle) and simulated there. The
normalization-independent flux transmission ratio (out/in) must agree.

Run: uv run pytest examples_2/test_gds_meep_equivalence.py -v
Skips when the meep-oracle conda env is absent.
"""
import json
import pathlib
import subprocess

import numpy as np
import pytest

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene

HERE = pathlib.Path(__file__).parent
ORACLE = pathlib.Path.home() / "miniforge3/envs/meep-oracle/bin/python"

pytestmark = pytest.mark.skipif(
    not ORACLE.exists(), reason="real pymeep (conda env meep-oracle) not installed"
)

LAYER = 4              # SiN fab-deck layer used by export_gds.py
WL = 0.674
FCEN = 1.0 / WL
FWIDTH = 0.2
FREQS = [FCEN - 0.05, FCEN, FCEN + 0.05]
RES = 25               # cells/um, both engines
DX = 1.0 / RES
PAD = 1.0              # cladding pad between device bbox and PML, um
PMLT = 0.8             # PML thickness, um
T_TOTAL = 120.0        # run time in meep units, both engines
N_CORE = Si3N4.n(FCEN).real
N_CLAD = SiO2.n(FCEN).real

# The generic mp.Source/LineSource excitation (not a mode source) makes the
# ratio a pure discretization-parity observable, identical setup both sides.
ORACLE_SCRIPT = r"""
import json, sys
import meep as mp

p = json.load(open(sys.argv[1]))
prisms = mp.get_GDSII_prisms(mp.Medium(index=p["n_core"]), p["gds"],
                             p["layer"], -100, 100)
# geometry_center puts the whole sim in the GDS coordinate frame: sources,
# monitors, and probes are all specified in raw GDS coordinates.
sim = mp.Simulation(
    cell_size=mp.Vector3(p["sx"], p["sy"]),
    resolution=p["res"],
    geometry=prisms,
    geometry_center=mp.Vector3(p["cx"], p["cy"]),
    default_material=mp.Medium(index=p["n_clad"]),
    boundary_layers=[mp.PML(p["pml"])],
    sources=[mp.Source(
        mp.GaussianSource(p["fcen"], fwidth=p["fwidth"]),
        component=mp.Ez,
        center=mp.Vector3(p["x_src"], p["y_c"]),
        size=mp.Vector3(0, p["y_span"]))],
)
def flux_plane(x):
    return sim.add_flux(p["fcen"], p["freqs"][-1] - p["freqs"][0], len(p["freqs"]),
                        mp.FluxRegion(center=mp.Vector3(x, p["y_c"]),
                                      size=mp.Vector3(0, p["y_span"])))
fin, fout = flux_plane(p["x_in"]), flux_plane(p["x_out"])
sim.run(until=p["t_total"])
print("RESULT " + json.dumps({"fin": mp.get_fluxes(fin), "fout": mp.get_fluxes(fout)}))
"""


def load_clean_polys(gds, cell):
    """Layer-LAYER polygons of `cell`, repaired to simple polygons.

    export_gds.py's matplotlib-contour polygons can self-intersect (both the
    strict meep-torch Polygon validator and shapely agree); shapely
    make_valid splits them into simple parts so BOTH engines get identical,
    well-defined geometry.
    """
    import gdstk
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.validation import make_valid

    lib = gdstk.read_gds(gds)
    c = next(x for x in lib.cells if x.name == cell)
    out = []
    for p in c.get_polygons():
        if (p.layer, p.datatype) != (LAYER, 0):
            continue
        sp = ShapelyPolygon(p.points)
        geoms = [sp] if sp.is_valid else [
            g for g in getattr(make_valid(sp), "geoms", [make_valid(sp)])
            if g.geom_type == "Polygon"]
        for g in geoms:
            if g.area > 1e-3:   # drop degenerate slivers from the repair
                out.append(np.asarray(g.exterior.coords)[:-1])
    return out


def write_clean_gds(verts_list, cell, path):
    import gdstk
    lib = gdstk.Library()
    c = lib.new_cell(cell)
    for v in verts_list:
        c.add(gdstk.Polygon(v, layer=LAYER))
    lib.write_gds(path)


def _domain(polys):
    pts = np.vstack(polys)
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    ox, oy = x0 - PAD - PMLT, y0 - PAD - PMLT
    sx, sy = (x1 + PAD + PMLT) - ox, (y1 + PAD + PMLT) - oy
    return x0, x1, ox, oy, sx, sy


def torch_ratio(polys, ox, oy, sx, sy, x_src, x_in, x_out, y_lo, y_hi):
    grid = Grid(nx=int(round(sx * RES)), ny=int(round(sy * RES)), dx=DX)
    items = [(Polygon(v - [ox, oy]), N_CORE ** 2) for v in polys]
    eps = rasterize_scene(items, grid, background=N_CLAD ** 2).eps_tm
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=int(round(PMLT * RES)))
    j0, j1 = int(round((y_lo - oy) * RES)), int(round((y_hi - oy) * RES))
    src = LineSource(grid, ix=int(round((x_src - ox) * RES)), j0=j0, j1=j1,
                     profile=GaussianPulse(FCEN, FWIDTH))
    mon_in = FluxMonitorTM(grid, FREQS, x_normal=True,
                           idx=int(round((x_in - ox) * RES)), lo=j0, hi=j1)
    mon_out = FluxMonitorTM(grid, FREQS, x_normal=True,
                            idx=int(round((x_out - ox) * RES)), lo=j0, hi=j1)
    steps = int(round(T_TOTAL / (grid.courant * DX)))
    state = init_tm_state(grid)
    _, (a, b) = run_tm(state, coeffs, steps, source=src, monitors=[mon_in, mon_out])
    return (mon_out.flux(b, DX) / mon_in.flux(a, DX)).cpu().numpy()


def meep_ratio(gds, ox, oy, sx, sy, x_src, x_in, x_out, y_lo, y_hi, tmp_path):
    params = dict(
        gds=str(gds), layer=LAYER, n_core=N_CORE, n_clad=N_CLAD,
        sx=sx, sy=sy, cx=ox + sx / 2, cy=oy + sy / 2, pml=PMLT, res=RES,
        fcen=FCEN, fwidth=FWIDTH, freqs=FREQS, x_src=x_src, x_in=x_in,
        x_out=x_out, y_c=(y_lo + y_hi) / 2, y_span=y_hi - y_lo, t_total=T_TOTAL,
    )
    pfile = tmp_path / "params.json"
    pfile.write_text(json.dumps(params))
    sfile = tmp_path / "oracle.py"
    sfile.write_text(ORACLE_SCRIPT)
    out = subprocess.run([str(ORACLE), str(sfile), str(pfile)],
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, f"oracle meep failed:\n{out.stderr[-2000:]}"
    line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT "))
    r = json.loads(line[len("RESULT "):])
    return np.array(r["fout"]) / np.array(r["fin"])


# gdsfactory writes a $$$CONTEXT_INFO$$$ sidecar cell, so name the cell
CASES = [                       # (device_dir, cell) -> <dir>/output/<cell>.gds
    ("y_splitter", "c_y_splitter"),
    ("edge_coupler", "c_edge_coupler"),
]


@pytest.mark.parametrize("ddir,cell", CASES, ids=[c[1] for c in CASES])
def test_gds_flux_ratio_parity(ddir, cell, tmp_path):
    fname = f"{cell}.gds"
    polys = load_clean_polys(HERE / ddir / "output" / fname, cell)
    assert polys, f"no layer-{LAYER} polygons in {fname}"
    clean_gds = tmp_path / fname
    write_clean_gds(polys, cell, clean_gds)
    x0, x1, ox, oy, sx, sy = _domain(polys)
    x_src, x_in, x_out = x0 - 0.6, x0 - 0.25, x1 + 0.5
    y_lo, y_hi = oy + PMLT + 0.2, oy + sy - PMLT - 0.2

    ours = torch_ratio(polys, ox, oy, sx, sy, x_src, x_in, x_out, y_lo, y_hi)
    theirs = meep_ratio(clean_gds, ox, oy, sx, sy, x_src, x_in, x_out, y_lo,
                        y_hi, tmp_path)

    assert np.all(theirs > 0) and np.all(ours > 0)
    assert np.allclose(ours, theirs, rtol=0.05), (
        f"flux ratio mismatch\n meep-torch: {ours}\n pymeep:     {theirs}")
