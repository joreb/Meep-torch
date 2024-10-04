import json
import pathlib

import torch
from meep_torch.core.grid import Grid
from meep_torch.ports.port import Port
from meep_torch.ports.smatrix import compute_smatrix

DATA = json.loads((pathlib.Path(__file__).parent / "data" / "meep_reference.json").read_text())
FCEN, FWIDTH = 0.15, 0.1


def test_straight_matches_meep():
    ref = DATA["straight"]
    grid = Grid(nx=200, ny=80, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    eps[:, 35:45] = 12.0
    ports = [Port(axis="x", index=40, lo=20, hi=60, direction=+1),
             Port(axis="x", index=160, lo=20, hi=60, direction=-1)]
    sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                         freqs=ref["freqs"], n_steps=3000, npml=10)
    for i in range(len(ref["freqs"])):
        assert abs(abs(sm.s[i, 1, 0]) - ref["S21_mag"][i]) < 0.03, (
            f"straight f={ref['freqs'][i]} |S21|={abs(sm.s[i,1,0])} vs meep {ref['S21_mag'][i]}")


def test_taper_matches_meep():
    ref = DATA["taper"]
    grid = Grid(nx=240, ny=100, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    for ix in range(grid.nx):
        # straight w=10 cells for x<10um, taper 10->20 cells over [10,14]um, then 20
        if ix < 100:
            w = 10
        elif ix >= 140:
            w = 20
        else:
            w = 10 + (ix - 100) * 10 // 40
        eps[ix, 50 - w // 2 : 50 + (w + 1) // 2] = 12.0
    ports = [Port(axis="x", index=40, lo=25, hi=75, direction=+1),
             Port(axis="x", index=200, lo=25, hi=75, direction=-1)]
    sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                         freqs=ref["freqs"], n_steps=3500, npml=10)
    for i in range(len(ref["freqs"])):
        assert abs(abs(sm.s[i, 1, 0]) - ref["S21_mag"][i]) < 0.03, (
            f"taper f={ref['freqs'][i]} |S21|={abs(sm.s[i,1,0])} vs meep {ref['S21_mag'][i]}")