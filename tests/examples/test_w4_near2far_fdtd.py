# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W4 T3: near2far from FDTD fields — Ez line dipole radiates isotropically."""
import math

import numpy as np

import meep_torch.mp as mp
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.near2far import (faces_from_accs, farfield_tm,
                                      near2far_monitors_tm)
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state

FCEN = 1.0


def _radial_flux(E, H, th):
    Sx = np.real(np.conj(E[:, 1]) * H[:, 2] - np.conj(E[:, 2]) * H[:, 1])
    Sy = np.real(np.conj(E[:, 2]) * H[:, 0] - np.conj(E[:, 0]) * H[:, 2])
    return Sx * np.cos(th) + Sy * np.sin(th)


def test_fdtd_ez_dipole_isotropic():
    grid = Grid(nx=200, ny=200, dx=0.05)         # 10x10 at 20 px/lambda, PML 1.0
    coeffs = build_tm_coeffs(grid, eps_inf=grid.zeros(200, 200) + 1.0, npml=20)
    src = LineSource(grid, ix=100, j0=100, j1=101, profile=GaussianPulse(FCEN, 0.2))
    mons = near2far_monitors_tm(grid, [FCEN], ix0=70, ix1=130, iy0=70, iy1=130)
    state = init_tm_state(grid)
    n_steps = int(round(80.0 / grid.dt))
    state, accs = run_tm(state, coeffs, n_steps, source=src,
                         monitors=[m for m, _ in mons])
    faces = faces_from_accs(grid, mons, accs)
    th = np.linspace(0, 2 * math.pi, 100, endpoint=False)
    r = 1000.0
    obs = np.stack([5.0 + r * np.cos(th), 5.0 + r * np.sin(th)], axis=1)
    E, H = farfield_tm(faces, [2 * math.pi * FCEN], obs)
    Sr = _radial_flux(E[0], H[0], th)
    assert (Sr > 0).all()
    assert Sr.std() / Sr.mean() < 0.02, f"anisotropy {Sr.std() / Sr.mean():.4f}"


def test_shim_add_near2far_ez():
    sim = mp.Simulation(cell_size=mp.Vector3(10, 10, 0), resolution=20,
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=0.2), component=mp.Ez,
                           center=mp.Vector3())],
        boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))
    n2f = sim.add_near2far(FCEN, 0, 1,
        mp.Near2FarRegion(center=mp.Vector3(1.5, 0), size=mp.Vector3(0, 3), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(-1.5, 0), size=mp.Vector3(0, 3), weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(0, 1.5), size=mp.Vector3(3, 0), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, -1.5), size=mp.Vector3(3, 0), weight=-1))
    sim.run(until=80)
    th = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    r = 1000.0
    ff = np.array([sim.get_farfield(n2f, mp.Vector3(r * math.cos(t), r * math.sin(t)))
                   for t in th])                  # (40, 6) complex
    E, H = ff[:, :3], ff[:, 3:]
    Sr = _radial_flux(E, H, th)
    assert (Sr > 0).all()
    assert Sr.std() / Sr.mean() < 0.02, f"anisotropy {Sr.std() / Sr.mean():.4f}"
