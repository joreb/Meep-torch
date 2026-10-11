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


def test_near2far_flux_closed_box_matches_near_flux_box():
    """Ez dipole inside a closed square: the far-box flux (DftNear2Far.flux)
    must equal the near flux-box power. Both are Re(E x H*) with no 1/2, as in
    meep; flux() once halved the far value (ratio 0.5)."""
    sim = mp.Simulation(cell_size=mp.Vector3(10, 10, 0), resolution=20,
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=0.2), component=mp.Ez,
                           center=mp.Vector3())],
        boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))
    sides = [(mp.Vector3(0, 1.5), mp.Vector3(3, 0), +1),
             (mp.Vector3(0, -1.5), mp.Vector3(3, 0), -1),
             (mp.Vector3(1.5, 0), mp.Vector3(0, 3), +1),
             (mp.Vector3(-1.5, 0), mp.Vector3(0, 3), -1)]
    n2f = sim.add_near2far(FCEN, 0, 1,
        *[mp.Near2FarRegion(center=c, size=s, weight=w) for c, s, w in sides])
    fl = sim.add_flux(FCEN, 0, 1,
        *[mp.FluxRegion(center=c, size=s, weight=w) for c, s, w in sides])
    sim.run(until=80)
    near = mp.get_fluxes(fl)[0]

    def far_side(d, center, size):
        return n2f.flux(d, mp.Volume(center=center, size=size), 0.5)[0]

    R = 50.0
    far = (far_side(mp.Y, mp.Vector3(0, R), mp.Vector3(2 * R, 0))
           - far_side(mp.Y, mp.Vector3(0, -R), mp.Vector3(2 * R, 0))
           + far_side(mp.X, mp.Vector3(R, 0), mp.Vector3(0, 2 * R))
           - far_side(mp.X, mp.Vector3(-R, 0), mp.Vector3(0, 2 * R)))
    assert near > 0
    assert abs(far - near) <= 0.03 * near, f"far {far:.4f} vs near {near:.4f}"


def test_get_farfield_phase_matches_get_dft_array():
    """get_farfield at a point outside the box must be get_dft_array's
    complex Ez there, not its conjugate. The kernels run e^{+i w t}; the
    output once skipped the conj back to meep's e^{-i w t}, which left |.|
    and fluxes intact but flipped Im (cavity-farfield)."""
    sim = mp.Simulation(cell_size=mp.Vector3(10, 10, 0), resolution=20,
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=0.2), component=mp.Ez,
                           center=mp.Vector3())],
        boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))
    n2f = sim.add_near2far(FCEN, 0, 1,
        mp.Near2FarRegion(center=mp.Vector3(1.0, 0), size=mp.Vector3(0, 2), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(-1.0, 0), size=mp.Vector3(0, 2), weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(0, 1.0), size=mp.Vector3(2, 0), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, -1.0), size=mp.Vector3(2, 0), weight=-1))
    mon = sim.add_dft_fields([mp.Ez], FCEN, 0, 1, center=mp.Vector3(0, 3),
                             size=mp.Vector3(4, 0))
    sim.run(until=80)
    dft = np.asarray(sim.get_dft_array(mon, mp.Ez, 0)).ravel()
    x, y, _, _ = sim.get_array_metadata(dft_cell=mon)
    ff = np.array([sim.get_farfield(n2f, mp.Vector3(xc, y[0]))[2] for xc in x])
    rel = np.linalg.norm(ff - dft) / np.linalg.norm(dft)
    rel_conj = np.linalg.norm(np.conj(ff) - dft) / np.linalg.norm(dft)
    assert rel < 0.3 < 0.8 < rel_conj, f"rel {rel:.3f}, rel(conj) {rel_conj:.3f}"
