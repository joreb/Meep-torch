# SPDX-License-Identifier: MIT
# straight-waveguide scenario, single source of truth for the devito/pylops
# stencil experiments. Mirrors examples/01_basics/straight-waveguide.py:
#   cell 16x8, resolution 10 -> dx=0.1, grid 160x80
#   PML 1.0 on every edge -> npml=10 cells
#   eps=12 block, width 1 in y (|y|<0.5), infinite in x
#   ContinuousSource freq=0.15, Ez, center (-7, 0)
#   run until=200
#
# The reference uses the meep_torch.core TM stepper (build_tm_coeffs +
# step_tm) on the SAME Grid we hand devito/pylops, so dx, dt (courant=0.5)
# and PML thickness match exactly -- only the PML *recursion* differs
# (meep CPML vs the simple conductive absorber the stencils use), so the
# comparison gate is field correlation / peak, not bitwise.
import math
import os

import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper_tm import init_tm_state, step_tm

# --- geometry ---------------------------------------------------------------
CELL_X, CELL_Y = 16.0, 8.0
RESOLUTION = 10            # pixels per unit
DX = 1.0 / RESOLUTION      # 0.1
NX = int(round(CELL_X / DX))   # 160
NY = int(round(CELL_Y / DX))   # 80
PML = 1.0
NPML = int(round(PML / DX))    # 10

# waveguide: eps=12 where |y| < 0.5, else eps=1
WG_EPS = 12.0
WG_HALF_W = 0.5

# source: ContinuousSource freq=0.15, Ez, at (-7, 0)
FREQ = 0.15
SRC_X = -7.0
UNTIL = 200.0

# cell centred at origin: x in [-8, 8], y in [-4, 4]
X_LO, Y_LO = -CELL_X / 2, -CELL_Y / 2


def device() -> str:
    return os.environ.get("MEEP_TORCH_DEVICE", "cpu")


def eps_field(dev: str | None = None) -> torch.Tensor:
    """Permittivity map (NX, NY): 12 in the waveguide, 1 elsewhere."""
    dev = dev or device()
    ys = Y_LO + (np.arange(NY) + 0.5) * DX
    in_wg = np.abs(ys) < WG_HALF_W
    eps = np.ones((NX, NY), dtype=np.float64)
    eps[:, in_wg] = WG_EPS
    return torch.from_numpy(eps).to(dev)


def source_mask(dev: str | None = None) -> torch.Tensor:
    """Soft Ez source mask (NX, NY): 1 at the (-7, 0) cell, 0 elsewhere."""
    dev = dev or device()
    ix = int(round((SRC_X - X_LO) / DX))   # 10
    iy = NY // 2                            # 40 (y=0)
    m = np.zeros((NX, NY), dtype=np.float64)
    m[ix, iy] = 1.0
    return torch.from_numpy(m).to(dev)


def source_amp(t_index: int, dt: float) -> float:
    """ContinuousSource amplitude cos(2*pi*f*t) at integer step t_index."""
    return math.cos(2.0 * math.pi * FREQ * (t_index * dt))


def grid(dev: str | None = None) -> Grid:
    return Grid(nx=NX, ny=NY, dx=DX, courant=0.5, device=dev or device())


def n_steps(dt: float) -> int:
    return int(round(UNTIL / dt))


def reference_ez(dev: str | None = None) -> np.ndarray:
    """meep_torch.core TMz reference run -> Ez array (NX, NY) on the CPU.

    Same Grid (dx, courant=0.5, PML thickness) the devito/pylops stencils
    use, so the comparison is against controlled numerics -- only the PML
    recursion differs (meep CPML vs the stencils' conductive absorber)."""
    dev = dev or device()
    g = grid(dev)
    c = build_tm_coeffs(g, eps_inf=eps_field(dev), npml=NPML)
    st = init_tm_state(g)
    mask = source_mask(dev)
    for t in range(n_steps(g.dt)):
        st = step_tm(st, c, source_amp(t, g.dt), src_mask=mask)
    return st.Ez.cpu().numpy()


def mp_straight_waveguide_ez() -> np.ndarray:
    """The actual examples/01_basics/straight-waveguide.py run, via the
    high-level meep_torch.mp API -> Ez array (NX, NY). This is the
    user-facing simulation; reference_ez() is the controlled-numerics golden
    the tests compare against. mp is a lazy import."""
    import meep_torch.mp as mp

    cell = mp.Vector3(CELL_X, CELL_Y, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 2 * WG_HALF_W, mp.inf),
                         center=mp.Vector3(), material=mp.Medium(epsilon=WG_EPS))]
    sources = [mp.Source(mp.ContinuousSource(frequency=FREQ),
                         component=mp.Ez, center=mp.Vector3(SRC_X, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(PML)],
                        geometry=geometry, sources=sources, resolution=RESOLUTION)
    sim.run(until=UNTIL)
    return sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)