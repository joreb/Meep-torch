# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Ring-resonator mode overlap (w vs 2w)
#
# meep-torch port of meep/python/examples/ring-mode-overlap.py. The meep
# tutorial excites a ring resonator (r=1, w=1, n=3.4 in air) at two
# frequencies -- fcen=0.118 (w) and 0.236 (2w) -- and computes the overlap
# integral of the two Ez field distributions with integrate2_field_function
# (<Ez_w | Ez_2w> = integral of Ez_w* Ez_2w dA). This is the nonlinear-optics
# mode-overlap that sets second-harmonic generation efficiency in a ring.
# Output: `output/ring-mode-overlap.png`.
#
# meep-torch has no integrate2_field_function; the equivalent is a direct
# domain integral of the two fields. (decompose / solve_1d is a 1D *slab* mode
# solver and does not apply to the ring's curved whispering-gallery modes, so
# it is not used here; the meep original does not use mode-decomposition
# either.)
#
# Each run drives the ring with a CW ramp at the target frequency to a steady
# state; the final Ez snapshot is the field distribution. Overlap =
# |sum(Ez1.conj() * Ez2) * dx^2|, reported in arbitrary units (as in meep).
#
# Run: `uv run python examples/06_modes_ports/ring-mode-overlap.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import CWRamp
from meep_torch.core.stepper_tm import init_tm_state, step_tm
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

R, W = 1.0, 1.0
N_SI = 3.4
EPS_SI = N_SI ** 2
PAD, DPML = 2.0, 1.0
DX = 0.1
FCEN1, FCEN2 = 0.118, 0.236          # meep tutorial frequencies (w and 2w)

# %% [markdown]
# ## Driven ring field at one frequency
# Build the Si annulus, drive a CW point source at the ring edge for ~20
# periods and return the steady-state Ez snapshot.

# %%
def circle(cx, cy, rad, n=128):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return Polygon(np.column_stack([cx + rad * np.cos(t), cy + rad * np.sin(t)]))


def ring_field(fcen):
    sxy = 2 * (R + W + PAD + DPML)         # 10.0 um
    nxy = int(round(sxy / DX))             # 100
    grid = Grid(nx=nxy, ny=nxy, dx=DX)
    npml = int(round(DPML / DX))           # 10
    c = nxy / 2 * DX                      # center in um
    # outer Si disk radius r+w, inner air disk radius r -> Si annulus in air
    rm = rasterize_scene([(circle(c, c, R + W), EPS_SI), (circle(c, c, R), 1.0)],
                          grid, background=1.0)
    eps = rm.eps_tm
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=npml)
    # Ez point source at (r+0.1, 0) -> (c + (r+0.1), c) in um
    src_ix = int(round((c + R + 0.1) / DX))
    src_iy = nxy // 2
    src_mask = torch.zeros(nxy, nxy, dtype=torch.float64, device=grid.device)
    src_mask[src_ix, src_iy] = 1.0
    ramp = CWRamp(fcen, t_ramp=1.0 / fcen)        # ramp over one period
    state = init_tm_state(grid)
    n_steps = int(round(20.0 / (fcen * grid.dt)))  # ~20 periods -> steady state
    for t in range(n_steps):
        state = step_tm(state, coeffs, ramp.amplitude(t), src_mask)
    return eps, state.Ez, grid

# %% [markdown]
# ## Two runs, overlap integral and plot
# The w and 2w field patterns side by side, with the overlap integral in the
# figure title and console.

# %%
def main():
    eps1, Ez1, grid = ring_field(FCEN1)
    eps2, Ez2, _ = ring_field(FCEN2)
    overlap = float((Ez1.conj() * Ez2).sum().abs()) * grid.dx ** 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, Ez, title in [(axes[0], Ez1, f"Ez at f={FCEN1} (w)"),
                         (axes[1], Ez2, f"Ez at f={FCEN2} (2w)")]:
        ax.imshow(eps1.T, origin="lower", cmap="binary")
        v = float(Ez.abs().max()) or 1.0
        ax.imshow(Ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
    fig.suptitle(f"Ring mode overlap |<Ez_w|Ez_2w>| = {overlap:.4e} (a.u.)")
    fig.tight_layout()
    fig.savefig(OUT / "ring-mode-overlap.png", dpi=150)
    plt.close(fig)
    print(f"ring-mode-overlap.png written; overlap integral |<Ez_w|Ez_2w>| = {overlap:.6e}")


if __name__ == "__main__":
    main()
