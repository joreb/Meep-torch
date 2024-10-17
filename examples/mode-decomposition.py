# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/mode-decomposition.py.

Linear taper (w1=1 -> w2=2 um) reflectance vs taper length, via the
S-matrix (compute_smatrix). The meep tutorial runs a straight-waveguide
reference, subtracts it (load_minus_flux_data), and reads |S11|^2; meep-torch's
compute_smatrix does that two-run subtraction internally, so a single call per
geometry yields the backward coefficient S11 directly.

Reduced from tests/plots/make_plots.mode_decomposition for runtime:
  lengths [2, 4, 8] um (vs 1,2,4,8), npml=40 (vs 60), n_steps=3000 (vs 16000).
  R is floored by PML absorption + dx=0.1 discretization; the trend
  (R ~ 1/L^2) is the point, not the absolute value. See make_plots for the
  converged dx=0.05 run that matches meep 1.34 to 7%.

Run: uv run python examples/mode-decomposition.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.grid import Grid
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.ports.port import Port
from meep_torch.ports.smatrix import compute_smatrix

OUT = pathlib.Path(__file__).parent
FCEN, FWIDTH = 1.0 / 6.67, 0.2 / 6.67  # meep tutorial: lcen=6.67, fwidth=0.2*fcen

w1, w2 = 1.0, 2.0
lengths_um = [2, 4, 8]


def taper_reflectance(L: float) -> float:
    grid = Grid(nx=340, ny=220, dx=0.1)
    cx_um = grid.nx * grid.dx / 2
    cy = grid.ny * grid.dx / 2
    x0u, x1u = cx_um - L / 2, cx_um + L / 2
    xs, xe = -1.0, grid.nx * grid.dx + 1.0
    poly = Polygon(np.array([
        [xs, cy - 0.5 * w1], [x0u, cy - 0.5 * w1], [x1u, cy - 0.5 * w2], [xe, cy - 0.5 * w2],
        [xe, cy + 0.5 * w2], [x1u, cy + 0.5 * w2], [x0u, cy + 0.5 * w1], [xs, cy + 0.5 * w1],
    ]))
    eps = rasterize_scene([(poly, 12.0)], grid, background=1.0).eps_tm
    ports = [Port(axis="x", index=90, lo=85, hi=135, direction=+1),
             Port(axis="x", index=250, lo=85, hi=135, direction=-1)]
    sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                         freqs=[FCEN], n_steps=3000, npml=40)
    return float(sm.s[0, 0, 0].abs() ** 2)


def main():
    R = []
    for L in lengths_um:
        r = taper_reflectance(L)
        R.append(r)
        print(f"  taper L={L} um: R={r:.3e}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(lengths_um, R, "o-", color="#1f77b4", label="meep-torch |S11|^2 (dx=0.1)")
    ax.loglog(lengths_um, [0.005 / Lt ** 2 for Lt in lengths_um], "k--",
              label=r"quadratic reference (1/Lt$^2$)")
    ax.set_xlabel("taper length Lt (um)")
    ax.set_ylabel("reflectance")
    ax.set_title("Linear taper reflectance vs length (Mode_Decomposition)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "mode-decomposition.png", dpi=150)
    plt.close(fig)
    print(f"mode-decomposition.png written; R(L=8) = {R[-1]:.3e}")


if __name__ == "__main__":
    main()