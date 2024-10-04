# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch equivalents of four meep Python tutorials, rendered to PNGs.

Run: uv run python tests/plots/make_plots.py          (~8 min, CPU)
Outputs land next to this file. Tutorials mirrored:
  Basics (straight waveguide field), Basics part 2 (bend flux),
  Mode_Decomposition (taper reflectance), Eigenmode_Source (one-way launch).
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import CWRamp, GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state, step_tm
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port
from meep_torch.ports.smatrix import compute_smatrix

OUT = pathlib.Path(__file__).parent
FCEN, FWIDTH = 0.15, 0.1


def _field_ax(ax, eps, Ez, title):
    v = float(Ez.abs().max()) or 1.0
    ax.imshow(eps.T, origin="lower", cmap="binary", alpha=0.9)
    ax.imshow(Ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("x (cells, dx=0.1 um)")
    ax.set_ylabel("y (cells)")


def straight_waveguide():
    grid = Grid(nx=160, ny=80, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    eps[:, 35:45] = 12.0
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=10)
    src = LineSource(grid, ix=20, j0=35, j1=45, profile=CWRamp(FCEN, t_ramp=50.0))
    state = init_tm_state(grid)
    for t in range(4000):
        state = step_tm(state, coeffs, src.amplitude(t), src.mask)
    fig, ax = plt.subplots(figsize=(8, 4))
    _field_ax(ax, eps, state.Ez, "Straight waveguide — CW Ez at f=0.15 (Basics)")
    fig.tight_layout()
    fig.savefig(OUT / "straight_waveguide.png", dpi=150)
    plt.close(fig)
    print("straight_waveguide.png")


def eigenmode_source():
    grid = Grid(nx=160, ny=80, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    eps[:, 35:45] = 12.0
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=10)
    port = Port(axis="x", index=60, lo=20, hi=60, direction=+1)
    src = ModeSourceTM(grid, coeffs, port, GaussianPulse(FCEN, FWIDTH), FCEN)
    state = init_tm_state(grid)
    for t in range(1400):
        state = step_tm(state, coeffs, src.amplitude(t), src.mask,
                        src.h_amplitude(t), **src.h_masks)
    fig, ax = plt.subplots(figsize=(8, 4))
    _field_ax(ax, eps, state.Ez,
              "Eigenmode TFSF source — one-way +x launch (Eigenmode_Source)")
    ax.axvline(60, color="k", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(OUT / "eigenmode_source.png", dpi=150)
    plt.close(fig)
    print("eigenmode_source.png")


def bend_flux():
    # 16x16 um cell, 1 um guide eps=12: horizontal arm -> 90 deg bend -> vertical arm
    grid = Grid(nx=160, ny=160, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    eps[:110, 60:70] = 12.0     # horizontal arm
    eps[100:110, 60:] = 12.0    # vertical arm
    ports = [
        Port(axis="x", index=30, lo=40, hi=90, direction=+1),
        Port(axis="y", index=130, lo=80, hi=130, direction=-1),
    ]
    freqs = [0.10 + 0.01 * i for i in range(11)]
    sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                         freqs=freqs, n_steps=6000, npml=10)
    T = (sm.s[:, 1, 0].abs() ** 2).numpy()
    R = (sm.s[:, 0, 0].abs() ** 2).numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs, T, "o-", color="#1f77b4", label="T = |S21|^2")
    ax.plot(freqs, R, "s-", color="#d62728", label="R = |S11|^2")
    ax.plot(freqs, 1.0 - T - R, "^-", color="#7f7f7f", label="loss (radiation)")
    ax.set_xlabel("frequency (1/um)")
    ax.set_ylabel("power fraction")
    ax.set_title("90-degree bend — S-parameter spectra (bend-flux)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "bend_flux.png", dpi=150)
    plt.close(fig)
    print("bend_flux.png")


def mode_decomposition():
    # linear taper w 1->2 um, reflectance vs taper length; meep tutorial reference
    lengths_um = [1, 2, 4, 8]
    # meep 1.34 mode-decomposition.py output on this machine (2026-07-28 run)
    meep_ref = [3.34e-4, 8.61e-5, 4.65e-5, 2.14e-5]
    R = []
    for L in lengths_um:
        grid = Grid(nx=240, ny=100, dx=0.1)
        eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
        lt = 10 * L
        x0, x1 = 120 - lt // 2, 120 + lt // 2
        for ix in range(grid.nx):
            if ix < x0:
                w = 10
            elif ix >= x1:
                w = 20
            else:
                w = 10 + (ix - x0) * 10 // lt
            eps[ix, 50 - w // 2: 50 + (w + 1) // 2] = 12.0
        ports = [Port(axis="x", index=40, lo=25, hi=75, direction=+1),
                 Port(axis="x", index=200, lo=25, hi=75, direction=-1)]
        sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                             freqs=[FCEN], n_steps=3500, npml=10)
        R.append(float(sm.s[0, 0, 0].abs() ** 2))
        print(f"  taper L={L} um: R={R[-1]:.3e}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(lengths_um, R, "o-", color="#1f77b4", label="meep-torch |S11|^2")
    ax.loglog(lengths_um, meep_ref, "s--", color="#7f7f7f",
              label="meep 1.34 tutorial (reference)")
    ax.set_xlabel("taper length (um)")
    ax.set_ylabel("reflectance")
    ax.set_title("Linear taper reflectance vs length (Mode_Decomposition)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "mode_decomposition.png", dpi=150)
    plt.close(fig)
    print("mode_decomposition.png")


if __name__ == "__main__":
    straight_waveguide()
    eigenmode_source()
    bend_flux()
    mode_decomposition()
    print("all plots written to", OUT)
