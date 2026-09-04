# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Waveguide crossing S-parameters
#
# Two perpendicular Si waveguides (eps = 12, width 0.5 um) cross at the cell
# center. This is the nominal "starting guess" geometry of meep's adjoint
# tutorial, characterized rather than optimized: excite the left port and
# report through, reflection, and crosstalk power fractions vs frequency.

# %%
"""meep-torch port of meep/python/examples/waveguide_crossing.py.

The meep tutorial uses the adjoint solver (meep.adjoint + nlopt + autograd) to
*optimize* a waveguide-crossing design region; meep-torch has no adjoint
optimizer. This port instead simulates the *nominal* crossing -- two
perpendicular Si waveguides (eps=12) of width 0.5 um crossing at the center, in
air -- and computes its S-parameters via compute_smatrix (TM, both x- and
y-normal ports supported for TM). This is the "starting guess" geometry the
meep optimizer begins from (the simple cross), characterised instead of
optimized.

Ports: left (input), right (through), top, bottom. Excite left; report
|S_right|^2 (through), |S_top|^2 = |S_bottom|^2 (crosstalk to the crossed
waveguide), |S_left|^2 (reflection), vs frequency.

Run: uv run python examples/08_devices/waveguide_crossing.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from meep_torch.core.grid import Grid
from meep_torch.ports.port import Port
from meep_torch.ports.smatrix import compute_smatrix

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Four-port S-matrix
#
# Ports span the full interior on each of the four sides so the 1D mode solver
# sees the guided mode; `compute_smatrix` runs the TM excitation and projects
# onto port modes at five frequencies around 1.55 um.

# %%
FCEN = 1.0 / 1.55
DX, NPML = 0.04, 25          # 1.0 um PML at res 25
W = 0.5
HALF = int(round(W / 2 / DX))   # 6 cells -> 0.24 um half-width


def main():
    nxy = 150                 # 6.0 um cell
    c = nxy // 2               # 75
    lo, hi = c - HALF, c + HALF
    # port span covers the full interior (waveguide + air cladding) so solve_1d
    # sees the guided mode; the waveguide itself is the eps=12 strip at the center.
    p_lo, p_hi = NPML + 2, nxy - NPML - 2
    grid = Grid(nx=nxy, ny=nxy, dx=DX)
    eps = torch.ones(nxy, nxy, dtype=torch.float64, device=grid.device)
    eps[:, lo:hi] = 12.0       # horizontal waveguide
    eps[lo:hi, :] = 12.0       # vertical waveguide (crossing junction = overlap)

    p_idx = NPML + 4           # inside PML + SRC_OFFSET margin
    ports = [
        Port(axis="x", index=p_idx, lo=p_lo, hi=p_hi, direction=+1),    # 0 left (input)
        Port(axis="x", index=nxy - p_idx, lo=p_lo, hi=p_hi, direction=-1),  # 1 right (through)
        Port(axis="y", index=p_idx, lo=p_lo, hi=p_hi, direction=+1),   # 2 bottom
        Port(axis="y", index=nxy - p_idx, lo=p_lo, hi=p_hi, direction=-1),  # 3 top
    ]
    freqs = [FCEN + f for f in (-0.05, -0.025, 0.0, 0.025, 0.05)]
    sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=0.15 * FCEN,
                         freqs=freqs, n_steps=5000, npml=NPML)
    T = (sm.s[:, 1, 0].abs() ** 2).cpu().numpy()        # through (right)
    R = (sm.s[:, 0, 0].abs() ** 2).cpu().numpy()        # reflection (left)
    Xt = (sm.s[:, 3, 0].abs() ** 2).cpu().numpy()       # crosstalk top
    Xb = (sm.s[:, 2, 0].abs() ** 2).cpu().numpy()       # crosstalk bottom

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs, T, "o-", color="#1f77b4", label="through |S21|^2")
    ax.plot(freqs, R, "s-", color="#d62728", label="reflection |S11|^2")
    ax.plot(freqs, Xt, "^-", color="#2ca02c", label="crosstalk top |S41|^2")
    ax.plot(freqs, Xb, "v-", color="#ff7f0e", label="crosstalk bottom |S31|^2")
    ax.set_xlabel("frequency (1/um)")
    ax.set_ylabel("power fraction")
    ax.set_title("Nominal waveguide crossing -- S-parameters (waveguide_crossing.py)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "waveguide_crossing.png", dpi=150)
    plt.close(fig)
    i0 = freqs.index(FCEN)
    print(f"waveguide_crossing.png written; at fcen: through |S21|^2={T[i0]:.4f}, "
          f"reflection |S11|^2={R[i0]:.4f}, crosstalk |S41|^2={Xt[i0]:.4f}")


if __name__ == "__main__":
    main()