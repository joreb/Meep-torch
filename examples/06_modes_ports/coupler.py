# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Directional coupler: coupling vs length
#
# meep-torch port of meep/python/examples/coupler.py. The meep tutorial reads
# a directional coupler from coupler.gds (two S-bent Si branches that are far
# apart at the ports and brought to a small gap in the central coupling
# section) and computes per-branch transmittance via eigenmode coefficients
# for a single fixed gap.
#
# This port builds the same S-bend coupler geometry parametrically (two
# constant-width branches that taper from a wide port separation down to a
# small coupling gap and back) and sweeps the coupling-section length L,
# plotting the cross-coupling |S_{upper-out, lower-in}|^2 vs L -- which
# oscillates as sin^2(pi*L/L_c), the two-mode interference the coupler is
# built on. The S-bends keep the ports far apart so each port sees an isolated
# single-waveguide mode (solve_1d needs the waveguide + cladding, not both
# branches). Output: `output/coupler.png`.
#
# Geometry: Si (eps=12) in air, w=0.5 um, coupling gap g=0.3 um, 2D TM. Each
# branch fills the full interior between the PMLs so the port plane (4 cells
# inside PML) is well within the waveguide.
#
# Run: `uv run python examples/06_modes_ports/coupler.py`

# %%
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

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
FCEN, FWIDTH = 1.0 / 1.55, 0.2 / 1.55
W, GAP = 0.5, 0.3
DX = 0.08
NPML = 10                     # 0.8 um
L_TRANS = 2.5                  # S-bend transition length (um)
Y_PORT = 1.8                   # branch center y at the ports (um)
Y_MID = GAP / 2 + W / 2        # 0.4 um branch center y in coupling section

# %% [markdown]
# ## Geometry helpers
# Cosine S-bend centerlines swept into constant-width strips replace the GDS
# import of the meep tutorial.

# %%
def _cosine_bend(x0, y0, x1, y1, n=16, include_ends=(True, True)):
    """Smooth (zero-slope-at-ends) S-bend centerline from (x0,y0) to (x1,y1)."""
    xs = np.linspace(x0, x1, n)
    s = (1 - np.cos(np.pi * (xs - x0) / (x1 - x0))) / 2   # 0->1, smooth
    ys = y0 + (y1 - y0) * s
    pts = list(zip(xs, ys))
    if not include_ends[0]:
        pts = pts[1:]
    if not include_ends[1]:
        pts = pts[:-1]
    return pts


def _strip_polygon(centerline, width):
    """Constant-width (y-offset) strip around a polyline centerline."""
    hw = width / 2
    xs = [p[0] for p in centerline]
    ys = [p[1] for p in centerline]
    top = list(zip(xs, [y + hw for y in ys]))
    bot = list(zip(reversed(xs), [y - hw for y in reversed(ys)]))
    return Polygon(np.array(top + bot, dtype=np.float64))

# %% [markdown]
# ## S-matrix per coupling length
# Builds the two-branch coupler for a given L, places four ports (in/out on
# each branch), and returns the cross and through power fractions from one
# S-matrix run.

# %%
def coupler_smatrix(coupling_len):
    interior = 2 * L_TRANS + coupling_len
    sx = interior + 2 * NPML * DX
    sy = 2 * (Y_PORT + W / 2 + 1.0) + 2 * NPML * DX
    nx = int(round(sx / DX))
    ny = int(round(sy / DX))
    grid = Grid(nx=nx, ny=ny, dx=DX)
    cy_um = ny * DX / 2
    # branch fills the full interior [NPML*dx, (nx-NPML)*dx]; S-bend+coupling centered in it
    x_left = NPML * DX
    x_bend1 = x_left + L_TRANS
    x_bend2 = x_bend1 + coupling_len
    x_right = x_left + interior            # = (nx - NPML) * dx (right PML edge)
    def centerline(y_port, y_mid):
        cl = _cosine_bend(x_left, y_port, x_bend1, y_mid)                 # incl both ends
        cl += _cosine_bend(x_bend1, y_mid, x_bend2, y_mid, n=3,
                           include_ends=(False, False)) if coupling_len > 0 else []
        cl += [(x_bend2, y_mid)]
        cl += _cosine_bend(x_bend2, y_mid, x_right, y_port, include_ends=(False, True))
        return cl
    poly_up = _strip_polygon([(x, y + cy_um) for x, y in centerline(Y_PORT, Y_MID)], W)
    poly_lo = _strip_polygon([(x, y + cy_um) for x, y in centerline(-Y_PORT, -Y_MID)], W)
    eps = rasterize_scene([(poly_up, 12.0), (poly_lo, 12.0)], grid, background=1.0).eps_tm

    left_idx = NPML + 4                     # 4 cells inside PML margin (>= SRC_OFFSET)
    right_idx = nx - NPML - 4
    up_c = int(round((cy_um + Y_PORT) / DX))   # upper branch y at port
    lo_c = int(round((cy_um - Y_PORT) / DX))   # lower branch y at port
    span = int(round(0.6 / DX))               # +-0.6 um: waveguide + cladding, isolated from other branch
    ports = [
        Port(axis="x", index=left_idx, lo=lo_c - span, hi=lo_c + span, direction=+1),   # 0 lower-left (in)
        Port(axis="x", index=left_idx, lo=up_c - span, hi=up_c + span, direction=+1),  # 1 upper-left
        Port(axis="x", index=right_idx, lo=lo_c - span, hi=lo_c + span, direction=-1),# 2 lower-right (through)
        Port(axis="x", index=right_idx, lo=up_c - span, hi=up_c + span, direction=-1),# 3 upper-right (cross)
    ]
    n_steps = int(round((0.7 * sx + 130) / grid.dt))   # transit (n~3.4) + source + decay
    sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                         freqs=[FCEN], n_steps=n_steps, npml=NPML)
    cross = float(sm.s[0, 3, 0].abs() ** 2)       # lower-in -> upper-out
    through = float(sm.s[0, 2, 0].abs() ** 2)     # lower-in -> lower-out
    return cross, through

# %% [markdown]
# ## Length sweep and plot
# The cross-coupling oscillates as sin^2(pi*L/L_c) with the through power
# complementary — the textbook directional-coupler power exchange.

# %%
def main():
    lengths = [1, 4, 7, 10, 13, 16]
    cross, through = [], []
    for L in lengths:
        cx, th = coupler_smatrix(L)
        cross.append(cx)
        through.append(th)
        print(f"  L={L} um: cross={cx:.4f}, through={th:.4f}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lengths, cross, "o-", color="#1f77b4", label="cross |S31|^2 (upper-out)")
    ax.plot(lengths, through, "s-", color="#d62728", label="through |S21|^2 (lower-out)")
    ax.plot(lengths, [max(0.0, 1 - c - t) for c, t in zip(cross, through)], "^-",
            color="#7f7f7f", label="radiation/reflection loss")
    ax.set_xlabel("coupling length L (um)")
    ax.set_ylabel("power fraction")
    ax.set_title("Directional coupler -- coupling vs length (coupler.py)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "coupler.png", dpi=150)
    plt.close(fig)
    ipeak = max(range(len(lengths)), key=lambda k: cross[k])
    print(f"coupler.png written; peak cross-coupling {cross[ipeak]:.4f} at L={lengths[ipeak]} um")


if __name__ == "__main__":
    main()
