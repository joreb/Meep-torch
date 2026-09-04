# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Ring resonator from a GDS file
#
# The geometry round-trips through GDSII: a ring is written with gdstk as two
# filled disks on separate layers (outer Si, inner SiO2), read back with
# `polygons_from_gds`, and composited painter's-style to carve the annulus.
# Resonances are then found from the FFT of a ringdown time series.

# %%
"""meep-torch port of meep/python/examples/ring_gds.py.

The meep tutorial builds a ring GDS (gdstk.ellipse with a hole), reads it back
with gdstk, rasterizes the ring as Si inside a SiO2 cladding, and finds the
resonant modes with Harminv (wavelength + Q).

Two adaptations for meep-torch:
  1. polygons_from_gds / Polygon read a single closed contour (no holes), so the
     ring is written as TWO filled disks on separate layers -- outer disk (Si)
     and inner disk (SiO2) -- and rasterized with painter's-algebra compositing
     (outer then inner) to carve the annulus. This mirrors meep's "outer Si
     cylinder minus inner air cylinder" construction exactly.
  2. meep-torch has no Harminv; resonances are found by FFT of the Ez time
     series recorded at a monitor point (a broadband Gaussian pulse excites the
     ring, the ringdown's spectrum peaks at the resonant frequencies).

The meep source is Hz (TE-like); this port uses Ez (TM), which the 2D TM engine
supports in all directions. The ring supports whispering-gallery modes for both
polarizations; TM modes are reported here.

Run: uv run python examples/08_devices/ring_gds.py
"""
import pathlib
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_tm import init_tm_state, step_tm
from meep_torch.geometry.gds import polygons_from_gds
from meep_torch.geometry.scene import rasterize_scene

OUT = pathlib.Path(__file__).parent / "output"

# %% [markdown]
# ## GDS write / read-back and simulation
#
# `write_ring_gds` emits the two-disk GDS; `main` reads it back, rasterizes
# the Si ring in SiO2 cladding, excites it with an odd pair of Ez point
# sources, and records 400 time units of ringdown at a monitor point. The
# spectrum's dominant peak gives the resonant wavelength.

# %%
R, W = 2.0, 0.5               # ring mean radius, waveguide width (um)
PAD, DPML = 2.0, 1.0
N_SI = 3.4                     # Si index (meep ring_gds Si=3.4)
N_CLAD = 1.4                   # SiO2 index
EPS_SI = N_SI ** 2
EPS_CLAD = N_CLAD ** 2
DX = 0.1                       # coarser than meep res=20 (dx=0.05) for <90s runtime
FCEN = 1.0 / 1.55
FWIDTH = 0.05 * FCEN           # meep bw=0.05


def write_ring_gds(path):
    import gdstk
    lib = gdstk.Library("ring")
    cell = lib.new_cell("RING")
    outer = R + W / 2          # 2.25
    inner = R - W / 2           # 1.75
    cell.add(gdstk.ellipse((0.0, 0.0), outer, layer=0))      # outer Si disk
    cell.add(gdstk.ellipse((0.0, 0.0), inner, layer=1))      # inner SiO2 disk (hole)
    lib.write_gds(str(path))
    return path


def main():
    sxy = 2 * (R + W / 2 + PAD + DPML)        # 11.0 um
    nxy = int(round(sxy / DX))                # 220
    grid = Grid(nx=nxy, ny=nxy, dx=DX)
    npml = int(round(DPML / DX))              # 20
    c = nxy // 2

    with tempfile.TemporaryDirectory() as td:
        gds = write_ring_gds(pathlib.Path(td) / "ring.gds")
        outer = polygons_from_gds(gds, layer=0)[0]
        inner = polygons_from_gds(gds, layer=1)[0]
    # background SiO2; outer disk -> Si; inner disk -> SiO2 (carves the annulus)
    rm = rasterize_scene([(outer, EPS_SI), (inner, EPS_CLAD)], grid,
                         background=EPS_CLAD)
    eps = rm.eps_tm
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=npml)

    # two opposite-sign Ez point sources at (+r+0.1, 0) and (-(r+0.1), 0): odd excitation
    src_off = int(round((R + 0.1) / DX))     # 42 cells from center
    src_mask = torch.zeros(nxy, nxy, dtype=torch.float64)
    src_mask[c + src_off, c] = 1.0
    src_mask[c - src_off, c] = -1.0
    pulse = GaussianPulse(FCEN, FWIDTH)

    mon_ix, mon_iy = c + src_off, c          # monitor at (r+0.1, 0)
    state = init_tm_state(grid)
    n_steps = 8000                            # 8000*0.05 = 400 time units, df=0.0025
    ts = np.empty(n_steps, dtype=np.float64)
    for t in range(n_steps):
        state = step_tm(state, coeffs, pulse.amplitude(t), src_mask)
        ts[t] = float(state.Ez[mon_ix, mon_iy])

    # FFT spectrum of the ringdown (skip the forced-source window for cleaner peaks)
    spec = np.abs(np.fft.rfft(ts)) ** 2
    freqs = np.fft.rfftfreq(n_steps, d=grid.dt)
    # restrict to the source band +- a margin
    band = (freqs > FCEN - 3 * FWIDTH) & (freqs < FCEN + 3 * FWIDTH)
    fb, sb = freqs[band], spec[band]
    peak_f = float(fb[np.argmax(sb)])
    peak_wvl = 1.0 / peak_f

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(fb, sb, color="#1f77b4")
    ax1.axvline(peak_f, color="k", ls=":", lw=1, label=f"peak f={peak_f:.4f} (wvl={peak_wvl:.3f} um)")
    ax1.set_xlabel("frequency (1/um)")
    ax1.set_ylabel("|FFT(Ez)|^2 (a.u.)")
    ax1.set_title("Ring resonator spectrum (ring_gds.py)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.imshow(eps.T, origin="lower", cmap="binary")
    v = float(state.Ez.abs().max()) or 1.0
    ax2.imshow(state.Ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.7)
    ax2.set_title("Ez ringdown snapshot")
    ax2.set_xlabel("x (cells)")
    ax2.set_ylabel("y (cells)")
    fig.tight_layout()
    fig.savefig(OUT / "ring_gds.png", dpi=150)
    plt.close(fig)
    print(f"ring_gds.png written; dominant resonance wvl={peak_wvl:.4f} um "
          f"(f={peak_f:.5f}, Si index {N_SI}, ring r={R} w={W})")


if __name__ == "__main__":
    main()