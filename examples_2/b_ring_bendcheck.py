# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (b, pre-check): LN ring bend-loss sweep at the SHG pair 1348 / 674 nm.

Peer 259d9a6d found r=50 is dead for SiN (402 dB/round-trip) and that 1348 nm
needs r >= 100 while 674 nm is lossless at r=50. This script does the same check
on the LN stack: for each radius, solve the bent cross-section with fd2d
(cylindrical transform, complex neff) at BOTH wavelengths and convert keff to
dB per round trip. The answer we want is the smallest radius that guides both.

Pure CPU (fd2d is numpy/scipy), so it does not contend with GPU FDTD runs.

  loss per round trip [dB] = 10*log10(e) * 2*k0*keff * (2*pi*R)
                           = 8.6858896 * 2*pi*f_spatial*keff * 2*pi*R  ... see below

with k0 = 2*pi*f (f in 1/um, fd2d convention) and keff = Im(neff): the field decays
as exp(-k0*keff*L) in amplitude, so POWER decays as exp(-2*k0*keff*L) and the loss
in dB over a round trip L = 2*pi*R is 10*log10(e) * 2*k0*keff*L.

STACK IS A PLACEHOLDER until the peer confirms (asked): film thickness, etch
depth, top width, cladding, cut (X/Z -> n_o vs n_e), and polarization. Every
number printed while STACK_CONFIRMED is False is labelled stack-assumed.

Run: uv run python examples_2/b_ring_bendcheck.py
"""
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from meep_torch.modes.fd2d import solve_2d

OUT = pathlib.Path(__file__).parent

# --- stack (PLACEHOLDER -- peer confirmation pending) ---------------------
STACK_CONFIRMED = False
H_FILM = 0.600          # LN film thickness [um]
H_ETCH = 0.300          # etch depth [um] -> slab remainder = H_FILM - H_ETCH
W_TOP = 1.200           # ridge top width [um]
N_LN_1348 = 2.2116      # n_e(1348 nm), Zelmon TFLN -- placeholder
N_LN_674 = 2.2812       # n_e(674 nm)  -- placeholder (SH sees higher index)
N_CLAD = 1.4469         # SiO2 at 1348 nm; 674 nm value applied per-wavelength below
N_CLAD_674 = 1.4562
POL = "TE"              # fundamental quasi-TE

WLS = {"pump 1348": (1.348, N_LN_1348, N_CLAD),
       "SH 674": (0.674, N_LN_674, N_CLAD_674)}
RADII = [20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0]
DX = 0.02               # 20 nm -- resolves the 300 nm slab (15 cells)
# A bend radiates OUTWARD (+x, away from the bend center), so the cladding on
# that side has to be wide enough to hold the leaky tail before the PML, or the
# PML clips it and keff comes out too small. PML is mandatory: with npml=(0,0)
# (the fd2d default) the walls are PEC, radiation has nowhere to go, and keff is
# identically 0.000e+00 for every radius -- which is what a first run produced.
PAD_X_IN, PAD_X_OUT = 2.5, 6.0   # cladding inward / outward of the ridge [um]
PAD_Y = 2.5
NPML_CELLS = 20                  # 0.4 um of PML at 20 nm cells
NPML = (NPML_CELLS, NPML_CELLS)
LOSS_BUDGET_DB = 0.1    # "guides" = below this per round trip
# keff below this is at/below the sparse eigensolve's resolution -- the solver
# returns 1e-11..1e-19 there, which is "lossless, magnitude unresolved", not a
# real number. Verified by re-solving at npml=30 and pad_out=10um: values above
# the floor reproduce to 3 digits, values below it do not mean anything.
KEFF_FLOOR = 1e-9


def build_eps(n_core, n_clad):
    """Ridge cross-section on its slab. Returns (eps, x_center_of_ridge).

    x increases OUTWARD from the bend center, so PAD_X_OUT (the leaky side) is
    the wide one. The unetched slab spans the full width, as on a real chip.
    """
    nx = int(round((W_TOP + PAD_X_IN + PAD_X_OUT) / DX))
    ny = int(round((H_FILM + 2 * PAD_Y) / DX))
    eps = np.full((nx, ny), n_clad ** 2)
    j0 = int(round(PAD_Y / DX))                       # film bottom
    j_slab = j0 + int(round((H_FILM - H_ETCH) / DX))  # top of the unetched slab
    j1 = j0 + int(round(H_FILM / DX))                 # film top
    eps[:, j0:j_slab] = n_core ** 2                   # slab spans the full width
    i0 = int(round(PAD_X_IN / DX))
    i1 = i0 + int(round(W_TOP / DX))
    eps[i0:i1, j_slab:j1] = n_core ** 2               # ridge
    return eps, (i0 + i1) / 2.0 * DX


def round_trip_db(keff, f, radius):
    """Power loss over one round trip [dB]. See the module docstring."""
    k0 = 2 * math.pi * f
    return 10 * math.log10(math.e) * 2 * k0 * abs(keff) * (2 * math.pi * radius)


def main():
    if not STACK_CONFIRMED:
        print("b_ring_bendcheck  *** STACK ASSUMED, NOT CONFIRMED BY PEER ***")
        print(f"b_ring_bendcheck  placeholder: LN {H_FILM*1e3:.0f}nm film / "
              f"{H_ETCH*1e3:.0f}nm etch / {W_TOP*1e3:.0f}nm top width, SiO2 clad, "
              f"{POL}, n_e-ish indices")
    results = {}
    for label, (wl, n_core, n_clad) in WLS.items():
        f = 1.0 / wl
        eps, xc = build_eps(n_core, n_clad)
        straight = solve_2d(eps, dx=DX, f=f, k=1, npml=NPML).neff[0]
        print(f"b_ring_bendcheck  {label} ({wl*1e3:.0f} nm): straight "
              f"neff={straight.real:.4f} (grid {eps.shape[0]}x{eps.shape[1]})")
        row = []
        for R in RADII:
            m = solve_2d(eps, dx=DX, f=f, k=1, npml=NPML,
                         bend_radius=R, bend_center=xc,
                         target_neff=straight.real).neff[0]
            keff = abs(float(m.imag))
            db = round_trip_db(keff, f, R)
            floored = keff < KEFF_FLOOR
            note = "  [below solver floor: lossless, magnitude unresolved]" if floored else ""
            row.append((R, float(m.real), db, floored))
            print(f"b_ring_bendcheck    R={R:6.1f} um: neff={m.real:.4f} "
                  f"keff={keff:.3e} -> {db:9.4f} dB/round-trip{note}")
        results[label] = row

    print("b_ring_bendcheck  --- smallest radius guiding BOTH "
          f"(<{LOSS_BUDGET_DB} dB/round-trip) ---")
    ok = None
    for i, R in enumerate(RADII):
        if all(results[k][i][2] < LOSS_BUDGET_DB for k in WLS):
            ok = R
            break
    if ok is None:
        print(f"b_ring_bendcheck  NONE of {RADII} guides both; extend the sweep")
    else:
        j = RADII.index(ok)
        per = {k: results[k][j][2] for k in WLS}
        print(f"b_ring_bendcheck  PROPOSED RADIUS r={ok:.0f} um "
              + ", ".join(f"{k}={v:.4f} dB/rt" for k, v in per.items()))
        # the binding wavelength is whichever is still lossy one step down
        if j > 0:
            prev = {k: results[k][j - 1][2] for k in WLS}
            worst = max(prev, key=prev.get)
            print(f"b_ring_bendcheck  binding constraint = {worst} "
                  f"({prev[worst]:.3f} dB/rt at r={RADII[j-1]:.0f} um, over the "
                  f"{LOSS_BUDGET_DB} dB budget)")

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, row in results.items():
        ax.semilogy([r[0] for r in row], [max(r[2], 1e-6) for r in row],
                    "o-", label=label)
        for R, _, db, floored in row:
            if floored:
                ax.plot(R, 1e-6, "kv", ms=5)
    ax.axhline(LOSS_BUDGET_DB, color="gray", ls="--", lw=0.9,
               label=f"{LOSS_BUDGET_DB} dB/round-trip budget")
    ax.axhline(1e-6, color="k", ls=":", lw=0.8,
               label="solver floor (v = lossless, unresolved)")
    ax.set_xlabel("bend radius [um]"); ax.set_ylabel("loss [dB per round trip]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    title = "LN ring bend loss, SHG pair 1348 / 674 nm"
    if not STACK_CONFIRMED:
        title += " (STACK ASSUMED)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / "b_ring_bendcheck.png", dpi=150)
    plt.close(fig)
    print("b_ring_bendcheck.png written")


if __name__ == "__main__":
    main()
