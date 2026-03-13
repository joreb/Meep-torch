# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (b, pre-check): phase-matched SH bus width scan for the TFLN ring.

WHY THIS EXISTS. The plain two-bus SHG ring fails on this platform. Measured
(b_ring_shg.py, RES=40): an SH bus at gap=0.20 couples |kappa|^2 = 0.586 of the
PUMP out per round trip, collapsing assembled pump Q from ~25400 to ~2900. The
pump is the LONGER wavelength, so it is less confined and has the LONGER
evanescent reach (measured gamma_pump = 3.99/um vs gamma_SH ~ 5.7/um): any gap
tight enough to extract the 674 nm SH couples the 1348 nm pump out even harder.
No gap fixes it -- at gap=0.25 the SH barely escapes (0.031) while the pump
still loses 0.393 per round trip.

THE FIX (peer d20473d3, approved): directional coupling requires PHASE MATCHING,
not just proximity. Make the SH bus a DIFFERENT WIDTH from the ring so that
  neff_bus(674)  ~= neff_ring(674)    -> SH couples out, and
  neff_bus(1348) != neff_ring(1348)   -> pump is phase-mismatched and stays in.
Then the SH bus is effectively invisible to the pump regardless of gap.

Coupled-mode theory for the mismatched case: with mismatch
delta = (2*pi/lam)*(neff_bus - neff_ring), the power transfer over interaction
length L is bounded by the Lorentzian factor

    kappa^2_max = kappa0^2 * 1 / (1 + (delta*L_int / pi)^2)          (approx)

i.e. a large delta*L_int suppresses coupling even when the modes overlap. We
require the PREDICTED pump cross-coupling below 0.01 (peer's criterion), using
the MEASURED kappa0^2 = 0.586 at gap 0.20 as the phase-matched ceiling.

Selection criteria (peer):
  1. |neff_bus(674) - neff_ring(674)| <= 0.002       (SH still couples)
  2. predicted pump kappa^2 < 0.01                    (pump does not leak)

Pure CPU (fd2d is numpy/scipy) -- runs alongside GPU FDTD without contention.

STACK ASSUMED, same as b_ring_shg.py: 600 nm LN film, 300 nm etch, SiO2 clad,
quasi-TE. Ring width is 1.2 um; only the BUS width is scanned.

Run: uv run python examples_2/b_ring_buswidth_scan.py
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

STACK_CONFIRMED = False
H_FILM, H_ETCH = 0.600, 0.300
N_LN = {1.348: 2.2116, 0.674: 2.2812}       # n_e placeholders, as in bendcheck
N_OX = {1.348: 1.4469, 0.674: 1.4562}
W_RING = 1.2                                 # ring width, fixed
DX = 0.02
PAD_X, PAD_Y = 2.5, 2.0
NPML = (20, 20)

# interaction length: the racetrack straight coupler plus the effective arc
# length where the guides stay within a coupling distance. LC=4 um straight;
# the arcs add roughly sqrt(R*lam) of effective length each side. Use the
# straight only -- it is the conservative (LONGER suppression needed) choice
# would be the opposite, so take straight + a modest arc allowance.
LC = 4.0
R_RING = 50.0
KAPPA0_PUMP = 0.586          # MEASURED phase-matched pump coupling at gap 0.20
TARGET_PUMP_K2 = 0.01        # peer criterion
SH_MATCH_TOL = 0.002         # peer criterion

WIDTHS = [round(0.30 + 0.05 * i, 2) for i in range(0, 19)]   # 0.30 .. 1.20 um


def ridge_eps(width, n_core, n_clad):
    nx = int(round((width + 2 * PAD_X) / DX))
    ny = int(round((H_FILM + 2 * PAD_Y) / DX))
    eps = np.full((nx, ny), n_clad ** 2)
    j0 = int(round(PAD_Y / DX))
    j_slab = j0 + int(round((H_FILM - H_ETCH) / DX))
    j1 = j0 + int(round(H_FILM / DX))
    eps[:, j0:j_slab] = n_core ** 2                # slab spans full width
    i0 = int(round(PAD_X / DX))
    i1 = i0 + int(round(width / DX))
    eps[i0:i1, j_slab:j1] = n_core ** 2            # ridge
    return eps


def neff(width, lam):
    eps = ridge_eps(width, N_LN[lam], N_OX[lam])
    m = solve_2d(eps, dx=DX, f=1.0 / lam, k=1, npml=NPML).neff[0]
    return float(m.real)


def l_interaction(lam):
    """Straight coupler plus the arc length that stays within coupling range."""
    return LC + 2.0 * math.sqrt(R_RING * lam)


def pump_k2(delta_neff, lam=1.348):
    """Predicted pump cross-coupling under phase mismatch (CMT Lorentzian)."""
    delta = 2 * math.pi / lam * abs(delta_neff)
    li = l_interaction(lam)
    return KAPPA0_PUMP / (1.0 + (delta * li / math.pi) ** 2)


def main():
    if not STACK_CONFIRMED:
        print("b_ring_buswidth_scan  *** STACK ASSUMED (600/300 nm TFLN, SiO2, TE) ***")
    ring = {lam: neff(W_RING, lam) for lam in (1.348, 0.674)}
    print(f"b_ring_buswidth_scan  ring W={W_RING} um: "
          f"neff(1348)={ring[1.348]:.4f} neff(674)={ring[0.674]:.4f}")
    print(f"b_ring_buswidth_scan  L_int: {l_interaction(1.348):.1f} um @1348 "
          f"(LC={LC} + arcs); kappa0^2(pump, measured @gap 0.20)={KAPPA0_PUMP}")
    print("b_ring_buswidth_scan  scanning bus width: need |dneff(674)| <= "
          f"{SH_MATCH_TOL} AND predicted pump kappa^2 < {TARGET_PUMP_K2}")

    rows = []
    for w in WIDTHS:
        n674 = neff(w, 0.674)
        n1348 = neff(w, 1.348)
        d674 = n674 - ring[0.674]
        d1348 = n1348 - ring[1.348]
        pk2 = pump_k2(d1348)
        ok_sh = abs(d674) <= SH_MATCH_TOL
        ok_pump = pk2 < TARGET_PUMP_K2
        rows.append((w, n674, d674, n1348, d1348, pk2, ok_sh, ok_pump))
        tag = "  <== BOTH OK" if (ok_sh and ok_pump) else ""
        print(f"b_ring_buswidth_scan    W={w:.2f}: neff674={n674:.4f} "
              f"(d={d674:+.4f}{' OK' if ok_sh else '   '}) "
              f"neff1348={n1348:.4f} (d={d1348:+.4f}) "
              f"pump k^2~{pk2:.4f}{' OK' if ok_pump else '   '}{tag}")

    winners = [r for r in rows if r[6] and r[7]]
    print("b_ring_buswidth_scan  --- result ---")
    if winners:
        # among winners prefer the smallest predicted pump leakage
        w = min(winners, key=lambda r: r[5])
        print(f"b_ring_buswidth_scan  PHASE-MATCHED SH BUS WIDTH = {w[0]:.2f} um: "
              f"dneff(674)={w[2]:+.4f} (couples), dneff(1348)={w[4]:+.4f} -> "
              f"pump kappa^2 ~ {w[5]:.4f} (was 0.586 at ring width)")
        print(f"b_ring_buswidth_scan  NEXT: one confirming FDTD run at W_bus="
              f"{w[0]:.2f}, SH gap 0.20")
    else:
        near = min(rows, key=lambda r: abs(r[2]))
        print("b_ring_buswidth_scan  NO WIDTH satisfies both criteria.")
        print(f"b_ring_buswidth_scan  closest SH match W={near[0]:.2f} "
              f"(dneff674={near[2]:+.4f}, pump k^2~{near[5]:.4f}). "
              f"FALLBACK = single bus at gap 0.25 (pump Q~4700).")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ws = [r[0] for r in rows]
    ax1.plot(ws, [r[2] for r in rows], "o-", label=r"$\Delta n_{eff}$ @674 (SH)")
    ax1.plot(ws, [r[4] for r in rows], "s-", label=r"$\Delta n_{eff}$ @1348 (pump)")
    ax1.axhspan(-SH_MATCH_TOL, SH_MATCH_TOL, color="green", alpha=0.15,
                label=f"SH match band +/-{SH_MATCH_TOL}")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("bus width [um]"); ax1.set_ylabel(r"$n_{eff,bus} - n_{eff,ring}$")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.semilogy(ws, [max(r[5], 1e-6) for r in rows], "o-", color="crimson")
    ax2.axhline(TARGET_PUMP_K2, color="gray", ls="--", lw=0.9,
                label=f"pump leak budget {TARGET_PUMP_K2}")
    ax2.axhline(KAPPA0_PUMP, color="k", ls=":", lw=0.9,
                label=f"measured at ring width ({KAPPA0_PUMP})")
    ax2.set_xlabel("bus width [um]"); ax2.set_ylabel(r"predicted pump $|\kappa|^2$")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    title = "Phase-matched SH bus width scan (TFLN ring W=1.2 um)"
    if not STACK_CONFIRMED:
        title += "  (STACK ASSUMED)"
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "b_ring_buswidth_scan.png", dpi=150)
    plt.close(fig)
    print("b_ring_buswidth_scan.png written")


if __name__ == "__main__":
    main()
