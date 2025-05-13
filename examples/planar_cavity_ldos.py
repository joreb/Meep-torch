# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/planar_cavity_ldos.py.

Purcell enhancement of a dipole in a planar dielectric cavity bounded by
lossless metallic walls, vs cavity thickness. The meep original computes
it in cylindrical (m=-1, Er) and 3D (Ex) coords and compares with the
Abram et al. 1998 analytic Purcell factor. meep-torch has no PEC wall
primitive and no cylindrical/3D LDOS monitor, so per the port instructions
this is re-cast as the equivalent 1D TM cavity:

  * the planar cavity is a uniform slab (eps = n^2) of thickness L along x,
    bounded by the Yee-grid PEC walls (Ez = 0 at the two end cells, which
    `build_tm_coeffs(..., npml=0)` produces for free -- no `mp.metal` needed);
    the trivial transverse direction is 3 cells with `y_pmc=True`.
  * the in-plane dipole -> an out-of-plane Ez point dipole at x = 0.27 L
    (a generic off-node position that couples to every m).
  * LDOS at fcen via `core/ldos.ldos_spectrum` from the source-cell Ez time
    series and the reconstructed J(t) (the same E/J-timing convention the
    `mp.dft_ldos` shim uses).

SUBSTITUTIONS:
  * PEC walls -> Yee end-cell Ez=0 walls (npml=0). Effective cavity length
    L_eff = (nx - 1) * dx (walls sit at the two end-cell centres); this is
    the half-cell wall-position offset also documented in metal-cavity-ldos.
  * lossless metallic walls -> walls stay lossless, but a small medium
    conductivity sigma_d = 0.05 is added to regularize the formally
    delta-function lossless-cavity LDOS into a finite Lorentzian so the
    finite-time DFT gives clean positive peaks. The resonance centre
    f_m = m / (2 n L_eff) is unchanged by such small loss (shift < 1e-3),
    so the peak LOCATIONS L = m * lambda_med / 2 are unaffected -- only the
    peak heights broaden. This is the quantity the gate checks.
  * cylindrical + 3D -> 1D TM (the m = m lambda/2 peak structure is the
    same 1D Fabry-Perot density-of-states result in all three coords).

Gate (printed): the LDOS local maxima fall at L_eff = m * lambda_med / 2
within 2% for m = 1..4.

Run: uv run python examples/planar_cavity_ldos.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.ldos import ldos_spectrum
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, SoftSource
from meep_torch.core.stepper import init_tm_state

OUT = pathlib.Path(__file__).parent

FCEN = 1.0                  # 1 / lambda_0
N_CAVITY = 2.4
LAMBDA_MED = 1.0 / N_CAVITY  # wavelength inside the cavity medium
RES = 100                    # pixels / vacuum-wavelength
DX = 1.0 / RES
DT = 0.5 * DX
T_END = 180.0               # run length (time units); >> pulse + a few ringdowns
N_STEPS = int(round(T_END / DT))
SIGMA_D = 0.05              # small medium loss: regularize LDOS peaks (see above)
FWIDTH = 0.2 * FCEN
NY = 3
SRC_FRAC = 0.27             # dipole at 0.27 L (off every low-m node)


def _cavity_ldos(nx: int) -> float:
    """LDOS at fcen for a PEC-wall cavity of nx cells (L_eff = (nx-1)*dx)."""
    g = Grid(nx=nx, ny=NY, dx=DX)
    eps = torch.full((nx, NY), N_CAVITY ** 2, dtype=g.dtype)
    sd = torch.full((nx, NY), SIGMA_D, dtype=g.dtype)
    coeffs = build_tm_coeffs(g, eps_inf=eps, npml=0, y_pmc=True, sigma_d=sd)
    ix = max(1, min(nx - 2, int(SRC_FRAC * nx)))
    mask = g.zeros(nx, NY)
    mask[ix, 1] = 1.0
    src = SoftSource(g, mask, GaussianPulse(FCEN, FWIDTH))
    _, (acc,) = run_tm(init_tm_state(g, y_pmc=True), coeffs, N_STEPS,
                       source=src, monitors=[PointProbe(ix, 1)])
    e = acc.detach().numpy()
    j = np.array([src.profile.amplitude((t + 0.5) * DT) for t in range(len(e))])
    return float(ldos_spectrum(e, j, DT, [FCEN])[0])


def _bulk_ldos() -> float:
    """LDOS at fcen in the open (PML) medium -- the Purcell denominator."""
    nx = 220
    g = Grid(nx=nx, ny=NY, dx=DX)
    eps = torch.full((nx, NY), N_CAVITY ** 2, dtype=g.dtype)
    sd = torch.full((nx, NY), SIGMA_D, dtype=g.dtype)
    coeffs = build_tm_coeffs(g, eps_inf=eps, npml=30, y_pmc=True, sigma_d=sd)
    ix = nx // 2
    mask = g.zeros(nx, NY)
    mask[ix, 1] = 1.0
    src = SoftSource(g, mask, GaussianPulse(FCEN, FWIDTH))
    _, (acc,) = run_tm(init_tm_state(g, y_pmc=True), coeffs, N_STEPS,
                       source=src, monitors=[PointProbe(ix, 1)])
    e = acc.detach().numpy()
    j = np.array([src.profile.amplitude((t + 0.5) * DT) for t in range(len(e))])
    return float(ldos_spectrum(e, j, DT, [FCEN])[0])


def _find_peaks(ys, rel_floor=0.05):
    """Indices of local maxima of ys whose value exceeds rel_floor * max."""
    floor = rel_floor * ys.max()
    peaks = []
    for i in range(1, len(ys) - 1):
        if ys[i] > ys[i - 1] and ys[i] >= ys[i + 1] and ys[i] >= floor:
            peaks.append(i)
    return peaks


def main():
    nxs = list(range(18, 89))           # covers m = 1..4 resonances with margin
    ldos = np.array([_cavity_ldos(nx) for nx in nxs])
    l_eff = np.array([(nx - 1) * DX for nx in nxs])

    ldos_bulk = _bulk_ldos()
    purcell = ldos / ldos_bulk

    pk = _find_peaks(ldos)
    rows = []
    for i in pk:
        nx = nxs[i]
        L = (nx - 1) * DX
        m = int(round(2 * N_CAVITY * L))      # nearest m with L = m lambda_med / 2
        L_theory = m * LAMBDA_MED / 2.0
        err = abs(L - L_theory) / L_theory
        rows.append((m, L, L_theory, err, purcell[i]))
        print(f"planar_cavity_ldos  m={m}: L_peak={L:.5f}  "
              f"L_theory={L_theory:.5f}  err={100*err:.2f}%  Purcell={purcell[i]:.3f}")

    max_err = max(r[3] for r in rows)
    assert max_err <= 0.02, f"peak-location gate failed: max err {100*max_err:.2f}% > 2%"
    assert len(rows) >= 4, f"expected >=4 peaks (m=1..4), got {len(rows)}"
    print(f"planar_cavity_ldos  GATE PASS: {len(rows)} peaks, max err {100*max_err:.2f}% <= 2%")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(l_eff, purcell, "b-", lw=1.2, label="meep-torch (1D TM)")
    for m in range(1, 5):
        ax.axvline(m * LAMBDA_MED / 2.0, color="g", ls="--", lw=0.8,
                   label="theory $m\\lambda/2n$" if m == 1 else None)
    ax.plot([l_eff[i] for i in pk], [purcell[i] for i in pk], "ro", ms=5)
    ax.set_xlabel("cavity thickness $L$ (vacuum units)")
    ax.set_ylabel("Purcell enhancement $\\mathrm{LDOS}/\\mathrm{LDOS}_{bulk}$")
    ax.set_title(f"planar cavity LDOS vs thickness (n={N_CAVITY}, PEC walls)\n"
                 f"peaks at $L=m\\lambda/2n$ within {100*max_err:.1f}%")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "planar_cavity_ldos.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT / 'planar_cavity_ldos.png'}")


if __name__ == "__main__":
    main()