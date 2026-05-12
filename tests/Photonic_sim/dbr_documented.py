# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch equivalent of Photonics_Simulation/src/docs/Testing_DBR_documented.py
(lines 1-112): a width-modulated SiN/SiO2 DBR grating.

Pure meep_torch + numpy + matplotlib (no tidy3d/sax/jax). Runnable:
    uv run python tests/Photonic_sim/dbr_documented.py
PNGs land next to this file.

Polarization is TE (user-confirmed; the notebook's class is misleadingly named
DBR_TM_SM). The strip effective index comes from the 2D mode solver
(modes/fd2d.solve_2d) -- the true 2D SiN/SiO2 strip mode, the meep-torch
equivalent of the notebook's tidy3d 2D solve -- NOT the 1D EIM two-step (which
underestimates neff and puts the Bragg wavelength at 0.65 instead of 0.695).
The 2D neff is sampled at ~15 wavelengths and fit to a Cauchy model
n = A + B/lam^2 + C/lam^4 (the notebook's fit form); the TMM spectrum uses the
Cauchy-evaluated neff per wavelength.

The notebook's TMM has a duty-cycle bug: `duty_cycle = min(l1/l2, l2/l1)` is a
ratio (~0.98), not the fraction l1/(l1+l2) (~0.495), which makes the effective
layer lengths l1_eff=0.198um, l2_eff=0.004um -- almost a uniform section, and
the reason the notebook needs N=21250 periods. This script uses the PHYSICAL
layer lengths l1=0.1, l2=0.102, so a modest N (~200) gives a clear stop band.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meep_torch.core.grid import Grid
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.modes.fd2d import solve_2d

# --- DBR geometry (notebook params) ---------------------------------------
W1, W2 = 1.14, 0.86         # waveguide widths (um)
L1, L2 = 0.1, 0.102          # layer lengths (um) -- physical, NOT notebook bug
H = 0.150                   # waveguide height (um)
LOSS = 2e-4                 # dB/um (notebook value)
N_PERIODS = 200             # periods; notebook uses 21250 (bug-inflated)
LAM_RANGE = (0.574, 0.774)  # notebook wavelength range (um)
DESIGN_LAM = 0.674
DX = 0.02                   # 2D solver discretization
PAD = 1.0                   # clad padding (um)
N_FIT = 15                  # fd2d samples for the Cauchy fit

OUT = pathlib.Path(__file__).parent


def _strip_solve(w, lam):
    """2D fd2d fundamental of a SiN strip (w x h) on SiO2 at lam.

    Returns (neff, |E|^2 2D array, x-edge coords, y-edge coords).
    """
    f = 1.0 / lam
    n_sin = Si3N4.n(f).real
    n_sio2 = SiO2.n(f).real
    nx, ny = int((w + 2 * PAD) / DX), int((H + 2 * PAD) / DX)
    g = Grid(nx=nx, ny=ny, dx=DX)
    core = Polygon(np.array([[PAD, PAD], [PAD + w, PAD], [PAD + w, PAD + H], [PAD, PAD + H]]))
    rm = rasterize_scene([(core, n_sin ** 2)], g, background=n_sio2 ** 2)
    ms = solve_2d(rm.eps_tm, DX, f, k=1)
    # Modes2D Ex, Ey are (Nx, Ny, k); mode 0 -> (Nx, Ny)
    Ex = ms.Ex[..., 0].numpy(); Ey = ms.Ey[..., 0].numpy()
    e2 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    return float(ms.neff[0].real), e2, nx, ny


def _cauchy_fit(lams, neff):
    """Fit n = A + B/lam^2 + C/lam^4 via polyfit in x=1/lam^2. Returns (A,B,C)."""
    x = 1.0 / lams ** 2
    c = np.polyfit(x, neff, 2)  # c[0]*x^2 + c[1]*x + c[2]
    return float(c[2]), float(c[1]), float(c[0])


def _cauchy_eval(lams, A, B, C):
    x = 1.0 / lams ** 2
    return A + B * x + C * x ** 2


def _tmm_rt(lams, n1, n2, N, loss):
    """Inline TMM R and T spectrum (physical l1/l2, per-lambda neff, loss)."""
    R = np.zeros_like(lams); T = np.zeros_like(lams)
    for i, lam in enumerate(lams):
        def Thw(ll, ne):
            beta = 2 * np.pi * ne / lam
            amp = 10 ** (-loss * ll / 20)
            return np.array([[amp * np.exp(1j * beta * ll), 0],
                             [0, amp * np.exp(-1j * beta * ll)]], dtype=complex)
        def Tis(x, y):
            a = (x + y) / (2 * np.sqrt(x * y)); b = (x - y) / (2 * np.sqrt(x * y))
            return np.array([[a, b], [b, a]], dtype=complex)
        Tp = Thw(L2, n2[i]) @ Tis(n2[i], n1[i]) @ Thw(L1, n1[i]) @ Tis(n1[i], n2[i])
        mat = np.linalg.matrix_power(Tp, N)
        S11 = mat[0, 1] / mat[0, 0]
        S21 = 1.0 / mat[0, 0]
        R[i] = abs(S11) ** 2
        T[i] = abs(S21) ** 2
    return R, T


def main():
    # --- 2D neff(lambda) via fd2d + Cauchy fit ------------------------------
    lams_fit = np.linspace(LAM_RANGE[0], LAM_RANGE[1], N_FIT)
    n1_fit = np.array([_strip_solve(W1, lam)[0] for lam in lams_fit])
    n2_fit = np.array([_strip_solve(W2, lam)[0] for lam in lams_fit])
    A1, B1, C1 = _cauchy_fit(lams_fit, n1_fit)
    A2, B2, C2 = _cauchy_fit(lams_fit, n2_fit)

    i_des = int(np.argmin(np.abs(lams_fit - DESIGN_LAM)))
    lam_bragg = 2 * (n1_fit[i_des] * L1 + n2_fit[i_des] * L2)

    # --- Figure 1: neff(lambda) + Cauchy overlay ---------------------------
    lams_smooth = np.linspace(LAM_RANGE[0], LAM_RANGE[1], 120)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lams_fit, n1_fit, "o", ms=4, label=f"w={W1} 2D fd2d neff")
    ax.plot(lams_smooth, _cauchy_eval(lams_smooth, A1, B1, C1), "k--", lw=1.2,
            label=f"w={W1} Cauchy  A={A1:.4f} B={B1:.3e} C={C1:.3e}")
    ax.plot(lams_fit, n2_fit, "s", ms=4, label=f"w={W2} 2D fd2d neff")
    ax.plot(lams_smooth, _cauchy_eval(lams_smooth, A2, B2, C2), "r--", lw=1.2,
            label=f"w={W2} Cauchy  A={A2:.4f} B={B2:.3e} C={C2:.3e}")
    ax.set_xlabel("wavelength (um)"); ax.set_ylabel("effective index")
    ax.set_title("DBR strip neff(lambda) [2D fd2d, TE]")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "dbr_neff_vs_wavelength.png", dpi=110)
    plt.close(fig)

    # --- Figure 2: DBR R/T spectrum (Cauchy neff per lambda) --------------
    span = 0.04
    lam_spec = np.linspace(lam_bragg - span, lam_bragg + span, 240)
    n1_s = _cauchy_eval(lam_spec, A1, B1, C1)
    n2_s = _cauchy_eval(lam_spec, A2, B2, C2)
    R, T = _tmm_rt(lam_spec, n1_s, n2_s, N_PERIODS, LOSS)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(lam_spec, R, label="R (|S11|^2)")
    ax1.axvline(lam_bragg, color="k", ls=":", lw=1, label=f"Bragg={lam_bragg:.4f}")
    ax1.set_ylabel("reflectance")
    ax1.set_title(f"DBR spectrum [TE], N={N_PERIODS} (notebook N=21250, duty_cycle bug)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.plot(lam_spec, T, color="C1", label="T (|S21|^2)")
    ax2.axvline(lam_bragg, color="k", ls=":", lw=1)
    ax2.set_xlabel("wavelength (um)"); ax2.set_ylabel("transmittance")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "dbr_spectrum.png", dpi=110)
    plt.close(fig)

    # --- Figure 3: stop-band zoom -----------------------------------------
    zoom = np.abs(lam_spec - lam_bragg) < 0.006
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lam_spec[zoom], R[zoom], label="R")
    ax.plot(lam_spec[zoom], T[zoom], label="T")
    ax.axvline(lam_bragg, color="k", ls=":", lw=1, label=f"Bragg={lam_bragg:.4f}")
    ax.set_xlabel("wavelength (um)"); ax.set_ylabel("power")
    ax.set_title(f"DBR stop-band zoom [TE], N={N_PERIODS}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "dbr_spectrum_zoom.png", dpi=110)
    plt.close(fig)

    # --- Figure 4: mode profiles (|E|^2) for w1 and w2 ---------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, w, lbl in zip(axs, [W1, W2], [f"w={W1}", f"w={W2}"]):
        _, e2, nx, ny = _strip_solve(w, DESIGN_LAM)
        ax.imshow(e2.T, extent=[0, nx * DX, 0, ny * DX], origin="lower",
                  cmap="hot", aspect="equal")
        ax.axvline(PAD, color="c", lw=0.5); ax.axvline(PAD + w, color="c", lw=0.5)
        ax.axhline(PAD, color="c", lw=0.5); ax.axhline(PAD + H, color="c", lw=0.5)
        ax.set_title(f"{lbl}  |E|^2  (lam={DESIGN_LAM})")
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    fig.tight_layout(); fig.savefig(OUT / "dbr_mode_profiles.png", dpi=110)
    plt.close(fig)

    print(f"Bragg lambda (design {DESIGN_LAM}): {lam_bragg:.4f} um")
    print(f"w={W1} Cauchy: A={A1:.4f} B={B1:.5e} C={C1:.5e}")
    print(f"w={W2} Cauchy: A={A2:.4f} B={B2:.5e} C={C2:.5e}")
    print(f"N periods: {N_PERIODS} (notebook N=21250, duty_cycle-bug-inflated)")
    print(f"stop-band peak R: {R.max():.4f} at lam={lam_spec[R.argmax()]:.4f}")
    print(f"wrote 4 PNGs to {OUT}")


if __name__ == "__main__":
    main()