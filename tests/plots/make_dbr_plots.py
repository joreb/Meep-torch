# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""DBR effective-index + TMM spectrum plots, pure meep_torch (TE).

Reproduces Photonics_Simulation/src/docs/Testing_DBR_documented.py lines
1-112 in pure meep_torch, polarization TE throughout (the user confirmed TE;
the notebook's `DBR_TM_SM` class name is ignored). Two figures land next to
this file:

  1. dbr_neff_vs_wavelength.png -- EIM slab-collapsed lateral neff(lambda) for
     the SiN/SiO2 h=0.150 guide at widths w=1.14 and w=0.86, lambda
     0.574-0.774 um (~50 pts), overlaid with a Cauchy fit
     n = A + B/lam^2 + C/lam^4 (polyfit in x = 1/lam^2 space).

  2. dbr_spectrum.png -- 1D TMM R = |S11|^2 and T = |S21|^2 vs lambda for the
     physical layer lengths l1=0.1, l2=0.102 um (NOT the notebook's
     `duty_cycle = min(l1/l2, l2/l1)` ratio bug, which yields near-uniform
     layers l1_eff=0.198/l2_eff=0.004 and forces N=21250). Widths w1=1.14,
     w2=0.86, TE lateral neff, optional loss 2e-4/um. Window centered on the
     Bragg lambda_B = 2*(n1*l1 + n2*l2), ~4x stop-band width.

Run: uv run python tests/plots/make_dbr_plots.py          (CPU, ~30 s)
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.modes.effindex import neff_table
from meep_torch.modes.fd1d import solve_1d

OUT = pathlib.Path(__file__).parent

# --- geometry (notebook values) -------------------------------------------------
H = 0.150              # waveguide height (um)
W1, W2 = 1.14, 0.86    # waveguide widths (um) -- notebook values
L1, L2 = 0.1, 0.102    # physical layer lengths (um) -- notebook values
DESIGN_LAM = 0.674     # notebook design wavelength (um)
LOSS = 0.0            # optional power loss per um (notebook had 2e-4); 0 keeps
                     # R+T=1 (lossless). Adding it as an imaginary neff in the
                     # propagation only, with real-n interface matrices, breaks
                     # passivity and gives R>1, so it is left off here.
DX_V = 0.005           # vertical slab collapse discretization
DX_L = 0.005           # lateral 1D solve discretization
PAD = 2.0             # lateral cladding pad (um)
N_PERIODS = 4000       # TMM periods -- deep visible stop band; TMM is cheap
LAM_LO, LAM_HI = 0.574, 0.774
N_LAM_NEFF = 50


# --- EIM helpers (TE) -----------------------------------------------------------
def _n_v(lam, pol="te"):
    """Vertical slab-collapsed effective index at lam (TE/TM)."""
    return float(
        neff_table([(2.0, SiO2), (H, Si3N4), (2.0, SiO2)], [1.0 / lam], pol, dx=DX_V)[0]
    )


def _lateral_neff(w, lam, pol="te"):
    """Slab-collapsed lateral mode index of a width-w guide at lam (EIM core)."""
    f = 1.0 / lam
    n_v = _n_v(lam, pol)
    n_sio2 = SiO2.n(f).real
    eps_line = np.concatenate([
        np.full(int(PAD / DX_L), n_sio2 ** 2),
        np.full(int(w / DX_L), n_v ** 2),
        np.full(int(PAD / DX_L), n_sio2 ** 2),
    ])
    return float(solve_1d(eps_line, DX_L, f, pol, k=1).neff[0])


def _cauchy_fit(lams, neff):
    """Fit n = A + B/lam^2 + C/lam^4 via polyfit in x = 1/lam^2 (n = A + B*x + C*x^2)."""
    x = 1.0 / lams ** 2
    c2b_a = np.polyfit(x, neff, 2)          # highest power first -> [C, B, A]
    C, B, A = c2b_a
    return float(A), float(B), float(C)


def _cauchy_eval(lams, A, B, C):
    return A + B / lams ** 2 + C / lams ** 4


# --- TMM (physical layer lengths, optional loss) -------------------------------
def _tmm_spectrum(lams, n1, n2, N, loss=0.0):
    """1D TMM R = |S11|^2 and T = |S21|^2 vs lambda.

    Transfer matrix M relates forward/backward amplitudes across the stack
    (matched outer media). With only a forward wave exiting the right port:
        S21 = 1 / M[0,0],  S11 = M[1,0] / M[0,0].
    Optional uniform power loss/um enters as a small imaginary neff
    (field decays as exp(-loss*z/2), so n -> n + 1j*loss*lam/(4*pi)).
    """
    R = np.empty_like(lams)
    T = np.empty_like(lams)
    for i, lam in enumerate(lams):
        k0 = 2.0 * np.pi / lam
        # complex neff with optional absorption
        nc1 = n1 + 1j * loss * lam / (4 * np.pi)
        nc2 = n2 + 1j * loss * lam / (4 * np.pi)

        def prop(l, nc):
            be = k0 * nc
            return np.array([[np.exp(1j * be * l), 0],
                             [0, np.exp(-1j * be * l)]], dtype=complex)

        def iface(x, y):
            a = (x + y) / (2 * np.sqrt(x * y))
            b = (x - y) / (2 * np.sqrt(x * y))
            return np.array([[a, b], [b, a]], dtype=complex)

        # one period: enter n1 -> l1 -> step to n2 -> l2 -> step back to n1
        Tp = iface(n1, n2) @ prop(L2, nc2) @ iface(n2, n1) @ prop(L1, nc1)
        M = np.linalg.matrix_power(Tp, N)
        R[i] = abs(M[1, 0] / M[0, 0]) ** 2
        T[i] = abs(1.0 / M[0, 0]) ** 2
    return R, T


def _stopband_fwhm(lams, R):
    """Half-width at half-max of the R peak (stop-band half-width), in um."""
    peak = R.argmax()
    half = 0.5 * R[peak]
    # walk left until R drops below half
    lo = peak
    while lo > 0 and R[lo] > half:
        lo -= 1
    hi = peak
    while hi < len(R) - 1 and R[hi] > half:
        hi += 1
    return 0.5 * (lams[hi] - lams[lo])


def figure1_neff_vs_wavelength():
    lams = np.linspace(LAM_LO, LAM_HI, N_LAM_NEFF)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fits = {}
    neff_by_w = {}
    for w, color in ((W1, "#1f77b4"), (W2, "#d62728")):
        neff = np.array([_lateral_neff(w, lam) for lam in lams])
        neff_by_w[w] = neff
        A, B, C = _cauchy_fit(lams, neff)
        fits[w] = (A, B, C, float(neff.min()), float(neff.max()))
        ax.plot(lams, neff, "-", color=color,
                label=f"w={w:.2f} um  EIM neff")
        ax.plot(lams, _cauchy_eval(lams, A, B, C), "k--", lw=1, alpha=0.6,
                label=f"w={w:.2f}  Cauchy  A={A:.5f} B={B:.5f} C={C:.5f}")
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("lateral effective index neff")
    ax.set_title("DBR slab-collapsed lateral neff(lambda) -- TE, SiN/SiO2 h=0.150 um")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "dbr_neff_vs_wavelength.png", dpi=150)
    plt.close(fig)
    print("dbr_neff_vs_wavelength.png")
    return lams, fits, neff_by_w


def figure2_spectrum():
    n1 = _lateral_neff(W1, DESIGN_LAM)
    n2 = _lateral_neff(W2, DESIGN_LAM)
    lam_B = 2 * (n1 * L1 + n2 * L2)

    # broad pre-sweep to find stop band, then window to ~4x FWHM
    pre = np.linspace(lam_B - 0.05, lam_B + 0.05, 401)
    R_pre, T_pre = _tmm_spectrum(pre, n1, n2, N_PERIODS, loss=LOSS)
    hw = _stopband_fwhm(pre, R_pre)
    win = 4 * hw if hw > 0 else 0.05
    lams = np.linspace(lam_B - win, lam_B + win, 401)
    R, T = _tmm_spectrum(lams, n1, n2, N_PERIODS, loss=LOSS)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lams, R, "-", color="#d62728", label="R = |S11|^2 (reflectance)")
    ax.plot(lams, T, "-", color="#1f77b4", label="T = |S21|^2 (transmittance)")
    ax.axvline(lam_B, color="k", ls=":", lw=1, label=f"lambda_B = {lam_B:.4f} um")
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("power fraction")
    ax.set_title(
        f"DBR TMM spectrum -- TE, l1={L1} l2={L2} um, w1={W1} w2={W2}, N={N_PERIODS}\n"
        f"(physical layer lengths; notebook duty_cycle=min-ratio bug avoided)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "dbr_spectrum.png", dpi=150)
    plt.close(fig)
    print("dbr_spectrum.png")
    return dict(n1=n1, n2=n2, lam_B=lam_B, lams=lams, R=R, T=T,
                R_peak=float(R.max()), R_peak_lam=float(lams[R.argmax()]),
                T_dip=float(T.min()), hw=hw)


def _verify(lams, fits, neff_by_w, spec):
    print("\n=== NUMERIC VERIFICATION ===")
    for w, (A, B, C, nmin, nmax) in fits.items():
        neff = neff_by_w[w]
        resid = np.abs(neff - _cauchy_eval(lams, A, B, C))
        print(f"w={w:.2f}: neff range [{nmin:.6f}, {nmax:.6f}], "
              f"Cauchy A={A:.6f} B={B:.6f} C={C:.6f}, max resid={resid.max():.2e}")
        assert resid.max() < 1e-3, f"Cauchy residual {resid.max():.2e} too large for w={w}"
    print(f"Bragg lambda_B = {spec['lam_B']:.6f} um  (n1={spec['n1']:.6f}, n2={spec['n2']:.6f})")
    print(f"R peak = {spec['R_peak']:.4f} at lam={spec['R_peak_lam']:.4f} um")
    print(f"T dip = {spec['T_dip']:.4f} (stop-band FWHM half-width ~{spec['hw']:.4f} um)")
    assert spec['R_peak'] <= 1.0 + 1e-6, f"R peak {spec['R_peak']} > 1 (passivity broken)"
    for png in ("dbr_neff_vs_wavelength.png", "dbr_spectrum.png"):
        p = OUT / png
        assert p.exists(), f"missing {png}"
        assert p.stat().st_size > 10_000, f"{png} too small ({p.stat().st_size} B)"
        print(f"{png}: {p.stat().st_size} B  OK")
    print("=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    lams, fits, neff_by_w = figure1_neff_vs_wavelength()
    spec = figure2_spectrum()
    _verify(lams, fits, neff_by_w, spec)
    print("plots written to", OUT)