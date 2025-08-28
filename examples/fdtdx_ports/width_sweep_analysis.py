# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Strip-waveguide neff vs width sweep (W9 big-fdtdx item 3).

A Si strip (n=3.48, h=220 nm) on oxide (n=1.44), air above -- the canonical SOI
waveguide. The full-vector 2D FD mode solver (meep_torch.modes.fd2d.solve_2d)
solves the transverse cross-section (width x height) at a fixed vacuum
wavelength (1.55 um) for each strip width W; the highest-Re(neff) mode is the
fundamental guided mode.

Two numeric gates (asserts; PNG for eyeballing):

  (1) monotonic: neff(W) is non-decreasing. A wider strip confines the mode
      better, so neff rises from just above n_clad (narrow, weakly guided)
      toward n_core (wide, bulk-like). A decrease would mean the solver lost
      the fundamental and returned a higher-order or radiative mode.
  (2) confined: every neff > n_clad (oxide 1.44) AND the mode is genuinely
      guided, not leaky -- Im(neff) (keff) is ~0. A leaky mode near cutoff gets
      a large keff (radiation into the PML); the gate rejects it so the sweep
      only walks the bound fundamental branch.

Run: uv run python examples/fdtdx_ports/width_sweep_analysis.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meep_torch.modes.fd2d import solve_2d

OUT = pathlib.Path(__file__).parent

# --- strip + wavelength ---
N_SI, N_OX, N_AIR = 3.48, 1.44, 1.0
LAM = 1.55                      # um (vacuum wavelength)
F = 1.0 / LAM                   # spatial frequency (1/um) for solve_2d
H = 0.22                        # strip height (um, SOI 220 nm)
RES = 32                        # cells per um
N = 2 * RES                     # cross-section cells per axis (2 um)
DX = 1.0 / RES
SS = 4                          # subpixel supersampling
NPML = 10                       # PML cells on each transverse wall
WIDTHS = (0.5, 0.7, 0.9, 1.1, 1.4, 1.8, 2.2)


def strip_eps(W):
    """Cell-center eps (N, N) for a centered Si strip of width W um on oxide."""
    yy = (np.arange(N * SS) + 0.5) / (RES * SS) - 1.0     # width  dir (-1..1 um)
    zz = (np.arange(N * SS) + 0.5) / (RES * SS) - 1.0     # height dir (-1..1 um)
    YY, ZZ = np.meshgrid(yy, zz, indexing="ij")
    fine = np.full((N * SS, N * SS), float(N_AIR) ** 2)
    fine[ZZ < -H / 2] = float(N_OX) ** 2                   # oxide below the strip
    fine[(np.abs(YY) < W / 2) & (np.abs(ZZ) < H / 2)] = float(N_SI) ** 2
    return fine.reshape(N, SS, N, SS).mean(axis=(1, 3))


def main():
    # k=8 + bound-mode selection: at wide W the shift-invert returns spurious
    # PML leaky modes (|keff|~0.5) whose Re(neff) sits ABOVE the true
    # fundamental. There is a clean keff gap (bound <= 1.1e-2, leaky >= 4e-2),
    # so the fundamental is the highest-Re(neff) mode among the bound set.
    neffs, keffs = [], []
    for W in WIDTHS:
        m = solve_2d(strip_eps(W), DX, F, k=8, npml=(NPML, NPML),
                    target_neff=N_SI)
        nk = np.asarray(m.neff)
        n_re = nk.real
        k_im = nk.imag
        bound = np.abs(k_im) < 5e-2
        i = int(np.argmax(np.where(bound, n_re, -1e9)))
        n0, k0 = float(n_re[i]), float(k_im[i])
        neffs.append(n0)
        keffs.append(k0)
        print(f"W={W:.2f} um  neff={n0:.4f}  keff={k0:+.2e}")
    neffs = np.array(neffs)
    keffs = np.array(keffs)

    # gate (1): monotonic non-decreasing (allow 1e-3 numerical noise).
    diffs = np.diff(neffs)
    monotonic = bool(np.all(diffs >= -1e-3))
    # gate (2): confined -- neff above n_clad and the mode bound (|keff| small).
    confined = bool(np.all(neffs > N_OX + 0.1) and np.all(np.abs(keffs) < 3e-2))

    print(f"gate1 monotonic={monotonic} (diffs={np.round(diffs, 4).tolist()})")
    print(f"gate2 confined={confined} "
          f"(min neff={neffs.min():.4f} > {N_OX}, max |keff|={np.abs(keffs).max():.2e})")

    assert monotonic, f"neff not monotonic: diffs={np.round(diffs, 4).tolist()}"
    assert confined, f"not all guided: neff={np.round(neffs, 4).tolist()} " \
                     f"keff={np.round(keffs, 4).tolist()}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(WIDTHS, neffs, "o-", label="neff (fundamental)")
    ax1.axhline(N_OX, ls="--", color="C1", label=f"n_clad={N_OX}")
    ax1.axhline(N_SI, ls="--", color="C2", label=f"n_core={N_SI}")
    ax1.set_xlabel("strip width W (um)")
    ax1.set_ylabel("neff")
    ax1.set_title("Strip neff vs width (1.55 um, h=220 nm)")
    ax1.legend()
    ax2.plot(WIDTHS, np.abs(keffs), "s-", color="C3")
    ax2.set_xlabel("strip width W (um)")
    ax2.set_ylabel("|keff| (loss)")
    ax2.set_title("Fundamental keff (0 = bound)")
    ax2.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT / "width_sweep_analysis.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT / 'width_sweep_analysis.png'}")


if __name__ == "__main__":
    main()