# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Photonics parity group 2: DBR reflectance -- ports S-matrix vs inline TMM.

Mirrors Photonics_Simulation's DBR (`src/photonic/dbr/__init__.py`,
`Testing_DBR_documented.ipynb`): a width-modulated SiN/SiO2 waveguide grating.
The notebook's TMM has a duty-cycle bug (`duty_cycle = min(l1/l2, l2/l1)`, a
ratio ~0.98, not the fraction l1/(l1+l2)~0.495), which makes its effective
layer lengths l1_eff=0.198um, l2_eff=0.004um -- almost a uniform section, which
is why the notebook needs N=21250 periods to build a stop band. The reference
TMM here uses the *physical* layer lengths l1=0.1, l2=0.102 (matching the 2D
FDTD geometry), so a small N shows a clear stop band.

The notebook's widths w1=1.14, w2=0.86 are wide enough that the slab-collapsed
(group 1) effective index barely changes between them (dn~0.01) -- a small-N
2D FDTD DBR with those widths has R below the npml=10 |S21| noise floor (and
the notebook's N=21250 is infeasible in 2D FDTD). This test uses w1=1.0,
w2=0.3 (dn~0.09) which keeps the Bragg wavelength at the notebook's design
(~0.674um) while giving a visible small-N stop band.

The 2D FDTD uses the slab-collapsed EIM core (eps = n_v^2, nondispersive at
the design wavelength) on a SiO2 clad, width-modulated, rasterized with Kottke
subpixel smoothing (staircase raster makes R(L) non-monotone). The gate is the
|S21| dip (stop-band) LOCATION, robust at npml=10 -- the |S11| magnitude would
need npml>=60 + a decayed window (far-PML echo lands in the backward coeff).
The 2D FDTD has extra broadband radiation loss at the near-cutoff w2 width
steps that the 1D TMM ignores, so |S21| baselines differ (~0.85 vs 1.0); the
dip *location* (Bragg wavelength) is the robust, model-independent quantity.
"""
import numpy as np
import pytest
import torch

from meep_torch.core.grid import Grid
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.modes.effindex import neff_table
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.port import Port
from meep_torch.ports.smatrix import compute_smatrix

# DBR geometry (notebook params where feasible; widths narrowed for contrast).
W1, W2 = 1.0, 0.3          # waveguide widths (um) -- narrowed from 1.14/0.86
L1, L2 = 0.1, 0.102         # layer lengths (um) -- notebook values
H = 0.150                  # waveguide height (um) -- notebook value
N_PERIODS = 8
DESIGN_LAM = 0.674
DX = 0.02
NPML = 10
SWEEP_LAMS = np.linspace(0.65, 0.71, 13)


def _n_v(lam):
    return float(neff_table([(2.0, SiO2), (H, Si3N4), (2.0, SiO2)], [1.0 / lam], "tm", dx=0.005)[0])


def _lateral_neff(w, lam):
    """Slab-collapsed lateral mode index of a width-w guide at lam (EIM core)."""
    f = 1.0 / lam
    n_v = _n_v(lam)
    n_sio2 = SiO2.n(f).real
    dx = 0.005
    eps_line = np.concatenate([
        np.full(int(2.0 / dx), n_sio2 ** 2),
        np.full(int(w / dx), n_v ** 2),
        np.full(int(2.0 / dx), n_sio2 ** 2),
    ])
    return float(solve_1d(eps_line, dx, f, "tm", k=1).neff[0])


def _tmm_s21(lams, N):
    """Inline TMM |S21|^2 spectrum (physical layer lengths, nondispersive at
    the design wavelength -- matches the 2D FDTD's nondispersive EIM core)."""
    n1 = _lateral_neff(W1, DESIGN_LAM)
    n2 = _lateral_neff(W2, DESIGN_LAM)
    out = []
    for lam in lams:
        def Thw(ll, ne):
            be = 2 * np.pi * ne / lam
            return np.array([[np.exp(1j * be * ll), 0], [0, np.exp(-1j * be * ll)]], dtype=complex)
        def Tis(x, y):
            a = (x + y) / (2 * np.sqrt(x * y)); b = (x - y) / (2 * np.sqrt(x * y))
            return np.array([[a, b], [b, a]], dtype=complex)
        # one period: l1 (n1) -> step -> l2 (n2) -> step
        Tp = Thw(L2, n2) @ Tis(n2, n1) @ Thw(L1, n1) @ Tis(n1, n2)
        T = np.linalg.matrix_power(Tp, N)
        out.append(abs(1.0 / T[0, 0]) ** 2)
    return np.array(out)


def _build_dbr_eps():
    n_v = _n_v(DESIGN_LAM)
    eps_core = n_v ** 2
    eps_clad = SiO2.n(1.0 / DESIGN_LAM).real ** 2
    yc = 1.5
    L_in = 0.3
    x_total = 2 * NPML * DX + 2 * L_in + N_PERIODS * (L1 + L2)
    nx = int(round(x_total / DX))
    ny = int(round(3.0 / DX))
    g = Grid(nx=nx, ny=ny, dx=DX)

    def rect(xs, xe, w):
        return Polygon(np.array([[xs, yc - w / 2], [xe, yc - w / 2],
                                 [xe, yc + w / 2], [xs, yc + w / 2]]))
    polys = []
    xc = NPML * DX
    polys.append(rect(xc, xc + L_in, W1)); xc += L_in
    for _ in range(N_PERIODS):
        polys.append(rect(xc, xc + L1, W1)); xc += L1
        polys.append(rect(xc, xc + L2, W2)); xc += L2
    polys.append(rect(xc, xc + L_in, W1))
    rm = rasterize_scene([(p, eps_core) for p in polys], g, background=eps_clad)
    pin = int((NPML * DX + 0.7 * L_in) / DX)
    pout = int((xc + 0.3 * L_in) / DX)
    lo = int((yc - 1.0) / DX); hi = int((yc + 1.0) / DX)
    ports = [Port("x", pin, lo, hi, +1), Port("x", pout, lo, hi, -1)]
    return rm.eps_tm, g, ports


@pytest.fixture(scope="module")
def dbr_spectra():
    eps, g, ports = _build_dbr_eps()
    freqs = (1.0 / SWEEP_LAMS).tolist()
    sm = compute_smatrix(eps, g, ports, pol="tm", fcen=1.0 / DESIGN_LAM,
                         fwidth=0.4, freqs=freqs, n_steps=4000, npml=NPML)
    s21 = np.array([abs(sm.s[i, 1, 0]).item() for i in range(len(SWEEP_LAMS))])
    tmm = _tmm_s21(SWEEP_LAMS, N_PERIODS)
    return s21, tmm


def test_dbr_has_a_stop_band(dbr_spectra):
    s21, _ = dbr_spectra
    # a clear dip (Bragg reflection) above the sweep noise
    assert s21.min() < s21.max() - 0.03, (
        f"no discernible DBR dip: |S21| range {s21.min():.4f}..{s21.max():.4f}")


def test_dbr_dip_location_matches_tmm(dbr_spectra):
    s21, tmm = dbr_spectra
    lam_fDTD = SWEEP_LAMS[int(s21.argmin())]
    lam_tmm = SWEEP_LAMS[int(tmm.argmin())]
    # Bragg-wavelength location is model-independent (2D radiation shifts the
    # baseline, not the Bragg condition); gate at 1.5% (~0.01um).
    assert abs(lam_fDTD - lam_tmm) < 0.012, (
        f"FDTD dip at {lam_fDTD:.4f} vs TMM dip at {lam_tmm:.4f} "
        f"(diff {abs(lam_fDTD - lam_tmm):.4f} > 0.012)")


def test_dbr_bragg_wavelength_near_notebook_design(dbr_spectra):
    # The Bragg wavelength (dip) sits at the notebook's design band ~0.674um.
    s21, _ = dbr_spectra
    lam_dip = SWEEP_LAMS[int(s21.argmin())]
    assert abs(lam_dip - DESIGN_LAM) < 0.02, (
        f"DBR dip at {lam_dip:.4f} far from notebook design {DESIGN_LAM}")