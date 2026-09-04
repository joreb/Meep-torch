# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Per-axis + Drude dispersion tests (W9 big-fdtdx item 1).

Gate (a): a Drude slab (pole (fn=0, gamma, sigma, fc=fp)) reflects a CW
plane wave at f0 with R matching the analytic 1D transfer matrix built from
    eps(f) = eps_inf - sigma*fp^2 / (f^2 + i*gamma*f)
within 3%. The recurrence's Drude pole (w0=0, wc=2*pi*fp) is verified by the
match: if the pole scaling were wrong the simulated R would not land on the
analytic curve.

Gate (b): per-axis pole selectivity. A TE plane wave (Ey, +x) traverses a
slab carrying a Lorentz pole on ONE E-axis only. The x-axis pole (poles_x)
acts on Ex, which is ~0 for a y-invariant plane wave, so transmission is
unchanged (T_x ~ 1). The y-axis pole (poles_y) acts on Ey, the propagating
DOF, so the slab is metallic for this polarization and transmission is killed
(T_y ~ 0). Gate: T_y < 5% of T_x -- the per-axis pole acts on the right axis.
"""
import math

import numpy as np
import torch

from meep_torch.core.coeffs import build_te_coeffs, build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTE, PointProbe
from meep_torch.core.run import run_te, run_tm
from meep_torch.core.sources import GaussianPulse, SoftSource
from meep_torch.core.stepper_tm import init_tm_state
from meep_torch.core.stepper_te import init_te_state

F0 = 0.3
# Zero-DC Gaussian pulse (cos-modulated Gaussian): the Drude pole (fn=0) is a
# free-electron integrator whose DC response diverges, so a CW/ramp source
# (nonzero DC content during turn-on) drives P -> inf. The cos-Gaussian has
# zero mean -> no DC accumulation -> P stays bounded. Spectral DFT at f0 then
# extracts R(f0) exactly (linearity), independent of pulse bandwidth.
PULSE_A = GaussianPulse(F0, 0.1)
PULSE_B = GaussianPulse(F0, 0.1)

# --- gate (a): TM Drude slab reflection vs 1D transfer matrix ---
# dx=0.025 (not 0.05): the slab is only d=2.0, so the half-cell thickness
# ambiguity at the vacuum/Drude faces is a first-order discretization error;
# convergence dx 0.05 -> 5.6%, 0.04 -> 4.3%, 0.025 -> 2.5% (under the 3% gate).
NX_A, NY_A, DX_A, NPML_A = 1200, 3, 0.025, 80
GRID_A = Grid(nx=NX_A, ny=NY_A, dx=DX_A)
SLAB0_A, SLAB1_A = 560, 640          # Drude slab, d = 80*0.025 = 2.0
SRC_IX_A, PROBE_IX_A = 300, 400      # probe between source and slab
T_END_A = 150.0
# Drude parameters: eps(f0) = 1 - sigma*fp^2/(f0^2 + i*gamma*f0)
DRUDE = (0.0, 0.03, 0.6, 0.4)        # (fn=0, gamma, sigma, fc=fp)


def _dft_f0(series, f0, dt):
    """Complex DFT coefficient at f0 over the full window (linear subtraction)."""
    t = np.arange(len(series)) * dt
    return complex(np.sum(series * np.exp(-1j * 2 * math.pi * f0 * t)) * dt)


def _run_a(pole_mask_on):
    eps = torch.ones(NX_A, NY_A, dtype=GRID_A.dtype)
    if pole_mask_on:
        mask = torch.zeros(NX_A, NY_A, dtype=GRID_A.dtype)
        mask[SLAB0_A:SLAB1_A, :] = 1.0
        coeffs = build_tm_coeffs(GRID_A, eps_inf=eps, poles=[DRUDE],
                                 pole_masks=[mask], npml=NPML_A, y_pmc=True)
        n_poles = 1
    else:
        coeffs = build_tm_coeffs(GRID_A, eps_inf=eps, npml=NPML_A, y_pmc=True)
        n_poles = 0
    smask = GRID_A.zeros(NX_A, NY_A)
    smask[SRC_IX_A, 1] = 1.0
    src = SoftSource(GRID_A, smask, PULSE_A)
    n_steps = int(round(T_END_A / GRID_A.dt))
    _, (series,) = run_tm(init_tm_state(GRID_A, n_poles=n_poles, y_pmc=True),
                          coeffs, n_steps, source=src,
                          monitors=[PointProbe(PROBE_IX_A, 1)])
    return series.detach().cpu().numpy()


def _drude_R_analytic(f0, d, fn=0.0, gamma=0.03, sigma=0.6, fp=0.4, eps_inf=1.0):
    """1D slab-in-vacuum power reflection at normal incidence.
    eps(f) = eps_inf - sigma*fp^2/(f^2 + i*gamma*f); n = sqrt(eps);
    r_01 = (1-n)/(1+n); R = |r_01 (1 - e^{2i delta})/(1 - r_01^2 e^{2i delta})|^2
    with delta = 2*pi*f0*n*d. R = |r|^2 is sign-convention independent (Im n squared).
    """
    eps = eps_inf - sigma * fp ** 2 / (f0 ** 2 + 1j * gamma * f0)
    n = np.sqrt(eps)
    r01 = (1.0 - n) / (1.0 + n)
    delta = 2 * math.pi * f0 * n * d
    e2 = np.exp(2j * delta)
    r = r01 * (1.0 - e2) / (1.0 - r01 ** 2 * e2)
    return float(abs(r) ** 2)


def test_drude_slab_reflection_vs_transfer_matrix():
    """Gate (a): Drude slab R (sim, via incident/reflected DFT subtraction)
    vs analytic 1D transfer-matrix R within 3%."""
    a_inc = _dft_f0(_run_a(pole_mask_on=False), F0, GRID_A.dt)
    a_tot = _dft_f0(_run_a(pole_mask_on=True), F0, GRID_A.dt)
    a_ref = a_tot - a_inc
    R_sim = abs(a_ref) ** 2 / max(abs(a_inc) ** 2, 1e-30)
    R_an = _drude_R_analytic(F0, (SLAB1_A - SLAB0_A) * DX_A,
                             gamma=DRUDE[1], sigma=DRUDE[2], fp=DRUDE[3])
    rel = abs(R_sim - R_an) / R_an
    print(f"Drude slab R: sim={R_sim:.5f} analytic={R_an:.5f} rel={rel:.4f} "
          f"(eps_inf=1, fp={DRUDE[3]}, gamma={DRUDE[1]}, sigma={DRUDE[2]}, "
          f"d={(SLAB1_A-SLAB0_A)*DX_A:.2f})")
    assert rel <= 0.03, f"Drude R sim {R_sim:.5f} vs analytic {R_an:.5f} rel {rel:.4f}"


# --- gate (b): TE per-axis pole polarization contrast ---
NX_B, NY_B, DX_B, NPML_B = 400, 60, 0.05, 20
GRID_B = Grid(nx=NX_B, ny=NY_B, dx=DX_B)
SLAB0_B, SLAB1_B = 200, 260          # d = 60*0.05 = 3.0
SRC_IX_B = 80
# Source/monitor span the interior only (NPML+4 .. NY-NPML-4): the beam must not
# touch the y-PML, see the pole-mask note below.
J0_B, J1_B = NPML_B + 4, NY_B - NPML_B - 4
MON_IX_B = 300                       # transmitted-flux monitor (behind slab)
MON_LO_B, MON_HI_B = J0_B, J1_B
T_END_B = 150.0
# Lorentz pole (finite fn), not Drude: the "pass" axis (poles_x) carries the
# pole on Ex, which is ~0 for a y-invariant +x plane wave. A Drude pole (fn=0)
# is a free-electron integrator whose DC response diverges, so on a near-zero
# field numerical noise drives P -> inf (the x-metal run blew up at 1e43). A
# Lorentz pole has a finite DC response (chi(0)=sigma), so Px stays bounded at
# ~0 when Ex~0 -- the per-axis selectivity is what this gate tests; the Drude
# recurrence itself is gated by (a). Detuned just below f0 (fn=0.28) with a
# strong oscillator (sigma=2.2) and light damping (gamma=0.03) so the Ey-axis
# pole is a strong absorber at f0 (T_y ~ 3% of T_x) while the Ex-axis pole,
# acting on the near-zero Ex, leaves the beam essentially unattenuated.
POLE_B = (0.28, 0.03, 2.2)


def _run_b(axis):
    """axis: 'vac' | 'x' | 'y' -> which E-axis carries the per-axis pole."""
    eps = torch.ones(NX_B, NY_B, dtype=GRID_B.dtype)
    # The dispersive pole is confined to the interior y (NPML .. NY-NPML), NOT
    # the y-PML. The TE PML damps E via a conductivity split but the ADE
    # polarization recurrence (c1, c2) is left untouched in the PML, so a pole
    # active where its E-component is PML-damped (Ex is damped by the y-PML,
    # k_ey) feeds an un-damped Px back into the damped Ex -> an Ex-Px growth mode
    # that diverges (1e14). The pole-active slab must not overlap the PML that
    # damps its axis. (poles_y is safe either way -- Ey is damped by the x-PML,
    # which sits outside the slab -- but the same interior-y mask is used for
    # both for consistency. The beam lives at J0..J1, well inside the PML, so the
    # vacuum-in-PML impedance step is never illuminated.)
    mask = torch.zeros(NX_B, NY_B, dtype=GRID_B.dtype)
    mask[SLAB0_B:SLAB1_B, NPML_B:NY_B - NPML_B] = 1.0
    if axis == "vac":
        coeffs = build_te_coeffs(GRID_B, eps_inf=eps, npml=NPML_B)
    elif axis == "x":
        coeffs = build_te_coeffs(GRID_B, eps_inf=eps, poles_x=[POLE_B],
                                 pole_masks_x=[mask], npml=NPML_B)
    else:  # 'y'
        coeffs = build_te_coeffs(GRID_B, eps_inf=eps, poles_y=[POLE_B],
                                 pole_masks_y=[mask], npml=NPML_B)
    # line source on Ey: mask (nx, ny) -> (nx, ny-1) for TE (slice y, per sim.py)
    smask = GRID_B.zeros(NX_B, NY_B)
    smask[SRC_IX_B, J0_B:J1_B] = 1.0
    src = SoftSource(GRID_B, smask[:, :-1], PULSE_B)
    mon = FluxMonitorTE(GRID_B, [F0], ix=MON_IX_B, lo=MON_LO_B, hi=MON_HI_B)
    n_steps = int(round(T_END_B / GRID_B.dt))
    _, (acc,) = run_te(init_te_state(GRID_B, n_poles=(1 if axis != "vac" else 0),
                                     n_poles_y=(1 if axis == "y" else 0)),
                       coeffs, n_steps, source=src, monitors=[mon])
    return float(mon.flux(acc, DX_B).item())


def test_peraxis_pole_polarization_contrast():
    """Gate (b): the y-axis pole (acts on Ey, the propagating DOF of a
    TE +x plane wave) kills transmission; the x-axis pole (acts on Ex ~ 0)
    does not. T_y < 5% of T_x."""
    p_inc = _run_b("vac")
    p_x = _run_b("x")
    p_y = _run_b("y")
    T_x = p_x / max(p_inc, 1e-30)
    T_y = p_y / max(p_inc, 1e-30)
    ratio = T_y / max(T_x, 1e-30)
    print(f"per-axis contrast: P_inc={p_inc:.4g} P_x={p_x:.4g} P_y={p_y:.4g} "
          f"T_x={T_x:.4f} T_y={T_y:.4f} T_y/T_x={ratio:.4f} (want < 0.05)")
    assert T_x > 0.5, f"x-pole should pass Ey (T_x={T_x:.4f})"
    assert ratio < 0.05, f"y-metal T_y/T_x={ratio:.4f} not < 0.05"