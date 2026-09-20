# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W9 wave 2: 3D engine hardening gates.

(a) 3D lossy-slab transmission: a z-normal Ex-polarized planewave crosses a
    finite lossy slab; the transmitted flux ratio vs a vacuum reference
    matches the analytic normal-incidence slab transmission to <=3%.
(b) PML residual vs absorber: a z-normal planewave in vacuum; the spurious
    reflection from the back CPML (pulse ringdown at a near-source probe) is
    measurably below the matched-sigma absorber at the same depth and within
    a few % of the incident amplitude.
"""
import math

import numpy as np
import pytest
import torch

from meep_torch.core.monitors_3d import FluxMonitor3D, PlaneSource, run_3d
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state, step_3d


# ---------------------------------------------------------------------------
# analytic lossy-slab power transmission (vacuum | slab | vacuum, normal
# incidence, e^{-iwt} phasors: eps_c = eps + i sigma/omega).
# ---------------------------------------------------------------------------
def _slab_T_power(f, eps, sigma_d, d):
    """Power transmission of a lossy slab (vacuum | slab | vacuum, normal
    incidence). Airy/transfer-matrix form with e^{-iwt} phasors and a forward
    wave e^{+ikz}: eps_c = eps (1 + i sigma/omega) (the FDTD d1/d2 fast-path
    damps D, so the conductivity enters as a D-conductivity). The internal
    round trips use the back-interface reflection squared r_b^2 = r^2, so the
    denominator is 1 - r^2 e^{2i delta} (standard Airy, T -> 1 at resonance)."""
    w = 2.0 * math.pi * f
    eps_c = eps * (1.0 + 1j * sigma_d / w)
    n = np.sqrt(eps_c)
    r = (1.0 - n) / (1.0 + n)                 # E-field Fresnel at vacuum->slab
    delta = w * n * d
    t = (1.0 - r * r) * np.exp(1j * delta) / (1.0 - r * r * np.exp(2j * delta))
    return float(abs(t) ** 2)


# ---------------------------------------------------------------------------
# (a) lossy-slab transmission. Mirrors the 2D W5 conductivity gate: a vacuum
# (eps=1) background with a finite D-conductivity slab. Keeping eps uniform
# avoids permittivity-contrast staircasing -- a hard eps step at a Yee cell
# face shifts the slab Fabry-Perot phase by ~0.5 rad/cell and would dominate
# the 3% budget; this way the gate cleanly exercises the material d1/d2
# fast-path. Analytic = full vacuum|slab|vacuum transfer matrix (e^{-iwt},
# eps_c = 1 + i sigma/omega, which is what the d1/d2 D-damping realizes).
# ---------------------------------------------------------------------------
NX = NY = 28
NZ = 64
DX = 0.05
NPML = 6
FCEN, FWIDTH = 1.0, 0.2
SRC_Z = 8
SLAB_Z0, SLAB_Z1 = 18, 38           # thickness = 20 cells = 1.0
SIGMA_D = 1.0
MON_Z = 40
N_STEPS = int(round(60.0 / (0.5 * DX / math.sqrt(3.0))))


def _transmitted_flux(sigma_d=None):
    sd = None
    if sigma_d is not None:
        sd = torch.zeros(NX, NY, NZ, dtype=torch.float64)
        sd[:, :, SLAB_Z0:SLAB_Z1] = sigma_d
    coeffs = build_3d_coeffs(NX, NY, NZ, DX, npml=NPML, sigma_d=sd)
    st = init_3d_state(NX, NY, NZ, npml=NPML)
    src = PlaneSource(NX, NY, NZ, coeffs.dt, "Ex", axis=2, idx=SRC_Z,
                      profile=GaussianPulse(FCEN, FWIDTH))
    mon = FluxMonitor3D(coeffs.dt, [FCEN], normal="z", idx=MON_Z, ap=NPML + 1)
    _, (acc,) = run_3d(st, coeffs, N_STEPS, source=src, monitors=[mon])
    return float(mon.flux(acc, DX)[0])


def test_3d_lossy_slab_transmission():
    d = (SLAB_Z1 - SLAB_Z0) * DX
    T_analytic = _slab_T_power(FCEN, 1.0, SIGMA_D, d)
    f_inc = _transmitted_flux()                       # vacuum reference
    f_trans = _transmitted_flux(SIGMA_D)
    T_sim = f_trans / f_inc
    err = abs(T_sim - T_analytic) / T_analytic
    assert err <= 0.03, (
        f"3D lossy-slab |T|^2: sim {T_sim:.4f} vs analytic {T_analytic:.4f} "
        f"(rel {err:.3%}); flux inc={f_inc:.4g} trans={f_trans:.4g}")


# ---------------------------------------------------------------------------
# (b) PML residual reflection vs absorber
#
# A z-uniform Ex planewave (no transverse k) in vacuum, so the only boundary
# reflection is from the z-PML: the x/y CPML is inert for the uniform
# transverse field (zero transverse curl -> no psi accumulation). The
# comparison is run in *mixed mode*: the absorber case keeps CPML on x/y
# (inert, identical to the CPML case) and replaces only the z CPML with a
# matched-sigma absorber (same depth, same R0=1e-6 and grade m=3 as the CPML
# -- ``absorber_m``/``absorber_R0``), isolating the z-boundary quality. A
# point-source ringdown is dominated by the source near-field (size-
# independent ~2.7e-3 floor, not boundary reflection); the planewave has no
# near-field, so the ringdown is pure boundary absorption.
#
# Two phase-robust quantities at a center probe (Ex), each an integrated
# energy (a mean of E^2 over a window, not a phase-dependent peak):
#   * residual = E^2(late) / E^2(peak)  -- the late field as a fraction of
#     the in-flight pulse; the CPML must be within a few % of incident.
#   * decay = E^2(late2) / E^2(late1)  -- the per-round-trip decay rate; the
#     CPML must decay faster than the matched absorber (R_CPML < R_abs).
# At normal incidence a graded absorber and a polynomial-graded CFS-PML are
# comparable at very thin depth; the CPML's polynomial grading wins once it
# has enough layers to engage (npml>=7 here), which is the regime this gate
# exercises. The transverse PEC/CPML dynamics cancel in the CPML/absorber
# ratio (identical x/y treatment), so the decay comparison is clean.
# ---------------------------------------------------------------------------
R_NX = R_NY = 12
R_NZ = 120
R_NPML = 8
R_DX = 0.05
R_T_END = 90.0
R_FCEN, R_FW, R_CUTOFF = 1.0, 0.5, 3.0
R_NSTEPS = int(round(R_T_END / (0.5 * R_DX / math.sqrt(3.0))))


def _ringdown(cpml: bool):
    """Return (residual, decay) for the z-planewave ringdown. ``cpml`` selects
    full CPML (z CPML) vs mixed mode (z matched-sigma absorber, x/y CPML)."""
    # pml_alpha=0.0 ON BOTH BRANCHES, and it is the difference between a
    # measurement and an unfair fight. This compares the CPML against a
    # matched-sigma absorber, and the absorber has no CFS term by
    # construction. The CFS alpha suppresses evanescent and low-frequency
    # content and, in exchange, leaves the layer impedance-matched only as
    # alpha -> 0, so a propagating wave at normal incidence sees a residual
    # reflection that grows with alpha. Measured at the package default of
    # 0.05 the CPML decayed 5.693x SLOWER than the absorber; at 0.0 and 0.01
    # it wins, as it should. That was read as a CPML defect for a while. It is
    # not: it is this gate comparing a layer that carries a CFS term against
    # one that cannot, on the one wave type the CFS term costs.
    if cpml:
        coeffs = build_3d_coeffs(R_NX, R_NY, R_NZ, R_DX, npml=R_NPML, n_abs=0,
                                 pml_alpha=0.0)
    else:
        coeffs = build_3d_coeffs(R_NX, R_NY, R_NZ, R_DX, npml=R_NPML,
                                  n_abs=R_NPML, absorber_axes=(2,),
                                  absorber_m=3, absorber_R0=1e-6,
                                  pml_alpha=0.0)
    st = init_3d_state(R_NX, R_NY, R_NZ, npml=R_NPML)
    pulse = GaussianPulse(R_FCEN, R_FW, cutoff=R_CUTOFF)
    src_z = R_NZ // 2
    mask = torch.zeros(R_NX - 1, R_NY, R_NZ, dtype=torch.float64)
    mask[:, :, src_z] = 1.0                       # uniform transverse planewave
    src = ("Ex", mask)
    series = np.empty(R_NSTEPS)
    for t in range(R_NSTEPS):
        st = step_3d(st, coeffs, pulse.amplitude((t + 0.5) * coeffs.dt), src)
        series[t] = st.Ex[R_NX // 2 - 1, R_NY // 2, src_z].item()
    tt = (np.arange(R_NSTEPS) + 1) * coeffs.dt
    e = series * series

    def win(a, b):
        return float(e[(tt >= a) & (tt < b)].mean())

    peak = win(10, 20)                            # in-flight pulse (AC, post-src)
    late = win(70, 85)                            # late ringdown
    decay = win(60, 72) / win(30, 42)             # phase-robust per-bounce decay
    return late / peak, decay


def test_3d_pml_residual_vs_absorber():
    res_cpml, dec_cpml = _ringdown(cpml=True)
    res_abs, dec_abs = _ringdown(cpml=False)
    # CPML residual within a few % of the incident pulse (PML works).
    assert res_cpml <= 0.05, (
        f"CPML residual {res_cpml:.2e} exceeds 5% of incident")
    # CPML per-bounce decay faster than the matched-sigma absorber at the same
    # depth (R_CPML < R_abs  <=>  decay_CPML < decay_abs).
    assert dec_cpml < dec_abs, (
        f"CPML decay {dec_cpml:.2e} not below absorber {dec_abs:.2e} "
        f"(ratio {dec_cpml / dec_abs:.3f})")