# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""3D ADE dispersion acceptance gates (port_3d-dispersion).

Gate (a): a Drude slab (pole (fn=0, gamma, sigma, fc=fp)) reflects a CW
plane wave at f0 with R matching the analytic 1D transfer matrix built from
    eps(f) = eps_inf - sigma*fp^2 / (f^2 + i*gamma*f)
within 3% -- the 3D mirror of test_peraxis_dispersion gate (a): a quasi-1D
z-propagating planewave on a 3 x 3 x 1200 grid with PMC transverse walls
(``pmc_axes=(0, 1)``, the 3D analogue of the 2D ``y_pmc``): tangential E is
live at the x/y walls and the transverse profiles are identity, so a
transverse-uniform field stays exactly uniform, every transverse derivative
is exactly zero, and the transverse CPML is inert. (Without PMC the PEC
zeroing of the Ey x-borders breaks uniformity and the npml=80 > nx=3
everywhere-active transverse sigma drains the wave -- e-fold ~42 voxels,
R_sim ~ 1e-10.) The Ey-polarized +z planewave keeps Ex/Ez
EXACTLY zero (all their curl terms vanish), so the isotropic Drude pole --
a free-electron integrator that diverges on noisy near-zero fields -- sees
only exact zeros on the quiet components and stays bounded.

Gate (b): stepper_3d vs stepper_3d_lean parity extended to a dispersive
case (Lorentz pole slab on a dielectric block), all six field components
to float64 round-off, mirroring test_stepper3d_lean.

Gate (c): autograd through a dispersive 3D run (Lorentz pole active around
the probed scatterer) vs central finite differences on eps voxels inside
the pole region, mirroring test_adjoint3d's FD gate.
"""
import cmath
import math

import torch

from meep_torch.core.monitors_3d import (DFTFieldsMonitor3D, PlaneSource,
                                         PointSource3D, run_3d)
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state
from meep_torch.core.stepper_3d_lean import build_lean, run_3d_lean

F0 = 0.3
# Zero-DC Gaussian pulse (cos-modulated): the Drude pole (fn=0) is a
# free-electron integrator whose DC response diverges, so the source must
# carry no DC content (same note as test_peraxis_dispersion).
PULSE_A = GaussianPulse(F0, 0.1)

# --- gate (a): quasi-1D 3D Drude slab reflection vs 1D transfer matrix ---
# dx=0.025 (not 0.05): the slab is only d=2.0, so the half-cell thickness
# ambiguity at the vacuum/Drude faces is a first-order discretization error
# (2D convergence: 0.05 -> 5.6%, 0.025 -> 2.5%, under the 3% gate).
NXT, NZ_A, DX_A, NPML_A = 3, 1200, 0.025, 80
SLAB0_A, SLAB1_A = 560, 640          # Drude slab, d = 80*0.025 = 2.0
SRC_IZ_A, PROBE_IZ_A = 300, 400      # probe between source and slab
T_END_A = 150.0
DRUDE = (0.0, 0.03, 0.6, 0.4)        # (fn=0, gamma, sigma, fc=fp)


def _run_a(pole_on):
    """Probe-plane DFT phasor of Ey at f0, with the Drude slab on/off."""
    eps = torch.ones(NXT, NXT, NZ_A, dtype=torch.float64)
    if pole_on:
        # Pole mask: slab only, interior in z (keeps the pole off the z-PML
        # that damps the propagating Ey -- the recurrence is not PML-split).
        mask = torch.zeros(NXT, NXT, NZ_A, dtype=torch.float64)
        mask[:, :, SLAB0_A:SLAB1_A] = 1.0
        coeffs = build_3d_coeffs(NXT, NXT, NZ_A, DX_A, eps=eps, npml=NPML_A,
                                 poles=[DRUDE], pole_masks=[mask],
                                 pmc_axes=(0, 1))
        n_poles = 1
    else:
        coeffs = build_3d_coeffs(NXT, NXT, NZ_A, DX_A, eps=eps, npml=NPML_A,
                                 pmc_axes=(0, 1))
        n_poles = 0
    src = PlaneSource(NXT, NXT, NZ_A, coeffs.dt, "Ey", 2, SRC_IZ_A, PULSE_A)
    mon = DFTFieldsMonitor3D(coeffs.dt, [F0], "Ey", plane=(2, PROBE_IZ_A))
    n_steps = int(round(T_END_A / coeffs.dt))
    _, (acc,) = run_3d(
        init_3d_state(NXT, NXT, NZ_A, npml=NPML_A, n_poles=n_poles),
        coeffs, n_steps, source=src, monitors=[mon])
    # acc channels: [dt*sum cos(wt) F, -dt*sum sin(wt) F] -> phasor
    # int F e^{-iwt} dt = ch0 + i*ch1
    return complex(acc[0, 0].sum() + 1j * acc[0, 1].sum())


def _drude_R_analytic(f0, d, fn=0.0, gamma=0.03, sigma=0.6, fp=0.4,
                      eps_inf=1.0):
    """1D slab-in-vacuum power reflection at normal incidence.
    eps(f) = eps_inf - sigma*fp^2/(f^2 + i*gamma*f); n = sqrt(eps);
    r_01 = (1-n)/(1+n); R = |r_01 (1 - e^{2i delta})/(1 - r_01^2 e^{2i delta})|^2
    with delta = 2*pi*f0*n*d. R = |r|^2 is sign-convention independent.
    """
    eps = eps_inf - sigma * fp ** 2 / (f0 ** 2 + 1j * gamma * f0)
    n = cmath.sqrt(eps)
    r01 = (1.0 - n) / (1.0 + n)
    delta = 2 * math.pi * f0 * n * d
    e2 = cmath.exp(2j * delta)
    r = r01 * (1.0 - e2) / (1.0 - r01 ** 2 * e2)
    return float(abs(r) ** 2)


def test_drude_slab_reflection_vs_transfer_matrix():
    """Gate (a): Drude slab R (sim, via incident/reflected DFT subtraction)
    vs analytic 1D transfer-matrix R within 3%."""
    a_inc = _run_a(pole_on=False)
    a_tot = _run_a(pole_on=True)
    a_ref = a_tot - a_inc
    R_sim = abs(a_ref) ** 2 / max(abs(a_inc) ** 2, 1e-30)
    R_an = _drude_R_analytic(F0, (SLAB1_A - SLAB0_A) * DX_A,
                             gamma=DRUDE[1], sigma=DRUDE[2], fp=DRUDE[3])
    rel = abs(R_sim - R_an) / R_an
    print(f"3D Drude slab R: sim={R_sim:.5f} analytic={R_an:.5f} rel={rel:.4f} "
          f"(eps_inf=1, fp={DRUDE[3]}, gamma={DRUDE[1]}, sigma={DRUDE[2]}, "
          f"d={(SLAB1_A-SLAB0_A)*DX_A:.2f})")
    assert rel <= 0.03, f"Drude R sim {R_sim:.5f} vs analytic {R_an:.5f} rel {rel:.4f}"


# --- gate (b): lean-vs-reference parity on a dispersive case ---
def test_lean_parity_dispersive():
    """Gate (b): the in-place folded-D lean stepper reproduces the
    functional stepper on a Lorentz-pole slab, all six components, rel
    < 1e-10 (mirrors test_stepper3d_lean on the dispersive extension)."""
    nx, ny, nz, npml, dx = 60, 52, 44, 10, 0.05
    eps = torch.ones(nx, ny, nz, dtype=torch.float64)
    eps[24:36, 20:32, 16:28] = 4.0                    # dielectric block
    # Lorentz pole (finite fn -> bounded DC response) on the block,
    # interior in all axes (well clear of the PML shells).
    mask = torch.zeros(nx, ny, nz, dtype=torch.float64)
    mask[24:36, 20:32, 16:28] = 1.0
    pole = (0.8, 0.05, 1.5)
    pulse = GaussianPulse(1.0, 0.5)

    cref = build_3d_coeffs(nx, ny, nz, dx, eps=eps, npml=npml,
                           poles=[pole], pole_masks=[mask])
    src = PlaneSource(nx, ny, nz, cref.dt, "Ey", 0, 14, pulse)
    st, _ = run_3d(init_3d_state(nx, ny, nz, npml=npml, n_poles=1),
                   cref, 400, source=src, monitors=[])

    clean = build_lean(nx, ny, nz, dx, eps, npml, poles=[pole],
                       pole_masks=[mask], dtype=torch.float64, device="cpu")
    src2 = PlaneSource(nx, ny, nz, clean.dt, "Ey", 0, 14, pulse)
    view, _ = run_3d_lean(clean, 400, source=src2, monitors=[])

    worst, worst_comp = 0.0, ""
    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        ref, got = getattr(st, comp), getattr(view, comp)
        rel = float((got - ref).abs().max() / ref.abs().max())
        if rel > worst:
            worst, worst_comp = rel, comp
    print(f"lean-vs-ref dispersive parity: worst rel {worst:.3e} ({worst_comp})")
    assert worst < 1e-10, f"lean stepper diverges on poles: {worst_comp} rel {worst:.3e}"


# --- gate (c): autograd through a dispersive 3D run vs central FD ---
POLE_C = (0.6, 0.1, 1.0)               # Lorentz pole around the probed block


def _loss_c(eps, n_steps):
    """|DFT Ez|^2 at a plane behind the scatterer, with the Lorentz pole
    active on the block region (eps voxels there feed the folded-D pole
    update, so the gradient path runs through the pole machinery)."""
    nx, ny, nz = eps.shape
    dx = 0.05
    mask = torch.zeros(nx, ny, nz, dtype=torch.float64)
    mask[12:18, 9:15, 8:12] = 1.0
    c = build_3d_coeffs(nx, ny, nz, dx, eps=eps, npml=6,
                        poles=[POLE_C], pole_masks=[mask])
    src = PointSource3D(c.dt, "Ez", nx // 2, ny // 2, nz // 2 - 3,
                        GaussianPulse(0.6, 0.25))
    mon = DFTFieldsMonitor3D(c.dt, [0.6], "Ez", plane=(0, nx - 10))
    _, (acc,) = run_3d(init_3d_state(nx, ny, nz, npml=6, n_poles=1),
                       c, n_steps, source=src, monitors=[mon])
    return (acc[0, 0] ** 2 + acc[0, 1] ** 2).sum()


def test_dispersive_fd_gradient_gate():
    """Gate (c): BPTT gradient through the dispersive stepper matches
    central FD on eps voxels inside the pole region, rel < 1e-4."""
    torch.manual_seed(0)
    nx, ny, nz = 30, 24, 20
    n_steps = 90
    eps = torch.full((nx, ny, nz), 1.5, dtype=torch.float64)
    eps[12:18, 9:15, 8:12] = 4.0
    eps = eps.requires_grad_(True)

    L = _loss_c(eps, n_steps)
    L.backward()
    assert eps.grad is not None and torch.isfinite(eps.grad).all()

    h = 1e-4
    for vox in ((14, 11, 9), (16, 12, 10)):        # both inside the pole mask
        g = float(eps.grad[vox])
        ep = eps.detach().clone(); ep[vox] += h
        em = eps.detach().clone(); em[vox] -= h
        g_fd = (_loss_c(ep, n_steps) - _loss_c(em, n_steps)) / (2 * h)
        rel = abs(g - g_fd) / max(abs(g_fd), 1e-30)
        print(f"FD gate {vox}: autograd {g:.6e} fd {g_fd:.6e} rel {rel:.3e}")
        assert rel < 1e-4, f"voxel {vox}: autograd {g} vs FD {g_fd}, rel {rel}"