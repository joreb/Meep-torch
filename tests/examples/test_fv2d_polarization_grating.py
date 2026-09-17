# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""The full-vector 2D stepper, gated on the problem it was written for.

`stepper_fv2d` carries all six components and a 3x3 symmetric eps because a
z-coupled permittivity breaks the TM/TE split -- it exists for the
polarization grating of Oh & Escuti (Opt. Lett. 33, 2287, 2008), and had no
test at all. It is also not reachable from the `mp` shim, whose scalar
eps_tm/eps_te grids cannot hold a tensor, so nothing in the suite touched it.

That matters beyond this file. Twelve steppers are being unified onto one
derived kernel, and a stepper with no gate cannot tell anyone that the
unification broke it: a change made for `stepper_tm`, which has 44 tests
watching, would land here with none.

The gate is a physical claim rather than a pinned float, and it separates two
effects that a single number would conflate:

  - BIREFRINGENCE converts handedness. A half-wave slab does it whether or not
    the director twists.
  - The TWIST deflects. The rotating director adds a geometric phase linear in
    y, which is a grating: the converted light leaves in order m = -1.

So a uniform half-wave plate must convert WITHOUT deflecting, and an isotropic
slab must do neither. Those two cases are what make the first one evidence.
"""
import math

import numpy as np
import pytest
import torch

from meep_torch.core.stepper_fv2d import (build_fv2d_coeffs, init_fv2d_state,
                                          step_fv2d)

RES, FCEN, NO, NE, PERIOD = 12, 1.0, 1.5, 1.7, 6.0


def _grating(twist=True, thickness_factor=1.0, isotropic=False, t_end=40.0):
    """Returns (converted fraction, peak converted order, per-order fractions).

    A (cos, sin) drive injects one circular state; the transmitted Ey/Ez line
    is split back into circular components and FFT'd along the periodic y.
    """
    d = thickness_factor * 0.5 / (NE - NO)          # half-wave at factor 1
    dx = 1.0 / RES
    nx, ny = int(10.0 * RES), int(PERIOD * RES)
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * dx
    phi = (math.pi * ys / PERIOD if twist
           else torch.zeros(ny, dtype=torch.float64))
    one = torch.ones(nx, ny, dtype=torch.float64)
    exx, eyy, ezz = one.clone(), one.clone(), one.clone()
    eyz = torch.zeros(nx, ny, dtype=torch.float64)
    i0, i1 = int(4.0 * RES), int((4.0 + d) * RES)
    delta = 0.0 if isotropic else NE ** 2 - NO ** 2
    eyy[i0:i1, :] = NO ** 2 + delta * torch.cos(phi)[None, :] ** 2
    ezz[i0:i1, :] = NO ** 2 + delta * torch.sin(phi)[None, :] ** 2
    eyz[i0:i1, :] = delta * (torch.cos(phi) * torch.sin(phi))[None, :]
    exx[i0:i1, :] = NO ** 2
    eps6 = (exx, torch.zeros_like(one), torch.zeros_like(one), eyy, eyz, ezz)

    c = build_fv2d_coeffs(eps6, dx, npml=RES)
    st = init_fv2d_state(nx, ny)
    mask = torch.zeros(nx, ny, dtype=torch.float64)
    mask[int(2.0 * RES), :] = 1.0
    omega = 2 * math.pi * FCEN
    n_steps = int(round(t_end / c.dt))
    accY = np.zeros(ny, dtype=complex)
    accZ = np.zeros(ny, dtype=complex)
    ix_out = nx - int(2.0 * RES)
    for t in range(n_steps):
        tt = (t + 0.5) * c.dt
        ramp = min(1.0, tt / 5.0)
        st = step_fv2d(st, c, math.cos(omega * tt) * ramp, mask, "Ey",
                       src2=("Ez", mask, math.sin(omega * tt) * ramp))
        if t >= n_steps // 2:                        # steady state only
            ph = np.exp(-2j * math.pi * FCEN * (t + 1.0) * c.dt) * c.dt
            accY += st.Ey[ix_out, :].cpu().numpy() * ph
            accZ += st.Ez[ix_out, :].cpu().numpy() * ph

    # e^{+iwt} phasors: the (cos, sin) drive is (1, -i) = aR, so aL is converted
    aR = (accY + 1j * accZ) / math.sqrt(2)
    aL = (accY - 1j * accZ) / math.sqrt(2)
    pR, pL = np.abs(np.fft.fft(aR)) ** 2, np.abs(np.fft.fft(aL)) ** 2
    orders = range(-2, 3)
    total = sum(pR[m % ny] + pL[m % ny] for m in orders)
    frac = {m: pL[m % ny] / total for m in orders}
    return sum(frac.values()), max(frac, key=frac.get), frac


def test_twisted_half_wave_grating_deflects_into_minus_one():
    """Full conversion into a single first order -- the geometric phase."""
    conv, peak, frac = _grating()
    assert peak == -1, f"converted power peaks at m={peak}, not -1: {frac}"
    assert conv > 0.9, f"conversion {conv:.3f} too low: {frac}"
    assert frac[0] < 0.02, f"m=0 leakage {frac[0]:.3f}: {frac}"


def test_a_uniform_half_wave_plate_converts_but_does_not_deflect():
    """Separates the two effects: birefringence converts, the twist deflects.

    Without this the first test could pass on a stepper that deflected for any
    reason at all, and "conversion" would be unattributed.
    """
    conv, peak, frac = _grating(twist=False)
    assert conv > 0.9, f"a half-wave plate must still convert: {conv:.3f}"
    assert peak == 0, f"an untwisted plate deflected to m={peak}: {frac}"


def test_an_isotropic_slab_neither_converts_nor_deflects():
    """The true null. A gate whose 'signal' cannot go to zero is not a gate."""
    conv, _, frac = _grating(isotropic=True)
    assert conv < 0.02, f"isotropic slab converted {conv:.3f}: {frac}"


def test_conversion_follows_the_half_wave_condition():
    """Half the thickness is a quarter-wave: conversion drops, order does not.

    This is the claim that the number tracks the physics rather than sitting
    at 1 for any birefringent slab.
    """
    full, peak_full, _ = _grating()
    half, peak_half, _ = _grating(thickness_factor=0.5)
    assert peak_half == peak_full == -1
    assert 0.4 < half < 0.8, f"quarter-wave conversion {half:.3f} off-physics"
    assert half < full - 0.2, f"thickness did not matter: {half:.3f} vs {full:.3f}"
