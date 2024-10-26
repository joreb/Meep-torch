# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W2 Wave 2: Bloch two-quadrature envelope stepper.

Launches a y-periodic planewave (envelope e = e_c + i e_s, physical
E = e * exp(i k_y y)) and checks the propagated field's k-spectrum peaks at the
set (k_x, k_y) -> propagation angle matches theta within 1 deg.
"""
import math

import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper_bloch import init_bloch_tm_state, step_bloch_tm

FCEN = 0.3
DX = 0.05
NX = 320                 # 16 um in x
NY = 195                 # Ly = 9.75 um (one y-period)
NPML = 20
LY = NY * DX
KY = 2 * math.pi / LY                                   # exact periodicity
THETA = math.asin(KY / (2 * math.pi * FCEN))            # ~19.97 deg
K0 = 2 * math.pi * FCEN
KX = math.sqrt(max(0.0, K0 * K0 - KY * KY))


def test_bloch_planewave_angle():
    g = Grid(nx=NX, ny=NY, dx=DX)
    coeffs = build_tm_coeffs(g, eps_inf=torch.ones(g.nx, g.ny, dtype=torch.float64),
                             npml=NPML, y_pmc=True)
    st = init_bloch_tm_state(g)
    ix = NPML + 5
    mask = g.zeros(g.nx, g.ny)
    mask[ix, :] = 1.0
    omega = 2 * math.pi * FCEN
    t_ramp = 5.0 / FCEN
    n_steps = int(round(40.0 / g.dt))                    # ~12 periods, past ramp
    for t in range(n_steps):
        ramp = min(1.0, (t + 0.5) * g.dt / t_ramp)
        env = math.cos(omega * (t + 0.5) * g.dt) * ramp   # +x envelope exp(-i w t)
        src_c = env
        src_s = -math.sin(omega * (t + 0.5) * g.dt) * ramp
        st = step_bloch_tm(st, coeffs, KY, src_c=src_c, src_s=src_s, src_mask=mask)

    # physical E = (Ez_c + i Ez_s) * exp(i k_y y); FFT2 over (x, y), interior away from PML
    y = (torch.arange(NY, dtype=torch.float64) + 0.5) * DX
    phase = torch.exp(1j * KY * y)
    x0, x1 = NPML + 40, NX - NPML - 10
    Ec = st.Ez_c[x0:x1, :].numpy(); Es = st.Ez_s[x0:x1, :].numpy()
    Ephys = (Ec + 1j * Es) * phase.numpy()[None, :]
    # 2D FFT (x real, y real -> use full fft for angle, zero-padded)
    nf = 4096
    spec = np.abs(np.fft.fft2(Ephys, s=(nf, nf)))
    kxs = np.fft.fftfreq(nf, DX) * 2 * math.pi
    kys = np.fft.fftfreq(nf, DX) * 2 * math.pi
    peak = np.unravel_index(spec.argmax(), spec.shape)
    kx_meas = kxs[peak[0] % nf]; ky_meas = kys[peak[1] % nf]
    angle_meas = math.degrees(math.atan2(abs(ky_meas), abs(kx_meas)))
    angle_set = math.degrees(THETA)
    assert abs(angle_meas - angle_set) < 1.0, \
        f"Bloch planewave angle {angle_meas:.2f} vs set {angle_set:.2f} (off {abs(angle_meas-angle_set):.2f} deg)"
    # k_y peak matches the set k_y within ~5% (FFT-bin resolution)
    assert abs(ky_meas) > 0.5 * KY, f"k_y peak {ky_meas:.3f} vs set {KY:.3f}"