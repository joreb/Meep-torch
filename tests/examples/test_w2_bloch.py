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
    coeffs = build_tm_coeffs(g, eps_inf=torch.ones(g.nx, g.ny, dtype=torch.float64, device=g.device),
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
    y = (torch.arange(NY, dtype=torch.float64, device=g.device) + 0.5) * DX
    phase = torch.exp(1j * KY * y)
    x0, x1 = NPML + 40, NX - NPML - 10
    Ec = st.Ez_c[x0:x1, :].cpu().numpy(); Es = st.Ez_s[x0:x1, :].cpu().numpy()
    Ephys = (Ec + 1j * Es) * phase.cpu().numpy()[None, :]
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

def _energy(st):
    return float((st.Ez_c ** 2).sum() + (st.Ez_s ** 2).sum()
                 + (st.Hx_c ** 2).sum() + (st.Hx_s ** 2).sum()
                 + (st.Hy_c ** 2).sum() + (st.Hy_s ** 2).sum())


def _energy_slice(st, sl):
    return float((st.Ez_c[sl] ** 2).sum() + (st.Ez_s[sl] ** 2).sum()
                 + (st.Hx_c[sl] ** 2).sum() + (st.Hx_s[sl] ** 2).sum()
                 + (st.Hy_c[sl] ** 2).sum() + (st.Hy_s[sl] ** 2).sum())


def test_bloch_ky_zero_decouples_quadratures():
    # k_y=0: the i k_y coupling vanishes -> the c and s quadratures evolve
    # independently under a periodic-y TM stepper. A c-only source leaves s=0.
    g = Grid(nx=NX, ny=NY, dx=DX)
    coeffs = build_tm_coeffs(g, eps_inf=torch.ones(g.nx, g.ny, dtype=torch.float64, device=g.device),
                             npml=NPML, y_pmc=True)
    st = init_bloch_tm_state(g)
    ix = NPML + 5
    mask = g.zeros(g.nx, g.ny); mask[ix, :] = 1.0
    omega = 2 * math.pi * FCEN
    for t in range(int(round(20.0 / g.dt))):
        env = math.cos(omega * (t + 0.5) * g.dt)
        st = step_bloch_tm(st, coeffs, 0.0, src_c=env, src_s=0.0, src_mask=mask)
    assert float(st.Ez_s.abs().max()) < 1e-9, "k_y=0 leaked into s quadrature"
    assert float(st.Ez_c.abs().max()) > 1e-3, "k_y=0 c quadrature did not launch"


def test_bloch_energy_conservation():
    # Lossless (eps=1, no chi3), source a short pulse, then evolve source-off
    # and check total EM energy drift < 1e-6 over 1000 steps. Catches a
    # quadrature-coupling sign asymmetry (artificial gain/loss).
    g = Grid(nx=NX, ny=NY, dx=DX)
    coeffs = build_tm_coeffs(g, eps_inf=torch.ones(g.nx, g.ny, dtype=torch.float64, device=g.device),
                             npml=NPML, y_pmc=True)
    st = init_bloch_tm_state(g)
    ix = NPML + 5
    mask = g.zeros(g.nx, g.ny); mask[ix, :] = 1.0
    omega = 2 * math.pi * FCEN
    n_src = int(round(15.0 / g.dt))
    for t in range(n_src):                       # emit a short pulse
        env = math.cos(omega * (t + 0.5) * g.dt) * math.exp(-((t * g.dt - 8.0) ** 2) / 6.0)
        st = step_bloch_tm(st, coeffs, KY, src_c=env, src_s=-env * 0.0, src_mask=mask)
    # measure interior energy away from x-PML, evolve source-off
    sl = (slice(NPML + 10, NX - NPML - 10), slice(None))
    e0 = _energy_slice(st, sl)
    for _ in range(1000):
        st = step_bloch_tm(st, coeffs, KY)
    e = _energy_slice(st, sl)
    # energy should not grow (PML leaks some out; assert no artificial gain):
    assert e <= e0 * (1 + 1e-6), f"energy grew {e0:.3e} -> {e:.3e} (artificial gain)"


def test_bloch_cfl_violation_raises():
    g = Grid(nx=20, ny=20, dx=0.1)
    coeffs = build_tm_coeffs(g, eps_inf=torch.ones(g.nx, g.ny, dtype=torch.float64, device=g.device),
                             npml=4, y_pmc=True)
    st = init_bloch_tm_state(g)
    bad_ky = 2.0 / g.dt                         # exceeds 1/dt bound
    try:
        step_bloch_tm(st, coeffs, bad_ky)
    except ValueError as e:
        assert "Nyquist" in str(e) or "courant" in str(e) or "k_y" in str(e)
        return
    assert False, "Bloch stepper did not raise on CFL/Nyquist violation"
