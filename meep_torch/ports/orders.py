# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Diffraction-order decomposition for Bloch-periodic 2D TM (planewave basis).

On a monitor line in the UNIFORM superstrate/substrate the exact mode basis is
plane waves exp(i (k_y + m G) y), G = 2*pi/period. The DFT envelope
e = E_c + i E_s (periodic over the cell) is projected onto the m-th harmonic;
order power = |a_m|^2 * Re(k_x,m)/k_x,inc with k_x,m = sqrt((n*k0)^2 - (k_y+mG)^2)
(evanescent orders drop, Re=0). meep get_eigenmode_coefficients-for-gratings in
the planewave limit. Forward (+x) amplitudes from E; backward from the E,H pair
or a two-run reference subtraction.
"""
import math

import numpy as np
import torch


class BlochDFTLine:
    """Accumulate the complex envelope DFT of Ez at a single frequency on an
    x-normal line ix (forward +x monitor). e(t) = Ez_c + i Ez_s; the DFT at the
    source angular frequency w0 is acc = sum_t e(t) exp(i w0 t) (proportional to
    the steady envelope E(y); the run-length scale cancels in ratios)."""

    def __init__(self, grid, ix: int, w0: float):
        self.ix, self.w0, self.dt = ix, w0, grid.dt
        self.acc = np.zeros(grid.ny, dtype=np.complex128)

    def sample(self, state, t_idx: int):
        e = (state.Ez_c[self.ix, :] + 1j * state.Ez_s[self.ix, :]).cpu().numpy()
        self.acc += e * np.exp(1j * self.w0 * (t_idx + 0.5) * self.dt)


def diffraction_orders(acc, ky: float, G: float, k0: float, n_medium: float,
                       dy: float, n_modes: int = 12):
    """Project the line DFT envelope onto diffraction orders; return
    (orders, angles_deg, powers) where powers are |a_m|^2 * Re(k_x,m)/k_x0
    (k_x0 = sqrt((n*k0)^2 - ky^2), the m=0 longitudinal wavenumber)."""
    ny = acc.shape[0]
    ys = (np.arange(ny) + 0.5) * dy
    kx0 = math.sqrt(max(0.0, (n_medium * k0) ** 2 - ky * ky))
    orders, angles, powers = [], [], []
    for m in range(-n_modes, n_modes + 1):
        kym = ky + m * G
        a = float(np.abs(np.sum(acc * np.exp(-1j * m * G * ys)) / ny))
        kxm2 = (n_medium * k0) ** 2 - kym * kym
        orders.append(m)
        if kxm2 <= 0 or kx0 <= 0:
            angles.append(float("nan")); powers.append(0.0)
        else:
            angles.append(math.degrees(math.asin(max(-1.0, min(1.0, kym / (n_medium * k0))))))
            powers.append(a * a * math.sqrt(kxm2) / kx0)
    return orders, np.array(angles), np.array(powers)