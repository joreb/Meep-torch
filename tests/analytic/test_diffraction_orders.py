# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Diffraction-order coefficients against a field we built ourselves.

`ports/orders.py` had the projection since it was written; nothing could
reach it, because `get_eigenmode_coefficients` raised for any band but 1.
This gates the wiring without an oracle: synthesise a monitor line that IS a
known superposition of orders, project it, and require the amplitudes back.

The forward/backward split is the part worth pinning. Order m's impedance
uses its LONGITUDINAL index n_x,m = k_x,m/k0, not the medium index n; get
that wrong and a pure forward wave leaks into the backward coefficient,
which is silent in |a|^2 and fatal in a grating's diffraction efficiency.
"""
import math

import numpy as np
import pytest
import torch

from meep_torch.ports.orders import band_to_order, orders_alpha


def _acc_from(E, H):
    """Pack complex (E, H) lines into the monitor accumulator layout.

    `mode_monitor.dft_fields` returns complex(Re, -Im) -- the monitors store
    meep's e^{+iwt} and the overlaps want its conjugate -- so the imaginary
    slots are negated here to make the round trip an identity.
    """
    acc = torch.zeros(1, 4, E.shape[-1], dtype=torch.float64)
    acc[0, 0] = torch.tensor(E.real); acc[0, 1] = torch.tensor(-E.imag)
    acc[0, 2] = torch.tensor(H.real); acc[0, 3] = torch.tensor(-H.imag)
    return acc


def test_band_to_order_sorts_by_longitudinal_wavenumber():
    G = 2 * math.pi / 1.0
    k0n = 2 * math.pi * 2.0
    # normal incidence: a degenerate +/-m pair is ONE band, so the orders run
    # 0, 1, 2, ... -- measured against meep on binary_grating, whose angles are
    # 0, 3.44, 6.89, 10.37 and not 0, 3.44, 3.44, 6.89.
    assert band_to_order([1, 2, 3, 4, 5], 0.0, G, k0n) == [0, 1, 2, 3, 4]
    # oblique lifts the degeneracy; the order nearest k_y = 0 sorts ahead
    assert band_to_order([1, 2, 3], 2.0, G, k0n) == [0, -1, 1]


@pytest.mark.parametrize("ky", [0.0, 1.3])
def test_orders_recover_a_known_superposition(ky):
    period, dx, f, n = 1.0, 1.0 / 64, 2.0, 1.0
    npts = int(round(period / dx))
    L, G, k0n = npts * dx, 2 * math.pi / period, n * 2 * math.pi * f
    ys = (np.arange(npts) + 0.5) * dx
    # ask which orders the bands actually ARE at this k_y rather than
    # hardcoding them: order 2 propagates at k_y = 0 and evanesces at 1.3, and
    # a synthetic field cannot be built for an evanescent order.
    bands = [1, 2, 3]
    amps = [1.0 + 0.0j, 0.35 - 0.20j, -0.15 + 0.40j]
    want = dict(zip(band_to_order(bands, ky, G, k0n), amps))

    E = np.zeros(npts, dtype=complex)
    H = np.zeros(npts, dtype=complex)
    for m, A in want.items():
        kym = ky + m * G
        nxm = math.sqrt(k0n ** 2 - kym ** 2) / (2 * math.pi * f)
        u = np.exp(1j * kym * ys) / math.sqrt(L)
        E += A * u
        H += -nxm * A * u                      # TM forward: Hy = -n_x Ez

    eps_line = np.full(npts, n * n)
    alpha = orders_alpha(_acc_from(E, H), eps_line, dx, "tm", ky, period,
                         [f], bands)
    got_m = band_to_order(bands, ky, G, k0n)
    for bi, m in enumerate(got_m):
        nxm = math.sqrt(k0n ** 2 - (ky + m * G) ** 2) / (2 * math.pi * f)
        # sqrt(n_x,m), no 0.5: meep's identity is |a+|^2 - |a-|^2 == the
        # monitor flux, and the flux carries no 1/2 either.
        expect = math.sqrt(nxm) * want[m]
        assert alpha[bi, 0, 0] == pytest.approx(expect, rel=1e-9, abs=1e-12)
        # a pure forward superposition must leave the backward slot empty
        assert abs(alpha[bi, 0, 1]) < 1e-9


def test_evanescent_order_carries_nothing():
    period, dx, f, n = 1.0, 1.0 / 64, 0.4, 1.0   # k0 n < G, so m = +/-1 evanesce
    npts = int(round(period / dx))
    E = np.ones(npts, dtype=complex)
    H = -np.ones(npts, dtype=complex)
    alpha = orders_alpha(_acc_from(E, H), np.full(npts, n * n), dx, "tm",
                         0.0, period, [f], [1, 2, 3])
    assert abs(alpha[0, 0, 0]) > 0.0             # specular propagates
    assert abs(alpha[1, 0, 0]) == 0.0            # +1 evanescent
    assert abs(alpha[2, 0, 0]) == 0.0            # -1 evanescent
