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

# --- band -> diffraction order, and the per-order coefficients ---------------
#
# `planewave_coeffs.planewave_alpha` is this with m = 0 hard-wired: it projects
# onto the CONSTANT profile u = 1/sqrt(L) and splits forward from backward with
# the impedance relation Hy = -n Ez. Order m differs in two places and nowhere
# else: the basis becomes exp(i (k_y + m G) y)/sqrt(L), and the impedance uses
# that order's LONGITUDINAL index n_x,m = k_x,m / k0 instead of the medium
# index n. At m = 0 and normal incidence k_x,0 = n k0, so n_x,0 = n and the two
# agree -- which is the check that the generalisation is a substitution rather
# than a rewrite.


def band_to_order(bands, ky: float, G: float, k0n: float, n_orders: int = 24):
    """meep's band index -> diffraction order m.

    meep sorts a periodic monitor's modes by DECREASING longitudinal
    wavenumber k_x,m = sqrt((n k0)^2 - (k_y + m G)^2), so band 1 is the
    specular order and the rest fan outwards; evanescent orders (k_x,m
    imaginary) sort last, least-evanescent first.

    A DEGENERATE PAIR IS ONE BAND, not two. At k_y = 0 the +/-m orders have
    identical k_x, and meep reports a single band for the pair -- measured
    against the oracle on `examples/11_upstream/binary_grating`, whose angles
    run 0, 3.44, 6.89, 10.37, ... one per order, where a +/-m pairing would
    give 0, 3.44, 3.44, 6.89, 6.89. The tutorial says the same thing
    independently: it halves every non-specular transmittance
    (`0.5 * tran if nm != 0`), which only makes sense if the reported
    coefficient already carries both partners.

    Oblique incidence lifts the degeneracy, and then every m is its own band.
    So: sort by decreasing k_x and collapse exact ties, keeping +m.

    An earlier version returned the +/-m pair as two bands. The analytic test
    could not catch it -- it checked this convention against itself; only the
    oracle could.
    """
    ms = list(range(-n_orders, n_orders + 1))

    def key(m):
        kx2 = k0n * k0n - (ky + m * G) ** 2
        if kx2 > 0:
            return (0, -math.sqrt(kx2), -m)      # propagating, +m before -m
        return (1, abs(ky + m * G), -m)          # evanescent, least first

    ms.sort(key=key)
    # collapse degenerate pairs: equal k_x is ONE band
    collapsed, seen = [], []
    for m in ms:
        kx2 = k0n * k0n - (ky + m * G) ** 2
        kx = math.sqrt(kx2) if kx2 > 0 else -abs(ky + m * G)
        if any(abs(kx - s) <= 1e-9 * max(1.0, abs(kx)) for s in seen):
            continue
        seen.append(kx)
        collapsed.append(m)
    ms = collapsed
    out = []
    for b in bands:
        if not (1 <= b <= len(ms)):
            raise ValueError(f"band {b} outside the {len(ms)} orders resolved "
                             f"at n_orders={n_orders}")
        out.append(ms[b - 1])
    return out


def orders_alpha(acc, eps_line, dx: float, pol: str, ky: float, period: float,
                 freqs, bands, n_orders: int = 24):
    """Forward/backward coefficients per band on a Bloch-periodic monitor.

    Returns ``(len(bands), nfreq, 2)`` complex, in meep's normalisation
    (``|a+|^2 - |a-|^2`` is the order's contribution to the monitor flux),
    matching `planewave_alpha` exactly when ``bands == [1]`` and ``ky == 0``.
    """
    import numpy as np

    from .mode_monitor import dft_fields
    from .planewave_coeffs import uniform_index

    n = uniform_index(eps_line)
    E, H = dft_fields(acc)
    E = E.detach().cpu().numpy()
    H = H.detach().cpu().numpy()
    npts = E.shape[-1]
    L = npts * dx
    G = 2.0 * math.pi / period
    ys = (np.arange(npts) + 0.5) * dx
    freqs = np.asarray(freqs, dtype=float).reshape(-1)

    out = np.zeros((len(bands), freqs.size, 2), dtype=np.complex128)
    for fi, f in enumerate(freqs):
        k0n = n * 2.0 * math.pi * f
        ms = band_to_order(bands, ky, G, k0n, n_orders)
        for bi, m in enumerate(ms):
            kym = ky + m * G
            kx2 = k0n * k0n - kym * kym
            if kx2 <= 0:                          # evanescent: carries no flux
                continue
            nxm = math.sqrt(kx2) / (2.0 * math.pi * f)
            # A collapsed degenerate pair is ONE band carrying BOTH partners:
            # meep reports |a|^2 summed over the pair. Project onto EACH member
            # and add the powers -- exact whether or not the structure is
            # symmetric. Scaling by sqrt(2) instead would assume |a_+m| =
            # |a_-m|, which holds for a symmetric grating at normal incidence
            # and fails for a field carrying only one partner.
            members = [m]
            if m != 0:
                kx2n = k0n * k0n - (ky - m * G) ** 2
                if kx2n > 0 and abs(kx2n - kx2) <= 1e-9 * max(1.0, kx2):
                    members.append(-m)
            tot_p = tot_m = 0.0
            lead_p = lead_m = 0.0 + 0.0j
            for mm in members:
                u = np.exp(-1j * (ky + mm * G) * ys) / math.sqrt(L)
                e = float(dx) * np.sum(E[fi] * u)
                h = float(dx) * np.sum(H[fi] * u)
                if pol == "tm":                   # (Ez, Hy), Hy = -n_x Ez fwd
                    a_p, a_m = 0.5 * (e - h / nxm), 0.5 * (e + h / nxm)
                    s = math.sqrt(nxm)            # see planewave_coeffs: no 0.5
                else:                             # (Ey, Hz), Ey = +(n_x/eps) Hz
                    a_p, a_m = 0.5 * (h + e * nxm), 0.5 * (h - e * nxm)
                    s = math.sqrt(1.0 / nxm)
                a_p, a_m = s * a_p, s * a_m
                tot_p += abs(a_p) ** 2
                tot_m += abs(a_m) ** 2
                if abs(a_p) > abs(lead_p): lead_p = a_p
                if abs(a_m) > abs(lead_m): lead_m = a_m
            # magnitude from the power sum, phase from the dominant partner
            ph_p = lead_p / abs(lead_p) if abs(lead_p) else 1.0
            ph_m = lead_m / abs(lead_m) if abs(lead_m) else 1.0
            out[bi, fi, 0] = math.sqrt(tot_p) * ph_p
            out[bi, fi, 1] = math.sqrt(tot_m) * ph_m
    return out


def order_kdom(bands, ky: float, G: float, freqs, n_medium: float,
               n_orders: int = 24):
    """meep's `kdom`: the dominant wavevector of each (band, freq), row-major.

    meep returns `len(bands) * nfreq` vectors in units where `|kdom| = n f`,
    so a caller recovers the diffraction angle as
    `acos(kdom.x / f)` (`examples/11_upstream/binary_grating`). That fixes the
    scaling: `kdom = (k_x,m, k_y,m) / 2 pi`.

    Evanescent orders have no real k_x; meep reports the propagating part, so
    `k_x` is reported as 0 there and the angle degenerates to 90 degrees
    rather than raising inside `acos`.
    """
    import numpy as np

    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    out = []
    for b in bands:
        for f in freqs:
            k0n = n_medium * 2.0 * math.pi * f
            (m,) = band_to_order([b], ky, G, k0n, n_orders)
            kym = ky + m * G
            kx2 = k0n * k0n - kym * kym
            kx = math.sqrt(kx2) if kx2 > 0 else 0.0
            out.append((kx / (2.0 * math.pi), kym / (2.0 * math.pi), 0.0))
    return out
