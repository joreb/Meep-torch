# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Mode-overlap decomposition of DFT line fields.

Bi-orthogonal <E1 x H2> convention, no conjugate (lossy-correct).
Forward-only: mode profiles detached.

axis_sign: TM Poynting S = -Ez*Hy x^ + Ez*Hx y^, so a +x wave stores
Hy = -neff*Ez (axis_sign=-1, the default) while a +y wave stores
Hx = +neff*Ez (pass axis_sign=+1 for y-normal monitors). With the matching
sign, a +axis forward wave maps to a_plus.
"""
from typing import Sequence

import numpy as np
import torch

from ..modes.fd1d import select_modes

_NORM_FLOOR = 1e-12


def dft_fields(acc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The stored accumulator as the phasor this module's overlaps expect.

    The monitors accumulate meep's e^{+i w t}; these overlaps and everything
    downstream of them (mode coefficients, S-parameters, the phase slope that
    gives the group index) were written against the e^{-i w t} phasor, so the
    imaginary part is negated here rather than at each call site. A conjugated
    phasor flips the SIGN of arg(S21), which is why the symptom was
    ng_est = -ng_ref -- |ng_est - ng_ref| / ng_ref = 2.007 -- and not a
    magnitude error: |S| is blind to it.
    """
    ReE, ImE, ReH, ImH = acc.unbind(1)
    return torch.complex(ReE, -ImE), torch.complex(ReH, -ImH)


def decompose(acc, eps_line, dx, freqs: Sequence[float], pol: str,
              n_modes: int, axis_sign: float = -1.0, bc: str = "pec",
              parity: int = 0):
    Ec, Hc = dft_fields(acc)
    nf = len(freqs)
    # The mode solve is a scipy eigenproblem, so solve_1d hands back CPU
    # profiles whatever device it was given, while the DFT accumulators live
    # on the engine device. Overlap them without moving one side and torch
    # raises "Expected all tensors to be on the same device".
    dev = Ec.device
    a_p = torch.zeros(nf, n_modes, dtype=torch.complex128, device=dev)
    a_m = torch.zeros(nf, n_modes, dtype=torch.complex128, device=dev)
    eps_line = eps_line.detach()
    for i, f in enumerate(freqs):
        # THE BASIS MUST MATCH THE SOURCE. When the port carries a parity the
        # source injects the parity-selected mode, so projecting onto the
        # unfiltered ordering decomposes in a different basis than the one
        # driven. For EVEN_Y on a symmetric guide the two coincide, which is
        # why that error would hide in exactly the case anyone would test.
        neffs, profs = select_modes(eps_line, dx, f, pol, n=n_modes,
                                    parity=parity, bc=bc)
        for m in range(n_modes):
            prof = profs[m].to(dev)
            neff = neffs[m]
            if pol == "tm":
                Em = prof
                Hm = axis_sign * neff * Em                  # stored Hy = -Hy_m for +x
            else:  # te (x-normal): prof is Hz_m, Em = Ey_m = neff*Hz_m/eps.
                # stored Hz is NOT sign-flipped vs the mode (unlike TM's Hy), so
                # Hm = +prof; axis_sign is unused here.
                Hm = prof
                Em = neff * prof / eps_line.to(dtype=prof.dtype, device=dev)
            N = float((Em * Hm).sum()) * dx
            if abs(N) < _NORM_FLOOR:
                raise ValueError(f"degenerate/leaky mode {m} at f={f}: norm {N:.3e}")
            ovl_EH = (Ec[i] * Hm.to(torch.complex128)).sum() * dx
            ovl_HE = (Em.to(torch.complex128) * Hc[i]).sum() * dx
            # Power-normalize: fd1d L2-normalizes profiles, so |a|^2 = P/P_m with
            # P_m = mode power proportional to |N|. Multiplying by sqrt(|N|) makes
            # |a|^2 proportional to power, so |S_ij|^2 is the power transmission
            # and |S21|=|S12| for reciprocal devices with different port modes
            # (e.g. a taper). The sqrt(|N|) factor is real, so S-matrix phase
            # (group index) is unaffected.
            pn = float(abs(N)) ** 0.5
            a_p[i, m] = (ovl_EH + ovl_HE) / (2.0 * N) * pn
            a_m[i, m] = (ovl_EH - ovl_HE) / (2.0 * N) * pn
    return a_p, a_m


def decompose_3d(mon, acc, eps_cs, dx, n_modes: int = 1,
                 target_neff: float | None = None):
    """3D analogue of :func:`decompose` for an x-normal DFT plane.

    ``mon`` is the ``FluxMonitor3D(normal="x")`` whose accumulator ``acc``
    (``(nfreq, 8, h0-l0, h1-l1)``) already holds the DFT'd tangential pairs
    (Ey, Hz) / (Ez, Hy) on the windowed plane; ``eps_cs`` is the FULL
    ``(ny, nz)`` cell-center cross-section at the monitor plane (the window
    is re-derived from ``mon`` and the sub-cross-section mode-solved with the
    full-vector ``solve_2d``; solver x = lab y, solver y = lab z, and the
    solver's Yee layout coincides with the FDTD in-plane staggering).

    Bi-orthogonal <E1 x H2> overlap, no conjugate, power-normalized per the
    2D convention: for a lossless cross-section ``0.5 * |a|**2`` is the modal
    power in the same units as ``mon.flux(acc, dx)``. Returns ``(a_p, a_m)``
    of shape ``(nfreq, n_modes)``; ``a_p`` is the +x-propagating wave.
    Forward-only (mode profiles detached, matching :func:`decompose`).
    """
    from ..modes.derivatives import ETA_0
    from ..modes.fd2d import solve_2d

    if mon.normal != "x":
        # ponytail: x-normal only -- every 3D port validation in the repo is
        # x-normal; add y/z by permuting the FluxMonitor3D._plane pair layout.
        raise NotImplementedError("decompose_3d supports x-normal monitors only")
    freqs = [float(f) for f in mon.freqs]
    acc = acc.detach().to(device="cpu", dtype=torch.float64)
    ReaE, ImaE, ReaH, ImaH, RebE, ImbE, RebH, ImbH = acc.unbind(1)
    Ea = torch.complex(ReaE, ImaE)   # Ey at (j+1/2, k)
    Ha = torch.complex(ReaH, ImaH)   # Hz at (j+1/2, k)
    Eb = torch.complex(RebE, ImbE)   # Ez at (j, k+1/2)
    Hb = torch.complex(RebH, ImbH)   # Hy at (j, k+1/2)
    eps_np = (eps_cs.detach().cpu().numpy() if isinstance(eps_cs, torch.Tensor)
              else np.asarray(eps_cs))
    ny, nz = eps_np.shape
    # Same window clipping as FluxMonitor3D._plane (m0, m1 = common min shape).
    m0, m1 = ny - 1, nz - 1
    if mon.win is not None:
        l0, h0, l1, h1 = mon.win
    else:
        l0, h0, l1, h1 = mon.ap, m0 - mon.ap, mon.ap, m1 - mon.ap
    l0, l1 = max(0, l0), max(0, l1)
    h0, h1 = min(m0, h0), min(m1, h1)
    if Ea.shape[-2:] != (h0 - l0, h1 - l1):
        raise ValueError(
            f"acc window {tuple(Ea.shape[-2:])} does not match the "
            f"({h0 - l0}, {h1 - l1}) window derived from mon and eps_cs")
    sub = np.real(eps_np[l0:h0 + 1, l1:h1 + 1]).astype(np.float64)
    nf = len(freqs)
    dev = Ea.device                       # solve_2d is scipy: CPU profiles
    a_p = torch.zeros(nf, n_modes, dtype=torch.complex128, device=dev)
    a_m = torch.zeros(nf, n_modes, dtype=torch.complex128, device=dev)
    dS = dx * dx
    for i, f in enumerate(freqs):
        modes = solve_2d(sub, dx, f, k=n_modes, target_neff=target_neff)
        for m in range(n_modes):
            sl = (slice(0, h0 - l0), slice(0, h1 - l1), m)
            Em_a = modes.Ex[sl].to(dev)   # lab Ey on the (j+1/2, k) sublattice
            Em_b = modes.Ey[sl].to(dev)   # lab Ez on the (j, k+1/2) sublattice
            Hm_b = modes.Hx[sl].to(dev) * ETA_0  # lab Hy, in FDTD units
            Hm_a = modes.Hy[sl].to(dev) * ETA_0  # lab Hz
            N = complex((Em_a * Hm_a - Em_b * Hm_b).sum()) * dS
            if abs(N) < _NORM_FLOOR:
                raise ValueError(f"degenerate/leaky mode {m} at f={f}: norm {abs(N):.3e}")
            ovl_EH = (Ea[i] * Hm_a - Eb[i] * Hm_b).sum() * dS
            ovl_HE = (Em_a * Ha[i] - Em_b * Hb[i]).sum() * dS
            pn = float(abs(N)) ** 0.5
            a_p[i, m] = (ovl_EH + ovl_HE) / (2.0 * N) * pn
            a_m[i, m] = (ovl_EH - ovl_HE) / (2.0 * N) * pn
    return a_p, a_m