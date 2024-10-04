# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Mode-overlap decomposition of DFT line fields.

Bi-orthogonal <E1 x H2> convention, no conjugate (lossy-correct).
Forward-only: mode profiles detached.

axis_sign defaults to -1.0: step_tm stores Hy/Hx with the opposite sign to
the fd1d eigenmode (Hy_m = +neff*Ez_m), so the monitor's accumulated H for a
+axis wave equals -neff*Ez; axis_sign=-1 (Hm = -neff*Em) aligns the overlap
so a +axis forward wave maps to a_plus. Same sign for y-normal (S_y = +Ez*Hx).
"""
from typing import Sequence

import torch

from ..modes.fd1d import solve_1d

_NORM_FLOOR = 1e-12


def dft_fields(acc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ReE, ImE, ReH, ImH = acc.unbind(1)
    return torch.complex(ReE, ImE), torch.complex(ReH, ImH)


def decompose(acc, eps_line, dx, freqs: Sequence[float], pol: str,
              n_modes: int, axis_sign: float = -1.0):
    Ec, Hc = dft_fields(acc)
    nf = len(freqs)
    a_p = torch.zeros(nf, n_modes, dtype=torch.complex128)
    a_m = torch.zeros(nf, n_modes, dtype=torch.complex128)
    for i, f in enumerate(freqs):
        modes = solve_1d(eps_line.detach(), dx, f, pol, k=n_modes)
        for m in range(n_modes):
            Em = modes.profiles[m]
            Hm = axis_sign * float(modes.neff[m]) * Em
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