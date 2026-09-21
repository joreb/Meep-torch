# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Mode coefficients on a flux monitor (meep get_eigenmode_coefficients).

Two monitor flavours reach ``Simulation.get_eigenmode_coefficients``:

* a *guided* monitor (non-uniform transverse eps) -- the S-parameter path
  (``EigenModeSource`` + mode ports -> ``compute_smatrix``);
* a *planewave* monitor on a Bloch-periodic plane whose transverse eps is
  uniform (the superstrate/substrate line of a grating or metasurface). The
  exact mode basis there is the diffraction orders exp(i (k_y + m G) y), and
  at NORMAL incidence (k_y = 0) meep's band 1 is the 0th (specular) order --
  verified against meep 1.34 for both ODD_Z+EVEN_Y and EVEN_Z+ODD_Y
  (tests/mp_parity/test_eig_coeffs_planewave.py).

This module implements the second. meep's normalization is
``|alpha_+|**2 - |alpha_-|**2 == the monitor flux``, so the projected
amplitudes are scaled by sqrt(n_x) (TM) / sqrt(1 / n_x) (TE) to land in the
same units as ``FluxMonitor*.flux``. The scale was sqrt(0.5 n_x) while the
flux carried its own 1/2; both lost it together when the DFT moved to meep's
unitary convention, and the identity is pinned by a test rather than a
comment now.

Sign conventions follow ``ports/mode_monitor.decompose``: an x-normal TM
monitor stores (Ez, Hy) with Hy = -n_x Ez for a +x wave; an x-normal TE
monitor stores (Ey, Hz) with Ey = +(n_x/eps) Hz for a +x wave.
"""
import math

import numpy as np

from .mode_monitor import dft_fields

_UNIFORM_RTOL = 1e-9


def uniform_index(eps_line) -> float:
    """Refractive index of a transverse eps line that must be uniform.

    Raises NotImplementedError when it is not: a structured line is a
    guided-mode problem, which has no flux-monitor decomposition path.
    """
    e = np.asarray(eps_line, dtype=float).reshape(-1)
    if e.size == 0:
        raise ValueError("empty transverse eps line")
    if float(e.max() - e.min()) > _UNIFORM_RTOL * float(e.max()):
        raise NotImplementedError(
            "get_eigenmode_coefficients on a flux monitor whose transverse "
            f"eps is not uniform (eps in [{e.min():.4f}, {e.max():.4f}]): only "
            "the planewave (diffraction-order) basis of a uniform "
            "superstrate/substrate line is implemented; for a guided mode "
            "drive the S-parameter path (EigenModeSource + mode ports)")
    return float(e.mean()) ** 0.5


def planewave_alpha(acc, eps_line, dx: float, pol: str):
    """0th-order (specular) forward/backward coefficients of an x-normal
    monitor at normal incidence.

    ``acc`` is the monitor's DFT accumulator ``(nfreq, 4, npts)``. Returns
    ``(a_p, a_m)``, complex ``(nfreq,)`` arrays in meep's normalization.
    """
    n = uniform_index(eps_line)
    E, H = dft_fields(acc)
    npts = E.shape[-1]
    u = 1.0 / math.sqrt(npts * dx)          # constant profile, sum(u^2 dx) = 1
    e = E.sum(-1).detach().cpu().numpy() * dx * u
    h = H.sum(-1).detach().cpu().numpy() * dx * u
    # No 0.5 under the root: meep's identity is |a+|^2 - |a-|^2 == the monitor
    # flux, and the flux lost its own 1/2 when the DFT moved to meep's unitary
    # convention (dft_flux::flux(), src/dft.cpp:551, carries none). With
    # sqrt(0.5 n) the identity measured 0.5000 instead of 1.
    if pol == "tm":                          # (Ez, Hy), Hy = -n Ez forward
        a_p, a_m = 0.5 * (e - h / n), 0.5 * (e + h / n)
        s = math.sqrt(n)
    else:                                    # (Ey, Hz), Ey = +(n/eps) Hz forward
        a_p, a_m = 0.5 * (h + e * n), 0.5 * (h - e * n)
        s = math.sqrt(1.0 / n)
    return s * a_p, s * a_m
