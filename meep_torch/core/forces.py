# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Maxwell-stress-tensor optical force on a closed box contour (2D TM).

Time-averaged stress from DFT phasors (e^{+i w t} convention of the
DFT monitors):

  <T_ij> = Re[ E_i conj(E_j) - delta_ij |E|^2 / 2
             + H_i conj(H_j) - delta_ij |H|^2 / 2 ]

(meep's normalisation: no outer 1/2, matching src/stress.cpp:90)

F = closed-contour sum of <T> . n_hat dl, outward normals, trapezoid
corner weights (same closure as near2far). TM fields: E = Ez z_hat,
H = (Hx, Hy); staggered H averaged to the Ez contour nodes.
"""
import numpy as np


def _h_at_nodes(planes, k, ii, jj):
    """Hx, Hy phasors averaged to Ez nodes (ii, jj) (arrays)."""
    hx = planes["Hx"][k]        # (nx, ny-1), node (i, j+1/2)
    hy = planes["Hy"][k]        # (nx-1, ny), node (i+1/2, j)
    hx_n = 0.5 * (hx[ii, jj - 1] + hx[ii, jj])
    hy_n = 0.5 * (hy[ii - 1, jj] + hy[ii, jj])
    return hx_n, hy_n


def mst_force_tm(planes, grid, *, ix0, ix1, iy0, iy1):
    """(Fx, Fy) per frequency for the index box [ix0, ix1] x [iy0, iy1].

    planes: {"Ez", "Hx", "Hy"} -> complex arrays (nfreq, ...) from
    DFTFieldsMonitor accumulators.
    """
    nfreq = planes["Ez"].shape[0]
    dl = grid.dx
    fx = np.zeros(nfreq)
    fy = np.zeros(nfreq)
    for k in range(nfreq):
        ez_p = planes["Ez"][k]

        def stress(ii, jj):
            ez = ez_p[ii, jj]
            hx, hy = _h_at_nodes(planes, k, ii, jj)
            e2 = np.abs(ez) ** 2
            h2 = np.abs(hx) ** 2 + np.abs(hy) ** 2
            # No outer 1/2: meep's stress_sum (src/stress.cpp:90) sums
            # real(w * F1 * conj(F2)) with none, exactly as dft_flux::flux()
            # carries none. The inner 1/2 is the stress tensor's own
            # delta_ij |F|^2 / 2 and stays. Dropping this was forced by the
            # DFT moving to meep's unitary weight: |E|^2 picked up 1/(2 pi)
            # while flux picked up 1/pi, so force/flux slipped by exactly 2.
            txx = np.abs(hx) ** 2 - 0.5 * (e2 + h2)
            tyy = np.abs(hy) ** 2 - 0.5 * (e2 + h2)
            txy = np.real(hx * np.conj(hy))
            return txx, tyy, txy

        for idx, lo, hi, normal, axis in (
            (ix1, iy0, iy1 + 1, +1, "x"),
            (ix0, iy0, iy1 + 1, -1, "x"),
            (iy1, ix0, ix1 + 1, +1, "y"),
            (iy0, ix0, ix1 + 1, -1, "y"),
        ):
            rng = np.arange(lo, hi)
            w = np.ones(len(rng))
            w[0] = w[-1] = 0.5
            if axis == "x":
                txx, tyy, txy = stress(np.full(len(rng), idx), rng)
                fx[k] += normal * float((txx * w).sum()) * dl
                fy[k] += normal * float((txy * w).sum()) * dl
            else:
                txx, tyy, txy = stress(rng, np.full(len(rng), idx))
                fx[k] += normal * float((txy * w).sum()) * dl
                fy[k] += normal * float((tyy * w).sum()) * dl
    return fx, fy
