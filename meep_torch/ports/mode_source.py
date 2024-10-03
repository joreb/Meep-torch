# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""TFSF one-way eigenmode injection (fdtdx MIT blueprint, reimplemented).

Forward-only: mode profiles are detached (fd1d solves in numpy).
"""
import torch

from ..core.grid import Grid
from ..modes.fd1d import solve_1d
from .port import Port

_H_SIGN = -1.0  # relative sign of the magnetic sheet; fixed by the one-way test


class ModeSourceTM:
    def __init__(self, grid, coeffs, port, profile, fcen, mode=0):
        self.profile, self.dt, self.d = profile, grid.dt, port.direction
        dx = grid.dx
        if port.axis == "x":
            eps_line = coeffs.eps_inf[port.index, port.lo:port.hi].detach()
        else:
            eps_line = coeffs.eps_inf[port.lo:port.hi, port.index].detach()
        if coeffs.c3.numel() and self._line_dispersive(coeffs.c3, port):
            raise ValueError("dispersive media on port line unsupported")
        modes = solve_1d(eps_line, dx, fcen, "tm", k=mode + 1)
        neff = float(modes.neff[mode])
        clad = float(torch.sqrt(torch.max(eps_line[0], eps_line[-1])))
        if neff <= clad + 1e-9:
            raise ValueError(f"no guided mode at port (neff={neff:.4f} <= cladding {clad:.4f})")
        prof = modes.profiles[mode].to(grid.dtype)
        self.neff = neff
        # M sheet sits one cell upstream of the J sheet in BOTH directions
        # (the col/row choice below already places it on the launch side), so the
        # half-cell propagation delay to the J sheet is direction-independent.
        self.delay = neff * dx / 2.0

        mask = grid.zeros(grid.nx, grid.ny)
        if port.axis == "x":
            mask[port.index, port.lo:port.hi] = neff * prof / dx      # Jz ∝ Hy_m
            hy = grid.zeros(grid.nx - 1, grid.ny)
            col = port.index - 1 if self.d > 0 else port.index
            hy[col, port.lo:port.hi] = self.d * prof / dx             # My ∝ Ez_m
            self.h_masks = {"src_hy_mask": hy}
        else:
            # y-normal TM: transverse pair is Ez/Hx with S_y = +Ez*Hx; the Jz
            # sheet carries -neff*prof so its sign matches decompose's
            # axis_sign=-1 (stored Hx sign) in tasks 5-6.
            mask[port.lo:port.hi, port.index] = -neff * prof / dx     # Jz ∝ -Hx_m
            hx = grid.zeros(grid.nx, grid.ny - 1)
            row = port.index - 1 if self.d > 0 else port.index
            hx[port.lo:port.hi, row] = self.d * prof / dx
            self.h_masks = {"src_hx_mask": hx}
        self.mask = mask

    @staticmethod
    def _line_dispersive(c3, port) -> bool:
        if port.axis == "x":
            return bool((c3[:, port.index, port.lo:port.hi] != 0).any())
        return bool((c3[:, port.lo:port.hi, port.index] != 0).any())

    def amplitude(self, t_idx: int) -> float:
        return self.profile.amplitude((t_idx + 0.5) * self.dt)

    def h_amplitude(self, t_idx: int) -> float:
        return _H_SIGN * self.profile.amplitude(t_idx * self.dt - self.delay)