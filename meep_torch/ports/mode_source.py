# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""TFSF one-way eigenmode injection (fdtdx MIT blueprint, reimplemented).

Forward-only: mode profiles are detached (fd1d solves in numpy).
"""
import torch

from ..core.grid import Grid
from ..modes.fd1d import solve_1d
from .port import Port, SRC_OFFSET

_H_SIGN = -1.0  # relative sign of the magnetic sheet; fixed by the one-way test
_H_SIGN_TE = +1.0  # TE magnetic-sheet sign; fixed by the TE one-way test (opposite stagger)
# TE source sheets sit farther upstream than TM: the weakly-guided TE mode
# (neff~1.3 near cutoff) has a long evanescent tail, so the monitor needs more
# separation from the source to read the radiated mode, not the near field.
_TE_SRC_OFFSET = 12


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

        # Source sheets sit SRC_OFFSET cells UPSTREAM of the monitor plane
        # (port.index) so the monitor reads the radiated mode, not the injected
        # current. M sheet is 0.5 cell upstream of J (launch side) in both
        # directions; shifting both by -d*SRC_OFFSET keeps that relative placement.
        j_idx = port.index - self.d * SRC_OFFSET
        mask = grid.zeros(grid.nx, grid.ny)
        if port.axis == "x":
            mask[j_idx, port.lo:port.hi] = neff * prof / dx          # Jz ∝ Hy_m
            hy = grid.zeros(grid.nx - 1, grid.ny)
            col = j_idx - 1 if self.d > 0 else j_idx
            hy[col, port.lo:port.hi] = self.d * prof / dx            # My ∝ Ez_m
            self.h_masks = {"src_hy_mask": hy}
        else:
            # y-normal TM: transverse pair is Ez/Hx with S_y = +Ez*Hx; the Jz
            # sheet carries -neff*prof so its sign matches decompose's
            # axis_sign=-1 (stored Hx sign) in tasks 5-6.
            mask[port.lo:port.hi, j_idx] = -neff * prof / dx         # Jz ∝ -Hx_m
            hx = grid.zeros(grid.nx, grid.ny - 1)
            row = j_idx - 1 if self.d > 0 else j_idx
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


class ModeSourceTE:
    """TE TFSF one-way eigenmode source (x-normal only this phase).

    Transverse pair Ey/Hz. mask is a Dy current (binds to step_te src_mask_ey);
    h_masks injects a Hz magnetic current. Mode profiles detached.
    """

    def __init__(self, grid, coeffs, port, profile, fcen, mode=0):
        if port.axis != "x":
            raise NotImplementedError("TE y-normal not supported (FluxMonitorTE is x-normal only)")
        self.profile, self.dt, self.d = profile, grid.dt, port.direction
        dx = grid.dx
        # eps at Ey nodes (nx, ny-1): coeffs.eps_y is already the half-y average.
        eps_line = coeffs.eps_y[port.index, port.lo:port.hi].detach()
        if coeffs.c3y.numel() and bool((coeffs.c3y[:, port.index, port.lo:port.hi] != 0).any()):
            raise ValueError("dispersive media on port line unsupported")
        modes = solve_1d(eps_line, dx, fcen, "te", k=mode + 1)
        neff = float(modes.neff[mode])
        clad = float(torch.sqrt(torch.max(eps_line[0], eps_line[-1])))
        if neff <= clad + 1e-9:
            raise ValueError(f"no guided mode at port (neff={neff:.4f} <= cladding {clad:.4f})")
        Hz_m = modes.profiles[mode].to(grid.dtype)
        Ey_m = (neff * Hz_m / eps_line).to(grid.dtype)
        self.neff = neff
        self.delay = neff * dx / 2.0  # direction-independent (M upstream of J both ways)

        j_idx = port.index - self.d * _TE_SRC_OFFSET
        # Dy (Ey) electric current: mask on (nx, ny-1) Ey nodes.
        mask = grid.zeros(grid.nx, grid.ny - 1)
        mask[j_idx, port.lo:port.hi] = Hz_m / dx                    # Jy ∝ Hz_m
        # Hz magnetic current on (nx-1, ny-1) Hz nodes, 0.5 cell upstream of J.
        hz = grid.zeros(grid.nx - 1, grid.ny - 1)
        col = j_idx - 1 if self.d > 0 else j_idx
        hz[col, port.lo:port.hi] = self.d * Ey_m / dx               # Mz ∝ Ey_m
        self.mask = mask
        self.h_masks = {"src_hz_mask": hz}

    def amplitude(self, t_idx: int) -> float:
        # step_te advances E to (t_idx+0.5)dt from Hz(t_idx); the E half-step is
        # driven by the source at t_idx*dt.
        return self.profile.amplitude(t_idx * self.dt)

    def h_amplitude(self, t_idx: int) -> float:
        # Hz advances to (t_idx+1)dt from E(t_idx+0.5); the magnetic sheet fires
        # a half-step later than the electric sheet (reversed stagger vs TM).
        return _H_SIGN_TE * self.profile.amplitude((t_idx + 0.5) * self.dt - self.delay)