# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math
from typing import Sequence

import torch

from .grid import Grid


class PointProbe:
    reduce = "stack"

    def __init__(self, ix: int, iy: int):
        self.ix, self.iy = ix, iy

    def sample(self, state, t_idx: int) -> torch.Tensor:
        # clone: a bare view pins that step's ENTIRE Ez tensor alive in the
        # stack-reduce buffer (functional stepper -> new Ez per step ->
        # n_steps full grids retained; 108 GB observed). Autograd-safe.
        return state.Ez[self.ix, self.iy].clone()


DFT_SCALE = 1.0 / math.sqrt(2.0 * math.pi)
"""meep's DFT weight per step, `src/dft.cpp:225`::

    data.dt_factor = dt / sqrt(2.0 * pi) * decimation_factor;

meep is self-consistently UNITARY; the plain convention we used is
self-consistently PLAIN, and neither is mathematically wrong. Parity is the
mandate, so meep's wins. Consequences, all measured before the change:
every flux we reported was exactly `pi` times meep's (3.141597 / 3.141587 /
3.141612 at fcen 0.5 / 1.0 / 2.0), because flux squares the DFT for `2 pi`
and meep's `dft_flux::flux()` (`dft.cpp:551`) carries no 1/2 where we carried
one. Field DFTs were `sqrt(2 pi)` large, so `Im(conj(E) D)` was `2 pi` large.
"""


def dft_time_offset(component: str) -> float:
    """Where in the step this component's DFT sample sits, in units of dt.

    Field TYPE, not first letter. The electric family is E AND D, the magnetic
    family H AND B; meep stamps each family with one time. Four call sites
    wrote `1.0 if component.startswith("E") else 0.5`, which silently gives D
    the MAGNETIC half-step. Measured in vacuum, where D/E must be exactly
    1 + 0j, that returned 0.99692 + 0.07846j -- a phase of omega*dt/2 to six
    digits. Nothing raises on it, and it is ruinous for absorbed power
    Im(conj(E) D): SiO2's Im(eps)/|eps| is 4.6e-4, so the phase error is ~68x
    the signal it competes with.
    """
    return 0.5 if component[0] in "HB" else 1.0


class DFTFieldsMonitor:
    """Running DFT of one full-grid field component at a set of frequencies.

    Per step accumulates dt * [cos(w t) F, -sin(w t) F] (the Re/Im of
    F * exp(-i w t), matching the FluxMonitor convention); the accumulated
    result has shape (nfreq, 2, nx, ny). Timing follows the TM stepper:
    E components sit at (t_idx+1)*dt, H components at (t_idx+1/2)*dt.
    """

    reduce = "sum"

    def __init__(self, grid: Grid, freqs: Sequence[float], component: str = "Ez"):
        self.component = component
        self.dt = grid.dt
        self.freqs = torch.tensor(freqs, dtype=grid.dtype, device=grid.device)

    def sample(self, state, t_idx: int) -> torch.Tensor:
        F = getattr(state, self.component)
        off = dft_time_offset(self.component)
        t = (t_idx + off) * self.dt
        w = 2 * math.pi * self.freqs[:, None, None]
        return self.dt * DFT_SCALE * torch.stack(
            [torch.cos(w * t) * F, torch.sin(w * t) * F], dim=1)


class FluxMonitorTM:
    pol = "tm"
    reduce = "sum"

    def __init__(
        self,
        grid: Grid,
        freqs: Sequence[float],
        *,
        x_normal: bool,
        idx: int,
        lo: int,
        hi: int,
    ):
        self.x_normal = x_normal
        self.idx, self.lo, self.hi = idx, lo, hi
        self.dt = grid.dt
        self.freqs = torch.tensor(freqs, dtype=grid.dtype, device=grid.device)
        if x_normal and not (1 <= idx < grid.nx - 1):
            raise ValueError(f"x-normal monitor ix={idx} must be interior")
        if not x_normal and not (1 <= idx < grid.ny - 1):
            raise ValueError(f"y-normal monitor iy={idx} must be interior")

    def sample(self, state, t_idx: int) -> torch.Tensor:
        if self.x_normal:
            E = state.Ez[self.idx, self.lo : self.hi]
            H = 0.5 * (
                state.Hy[self.idx - 1, self.lo : self.hi]
                + state.Hy[self.idx, self.lo : self.hi]
            )
        else:
            E = state.Ez[self.lo : self.hi, self.idx]
            H = 0.5 * (
                state.Hx[self.lo : self.hi, self.idx - 1]
                + state.Hx[self.lo : self.hi, self.idx]
            )
        tE = (t_idx + 1.0) * self.dt
        tH = (t_idx + 0.5) * self.dt
        w = 2 * math.pi * self.freqs[:, None]
        return self.dt * DFT_SCALE * torch.stack(
            [
                torch.cos(w * tE) * E,
                torch.sin(w * tE) * E,
                torch.cos(w * tH) * H,
                torch.sin(w * tH) * H,
            ],
            dim=1,
        )

    def flux(self, acc: torch.Tensor, dx: float) -> torch.Tensor:
        ReE, ImE, ReH, ImH = acc.unbind(1)
        s = (ReE * ReH + ImE * ImH).sum(-1) * dx      # Re(Ê Ĥ*) integrated
        return -s if self.x_normal else s  # meep's dft_flux::flux() (dft.cpp:551) sums real(E conj(H)) with NO 1/2


class FluxMonitorTE:
    """TE flux line, either orientation, as FluxMonitorTM already does.

    x-normal takes ``ix`` and reads ``Ey`` against ``Hz`` averaged along x;
    y-normal takes ``iy`` and reads ``Ex`` against ``Hz`` averaged along y.
    The two differ ONLY in that slice and in the sign of the Poynting
    component -- ``Sx = +Ey Hz`` against ``Sy = -Ex Hz`` -- which is why they
    are one class and were two.
    """

    pol = "te"
    reduce = "sum"

    def __init__(self, grid: Grid, freqs, *, ix: int | None = None,
                 iy: int | None = None, lo: int, hi: int):
        if (ix is None) == (iy is None):
            raise ValueError("pass exactly one of ix= (x-normal) or iy= (y-normal)")
        self.x_normal = ix is not None
        self.idx = ix if self.x_normal else iy
        n = grid.nx if self.x_normal else grid.ny
        if not (1 <= self.idx < n - 1):
            raise ValueError(f"index {self.idx} must be interior")
        self.ix = self.iy = self.idx        # both names kept for callers
        self.lo, self.hi = lo, hi
        self.dt = grid.dt
        self.freqs = torch.tensor(freqs, dtype=grid.dtype, device=grid.device)

    def sample(self, state, t_idx: int) -> torch.Tensor:
        i, lo, hi = self.idx, self.lo, self.hi
        if self.x_normal:
            E = state.Ey[i, lo:hi]
            H = 0.5 * (state.Hz[i - 1, lo:hi] + state.Hz[i, lo:hi])
        else:
            E = state.Ex[lo:hi, i]
            H = 0.5 * (state.Hz[lo:hi, i - 1] + state.Hz[lo:hi, i])
        # step_te advances E to (t_idx+1/2)*dt and Hz to (t_idx+1)*dt
        tE = (t_idx + 0.5) * self.dt
        tH = (t_idx + 1.0) * self.dt
        w = 2 * torch.pi * self.freqs[:, None]
        return self.dt * DFT_SCALE * torch.stack(
            [torch.cos(w * tE) * E, torch.sin(w * tE) * E,
             torch.cos(w * tH) * H, torch.sin(w * tH) * H], dim=1)

    def flux(self, acc: torch.Tensor, dx: float) -> torch.Tensor:
        ReE, ImE, ReH, ImH = acc.unbind(1)
        # meep's dft_flux::flux() (dft.cpp:551) sums real(E conj(H)), NO 1/2
        s = (ReE * ReH + ImE * ImH).sum(-1) * dx
        return s if self.x_normal else -s


def FluxMonitorTEY(grid: Grid, freqs, *, iy: int, lo: int, hi: int):
    """Back-compatible spelling for the y-normal FluxMonitorTE."""
    return FluxMonitorTE(grid, freqs, iy=iy, lo=lo, hi=hi)


def cyl_flux(dft, *, z_normal: bool, idx: int, dr: float, dz: float,
             lo: int = 0, hi: int | None = None, r0: float = 0.0):
    """Poynting flux through a z-normal disc or an r-normal cylinder (BOR).

    ``dft`` maps the six BOR components to their DFT arrays, each with the
    leading axis over frequency and the Yee shapes of ``core/stepper_bor``::

        Er(i+1/2, j)   Ephi(i, j)          Ez(i, j+1/2)
        Hr(i, j+1/2)   Hphi(i+1/2, j+1/2)  Hz(i+1/2, j)

    BOTH terms of the Poynting component are summed::

        S_z = Er conj(Hphi) - Ephi conj(Hr)
        S_r = Ephi conj(Hz) - Ez conj(Hphi)

    An ``Er`` source at m != 0 drives the second pair as hard as the first, so
    keeping only one term reads as very nearly zero rather than as a factor.

    The surface element is derived, not guessed. A BOR field is
    ``F(r, z) e^{i m phi}`` and a Poynting component is a product
    ``F conj(G)``, so the phi dependence cancels and

        int_0^{2 pi} e^{i m phi} e^{-i m phi} dphi = 2 pi      for EVERY m,

    while over a radial cell CENTRED at r,

        int_{r - dr/2}^{r + dr/2} rho drho = r dr               exactly,

    so the midpoint rule carries no error and needs no gradient correction.
    (meep carries a second ``dV1`` term at ``loop_in_chunks.cpp:511`` because
    its loop is corner-indexed; sampling at cell centres removes it.)

    Each term is evaluated at ITS OWN radius, because the Yee layout puts the
    two pairs half a cell apart -- ``Er``/``Hphi`` at ``r_{i+1/2}``,
    ``Ephi``/``Hr`` at ``r_i`` -- and the H member is averaged from its two
    half-cell neighbours onto the E member's position along the normal.
    """
    import numpy as np

    def _avg_to_int(a, axis, k):
        """H sampled at half indices, averaged onto integer index ``k``."""
        n = a.shape[axis] - 1
        k = min(max(k - 1, 0), n - 1)
        sl = [slice(None)] * a.ndim
        sl[axis] = k
        lo_ = tuple(sl)
        sl[axis] = k + 1
        return 0.5 * (a[lo_] + a[tuple(sl)])

    def _term(E, H, r):
        """sum over the surface of Re(E conj(H)) * 2 pi r dl."""
        k = min(E.shape[-1], H.shape[-1], (hi if hi is not None else 10**9) - lo)
        a, b, w = E[..., lo:lo + k], H[..., lo:lo + k], r[lo:lo + k]
        return (np.real(a * np.conj(b)) * w).sum(-1)

    if z_normal:
        # lo:hi index r. Er/Hphi live at r_{i+1/2}, Ephi/Hr at r_i.
        nh = dft["Er"].shape[1]
        r_half = (np.arange(nh) + 0.5) * dr * dr * 2.0 * np.pi
        r_int = np.arange(dft["Ephi"].shape[1]) * dr * dr * 2.0 * np.pi
        er = dft["Er"][:, :, idx]
        hp = _avg_to_int(dft["Hphi"], 2, idx)
        ep = dft["Ephi"][:, :, idx]
        hr = _avg_to_int(dft["Hr"], 2, idx)
        return _term(er, hp, r_half) - _term(ep, hr, r_int)

    # r-normal at integer r index ``idx``; lo:hi index z. Both pairs sit at
    # the same radius r0, so the weight is the constant 2 pi r0 dz.
    w_p = np.full(dft["Ephi"].shape[2], 2.0 * np.pi * r0 * dz)
    w_z = np.full(dft["Ez"].shape[2], 2.0 * np.pi * r0 * dz)
    ep = dft["Ephi"][:, idx, :]
    hz = _avg_to_int(dft["Hz"], 1, idx)
    ez = dft["Ez"][:, idx, :]
    hp = _avg_to_int(dft["Hphi"], 1, idx)
    return _term(ep, hz, w_p) - _term(ez, hp, w_z)


