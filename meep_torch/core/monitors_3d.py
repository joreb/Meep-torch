# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""3D monitors + forward-only runner (W9 wave 2). Same ``e^{-iwt}`` DFT
phasor convention as the 2D monitors (``dt * [cos(wt) F, -sin(wt) F]``,
E at ``(t+1) dt``, H at ``(t+1/2) dt``, ``reduce="sum"``). W9 is
forward-only per the master spec, so ``run_3d`` carries no checkpointing.
"""
import math
from typing import Sequence

import torch

from .stepper_3d import step_3d

_COMP_NODE = {"Ex": "Ex", "Ey": "Ey", "Ez": "Ez", "Hx": "Hx", "Hy": "Hy", "Hz": "Hz"}


class PlaneSource:
    """Soft area current on an axis-normal plane, launching a transverse
    planewave. ``axis`` is the propagation axis (0=x,1=y,2=z); ``comp`` is the
    E component injected (must be transverse to ``axis``); ``profile.amplitude``
    gives the time waveform. ``lo``/``hi`` bound the transverse window of the
    source (defaults fill the full cross-section)."""

    def __init__(self, nx, ny, nz, dt, comp, axis, idx, profile,
                 lo=None, hi=None):
        shape = {"Ex": (nx - 1, ny, nz), "Ey": (nx, ny - 1, nz),
                 "Ez": (nx, ny, nz - 1)}[comp]
        mask = torch.zeros(shape, dtype=torch.float64)
        sl = [slice(None)] * 3
        sl[axis] = idx
        mask[tuple(sl)] = 1.0
        if lo is not None and hi is not None:           # aperture on transverse axes
            sel = [slice(None)] * 3
            for a in range(3):
                if a != axis:
                    sel[a] = slice(lo, hi)
            window = mask[tuple(sel)].clone()
            mask = torch.zeros_like(mask)
            mask[tuple(sel)] = window
        self.profile = profile
        self.dt = dt
        self.src = (comp, mask)

    def amplitude(self, t_idx: int) -> float:
        return self.profile.amplitude((t_idx + 0.5) * self.dt)


class PointSource3D:
    """Soft point current (``step_3d`` point-source tuple) wrapped in the
    same ``.amplitude(t_idx)`` + ``.src`` interface as ``PlaneSource`` so
    ``run_3d`` drives both uniformly. ``comp`` is the injected E component,
    ``(i, j, k)`` the node index, ``profile.amplitude`` the time waveform."""
    _COMP = {"Ex", "Ey", "Ez"}

    def __init__(self, dt, comp, i, j, k, profile):
        if comp not in self._COMP:
            raise ValueError(f"PointSource3D comp must be Ex/Ey/Ez, got {comp!r}")
        self.dt = dt
        self.profile = profile
        self.src = (comp, int(i), int(j), int(k))

    def amplitude(self, t_idx: int) -> float:
        return self.profile.amplitude((t_idx + 0.5) * self.dt)


class DFTFieldsMonitor3D:
    """Running DFT of one component on a plane (or full grid). Returns
    ``(nfreq, 2, ...)`` per step; ``reduce="sum"`` accumulates the phasor."""
    reduce = "sum"

    def __init__(self, dt, freqs, component="Ez", plane=None, dtype=torch.float64):
        self.component = component
        self.dt = dt
        self.freqs = torch.tensor(freqs, dtype=dtype)
        self.plane = plane          # (axis, idx) or None (full grid)

    def sample(self, state, t_idx: int) -> torch.Tensor:
        F = getattr(state, _COMP_NODE[self.component])
        if self.plane is not None:
            axis, idx = self.plane
            sl = [slice(None)] * 3
            sl[axis] = idx
            F = F[tuple(sl)]
        off = 1.0 if self.component.startswith("E") else 0.5
        t = (t_idx + off) * self.dt
        shape = [1, 1] + [1] * F.ndim
        w = (2 * math.pi * self.freqs).reshape(shape)
        return self.dt * torch.stack([torch.cos(w * t) * F, -torch.sin(w * t) * F], dim=1)


class FluxMonitor3D:
    """Time-averaged Poynting flux through an axis-normal plane.

    z-normal: ``S_z = 0.5 Re[ Ex conj(Hy) - Ey conj(Hx) ]`` on the plane at
    integer index ``idx`` (Ex/Ey at z=idx, Hx/Hy averaged from idx±1/2). x/y
    normals follow by cyclic rotation. ``ap`` is the aperture margin (cells
    excluded from each transverse edge) so the side-PML region is kept out of
    the integral. ``reduce="sum"``; ``flux(acc, dx)`` returns the signed
    power through the plane.
    """
    reduce = "sum"

    def __init__(self, dt, freqs, *, normal, idx, ap=0, dtype=torch.float64):
        if normal not in ("x", "y", "z"):
            raise ValueError(f"normal must be x/y/z, got {normal!r}")
        self.normal = normal
        self.idx = idx
        self.ap = ap
        self.dt = dt
        self.freqs = torch.tensor(freqs, dtype=dtype)

    def _plane(self, state):
        # Returns (E_a, H_a, E_b, H_b) plane samples for the two transverse
        # E-H pairs whose cross product gives the normal Poynting.
        # S_n = 0.5 Re[ E_a conj(H_a) - E_b conj(H_b) ] for all normals.
        n = self.normal
        ap = self.ap
        if n == "z":                                   # S_z = Ex Hy - Ey Hx
            k = self.idx
            Ea = state.Ex[:, :, k]                             # (nx-1, ny)
            Ha = 0.5 * (state.Hy[:, :, k - 1] + state.Hy[:, :, k])
            Eb = state.Ey[:, :, k]                            # (nx, ny-1)
            Hb = 0.5 * (state.Hx[:, :, k - 1] + state.Hx[:, :, k])
        elif n == "x":                                # S_x = Ey Hz - Ez Hy
            i = self.idx
            Ea = state.Ey[i, :, :]                            # (ny-1, nz)
            Ha = 0.5 * (state.Hz[i - 1, :, :] + state.Hz[i, :, :])
            Eb = state.Ez[i, :, :]                           # (ny, nz-1)
            Hb = 0.5 * (state.Hy[i - 1, :, :] + state.Hy[i, :, :])
        else:                                          # S_y = Ez Hx - Ex Hz
            j = self.idx
            Ea = state.Ez[:, j, :]                           # (nx, nz-1)
            Ha = 0.5 * (state.Hx[:, j - 1, :] + state.Hx[:, j, :])
            Eb = state.Ex[:, j, :]                           # (nx-1, nz)
            Hb = 0.5 * (state.Hz[:, j - 1, :] + state.Hz[:, j, :])
        # The two pairs sit on offset sublattices (shapes differ by one
        # cell on each axis); slice both to the common min shape so they
        # stack. The aperture drops the side-PML fringe from the integral.
        tensors = [Ea, Ha, Eb, Hb]
        m0 = min(t.shape[0] for t in tensors)
        m1 = min(t.shape[1] for t in tensors)
        r0, r1 = ap, m0 - ap
        c0, c1 = ap, m1 - ap
        return tuple(t[r0:r1, c0:c1] for t in tensors)

    def sample(self, state, t_idx: int) -> torch.Tensor:
        Ea, Ha, Eb, Hb = self._plane(state)
        tE = (t_idx + 1.0) * self.dt
        tH = (t_idx + 0.5) * self.dt
        wE = (2 * math.pi * self.freqs)[:, None, None]
        wH = (2 * math.pi * self.freqs)[:, None, None]
        return self.dt * torch.cat([
            (torch.cos(wE * tE) * Ea).unsqueeze(1),
            (-torch.sin(wE * tE) * Ea).unsqueeze(1),
            (torch.cos(wH * tH) * Ha).unsqueeze(1),
            (-torch.sin(wH * tH) * Ha).unsqueeze(1),
            (torch.cos(wE * tE) * Eb).unsqueeze(1),
            (-torch.sin(wE * tE) * Eb).unsqueeze(1),
            (torch.cos(wH * tH) * Hb).unsqueeze(1),
            (-torch.sin(wH * tH) * Hb).unsqueeze(1),
        ], dim=1)

    def flux(self, acc: torch.Tensor, dx: float) -> torch.Tensor:
        ReaE, ImaE, ReaH, ImaH, RebE, ImbE, RebH, ImbH = acc.unbind(1)
        # S_n = 0.5 Re(Ea conj(Ha) - Eb conj(Hb)); integrate over the plane
        # (all spatial axes after the freq axis), same formula all normals.
        sa = (ReaE * ReaH + ImaE * ImaH)
        sb = (RebE * RebH + ImbE * ImbH)
        return 0.5 * (sa - sb).sum(tuple(range(1, sa.ndim))) * dx * dx


def run_3d(state, coeffs, n_steps: int, *, source=None, monitors=()):
    """Forward-only 3D runner (no checkpointing). ``source`` exposes
    ``.amplitude(t_idx)`` and ``.src`` (a ``step_3d`` source tuple)."""
    monitors = list(monitors)
    acc = [None] * len(monitors)
    for t in range(n_steps):
        if source is not None:
            state = step_3d(state, coeffs, source.amplitude(t), source.src)
        else:
            state = step_3d(state, coeffs)
        for i, mon in enumerate(monitors):
            out = mon.sample(state, t)
            if acc[i] is None:
                acc[i] = out
            elif mon.reduce == "sum":
                acc[i] = acc[i] + out
            else:
                acc[i] = torch.cat([acc[i], out])
    return state, acc