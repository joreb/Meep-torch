# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Memory-lean FORWARD-ONLY 3D FDTD stepper (docs/plans/2026-08-16).

Same physics as ``step_3d`` (CPML path, sigma = 0), engineered for GPU
memory: in-place field updates (no double buffering), CPML psi stored
only on its PML shells, damp collapsed to scalars, eps averages
precomputed as reciprocals, and a per-shape scratch pool. Peak memory
~ fields + inv_eps + shells + a few scratch buffers, instead of the
functional stepper's ~2.5x-static blowup.

Forward-only by contract: everything mutates under torch.no_grad().
Parity-gated against step_3d in tests/examples/test_stepper3d_lean.py.
"""
import math
from dataclasses import dataclass, field

import torch

from .pml import cpml_profiles

__all__ = ["build_lean", "run_3d_lean"]


def _prof(n, npml, offset, dt, dx, dtype, device):
    return cpml_profiles(n, npml, offset, dt, dx, dtype=dtype, device=device)


def _avg(v, axis):
    sl0 = [slice(None)] * 3
    sl1 = [slice(None)] * 3
    sl0[axis] = slice(None, -1)
    sl1[axis] = slice(1, None)
    return 0.5 * (v[tuple(sl0)] + v[tuple(sl1)])


@dataclass
class _Axis:
    b_lo: torch.Tensor; a_lo: torch.Tensor; k_lo: torch.Tensor
    b_hi: torch.Tensor; a_hi: torch.Tensor; k_hi: torch.Tensor
    m_lo: int
    m_hi: int


@dataclass
class Lean3D:
    dt: float
    dx: float
    nx: int
    ny: int
    nz: int
    npml: int
    inv_eps: dict            # 'x','y','z' -> reciprocal averaged eps
    axH: dict                # axis -> _Axis for H-side derivatives (half nodes)
    axE: dict                # axis -> _Axis for E-side derivatives (int nodes)
    F: dict = field(default_factory=dict)      # fields Ex..Hz (in-place)
    psi: dict = field(default_factory=dict)    # name -> (lo, hi) shell tensors
    scratch: dict = field(default_factory=dict)

    def buf(self, shape):
        # 2-slot ring per shape: an H/E component's two curl terms can share
        # a shape and must NOT alias (the first parity failure was exactly
        # ce_z overwriting ce_y through a single shared buffer).
        ring = self.scratch.setdefault(shape, {"bufs": [], "i": 0})
        if len(ring["bufs"]) < 2:
            dev = self.inv_eps["x"].device
            t = torch.empty(*shape, dtype=self.inv_eps["x"].dtype, device=dev)
            ring["bufs"].append(t)
        else:
            t = ring["bufs"][ring["i"] % 2]
        ring["i"] += 1
        return t


def build_lean(nx, ny, nz, dx, eps, npml, dtype=torch.float32,
               device="cuda") -> Lean3D:
    dt = 0.5 * dx / math.sqrt(3.0)
    e = eps.to(dtype=dtype, device=device)
    inv = {k: 1.0 / _avg(e, ax) for k, ax in (("x", 0), ("y", 1), ("z", 2))}
    dims = (nx, ny, nz)

    def mk(n, offset):
        b, a, k = _prof(n, npml, offset, dt, dx, dtype, device)
        if offset == 0.0:
            b, a, k = b[1:-1], a[1:-1], k[1:-1]
        # derive shell widths EXACTLY from the profile content
        active = ((b != 0) | (a != 0) | (k != 1)).nonzero().flatten()
        half = len(b) // 2
        lo_idx = active[active < half]
        hi_idx = active[active >= half]
        m_lo = int(lo_idx.max()) + 1 if len(lo_idx) else 1
        m_hi = len(b) - int(hi_idx.min()) if len(hi_idx) else 1
        return _Axis(b[:m_lo].clone(), a[:m_lo].clone(), k[:m_lo].clone(),
                     b[-m_hi:].clone(), a[-m_hi:].clone(), k[-m_hi:].clone(),
                     m_lo, m_hi)

    axH = {ax: mk(dims[ax] - 1, 0.5) for ax in range(3)}
    axE = {ax: mk(dims[ax], 0.0) for ax in range(3)}

    c = Lean3D(dt=dt, dx=dx, nx=nx, ny=ny, nz=nz, npml=npml,
               inv_eps=inv, axH=axH, axE=axE)
    z = lambda *s: torch.zeros(*s, dtype=dtype, device=device)
    c.F = dict(Ex=z(nx - 1, ny, nz), Ey=z(nx, ny - 1, nz), Ez=z(nx, ny, nz - 1),
               Hx=z(nx, ny - 1, nz - 1), Hy=z(nx - 1, ny, nz - 1),
               Hz=z(nx - 1, ny - 1, nz))
    # psi shells: name -> (lo, hi); shapes follow the derivative outputs
    def shells(shape, axis, m):
        m_lo, m_hi = m
        s_lo = list(shape); s_lo[axis] = m_lo
        s_hi = list(shape); s_hi[axis] = m_hi
        return (z(*s_lo), z(*s_hi))
    m5 = lambda ax: (axH[ax].m_lo, axH[ax].m_hi)
    mE = lambda ax: (axE[ax].m_lo, axE[ax].m_hi)
    c.psi = {
        "Ez_y": shells((nx, ny - 1, nz - 1), 1, m5(1)),
        "Ey_z": shells((nx, ny - 1, nz - 1), 2, m5(2)),
        "Ex_z": shells((nx - 1, ny, nz - 1), 2, m5(2)),
        "Ez_x": shells((nx - 1, ny, nz - 1), 0, m5(0)),
        "Ey_x": shells((nx - 1, ny - 1, nz), 0, m5(0)),
        "Ex_y": shells((nx - 1, ny - 1, nz), 1, m5(1)),
        "Hz_y": shells((nx - 1, ny - 2, nz), 1, mE(1)),
        "Hy_z": shells((nx - 1, ny, nz - 2), 2, mE(2)),
        "Hx_z": shells((nx, ny - 1, nz - 2), 2, mE(2)),
        "Hz_x": shells((nx - 2, ny - 1, nz), 0, mE(0)),
        "Hy_x": shells((nx - 2, ny, nz - 1), 0, mE(0)),
        "Hx_y": shells((nx, ny - 2, nz - 1), 1, mE(1)),
    }
    return c


def _shape1(v, axis):
    s = [1, 1, 1]
    s[axis] = -1
    return v.reshape(s)


def _deriv_ce(c, Fname, axis, ax: _Axis, psi_name):
    """Curl term D/k + psi computed IN PLACE on a scratch buffer.
    Returns the buffer (valid until the next two _deriv_ce calls)."""
    F = c.F[Fname]
    sl0 = [slice(None)] * 3
    sl1 = [slice(None)] * 3
    sl0[axis] = slice(None, -1)
    sl1[axis] = slice(1, None)
    out_shape = list(F.shape)
    out_shape[axis] -= 1
    D = c.buf(tuple(out_shape))
    torch.sub(F[tuple(sl1)], F[tuple(sl0)], out=D)
    D.div_(c.dx)
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    lo[axis] = slice(0, ax.m_lo)
    hi[axis] = slice(out_shape[axis] - ax.m_hi, out_shape[axis])
    p_lo, p_hi = c.psi[psi_name]
    for sl, p, b, a, k in ((lo, p_lo, ax.b_lo, ax.a_lo, ax.k_lo),
                           (hi, p_hi, ax.b_hi, ax.a_hi, ax.k_hi)):
        Ds = D[tuple(sl)]
        p.mul_(_shape1(b, axis)).add_(Ds * _shape1(a, axis))
        Ds.div_(_shape1(k, axis)).add_(p)
    return D


def _interior_sub(c, A, B, sliceA, sliceB, out_shape, pads):
    """out (zero borders on `pads` axes) with interior = A[sliceA]-B[sliceB]."""
    out = c.buf(tuple(out_shape))
    out.zero_()
    inner = [slice(None)] * 3
    for ax in pads:
        inner[ax] = slice(1, -1)
    torch.sub(A[tuple(sliceA)], B[tuple(sliceB)], out=out[tuple(inner)])
    return out


@torch.no_grad()
def step_lean(c: Lean3D, src_amp=0.0, src=None):
    F = c.F
    dt = c.dt
    # --- H updates ---
    ce_y = _deriv_ce(c, "Ez", 1, c.axH[1], "Ez_y")
    ce_z = _deriv_ce(c, "Ey", 2, c.axH[2], "Ey_z")
    F["Hx"].sub_((ce_y - ce_z).mul_(dt))
    ce_z2 = _deriv_ce(c, "Ex", 2, c.axH[2], "Ex_z")
    ce_x = _deriv_ce(c, "Ez", 0, c.axH[0], "Ez_x")
    F["Hy"].sub_((ce_z2 - ce_x).mul_(dt))
    ce_x2 = _deriv_ce(c, "Ey", 0, c.axH[0], "Ey_x")
    ce_y2 = _deriv_ce(c, "Ex", 1, c.axH[1], "Ex_y")
    F["Hz"].sub_((ce_x2 - ce_y2).mul_(dt))

    # --- E updates (PEC outer edge: zero-padded borders) ---
    ch_y = _deriv_ce(c, "Hz", 1, c.axE[1], "Hz_y")     # (nx-1, ny-2, nz)
    ch_z = _deriv_ce(c, "Hy", 2, c.axE[2], "Hy_z")     # (nx-1, ny, nz-2)
    nx, ny, nz = c.nx, c.ny, c.nz
    curl = _interior_sub(c, ch_y, ch_z,
                         (slice(None), slice(None), slice(1, -1)),
                         (slice(None), slice(1, -1), slice(None)),
                         (nx - 1, ny, nz), (1, 2))
    F["Ex"].add_(curl.mul_(c.inv_eps["x"]).mul_(dt))
    ch_z2 = _deriv_ce(c, "Hx", 2, c.axE[2], "Hx_z")    # (nx, ny-1, nz-2)
    ch_x = _deriv_ce(c, "Hz", 0, c.axE[0], "Hz_x")     # (nx-2, ny-1, nz)
    curl = _interior_sub(c, ch_z2, ch_x,
                         (slice(1, -1), slice(None), slice(None)),
                         (slice(None), slice(None), slice(1, -1)),
                         (nx, ny - 1, nz), (0, 2))
    F["Ey"].add_(curl.mul_(c.inv_eps["y"]).mul_(dt))
    ch_x2 = _deriv_ce(c, "Hy", 0, c.axE[0], "Hy_x")    # (nx-2, ny, nz-1)
    ch_y2 = _deriv_ce(c, "Hx", 1, c.axE[1], "Hx_y")    # (nx, ny-2, nz-1)
    curl = _interior_sub(c, ch_x2, ch_y2,
                         (slice(None), slice(1, -1), slice(None)),
                         (slice(1, -1), slice(None), slice(None)),
                         (nx, ny, nz - 1), (0, 1))
    F["Ez"].add_(curl.mul_(c.inv_eps["z"]).mul_(dt))

    # --- soft current injection ---
    if src is not None and src_amp != 0.0:
        comp = src[0]
        key = comp[1].lower()
        if len(src) == 2:
            F[comp].add_(src[1] * (dt * src_amp) * c.inv_eps[key])
        else:
            i, j, k = src[1], src[2], src[3]
            F[comp][i, j, k] += dt * src_amp * c.inv_eps[key][i, j, k]


class _StateView:
    """Duck-typed State3D view over the in-place field dict (monitors)."""

    def __init__(self, F):
        self.__dict__.update(F)


@torch.no_grad()
def run_3d_lean(c: Lean3D, n_steps: int, *, source=None, monitors=(),
                compile: bool = False):
    """``compile=True`` wraps the step in torch.compile: measured 2.19x
    per-step speedup on an GPU (kernel fusion; ~1 min warmup)."""
    monitors = list(monitors)
    acc = [None] * len(monitors)
    view = _StateView(c.F)
    step = torch.compile(step_lean) if compile else step_lean
    for t in range(n_steps):
        amp = source.amplitude(t) if source is not None else 0.0
        step(c, amp, source.src if source is not None else None)
        for i, mon in enumerate(monitors):
            smp = mon.sample(view, t)
            if mon.reduce == "sum":
                acc[i] = smp if acc[i] is None else acc[i] + smp
            else:
                acc[i] = (torch.cat([acc[i], smp[None]])
                          if acc[i] is not None else smp[None])
    return view, acc
