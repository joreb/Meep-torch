# SPDX-License-Identifier: MIT
# 2D TMz FDTD built with devito (symbolic PDE -> C/JIT codegen), run on the
# straight-waveguide scenario from scenario.py. GPU via devito's nvidiaX
# platform; CPU otherwise. devito is a lazy import -- this module loads
# without it; only run_devito() needs it.
#
# Stencil: standard Yee leapfrog, mu=1.
#   Hx^{n+1/2} = Hx - dt * dEz/dy
#   Hy^{n+1/2} = Hy + dt * dEz/dx
#   Ez^{n+1}   = Ez + (dt/eps) * (dHy/dx - dHx/dy) - (dt/eps) * J
# PML: simple conductive absorber (quadratic sigma profile over NPML cells),
# NOT meep's CPML recursion -- field-correlation gate, not bitwise.
import math

import numpy as np

from . import scenario as S


def _pml_sigma(axis_len: int, npml: int) -> np.ndarray:
    """Quadratic conductivity ramp: 0 in the interior, rising into each edge."""
    sigma = np.zeros(axis_len)
    for edge in (slice(0, npml), slice(axis_len - npml, axis_len)):
        idx = np.arange(axis_len)[edge]
        dist = np.minimum(idx, axis_len - 1 - idx)   # distance to nearest edge
        # ponytail: simple quadratic, no tuning -- tune the prefactor on the box
        sigma[edge] = 0.8 * ((npml - dist) / npml) ** 2
    return sigma


def run_devito(gpu: bool = False) -> np.ndarray:
    """Run the straight-waveguide with devito -> Ez array (NX, NY) on CPU."""
    from devito import (Grid, TimeFunction, Function, Eq, Operator,
                        SparseTimeFunction, configuration)

    if gpu:
        # devito GPU needs platform=nvidiaX (Linux + CUDA toolchain).
        configuration["platform"] = "nvidiaX"

    grid = Grid(shape=(S.NX, S.NY), extent=(S.CELL_X, S.CELL_Y))
    dt = S.DX * 0.5   # match scenario courant=0.5

    Ez = TimeFunction(name="Ez", grid=grid, space_order=1, time_order=1)
    Hx = TimeFunction(name="Hx", grid=grid, space_order=1, time_order=1,
                      staggered=(0, 0, 0.5))   # y-staggered
    Hy = TimeFunction(name="Hy", grid=grid, space_order=1, time_order=1,
                      staggered=(0, 0.5, 0))   # x-staggered

    eps = Function(name="eps", grid=grid, space_order=0)
    # scenario eps is (NX=x, NY=y); devito indexes [x, y] with the same shape.
    eps.data[:] = S.eps_field("cpu").numpy()

    damp_h = Function(name="damp_h", grid=grid, space_order=0)
    damp_e = Function(name="damp_e", grid=grid, space_order=0)
    sx = _pml_sigma(S.NX, S.NPML)
    sy = _pml_sigma(S.NY, S.NPML)
    sx_g, sy_g = np.meshgrid(sx, sy, indexing="ij")
    sigma = sx_g + sy_g
    damp_h.data[:] = np.exp(-sigma * dt)
    damp_e.data[:] = np.exp(-sigma * dt)

    # soft Ez source at (-7, 0) in meep coords. devito Grid origin sits at the
    # extent corner (0,0), so meep x=-7 -> local 1.0, y=0 -> local 4.0. The
    # source sits in the eps=1 vacuum (|y|>0.5), so the J scale is just dt.
    n = S.n_steps(dt)
    src = SparseTimeFunction(name="src", grid=grid, npoint=1, nt=n,
                             coordinates=(np.array([S.SRC_X + S.CELL_X / 2]),
                                          np.array([S.CELL_Y / 2])))
    ts = np.arange(n) * dt
    src.data[:, 0] = np.cos(2.0 * math.pi * S.FREQ * ts)

    eq_hx = Eq(Hx.forward, damp_h * Hx - dt * Ez.dy)
    eq_hy = Eq(Hy.forward, damp_h * Hy + dt * Ez.dx)
    eq_ez = Eq(Ez.forward,
               damp_e * Ez + (dt / eps) * (Hy.dx - Hx.dy))

    # J enters as -dt*J (meep step_source sign, see scenario.source_amp).
    op = Operator([eq_hx, eq_hy, eq_ez] +
                  [src.inject(field=Ez.forward, expr=-dt * src)],
                  subs=grid.spacing_map)

    op.apply(time=n - 1, dt=dt)
    # devito lays out [time, x, y]; scenario convention is (NX, NY) = (x, y).
    return np.asarray(Ez.data[0]).copy()