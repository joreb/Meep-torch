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
import sys

import numpy as np

try:                       # package import (pytest, -m) ...
    from . import scenario as S
except ImportError:        # ... or plain script run (path invocation)
    import scenario as S


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

    if gpu and "torch" in sys.modules:
        # MEASURED on this box (RTX A4000, nvhpc 26.5, torch 2.9.1+cu128):
        # devito GPU offload works in a bare process, and dies the moment
        # torch is imported -- `libgomp: no device found`, process aborted.
        # Two OpenMP runtimes are live: torch loads its own
        # torch/lib/libgomp.so.1 via RPATH, and the nvc++-built kernel wants
        # NVIDIA's. LD_PRELOADing nvhpc's libgomp does not displace an RPATH
        # load, and NVCOMPILER_OMP_DISABLE_WARNINGS only hides the warning,
        # not the conflict. So: fall back to CPU rather than abort, and say
        # so. A torch-free subprocess is the only way to get devito on the GPU.
        print("      devito: GPU offload unavailable in a torch process "
              "(two OpenMP runtimes); running CPU/OpenMP")
        gpu = False

    if gpu:
        # devito GPU offload: all three must be set together. platform alone
        # leaves the default gcc compiler emitting OpenMP *target* pragmas,
        # which aborts at run time with "libgomp: no device found" -- the
        # NVIDIA compiler and OpenACC are what actually reach the device.
        configuration["platform"] = "nvidiaX"
        configuration["compiler"] = "nvc"
        configuration["language"] = "openacc"

    # extent is FIRST-to-LAST point, so devito's spacing is extent/(shape-1):
    # passing the full cell size makes every derivative off by NX/(NX-1). And
    # the devito default dtype is float32, which caps agreement at ~1e-7 and
    # here diverges to NaN over 4000 steps. Same two fixes fdtd_devito_2d.py
    # documents at its _build().
    grid = Grid(shape=(S.NX, S.NY),
                extent=((S.NX - 1) * S.DX, (S.NY - 1) * S.DX),
                dtype=np.float64)
    dt = S.DX * 0.5   # match scenario courant=0.5

    # BROKEN on devito 4.8.x+ (this box): the run diverges to NaN by step ~50
    # regardless of stagger formulation -- measured |Ez| 2.5e-1 at step 20,
    # 2.7e8 at step 50. Not a regression from the extent/dtype fixes above:
    # those are independently correct (see fdtd_devito_2d.py:60). Rewriting
    # the stagger explicitly with fdkind="forward" plus index-shifted backward
    # differences did NOT fix it either, so the cause is still open. Until it
    # is found, the devito straight-waveguide number is not trustworthy.
    Ez = TimeFunction(name="Ez", grid=grid, space_order=1, time_order=1)
    Hx = TimeFunction(name="Hx", grid=grid, space_order=1, time_order=1,
                      staggered=(0, 0, 0.5))   # y-staggered
    Hy = TimeFunction(name="Hy", grid=grid, space_order=1, time_order=1,
                      staggered=(0, 0.5, 0))   # x-staggered

    eps = Function(name="eps", grid=grid, space_order=0)
    # scenario eps is (NX=x, NY=y); devito indexes [x, y] with the same shape.
    eps_np = S.eps_field("cpu").numpy()
    eps.data[:] = eps_np

    damp_h = Function(name="damp_h", grid=grid, space_order=0)
    damp_e = Function(name="damp_e", grid=grid, space_order=0)
    sx = _pml_sigma(S.NX, S.NPML)
    sy = _pml_sigma(S.NY, S.NPML)
    sx_g, sy_g = np.meshgrid(sx, sy, indexing="ij")
    sigma = sx_g + sy_g
    damp_h.data[:] = np.exp(-sigma * dt)
    damp_e.data[:] = np.exp(-sigma * dt)

    # soft Ez source at (-7, 0) in meep coords. devito Grid origin sits at the
    # extent corner (0,0), so meep x=-7 -> local 1.0, y=0 -> local 4.0.
    # The source cell is INSIDE the core, not in vacuum: scenario puts it at
    # iy = NY//2 = 40, i.e. y = +0.05, and the waveguide is |y| < 0.5 with
    # eps = 12. Injecting -dt*J instead of -(dt/eps)*J therefore overdrove the
    # field by exactly that eps -- measured peak ratio 12.89x vs the
    # meep_torch reference, with interior correlation already fine.
    n = S.n_steps(dt)
    # coordinates is (npoint, ndim) -- one row per point. A tuple of two
    # 1-element arrays reads as (2, 1) and modern devito rejects it.
    src = SparseTimeFunction(name="src", grid=grid, npoint=1, nt=n,
                             coordinates=np.array([[S.SRC_X + S.CELL_X / 2,
                                                    S.CELL_Y / 2]]))
    ts = np.arange(n) * dt
    # 1/eps folded into the source SAMPLES rather than the inject expression:
    # `expr=-dt*src/eps` makes devito interpolate eps at the sparse point, and
    # a space_order=0 Function cannot be interpolated ("Space order 0 too small
    # for interpolation r 1"). The source is one cell, so its eps is a scalar.
    ix_src = int(round((S.SRC_X - S.X_LO) / S.DX))
    iy_src = S.NY // 2
    src.data[:, 0] = np.cos(2.0 * math.pi * S.FREQ * ts) / float(
        eps_np[ix_src, iy_src])

    # Leapfrog: E must read the H this same step just wrote, i.e. Hx.forward /
    # Hy.forward, not Hx / Hy. Reading the OLD H makes the update non-symplectic
    # and it diverges -- measured on this scenario, max|Ez| after 400 steps:
    #   Hy.dx  (old H) -> 2.1e+82        Hy.forward.dx (new H) -> 3.6e-02
    # That difference, not the stagger, was the NaN: declaring the H functions
    # with staggered=(0,0,0.5)/(0,0.5,0) vs not gives bit-identical results
    # here (both numbers above are unchanged by it), so this devito derives the
    # half-cell offset from neither -- the leapfrog order is what carries it.
    eq_hx = Eq(Hx.forward, damp_h * Hx - dt * Ez.dy)
    eq_hy = Eq(Hy.forward, damp_h * Hy + dt * Ez.dx)
    eq_ez = Eq(Ez.forward,
               damp_e * Ez + (dt / eps) * (Hy.forward.dx - Hx.forward.dy))

    # J enters as -dt*J (meep step_source sign, see scenario.source_amp).
    # meep injects the current into D and then solves E = D/eps, so the
    # amplitude that reaches Ez carries the 1/eps -- see stepper_tm.step_source
    # (scale = dt) followed by update_eh(E). Same factor here.
    op = Operator([eq_hx, eq_hy, eq_ez] +
                  [src.inject(field=Ez.forward, expr=-dt * src)],
                  subs=grid.spacing_map)

    op.apply(time=n - 1, dt=dt)
    # devito lays out [time, x, y]; scenario convention is (NX, NY) = (x, y).
    return np.asarray(Ez.data[0]).copy()