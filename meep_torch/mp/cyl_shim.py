# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Cylindrical (r, z) Simulation runs -- the adapter between the meep
Simulation surface (``dimensions=mp.CYLINDRICAL``, ``m=...``) and
core/stepper_bor.py, the six-component body-of-revolution FDTD.

The engine was already there (core/stepper_bor.py, core/stepper_cyl.py,
core/near2far_cyl.py, gated by tests/examples/test_w75_bor.py and
test_w775_cyl_n2f.py); what this module adds is the meep-shaped wrapper:
cell/geometry/source/boundary translation, the run loop with step
functions, and the near2far + LDOS accessors.

Coordinates. meep's cylindrical cell is ``Vector3(sr, 0, sz)`` with r in
[0, sr] (corner origin) and z in [-sz/2, sz/2] (center origin). Node
(i, j) sits at r = i dr, z = -sz/2 + j dr; the Yee staggering (Er at
i+1/2, Ez at j+1/2, ...) follows core/stepper_bor.

Scope (everything else raises with a named reason):
  * geometry: mp.Block only (r-z rectangles), non-dispersive media
  * sources: point Er/Ep/Ez sources sharing one time profile
  * boundaries: PML/Absorber -> the stepper's matched absorber ramp
  * step functions: mp.Harminv, stop_when_fields_decayed
  * monitors: add_near2far / get_farfield, dft_ldos
"""
import math

import numpy as np
import torch

from ..core.ldos import ldos_spectrum
from ..core.near2far_cyl import farfield_bor_m_complex
from ..core.stepper_bor import build_bor_coeffs, init_bor_state, step_bor
from .run_control import (_eval_step_func, fire_per_step,
                          fire_scheduled, plan_hooks)
from .vec import Vector3

# meep enum / component sentinels for the cylindrical coordinate system
# (mp.visualization uses the same CYLINDRICAL = -2 literal).
CYLINDRICAL = -2
Er, Ep, Hr, Hp = "Er", "Ep", "Hr", "Hp"

# meep component name -> core/stepper_bor BORState field name
_FIELD = {"Er": "Er", "Ep": "Ephi", "Ez": "Ez",
          "Hr": "Hr", "Hp": "Hphi", "Hz": "Hz"}
# per-component (r, z) half-cell offsets in index units
_OFF = {"Er": (0.5, 0.0), "Ep": (0.0, 0.0), "Ez": (0.0, 0.5),
        "Hr": (0.0, 0.5), "Hp": (0.5, 0.5), "Hz": (0.5, 0.0)}

_NIE = NotImplementedError


def needs_cyl(sim) -> bool:
    """True for a cylindrical Simulation (``dimensions = mp.CYLINDRICAL``)."""
    return getattr(sim, "dimensions", 2) == CYLINDRICAL


# --------------------------------------------------------------- geometry
class CylGrid:
    """(r, z) node grid of a cylindrical cell."""

    def __init__(self, sim):
        cs = sim.cell_size
        cs = cs if isinstance(cs, Vector3) else Vector3(*cs)
        self.dr = 1.0 / float(sim.resolution)
        self.sr, self.sz = float(cs.x), float(cs.z)
        if self.sr <= 0 or self.sz <= 0:
            raise _NIE("cylindrical cells need a nonzero r and z extent "
                       "(cell_size = Vector3(sr, 0, sz)); the 1D radial-only "
                       "case lives in core/stepper_cyl.py")
        self.nr = max(3, int(round(self.sr / self.dr)))
        self.nz = max(3, int(round(self.sz / self.dr)))

    def index(self, comp: str, pt) -> tuple[int, int]:
        """Nearest (i, j) index of meep point ``pt`` on component ``comp``'s
        staggered subgrid, clamped to that subgrid's shape."""
        offr, offz = _OFF[comp]
        i = int(round(float(pt.x) / self.dr - offr))
        j = int(round((float(pt.z) + self.sz / 2.0) / self.dr - offz))
        ni = self.nr - (1 if offr else 0)
        nj = self.nz - (1 if offz else 0)
        return max(0, min(ni - 1, i)), max(0, min(nj - 1, j))


def cyl_eps(sim, g: CylGrid) -> torch.Tensor:
    """(nr, nz) node eps from the geometry, area-averaged by the ordinary 2D
    rasterizer: r -> grid x, z -> grid y, shifted a half pixel so rasterizer
    cell (i, j) is the cell CENTERED on node (i, j) -- the same convention the
    ring-cyl / disc-BOR gates use."""
    from ..core.grid import Grid
    from ..geometry.scene import rasterize_scene
    from .coords import GridSpec
    from .geometry import Block, build_scene_items

    items = []
    for obj in getattr(sim, "_geometry", []):
        if not isinstance(obj, Block):
            raise _NIE(
                f"cylindrical geometry: {type(obj).__name__} is not supported "
                "(the (r, z) rasterizer handles mp.Block rectangles only)")
        med = obj.material
        if getattr(med, "E_susceptibilities", None):
            raise _NIE("cylindrical runs support non-dispersive media only "
                       "(core/stepper_bor has no polarization currents)")
        c, s = obj.center, obj.size
        items.append(Block(
            center=Vector3(float(c.x) + g.dr / 2 - g.nr * g.dr / 2,
                           float(c.z) + g.dr / 2, 0),
            size=Vector3(s.x, s.z, 0), material=med))
    gs = GridSpec(nx=g.nr, ny=g.nz, dx=g.dr, sx=g.nr * g.dr, sy=g.nz * g.dr)
    # rasterize_scene builds its background on grid.device, so a CPU grid
    # here yields a CPU eps even for empty geometry -- which is why the
    # vacuum test passed while a real example crashed.
    grid = Grid(nx=g.nr, ny=g.nz, dx=g.dr, dtype=torch.float64,
                device=getattr(sim, "device", "cpu") or "cpu")
    eps_items, chi3_items = build_scene_items(items, gs, 2)
    if chi3_items:
        raise _NIE("cylindrical runs with chi3 (Kerr) media are not supported")
    bg, _ = sim._default.resolve()
    return rasterize_scene(eps_items, grid, background=bg).eps_tm


def _absorber_cells(sim, g: CylGrid) -> tuple[int, int]:
    """(n_abs_r, n_abs_z) from the boundary layers. The stepper puts the
    radial ramp on the outer r side only (the axis is not a boundary) and on
    both z ends."""
    nr = nz = 0
    for bl in getattr(sim, "boundary_layers", []):
        d = int(round(float(bl.thickness) / g.dr))
        drc = getattr(bl, "direction", None)
        if drc in (None, -1, 3):        # ALL or R
            nr = max(nr, d)
        if drc in (None, -1, 2):        # ALL or Z
            nz = max(nz, d)
    return nr, nz


# ---------------------------------------------------------------- sources
def _src_terms(sim, g: CylGrid):
    """([(field, i, j, complex factor)], profile) for step_bor."""
    specs = list(getattr(sim, "_sources", []))
    if not specs:
        raise _NIE("a cylindrical run needs at least one source")
    terms, prof = [], None
    for spec in specs:
        cls = type(spec).__name__
        if cls != "Source":
            raise _NIE(f"{cls} in a cylindrical run is not supported "
                       "(point mp.Source only)")
        comp = spec.component
        if comp not in _FIELD or comp.startswith("H"):
            raise _NIE(f"cylindrical source component {comp!r} is not "
                       "supported (Er / Ep / Ez only)")
        size = spec.size or Vector3()
        if float(size.x) or float(size.z):
            raise _NIE("extended (line/plane) sources in a cylindrical run "
                       "are not supported; use a point mp.Source")
        p = spec.src.profile()
        if prof is None:
            prof = p
        elif not _same_profile(prof, p):
            raise _NIE("cylindrical runs drive one shared time profile; "
                       "sources with differing src= profiles are not wired")
        i, j = g.index(comp, spec.center or Vector3())
        terms.append((_FIELD[comp], i, j, complex(spec.amplitude)))
    return terms, prof


def _same_profile(a, b) -> bool:
    if type(a) is not type(b):
        return False
    keys = ("fcen", "w", "t0", "t_ramp")
    return all(getattr(a, k, None) == getattr(b, k, None) for k in keys)


# -------------------------------------------------------------- the run
class CylRun:
    """Live cylindrical run state: the stepper plus whatever the registered
    monitors and step functions need to sample every step."""

    def __init__(self, sim):
        self.sim = sim
        self.g = g = CylGrid(sim)
        n_abs_r, n_abs_z = _absorber_cells(sim, g)
        # cyl_eps goes through the ordinary rasterizer, so its eps lands on
        # the Simulation's device; the coefficient and state tensors have to
        # follow or step_bor mixes a CUDA eps_r with CPU ri/rh/sigma.
        dev = getattr(sim, "device", "cpu") or "cpu"
        self.c = build_bor_coeffs(g.nr, g.nz, g.dr, int(sim.m),
                                  eps=cyl_eps(sim, g),
                                  n_abs_r=n_abs_r, n_abs_z=n_abs_z,
                                  device=dev)
        self.state = init_bor_state(g.nr, g.nz, device=dev)
        self.terms, self.prof = _src_terms(sim, g)
        self.cutoff = sim._source_cutoff()
        self.t = 0
        self.j_series = []          # source current, for LDOS
        self.dft = None             # {comp: (nfreq, ...) complex} for near2far
        self.dft_freqs = None
        self.ldos = None            # (freqs, [E at the source point])

    @property
    def dt(self) -> float:
        return self.c.dt

    def field(self, comp: str, pt) -> complex:
        i, j = self.g.index(comp, pt)
        return complex(getattr(self.state, _FIELD[comp])[i, j].item())

    def step(self):
        tt = (self.t + 0.5) * self.c.dt
        amp = self.prof.amplitude(tt) if tt < self.cutoff else 0.0
        self.state = step_bor(self.state, self.c, amp, self.terms)
        self.t += 1
        self._sample(amp)

    def _sample(self, amp):
        if self.ldos is not None:
            comp, pt = self.ldos[1], self.ldos[2]
            self.ldos[3].append(self.field(comp, pt).real)
            self.j_series.append(amp)
        if self.dft_freqs is None:
            return
        f = self.dft_freqs
        for comp in ("Er", "Ez", "Ephi", "Hr", "Hphi", "Hz"):
            F = getattr(self.state, comp).cpu().numpy()
            off = 1.0 if comp.startswith("E") else 0.5
            ph = np.exp(-2j * np.pi * f * (self.t - 1 + off) * self.c.dt) * self.c.dt
            contrib = F[None, ...] * ph.reshape((-1,) + (1,) * F.ndim)
            self.dft[comp] = contrib if self.dft[comp] is None \
                else self.dft[comp] + contrib


def _scope_check(sim, step_funcs):
    if getattr(sim, "symmetries", None) or getattr(sim, "_sym_cx", 0) \
            or getattr(sim, "_sym_cy", 0):
        raise _NIE("symmetries in a cylindrical run are not supported "
                   "(the r -> -r mirror is already implicit in the BOR ansatz)")
    if getattr(sim, "k_point", False) not in (False, None):
        raise _NIE("k_point (Bloch) with dimensions=CYLINDRICAL is not "
                   "supported")
    # Step functions are NOT restricted here any more. This used to allow
    # Harminv and reject everything else, because run_cyl ran its own loop that
    # called every step function on every step and understood nothing else --
    # no at_every, no after_sources, no at_end. The cartesian path already has
    # the whole protocol in run_control.plan_hooks/fire_*, so the cylindrical
    # loop now wires itself to that instead of reimplementing a worse version.
    # perturbation_theory.py and ring-cyl.py both use after_sources(Harminv),
    # which is the ordinary meep idiom and was rejected as unsupported.


def run_cyl(sim, step_funcs, until=None, until_after_sources=None,
            n_steps=None):
    """meep ``Simulation.run`` for a cylindrical cell. Returns None (meep)."""
    _scope_check(sim, step_funcs)
    run = getattr(sim, "_cyl", None)
    if run is None:
        run = sim._cyl = CylRun(sim)
    dt = run.dt

    conds, gate, n_total = None, 0.0, n_steps
    stop = until if until is not None else until_after_sources
    if callable(stop) or isinstance(stop, (list, tuple)):
        conds = list(stop) if isinstance(stop, (list, tuple)) else [stop]
        if until is None:
            gate = run.cutoff
        n_total = None
    elif n_total is None:
        if until is not None:
            n_total = max(1, int(round(float(until) / dt)))
        else:
            n_total = max(1, int(round((run.cutoff
                                        + float(until_after_sources or 0)) / dt)))

    begin, per_step, scheduled = plan_hooks(step_funcs, run.cutoff)
    for g in begin:
        _eval_step_func(sim, g, "step")
    t0 = run.t
    fire_scheduled(sim, scheduled, run.t * dt, dt)
    while n_total is None or run.t - t0 < n_total:
        run.step()
        t = run.t * dt
        fire_per_step(sim, per_step, t, dt)
        fire_scheduled(sim, scheduled, t, dt)
        if conds is not None:
            if t >= gate - 1e-9 and any(bool(c(sim)) for c in conds):
                break
    for func in step_funcs:      # meep run-end protocol: todo="finish"
        _eval_step_func(sim, func, "finish")


# --------------------------------------------------------- field probes
def get_field_point(sim, c, pt):
    """meep ``Simulation.get_field_point`` on the cylindrical state."""
    run = getattr(sim, "_cyl", None)
    if run is None:
        return 0.0
    name = c if isinstance(c, str) else ""
    if name.startswith("B"):
        name = "H" + name[1:]            # mu = 1 engine
    if name not in _FIELD:
        raise _NIE(f"cylindrical get_field_point: component {c!r} is not "
                   "carried by the BOR state (Er/Ep/Ez/Hr/Hp/Hz)")
    return run.field(name, pt)


# ---------------------------------------------------------------- LDOS
def dft_ldos(sim, fcen, df, nfreq, component="Ez"):
    """meep ``Simulation.dft_ldos`` for a cylindrical run: records the field
    at the (single) source point and the driving current, so ``ldos_data``
    is core/ldos.ldos_spectrum of the pair."""
    run = getattr(sim, "_cyl", None)
    if run is None:
        run = sim._cyl = CylRun(sim)
    spec = sim._sources[0]
    comp = spec.component
    freqs = np.array([fcen]) if nfreq == 1 else \
        np.linspace(fcen - df / 2, fcen + df / 2, nfreq)
    run.ldos = [freqs, comp, spec.center or Vector3(), []]
    return run


def ldos_data(sim):
    run = getattr(sim, "_cyl", None)
    if run is None or run.ldos is None:
        raise RuntimeError("no cylindrical dft_ldos monitor was added")
    freqs, _, _, series = run.ldos
    return list(ldos_spectrum(np.array(series), np.array(run.j_series),
                              run.dt, freqs))


# ------------------------------------------------------------- near2far
def add_near2far(sim, fcen, df, nfreq, *regions):
    """meep ``Simulation.add_near2far`` for a cylindrical run: the regions
    only fix the (r, z) box the surface-equivalence integral runs over; the
    DFT itself accumulates on the whole (r, z) plane."""
    run = getattr(sim, "_cyl", None)
    if run is None:
        run = sim._cyl = CylRun(sim)
    if run.dft_freqs is not None:
        raise _NIE("one near2far monitor per cylindrical run is supported")
    freqs = np.array([fcen]) if nfreq == 1 else \
        np.linspace(fcen - df / 2, fcen + df / 2, nfreq)
    run.dft_freqs = freqs
    run.dft = {k: None for k in ("Er", "Ez", "Ephi", "Hr", "Hphi", "Hz")}
    g = run.g
    rs, zs = [], []
    for reg in regions:
        c = reg.center
        s = reg.size or Vector3()
        rs += [float(c.x) - float(s.x) / 2, float(c.x) + float(s.x) / 2]
        zs += [float(c.z) - float(s.z) / 2, float(c.z) + float(s.z) / 2]
    run.n2f_box = (int(round(max(rs) / g.dr)),
                   int(round((min(zs) + g.sz / 2) / g.dr)),
                   int(round((max(zs) + g.sz / 2) / g.dr)))
    return run


def get_farfield(sim, n2f, point, tol=None):
    """meep ``Simulation.get_farfield``: the six far fields (Er, Ep, Ez, Hr,
    Hp, Hz) at ``point`` = Vector3(r, 0, z) in the far zone.

    Far-zone spherical fields E_theta, E_phi come from
    core/near2far_cyl.farfield_bor_m_complex; the radiation prefactor is
    i k exp(i k R) / (4 pi R) and H = rhat x E (Z0 = 1 in meep units),
    i.e. H_theta = -E_phi, H_phi = E_theta. At the phi = 0 half-plane
    thetahat = cos(theta) rhat - sin(theta) zhat, so
    (F_r, F_p, F_z) = (F_theta cos, F_phi, -F_theta sin).
    """
    run = getattr(sim, "_cyl", None)
    if run is None or run.dft_freqs is None:
        raise RuntimeError("no cylindrical near2far monitor was added")
    if len(run.dft_freqs) != 1:
        raise _NIE("cylindrical get_farfield returns one frequency; "
                   "multi-frequency near2far is not wired")
    r, z = float(point.x), float(point.z)
    R = math.hypot(r, z)
    theta = math.atan2(r, z)
    ir, jz0, jz1 = run.n2f_box
    k = 2 * math.pi * float(run.dft_freqs[0])
    Et, Epf = farfield_bor_m_complex(run.dft, run.c, int(sim.m), ir=ir,
                                     jz0=jz0, jz1=jz1,
                                     thetas=np.array([theta]), k=k)
    Et, Epf = Et[0, 0], Epf[0, 0]
    pref = 1j * k * np.exp(1j * k * R) / (4 * np.pi * R)
    Et, Epf = pref * Et, pref * Epf
    Ht, Hpf = -Epf, Et
    ct, st = math.cos(theta), math.sin(theta)
    return [Et * ct, Epf, -Et * st, Ht * ct, Hpf, -Ht * st]
