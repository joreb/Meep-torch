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
  * cells: (r, z) via core/stepper_bor, or zero z extent -- a run with no
    z dependence -- via core/stepper_cyl (radial, {Ez, Hr, Hphi}, m != 0)
  * geometry: mp.Block only (r-z rectangles), non-dispersive media
  * sources: point Er/Ep/Ez sources sharing one time profile
  * boundaries: PML/Absorber -> the stepper's matched absorber ramp
  * step functions: the full meep protocol, shared with the cartesian
    path via run_control.plan_hooks (at_every, at_beginning, at_end,
    after_sources, to_appended, Harminv, stop_when_fields_decayed)
  * monitors: add_near2far / get_farfield, dft_ldos
"""
import math

import numpy as np
import torch

from ..core.ldos import ldos_spectrum
from ..core.near2far_cyl import farfield_bor_m_complex
from ..core.sources import source_measure
from ..core.stepper_bor import build_bor_coeffs, init_bor_state, step_bor
from ..core.stepper_cyl import (build_cyl_coeffs, init_cyl_state,
                                step_cyl)
from .run_control import (_eval_step_func, fire_per_step,
                          fire_scheduled, plan_hooks)
from .vec import Vector3
from ..core.monitors import dft_time_offset

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
        if self.sr <= 0 or self.sz < 0:
            raise _NIE("cylindrical cells need a positive r extent "
                       "(cell_size = Vector3(sr, 0, sz), or Vector3(sr, 0, 0) "
                       "for a run with no z dependence)")
        # A ZERO z EXTENT IS A LEGAL MEEP CELL, not a malformed one. Upstream
        # ring-cyl.py passes Vector3(sr, 0, 0): a body of revolution with no z
        # dependence, solved in r and the azimuthal index m alone. pymeep runs
        # it and reports "Working in Cylindrical dimensions". This used to
        # raise and point at core/stepper_cyl.py, which is the radial stepper
        # that was already written for exactly this tutorial -- implemented,
        # unit-tested, and reachable from nothing.
        self.radial = self.sz == 0.0
        self.nr = max(3, int(round(self.sr / self.dr)))
        self.nz = 1 if self.radial else max(3, int(round(self.sz / self.dr)))

    def index(self, comp: str, pt) -> tuple[int, int]:
        """Nearest (i, j) index of meep point ``pt`` on component ``comp``'s
        staggered subgrid, clamped to that subgrid's shape."""
        offr, offz = _OFF[comp]
        i = int(round(float(pt.x) / self.dr - offr))
        j = 0 if self.radial else int(
            round((float(pt.z) + self.sz / 2.0) / self.dr - offz))
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
        # CUTOFF 0 MEANS "NEVER ENDS", NOT "ENDED AT t = 0".
        # _source_cutoff returns the latest GaussianPulse.t_end, and its
        # docstring says "CW never ends -> 0" -- a sentinel for infinity. Read
        # as a turn-off time it silenced every continuous source in a
        # cylindrical run: tt < 0.0 is never true, so the amplitude was 0.0 for
        # every step, the run completed, the fields stayed exactly zero and
        # nothing raised. The cartesian path does not gate on this at all,
        # which is why only cylindrical runs were affected.
        amp = (self.prof.amplitude(tt)
               if (not self.cutoff or tt < self.cutoff) else 0.0)
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
            off = dft_time_offset(comp)
            ph = np.exp(-2j * np.pi * f * (self.t - 1 + off) * self.c.dt) * self.c.dt
            contrib = F[None, ...] * ph.reshape((-1,) + (1,) * F.ndim)
            self.dft[comp] = contrib if self.dft[comp] is None \
                else self.dft[comp] + contrib


#: The radial (d/dz = 0) run carries the {Ez, Hr, Hphi} set only -- the
#: complementary {Er, Ep, Hz} have no equations in core/stepper_cyl.
_RADIAL_SET = ("Ez", "Hr", "Hp")


class CylRunRadial(CylRun):
    """A cylindrical run with no z dependence, on core/stepper_cyl.

    Everything that is not the stepper itself -- the source cutoff, the step
    counter, the LDOS/DFT sampling, and the whole meep step-function protocol
    that run_cyl now shares with the cartesian path -- is inherited. Only the
    three things that are genuinely different are overridden: what is built,
    what one step does, and how a point is probed.
    """

    def __init__(self, sim):
        self.sim = sim
        self.g = g = CylGrid(sim)
        n_abs_r, _ = _absorber_cells(sim, g)      # the axis is not a boundary
        dev = getattr(sim, "device", "cpu") or "cpu"
        # eps comes from the SAME rasterizer as the (r, z) case, evaluated on
        # its single z row. A shim that assembles a permittivity profile by
        # hand is a shim computing physics; the raw example may do that, this
        # may not.
        # The rasterizer's Grid demands ny >= 3, and a radial run has one z
        # row. Rasterize on three identical rows and take the middle one: the
        # geometry has no z dependence here (a cylindrical Block spans z), so
        # every row is the same and the middle one is free of any edge
        # treatment the outer two might carry.
        import copy
        g3 = copy.copy(g)
        g3.nz = 3
        eps = cyl_eps(sim, g3)[:, 1].contiguous()
        # build_cyl_coeffs raises for m = 0 on its own (the axis update lands
        # with W7.5), which is the right refusal to inherit rather than
        # reimplement: better to raise than to return a plausible field.
        self.c = build_cyl_coeffs(g.nr, g.dr, int(sim.m), eps=eps,
                                  n_absorber=n_abs_r, device=dev)
        self.state = init_cyl_state(g.nr, device=dev)
        self.src_ix, self.prof = _radial_source(sim, g)
        # THE SOURCE CARRIES A MEASURE, derived rather than typed. meep's rule
        # (src/sources.cpp:501) multiplies the amplitude by the resolution once
        # for every grid direction the source is a DELTA in and that the grid
        # actually has. A radial run's directions are R and Z; sz = 0 makes Z a
        # nosize direction, so a point source is a delta in R alone and the
        # measure is 1/dr. step_cyl takes a bare amplitude, so it is applied
        # here, through the same core.source_measure every cartesian path uses
        # -- a second copy of the rule is a second thing that can disagree, and
        # a wrong measure scales the field smoothly with nothing raising.
        self.measure = source_measure((1,), g.dr, ndim=1)
        self.cutoff = sim._source_cutoff()
        self.t = 0
        self.j_series = []
        self.dft = None
        self.dft_freqs = None
        self.ldos = None

    def step(self):
        tt = (self.t + 0.5) * self.c.dt
        # CUTOFF 0 MEANS "NEVER ENDS", NOT "ENDED AT t = 0".
        # _source_cutoff returns the latest GaussianPulse.t_end, and its
        # docstring says "CW never ends -> 0" -- a sentinel for infinity. Read
        # as a turn-off time it silenced every continuous source in a
        # cylindrical run: tt < 0.0 is never true, so the amplitude was 0.0 for
        # every step, the run completed, the fields stayed exactly zero and
        # nothing raised. The cartesian path does not gate on this at all,
        # which is why only cylindrical runs were affected.
        amp = (self.prof.amplitude(tt)
               if (not self.cutoff or tt < self.cutoff) else 0.0)
        self.state = step_cyl(self.state, self.c, amp * self.measure,
                              self.src_ix)
        self.t += 1
        self._sample(amp)

    def field(self, comp: str, pt) -> complex:
        if comp not in _RADIAL_SET:
            raise _NIE(f"component {comp!r} is not carried by a radial "
                       "cylindrical run (Ez / Hr / Hp only; the cell has no "
                       "z extent, so there is no Er / Ep / Hz to report)")
        i, _ = self.g.index(comp, pt)
        arr = getattr(self.state, _FIELD[comp])
        return complex(arr[min(i, arr.shape[0] - 1)].item())


def _radial_source(sim, g: CylGrid):
    """(src_ix, profile) for step_cyl, reusing the (r, z) source validation."""
    terms, prof = _src_terms(sim, g)
    if len(terms) != 1:
        raise _NIE("a radial cylindrical run drives one point source "
                   f"(got {len(terms)})")
    comp, i, _j, factor = terms[0]
    if comp != "Ez":
        raise _NIE(f"radial cylindrical source component {comp!r} is not "
                   "supported (step_cyl drives Dz, so mp.Ez only)")
    if abs(complex(factor) - 1.0) > 1e-12:
        raise _NIE("a radial cylindrical source carries no amplitude factor "
                   f"(got {factor!r})")
    return i, prof


def _cyl_centres(arr, comp, target):
    """A cylindrical component array on the grid meep reports it on.

    meep has no ``get_array_metadata`` in cylindrical, so the grid was
    measured rather than read: probing ``get_field_point`` puts every
    component on the CELL-CENTRE grid, r = (i + 0.5) dr and
    z = (j + 0.5) dr - sz/2, to 1e-15 for all six -- the same convention
    ``mp/simulation.py`` already applies to cartesian runs. An axis where
    ``_OFF`` is 0.0 holds nodes and is averaged onto the midpoints; an axis
    already at the half cell is left alone. Either way the axis comes out one
    short of meep's length, so the outer row is edge-clamped, as ``_to_centres``
    does.
    """
    import torch
    for ax, off in enumerate(_OFF[comp][:arr.dim()]):
        if off == 0.0:
            n = arr.shape[ax] - 1
            arr = 0.5 * (arr.narrow(ax, 0, n) + arr.narrow(ax, 1, n))
    for ax, n in enumerate(target):
        while arr.shape[ax] < n:
            arr = torch.cat([arr, arr.narrow(ax, arr.shape[ax] - 1, 1)], dim=ax)
    return arr


def cyl_array(sim, component, center=None, size=None):
    """meep ``Simulation.get_array`` on a cylindrical run.

    Reached because the hook rewire made at_beginning(mp.output_epsilon) fire
    at all: ring-cyl.py opens with it, and the cartesian get_array walks
    self.gs, which is None here. The permittivity a cylindrical run actually
    solves with lives on the stepper coefficients, so report THAT rather than
    rasterizing a second copy -- two arrays that can disagree is the defect
    this codebase keeps paying for.
    """
    if center is not None or size is not None:
        raise _NIE("get_array(center=, size=) sub-volumes are not supported "
                   "for a cylindrical run")
    name = component if isinstance(component, str) else ""
    run = getattr(sim, "_cyl", None)
    if name == "Dielectric":
        g = run.g if run is not None else CylGrid(sim)
        if g.radial:
            # The radial coefficients carry the one eps the stepper solves
            # with, so report that rather than rasterizing a second copy.
            # Only the RADIAL table has a plain `.eps`; build_bor_coeffs
            # splits its permittivity across eps_r / eps_p / eps_z, so
            # reading `run.c.eps` unconditionally would raise on every
            # (r, z) run -- a fault introduced by writing the radial case
            # first and assuming the other matched it.
            if run is not None:
                return run.c.eps.detach().cpu().numpy()
            import copy
            g3 = copy.copy(g)
            g3.nz = 3
            return cyl_eps(sim, g3)[:, 1].detach().cpu().numpy()
        return cyl_eps(sim, g).detach().cpu().numpy()
    if name in _FIELD:
        if run is None:
            raise RuntimeError("run() before get_array(fields)")
        arr = getattr(run.state, _FIELD[name])
        g = run.g
        if g.radial:
            return _cyl_centres(arr, name, (g.nr,)).detach().cpu().numpy()
        # meep returns the (r, z) array z-major -- shape (nz, nr), measured
        # against an asymmetric cell so the two lengths cannot be confused.
        return _cyl_centres(arr, name, (g.nr, g.nz)).T.detach().cpu().numpy()
    raise _NIE(f"cylindrical get_array: component {component!r} is not "
               "carried by the state (Dielectric / Er / Ep / Ez / Hr / Hp / Hz)")


def cyl_energy_in_box(sim, kind, box=None, center=None, size=None):
    """meep ``*_energy_in_box`` on a cylindrical run.

    THE MEASURE IS THE POINT. A cylindrical cell's volume element is
    2 pi r dr dz, not dr dz, so an energy integral that sums cells uniformly
    is wrong by a factor that varies across the domain -- it under-weights the
    outside and over-weights the axis, which no single constant can repair.
    perturbation_theory.py divides its own numerator, written with explicit
    2 pi r factors, by this integral, so the two measures have to agree or the
    ratio is silently wrong rather than loudly broken.

    For a radial run (d/dz = 0) the z integral is per unit length and the
    element is 2 pi r dr.
    """
    run = getattr(sim, "_cyl", None)
    if run is None:
        raise RuntimeError("run() before an energy integral")
    g = run.g
    if not g.radial:
        raise _NIE("energy integrals on an (r, z) cylindrical run are not "
                   "wired yet (the radial case is)")
    if kind != "electric":
        raise _NIE(f"{kind}_energy_in_box on a radial cylindrical run is not "
                   "wired (the radial H pair lives on staggered r grids)")
    if box is not None:
        center, size = box.center, box.size
    ez = run.state.Ez.detach().cpu().numpy()
    eps = run.c.eps.detach().cpu().numpy()
    r = np.arange(len(ez)) * g.dr
    i0, i1 = 0, len(ez)
    if center is not None or size is not None:
        c = float((center or Vector3()).x)
        w = float((size or Vector3()).x)
        i0 = max(0, int(round((c - w / 2.0) / g.dr)))
        i1 = min(len(ez), int(round((c + w / 2.0) / g.dr)) + 1)
    sl = slice(i0, max(i1, i0 + 1))
    u = 0.5 * eps[sl] * np.abs(ez[sl]) ** 2
    return float(np.sum(u * 2.0 * np.pi * r[sl]) * g.dr)


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
        cls = CylRunRadial if CylGrid(sim).radial else CylRun
        run = sim._cyl = cls(sim)
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
