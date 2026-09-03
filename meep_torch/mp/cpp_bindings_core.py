# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""C++ SWIG-binding surface parity: meep's core object model over the tensor
engine. `structure` holds the rasterized materials and stepping coefficients,
`fields(structure)` owns the field state and drives step_tm one step at a
time (bit-for-bit the same evolution as core.run.run_tm), `boundary_region`
encodes a PML/absorber band, `h5file` is a thin h5py wrapper (h5py imported
lazily: it is not a dependency). Plus the unowned lowercase leftovers:
absorber, cond_profile, flux_vol, initialize.

Genuine adapters only; whole-domain tensors -- chunked domain decomposition,
MPI comms, eigenmode/CW solves and HDF5 field output raise NotImplementedError.
structure/fields are the canonical engine owners behind mp.Simulation
(init_sim builds structure then fields(structure); run() steps the fields).
The stepper follows the fields' allocated component set (step_tm, or step_te
built from a Simulation); the direct meep-style constructor surface
(set_epsilon, add_point_source, ...) remains TM."""
import math
import time

import numpy as np
import torch

from ..core.coeffs import build_te_coeffs, build_tm_coeffs
from ..core.grid import Grid
from ..core.monitors import DFTFieldsMonitor, FluxMonitorTM
from ..core.pml import absorber_sigma
from ..core.run import _run_chunk
from ..core.sources import SoftSource
from ..core import stepper as _tm_phases
from ..core import stepper_te as _te_phases
from ..core.stepper import init_tm_state, step_tm
from ..core.stepper_te import init_te_state, step_te
from .coords import GridSpec, cell_index, cell_index_y, pml_npml, span_index_range
from .run_control import Volume
from .simulation import Simulation, X, Y
from .vec import Vector3

_TM_COMPONENTS = ("Ez", "Hx", "Hy", "Dz")
_TE_COMPONENTS = ("Hz", "Ex", "Ey", "Dx", "Dy")


def _stub_raises(cls, why: str, names: str):
    """Attach honest NotImplementedError stubs for a group of SWIG methods."""
    for name in names.split():
        def m(self, *a, _n=name, _w=why, _c=cls.__name__, **k):
            raise NotImplementedError(f"{_c}.{_n} not supported: {_w}")
        m.__name__ = name
        m.__doc__ = f"Not supported: {why}."
        setattr(cls, name, m)


def _gs_of(grid: Grid) -> GridSpec:
    return GridSpec(grid.nx, grid.ny, grid.dx, grid.nx * grid.dx, grid.ny * grid.dx)


def _clamp_ij(pt: Vector3, gs: GridSpec) -> tuple[int, int]:
    ix = max(0, min(gs.nx - 1, cell_index(pt.x, gs)))
    iy = max(0, min(gs.ny - 1, cell_index_y(pt.y, gs)))
    return ix, iy


def _plane_spec(d: int, where: Volume, gs: GridSpec) -> tuple[bool, int, int, int]:
    """Direction + zero-thickness Volume -> (x_normal, idx, lo, hi) line span."""
    c = where.center or Vector3()
    s = where.size or Vector3()
    if d == X:
        idx = cell_index(c.x, gs)
        lo, hi = span_index_range(max(0.0, c.y - s.y / 2 + gs.sy / 2),
                                  min(gs.sy, c.y + s.y / 2 + gs.sy / 2), gs, "y")
        return True, idx, lo, hi
    if d == Y:
        idx = max(0, min(gs.ny - 1, cell_index_y(c.y, gs)))
        lo, hi = span_index_range(max(0.0, c.x - s.x / 2 + gs.sx / 2),
                                  min(gs.sx, c.x + s.x / 2 + gs.sx / 2), gs, "x")
        return False, idx, lo, hi
    raise ValueError(f"direction must be mp.X or mp.Y, got {d!r}")


class boundary_region:
    """meep::boundary_region -- one PML (or absorber) band spec. `direction`
    / `side` of None mean all directions/sides (the engine applies a uniform
    band on every wall; per-wall bands are not supported and raise when a
    structure is built with a direction-restricted region)."""

    NOTHING_SPECIAL, PML = 0, 1

    def __init__(self, kind=PML, thickness=1.0, direction=None, side=None,
                 strength=1.0):
        self.kind = kind
        self.thickness = float(thickness)
        self.direction = direction
        self.side = side
        self.strength = float(strength)

    def check_ok(self, s=None):
        if not (self.thickness >= 0 and math.isfinite(self.thickness)):
            raise ValueError(f"bad boundary_region thickness {self.thickness}")

    def boundary_region_assign(self, other):
        """SWIG operator=: copy the other region's spec into self."""
        self.__dict__.update(vars(other))
        return self

    def br_apply(self, s):
        """Apply this band to a structure (rebuilds its coefficients)."""
        s._apply_boundary(self)

    def npml(self, gs: GridSpec) -> int:
        return pml_npml(self.thickness, gs)


class absorber(boundary_region):
    """meep::absorber -- a plain conductivity absorber band (mapped to the
    engine's quadratic sigma_d = sigma_b ramp, core.pml.absorber_sigma)."""

    def __init__(self, thickness=1.0, direction=None, side=None):
        super().__init__(kind=boundary_region.NOTHING_SPECIAL,
                         thickness=thickness, direction=direction, side=side)


class cond_profile:
    """SWIG wrapper of the C PML profile function pointer: a callable
    u in [0,1] -> profile value (default: meep's quadratic u^2)."""

    def __init__(self, fn=None):
        self.fn = fn if fn is not None else (lambda u: u * u)

    def __call__(self, u):
        return self.fn(u)


class structure:
    """meep::structure -- the materials half of a simulation: eps/conductivity
    /mu/chi2/chi3 tensors + dispersive poles + boundary bands, compiled to the
    engine's TMCoeffs (rebuilt lazily after every set_* mutation).

    Construct from a shim `mp.Simulation` (adopts its rasterized materials and
    coefficient tensors unchanged) or meep-style: structure(gv, eps, br, ...)
    with `gv` a core Grid (or anything with nx/ny/dx), `eps` a scalar, an
    (nx, ny) array/tensor, or a callable(Vector3) -> eps evaluated at cell
    centers in meep (center-origin) coordinates."""

    def __init__(self, gv=None, eps=None, br=None, s=None, num_chunks=1,
                 Courant=0.5):
        if num_chunks not in (0, 1):
            raise NotImplementedError(
                "num_chunks > 1: chunked domain decomposition is out of scope "
                "(whole-domain tensors)")
        if s is not None and getattr(s, "direction", None) is not None:
            raise NotImplementedError(
                "structure(symmetry=...): pass symmetries to mp.Simulation and "
                "build structure(sim) instead")
        self._outdir = ""
        self._chi2 = None
        self._nabs = 0
        self._eps_te = None
        if isinstance(gv, Simulation):
            self._adopt(gv)
            return
        if gv is None:
            raise ValueError("structure needs a grid volume (core Grid) or an mp.Simulation")
        if not isinstance(gv, Grid):
            gv = Grid(nx=int(gv.nx), ny=int(gv.ny), dx=float(gv.dx), courant=Courant)
        self.grid = gv
        self.gs = _gs_of(gv)
        self._eps = self._field(eps if eps is not None else 1.0)
        self._sd = self._sb = self._mu = None
        self._chi3 = None
        self._poles, self._pole_masks = [], []
        self.npml = 0
        self._y_pmc = False
        self._y_mirror = 0
        self._x_mirror = 0
        self._coeffs = self._coeffs_tm = self._coeffs_te = None
        for region in (br if isinstance(br, (list, tuple)) else [br] if br else []):
            self._apply_boundary(region)

    def _adopt(self, sim: Simulation):
        if sim.dimensions == 3:
            raise NotImplementedError("structure(sim): 2D Simulations only")
        if sim.structure is None:
            sim._init_structure()
        sim._coeffs                       # compile into sim.structure's slot first
        self.__dict__.update(sim.structure.__dict__)  # shared slots -> bit-for-bit with sim runs

    @classmethod
    def _from_sim(cls, sim: Simulation):
        """The structure Simulation._init_structure owns, built from the
        just-rasterized sim attributes (sim folded any Absorber ramp into
        _cond already -> _nabs stays 0)."""
        st = cls.__new__(cls)
        st._outdir = ""
        st._chi2 = None
        st._nabs = 0
        st.grid, st.gs = sim.grid, sim.gs
        # both orientations stored; fields' component set picks at compile
        # time (meep: structure holds the full eps, components choose)
        st._eps = sim._rm.eps_tm
        st._eps_te = sim._rm.eps_te_xx
        st._chi3 = sim._rm.chi3
        st._poles = list(sim._rm.poles)
        st._pole_masks = list(sim._rm.pole_masks)
        st._sd, st._sb, st._mu = sim._cond if sim._cond is not None else (None,) * 3
        st.npml = sim._npml
        st._y_pmc = sim._y_pmc
        st._y_mirror = sim._sym_py if sim._sym_cy else 0
        st._x_mirror = sim._sym_px if sim._sym_cx else 0
        st._coeffs = st._coeffs_tm = st._coeffs_te = None
        return st

    def _field(self, v) -> torch.Tensor:
        g = self.grid
        if callable(v) and not isinstance(v, torch.Tensor):
            gs = self.gs
            out = g.zeros(g.nx, g.ny)
            for i in range(g.nx):
                x = (i + 0.5) * gs.dx - gs.sx / 2
                for j in range(g.ny):
                    out[i, j] = float(v(Vector3(x, (j + 0.5) * gs.dx - gs.sy / 2)))
            return out
        if isinstance(v, (int, float)):
            return torch.full((g.nx, g.ny), float(v), dtype=g.dtype, device=g.device)
        t = torch.as_tensor(np.asarray(v), dtype=g.dtype, device=g.device)
        if t.shape != (g.nx, g.ny):
            raise ValueError(f"expected shape {(g.nx, g.ny)}, got {tuple(t.shape)}")
        return t

    def _apply_boundary(self, region: boundary_region):
        if region.direction is not None or region.side is not None:
            raise NotImplementedError(
                "per-direction/per-side boundary bands: the engine applies a "
                "uniform band on every wall")
        n = pml_npml(region.thickness, self.gs)
        if isinstance(region, absorber) or region.kind != boundary_region.PML:
            self._nabs = max(self._nabs, n)
        else:
            self.npml = max(self.npml, n)
        self._coeffs = self._coeffs_tm = self._coeffs_te = None

    @property
    def coeffs(self):
        """Default (TM component set) view; fields asks coeffs_for() with its
        allocated set -- meep style: the structure holds the materials, the
        required components choose the compiled form."""
        return self.coeffs_for(_TM_COMPONENTS)

    def coeffs_for(self, components):
        te = "Hz" in components
        slot = "_coeffs_te" if te else "_coeffs_tm"
        cached = getattr(self, slot, None)
        if cached is not None:
            return cached
        # legacy single-slot cache (adopted structures): honor it for TM
        if not te and getattr(self, "_coeffs", None) is not None:
            setattr(self, slot, self._coeffs)
            return self._coeffs
        extra = {}
        sd, sb, mu = self._sd, self._sb, self._mu
        if self._nabs:
            g = self.grid
            kw = dict(dtype=g.dtype, device=g.device)
            ramp = absorber_sigma(g.nx, self._nabs, g.dx, **kw)[:, None].expand(g.nx, g.ny).clone()
            if not self._y_pmc:
                ramp = torch.maximum(
                    ramp, absorber_sigma(g.ny, self._nabs, g.dx, **kw)[None, :].expand(g.nx, g.ny))
            sd = ramp if sd is None else sd + ramp
            sb = ramp if sb is None else sb + ramp
        if sd is not None or sb is not None or mu is not None:
            extra = dict(sigma_d=sd, sigma_b=sb, mu=mu)
        if te:
            eps = self._eps_te if self._eps_te is not None else self._eps
            built = build_te_coeffs(
                self.grid, eps_inf=eps, poles=self._poles,
                pole_masks=self._pole_masks or None, chi3=self._chi3,
                npml=self.npml, **extra)
        else:
            built = build_tm_coeffs(
                self.grid, eps_inf=self._eps, poles=self._poles,
                pole_masks=self._pole_masks or None, chi2=self._chi2,
                chi3=self._chi3, npml=self.npml, y_pmc=self._y_pmc,
                y_mirror=self._y_mirror, x_mirror=self._x_mirror, **extra)
        setattr(self, slot, built)
        return built

    # --- material mutation (each invalidates the compiled coefficients) ---
    def set_epsilon(self, eps, use_anisotropic_averaging=False, tol=1e-4,
                    maxeval=0):
        self._eps = self._field(eps)
        self._coeffs = self._coeffs_tm = self._coeffs_te = None

    def set_materials(self, mat_func=None, *a, **k):
        if mat_func is None or not (callable(mat_func) or isinstance(mat_func, (int, float))):
            raise NotImplementedError(
                "set_materials with geometric objects: build via mp.Simulation "
                "and structure(sim); a callable/scalar maps to set_epsilon")
        self.set_epsilon(mat_func)

    def set_chi1inv(self, c=None, fn=None, **k):
        if fn is None:
            c, fn = None, c
        self._eps = 1.0 / self._field(fn)
        self._coeffs = self._coeffs_tm = self._coeffs_te = None

    def set_mu(self, mu):
        self._mu = self._field(mu)
        self._coeffs = self._coeffs_tm = self._coeffs_te = None

    def set_conductivity(self, c, cond):
        name = c if isinstance(c, str) else ""
        if name.startswith(("E", "D")):
            self._sd = self._field(cond)
        elif name.startswith(("H", "B")):
            self._sb = self._field(cond)
        else:
            raise ValueError(f"set_conductivity component {c!r}: expected an E/D or H/B component")
        self._coeffs = self._coeffs_tm = self._coeffs_te = None

    def _chi_field(self, args):
        # SWIG overloads: set_chiN(value) or set_chiN(component, value)
        return self._field(args[-1])

    def set_chi2(self, *args):
        self._chi2 = self._chi_field(args)
        self._coeffs = self._coeffs_tm = self._coeffs_te = None

    def set_chi3(self, *args):
        self._chi3 = self._chi_field(args)
        self._coeffs = self._coeffs_tm = self._coeffs_te = None

    def add_susceptibility(self, sus, field_type=None, mask=None):
        """Append one dispersive pole. `sus` is an mp Susceptibility (via its
        to_pole()) or an engine pole tuple (fn, gamma, sigma[, fc])."""
        pole = sus.to_pole() if hasattr(sus, "to_pole") else tuple(sus)
        if field_type is not None and str(field_type).startswith(("H", "B")):
            raise NotImplementedError("magnetic susceptibilities: E-field poles only here")
        self._poles.append(tuple(pole))
        self._pole_masks.append(mask if mask is None else self._field(mask))
        if any(m is None for m in self._pole_masks):
            # build_tm_coeffs wants all-or-no masks; None -> uniform
            self._pole_masks = [torch.ones_like(self._eps) if m is None else m
                                for m in self._pole_masks]
        self._coeffs = self._coeffs_tm = self._coeffs_te = None

    def remove_susceptibilities(self):
        self._poles, self._pole_masks = [], []
        self._coeffs = self._coeffs_tm = self._coeffs_te = None

    # --- queries ---
    def _at(self, pt: Vector3, t: torch.Tensor) -> float:
        ix, iy = _clamp_ij(pt, self.gs)
        return float(t[ix, iy])

    def get_eps(self, pt):
        return self._at(pt, self._eps)

    def get_inveps(self, pt):
        return 1.0 / self._at(pt, self._eps)

    def get_chi1inv(self, c=None, d=None, pt=None, **k):
        pt = next((v for v in (pt, d, c) if isinstance(v, Vector3)), None)
        if pt is None:
            raise TypeError("get_chi1inv needs a point (Vector3)")
        return 1.0 / self._at(pt, self._eps)

    def get_mu(self, pt):
        return 1.0 if self._mu is None else self._at(pt, self._mu)

    def max_eps(self):
        return float(self._eps.max())

    def has_chi(self, c=None, d=None):
        return bool(self._poles) or self._chi2 is not None or (
            self._chi3 is not None and bool((self._chi3 != 0).any()))

    def mix_with(self, other, frac):
        """meep mix_with: linear interpolation of chi1inv (inverse eps)."""
        if not self.equal_layout(other):
            raise ValueError("mix_with needs an equal grid layout")
        f = float(frac)
        self._eps = 1.0 / ((1.0 - f) / self._eps + f / other._eps)
        self._coeffs = self._coeffs_tm = self._coeffs_te = None

    def equal_layout(self, other):
        return (self.grid.nx, self.grid.ny, self.grid.dx) == \
               (other.grid.nx, other.grid.ny, other.grid.dx)

    def estimated_cost(self, process=0):
        """Stepping cost estimate: cell count (one whole-domain region)."""
        return float(self.grid.nx * self.grid.ny)

    def print_layout(self):
        print(f"structure: 1 whole-domain region, {self.grid.nx}x{self.grid.ny} "
              f"cells, dx={self.grid.dx}")

    def set_output_directory(self, d):
        self._outdir = str(d)


_stub_raises(structure, "chunked domain decomposition / MPI is out of scope "
             "(whole-domain tensors)",
             "dump_chunk_layout load_chunk_layout get_binary_partition "
             "get_chunk_owners get_chunk_volumes")
_stub_raises(structure, "structure HDF5 dump/load is not implemented", "dump load")


class _PointDFT:
    """Running DFT of Ez at one cell (fields.add_dft_pt)."""

    reduce = "sum"

    def __init__(self, grid: Grid, freqs, ix: int, iy: int):
        self.ix, self.iy = ix, iy
        self.dt = grid.dt
        self.freqs = torch.tensor(freqs, dtype=grid.dtype, device=grid.device)

    def sample(self, state, t_idx: int) -> torch.Tensor:
        t = (t_idx + 1.0) * self.dt
        w = 2 * math.pi * self.freqs
        v = state.Ez[self.ix, self.iy]
        return self.dt * torch.stack([torch.cos(w * t) * v, -torch.sin(w * t) * v], dim=-1)


class _DftHandle:
    """Handle returned by fields.add_dft*: names the registered monitors."""

    def __init__(self, f, keys, freqs):
        self._f = f
        self.keys = list(keys)
        self.freqs = list(freqs)

    def flux(self):
        mon, acc = self._f._mon(self.keys[0])
        return mon.flux(acc, self._f.s.grid.dx).detach().cpu().numpy().tolist()


class flux_vol:
    """meep::flux_vol -- instantaneous flux through a plane volume of fields
    `f_` along direction `d_`. flux() integrates the current Poynting flux
    from the live field state; the per-step update hooks belong to the C++
    time loop and raise."""

    def __init__(self, f_, d_, where_):
        self.f_, self.d_, self.where_ = f_, d_, where_

    def flux(self):
        return self.f_._instant_flux(self.d_, self.where_)


_stub_raises(flux_vol, "internal C++ time-loop accumulation hook (fields.step "
             "computes flux from the live state instead)", "update update_half")


class fields:
    """meep::fields -- field state + stepping over a `structure`. step()
    advances one engine step (core.stepper.step_tm) with the structure's
    coefficients; DFT monitors registered via add_dft* accumulate inside
    step() exactly as core.run.run_tm folds them."""

    def __init__(self, s, m=0, beta=0, zero_fields_near_cylorigin=True,
                 components=None):
        if m or beta:
            raise NotImplementedError("cylindrical m / bloch beta: 2D cartesian only")
        self.s = s
        # meep require_component flow: the SOURCES (spec) define the required
        # component set; the structure holds the materials for either set.
        self._components = tuple(components) if components else _TM_COMPONENTS
        c = self._mcoeffs
        self._n_mpoles = (0 if getattr(c, "mc1x", None) is None
                          else c.mc1x.shape[0])
        self._state = None
        for comp in self._components:
            self.alloc_f(comp)
        self._t = 0
        self._sources = []
        self._monitors = {}          # name -> [monitor, accumulator]
        self._outdir = ""

    def alloc_f(self, c):
        """meep fields::alloc_f (fields.cpp:480): ensure component ``c``'s
        tensor is allocated in the state; True when this call newly allocated
        it, False when it already existed (meep's non-null pointer check,
        tensorized). The fused state constructors allocate a full component
        set at once, so the first alloc_f of any component materializes the
        whole set and the rest report False. A component outside the set the
        structure's coefficients define raises (the 2D TM/TE sets are fixed;
        force_all_components is not supported)."""
        if c not in self._components:
            raise NotImplementedError(
                f"component {c!r} is not in this fields' required component "
                f"set {self._components} (fixed 2D sets)")
        if self._state is not None:
            return False
        self._state = self._fresh_state()
        return True

    def _has(self, comp):
        """True when ``comp`` is allocated in the state (meep's null-pointer
        test on f[c]; an absent namedtuple field or a None slot is null)."""
        return getattr(self._state, comp, None) is not None

    @property
    def _mcoeffs(self):
        return self.s.coeffs_for(self._components)

    def _fresh_state(self):
        c = self._mcoeffs
        if "Hz" in self._components:                 # TE component set
            return init_te_state(self.s.grid, n_poles=c.c1x.shape[0])
        return init_tm_state(self.s.grid, n_poles=c.c1.shape[0], y_pmc=c.y_pmc,
                             n_mpoles=self._n_mpoles,
                             y_mirror=c.y_mirror_lo, x_mirror=c.x_mirror_lo)

    # --- stepping (the physics core) ---
    def _phase_mod(self):
        # component-generic dispatch, meep step_generic style: the allocated
        # field set picks the kernel (Ez/Hx/Hy -> TM, Hz/Ex/Ey -> TE)
        return _te_phases if self._has("Hz") else _tm_phases

    def _step_kernel(self):
        """Fused kernel for the allocated component set -- the tensorized
        analog of meep's component-generic step_generic instantiations: the
        dispatch picks the TM or TE fused compute backend (they stay
        separate; no merged generic kernel)."""
        return step_te if self._has("Hz") else step_tm

    def phase_material(self):
        """meep phase_material: phased-material interpolation -- no-op
        (is_phasing() is always False)."""

    def step_db(self, ft):
        """meep step_db(ft): advance B/H (ft="B", from curl E) or D (ft="D",
        from curl H). The CPML recursive-convolution psi updates and sigma
        damping live here, where meep applies its sigma (core.stepper /
        core.stepper_te phase kernels)."""
        self._state = self._phase_mod().step_db(self._state, self._mcoeffs, ft)

    def step_source(self, ft="D"):
        """meep step_source(ft): inject the registered soft current sources at
        the current step -- electric currents into D (ft="D"), magnetic TFSF
        currents (source.h_masks) into B (ft="B")."""
        mod = self._phase_mod()
        m = self._mcoeffs
        t = self._t
        if ft == "B":
            for src in self._sources:
                hm = getattr(src, "h_masks", None)
                if hm:
                    self._state = mod.step_source(self._state, m, "B",
                                                  src.h_amplitude(t), **hm)
            return
        e_srcs = [s for s in self._sources if getattr(s, "mask", None) is not None]
        if not e_srcs:
            return
        if len(e_srcs) == 1:
            self._state = mod.step_source(self._state, m, "D",
                                          e_srcs[0].amplitude(t),
                                          src_mask=e_srcs[0].mask)
            return
        # several sources: sum amp_i(t) * mask_i into one soft injection
        # (meep sums current sources the same way)
        mask = e_srcs[0].amplitude(t) * e_srcs[0].mask
        for src in e_srcs[1:]:
            mask = mask + src.amplitude(t) * src.mask
        self._state = mod.step_source(self._state, m, "D", 1.0, src_mask=mask)

    def update_pols(self, ft="E"):
        """meep update_pols: advance the ADE Lorentz polarizations from the
        pre-update E."""
        self._state = self._phase_mod().update_pols(self._state, self._mcoeffs)

    def update_eh(self, ft):
        """meep update_eh(ft): form H from B and the magnetic polarizations
        (ft="H"), or solve E from D - P including the instantaneous chi2/chi3
        fixed point (ft="E")."""
        self._state = self._phase_mod().update_eh(self._state, self._mcoeffs, ft)

    def step_boundaries(self, ft=None):
        """meep step_boundaries: chunk-boundary / periodic field exchange --
        a no-op here: whole-domain tensors, with the PEC walls and CPML baked
        into the phase kernels' curls."""

    def step(self):
        """One engine step in meep's fields::step phase order (src/step.cpp):
        phase_material -> step_db(B) -> step_source(B) -> update_eh(H) ->
        step_boundaries(H) -> step_db(D) -> step_source(D) -> update_pols ->
        update_eh(E) -> step_boundaries(E). A TE component set (Hz/Ex/Ey;
        Hz staggered on integer steps by the engine's TM-TE duality) enters
        the same cycle at the D half. Identical evolution to core.run.run_tm.
        """
        self.phase_material()
        if self._has("Hz"):
            self.step_db("D")
            self.step_source("D")
            self.update_pols()
            self.update_eh("E")
            self.step_boundaries("E")
            self.step_db("B")
            self.step_source("B")
            self.update_eh("H")
            self.step_boundaries("H")
        else:
            self.step_db("B")
            self.step_source("B")
            self.update_eh("H")
            self.step_boundaries("H")
            self.step_db("D")
            self.step_source("D")
            self.update_pols()
            self.update_eh("E")
            self.step_boundaries("E")
        t = self._t
        for name, (mon, _) in list(self._monitors.items()):
            smp = mon.sample(self._state, t)
            self._fold(name, mon, smp if mon.reduce == "sum" else smp.unsqueeze(0))
        self._t += 1

    def _fold(self, name, mon, out):
        entry = self._monitors[name]
        if entry[1] is None:
            entry[1] = out
        elif mon.reduce == "sum":
            entry[1] = entry[1] + out
        else:
            entry[1] = torch.cat([entry[1], out])

    def step_n(self, k: int):
        """Advance ``k`` engine steps in one chunked call.

        This is the GPU/autograd analog of meep's chunked time-stepping:
        the ``k`` steps run through ``core.run._run_chunk`` -- the same
        kernel ``core.run.run_tm`` wraps and ``torch.utils.checkpoint``
        recomputes chunk-by-chunk on the adjoint backward pass -- so DFT
        monitors fold per chunk instead of per Python-level step.
        ``Simulation._run_until`` sizes ``k`` to the finest step-function /
        stop-condition granularity; with none active a whole run is a
        single chunk. Evolution and DFT accumulation are bit-for-bit
        identical to ``k`` calls of ``step()``.
        """
        if len(self._sources) > 1:
            # ponytail: per-step phase loop for the multi-source combined
            # injection; teach _run_chunk a multi-source path if this gets hot
            for _ in range(k):
                self.step()
            return
        names = list(self._monitors)
        mons = [self._monitors[n][0] for n in names]
        src = self._sources[0] if self._sources else None
        self._state, outs = _run_chunk(self._step_kernel(), self._state,
                                       self._mcoeffs, self._t, k, src, mons)
        for name, mon, out in zip(names, mons, outs):
            self._fold(name, mon, out)
        self._t += k

    def time(self):
        return self._t * self._mcoeffs.dt

    def round_time(self):
        return self._t * self._mcoeffs.dt

    def reset(self):
        """Zero the fields, rewind time and drop DFT accumulations."""
        self._state = self._fresh_state()
        self._t = 0
        for entry in self._monitors.values():
            entry[1] = None

    def zero_fields(self):
        self._state = self._fresh_state()

    # --- sources ---
    def _profile_of(self, src):
        return src.profile() if hasattr(src, "profile") else src

    def add_point_source(self, c, src, pt, amplitude=1.0):
        self._add_mask_source(c, src, *_clamp_ij(pt, self.s.gs),
                              None, None, amplitude)

    def add_volume_source(self, c, src, where, amplitude=1.0):
        v = where if isinstance(where, Volume) else Volume(center=where)
        gs = self.s.gs
        cen, sz = v.center or Vector3(), v.size or Vector3()
        i0, i1 = span_index_range(max(0.0, cen.x - sz.x / 2 + gs.sx / 2),
                                  min(gs.sx, cen.x + sz.x / 2 + gs.sx / 2), gs, "x")
        j0, j1 = span_index_range(max(0.0, cen.y - sz.y / 2 + gs.sy / 2),
                                  min(gs.sy, cen.y + sz.y / 2 + gs.sy / 2), gs, "y")
        self._add_mask_source(c, src, i0, j0, i1, j1, amplitude)

    def _add_mask_source(self, c, src, i0, j0, i1=None, j1=None, amplitude=1.0):
        if c not in ("Ez", "Ex"):                # TM: Jz only (Ex = 1D alias)
            raise NotImplementedError(f"component {c!r}: TM Ez current sources only")
        g = self.s.grid
        mask = g.zeros(g.nx, g.ny)
        mask[i0:(i1 if i1 is not None else i0 + 1),
             j0:(j1 if j1 is not None else j0 + 1)] = 1.0
        if amplitude != 1.0:
            mask = mask * float(amplitude)
        self._sources.append(SoftSource(g, mask, self._profile_of(src)))

    def add_source(self, src):
        """Register a prebuilt engine source (``.amplitude(t)`` + ``.mask``,
        optionally ``.h_masks``) -- the registration path Simulation.init_sim
        uses (meep's fields.add_source analog for tensorized profiles)."""
        self._sources.append(src)

    def remove_sources(self):
        self._sources = []

    def last_source_time(self):
        return max((float(getattr(getattr(s, "profile", None), "t_end", 0.0))
                    for s in self._sources), default=0.0)

    # --- DFT monitors (accumulate inside step(), run_tm-style) ---
    def _register(self, name, mon):
        self._monitors[name] = [mon, None]
        return name

    def _mon(self, name):
        mon, acc = self._monitors[name]
        if acc is None:
            raise RuntimeError(f"monitor {name!r}: no steps accumulated yet")
        return mon, acc

    def add_dft(self, c, freqs, where=None, decimation_factor=0):
        if where is not None:
            raise NotImplementedError("add_dft with a sub-volume: whole-cell only")
        name = f"dft{len(self._monitors)}"
        self._register(name, DFTFieldsMonitor(self.s.grid, list(freqs), component=c))
        return _DftHandle(self, [name], freqs)

    def add_dft_fields(self, components, freqs, where=None, decimation_factor=0):
        if where is not None:
            raise NotImplementedError("add_dft_fields with a sub-volume: whole-cell only")
        keys = []
        base = f"dft{len(self._monitors)}"
        for c in components:
            keys.append(self._register(
                f"{base}:{c}", DFTFieldsMonitor(self.s.grid, list(freqs), component=c)))
        return _DftHandle(self, keys, freqs)

    def add_dft_flux(self, d, where, freqs, decimation_factor=0):
        x_normal, idx, lo, hi = _plane_spec(d, where, self.s.gs)
        mon = FluxMonitorTM(self.s.grid, list(freqs), x_normal=x_normal,
                            idx=idx, lo=lo, hi=hi)
        name = self._register(f"flux{len(self._monitors)}", mon)
        return _DftHandle(self, [name], freqs)

    def add_dft_flux_plane(self, where, freqs, decimation_factor=0):
        s = where.size or Vector3()
        return self.add_dft_flux(X if abs(s.x) <= abs(s.y) else Y, where, freqs)

    def add_dft_pt(self, c, pt, freqs, decimation_factor=0):
        if c != "Ez":
            raise NotImplementedError("add_dft_pt: Ez only")
        ix, iy = _clamp_ij(pt, self.s.gs)
        name = self._register(f"pt{len(self._monitors)}",
                              _PointDFT(self.s.grid, list(freqs), ix, iy))
        return _DftHandle(self, [name], freqs)

    def get_dft_array(self, handle, component=None, num_freq=0):
        key = handle.keys[0] if component is None else \
            next(k for k in handle.keys if k.endswith(f":{component}"))
        _, acc = self._mon(key)
        a = acc[num_freq].detach().cpu().numpy()
        return a[0] + 1j * a[1]

    def update_dfts(self):
        """DFT monitors accumulate inside step(); they are always current."""

    def clear_dft_monitors(self):
        self._monitors = {}

    def remove_fluxes(self):
        self._monitors = {k: v for k, v in self._monitors.items()
                          if not isinstance(v[0], FluxMonitorTM)}

    def dft_maxfreq(self):
        return max((float(e[0].freqs.max()) for e in self._monitors.values()
                    if hasattr(e[0], "freqs")), default=0.0)

    def dft_monitor_size(self):
        return sum(a.numel() * a.element_size()
                   for _, a in self._monitors.values() if a is not None)

    # --- flux volumes / instantaneous flux ---
    def add_flux_vol(self, d, where):
        return flux_vol(self, d, where)

    def add_flux_plane(self, where):
        s = where.size or Vector3()
        return flux_vol(self, X if abs(s.x) <= abs(s.y) else Y, where)

    def _instant_flux(self, d, where):
        """Poynting flux through a plane line from the live state (TM:
        Sx = -Ez*Hy, Sy = +Ez*Hx; H Yee-averaged onto the E line)."""
        x_normal, idx, lo, hi = _plane_spec(d, where, self.s.gs)
        st, dx = self._state, self.s.grid.dx
        if x_normal:
            E = st.Ez[idx, lo:hi]
            H = 0.5 * (st.Hy[idx - 1, lo:hi] + st.Hy[idx, lo:hi])
            return float(-(E * H).sum() * dx)
        E = st.Ez[lo:hi, idx]
        H = 0.5 * (st.Hx[lo:hi, idx - 1] + st.Hx[lo:hi, idx])
        return float((E * H).sum() * dx)

    def flux_in_box(self, d, where):
        return self._instant_flux(d, where)

    # --- field access ---
    def get_field(self, c, pt):
        if c not in _TM_COMPONENTS:
            raise NotImplementedError(f"component {c!r}: TM fields hold {_TM_COMPONENTS}")
        t = getattr(self._state, c)
        ix, iy = _clamp_ij(pt, self.s.gs)
        return float(t[min(ix, t.shape[0] - 1), min(iy, t.shape[1] - 1)])

    def get_field_from_comp(self, c, pt):
        return self.get_field(c, pt)

    def have_component(self, c):
        return c in _TM_COMPONENTS

    def require_component(self, c):
        if not self.have_component(c):
            raise NotImplementedError(f"component {c!r} does not exist on a TM run")

    def require_source_components(self):
        """No-op: every TM component is always allocated."""

    def initialize_field(self, c, fn):
        if c != "Ez":
            raise NotImplementedError("initialize_field: Ez only")
        self._state = self._state._replace(Ez=self.s._field(fn))

    def get_array_slice(self, where=None, component="Ez", *a, **k):
        if where is not None and not (isinstance(where, Volume) and where.size is None):
            raise NotImplementedError("get_array_slice sub-volumes: whole-cell only")
        self.require_component(component)
        return getattr(self._state, component).detach().cpu().numpy()

    def get_array_slice_dimensions(self, *a, **k):
        return (self.s.grid.nx, self.s.grid.ny)

    def max_abs(self, c):
        self.require_component(c)
        return float(getattr(self._state, c).abs().max())

    # --- energy / geometry queries ---
    def field_energy(self):
        """Total EM energy 0.5*integral(eps E^2 + H^2) dA (mu must be 1)."""
        if self.s._mu is not None:
            raise NotImplementedError("field_energy with mu != 1")
        st, dx = self._state, self.s.grid.dx
        u = 0.5 * (self.s._eps * st.Ez ** 2).sum() + \
            0.5 * (st.Hx ** 2).sum() + 0.5 * (st.Hy ** 2).sum()
        return float(u * dx * dx)

    def total_energy(self):
        return self.field_energy()

    def total_volume(self):
        return self.s.gs.sx * self.s.gs.sy

    def count_volume(self, c=None):
        return self.s.grid.nx * self.s.grid.ny * self.s.grid.dx ** 2

    def normal_direction(self, where):
        s = where.size or Vector3()
        if s.x == 0:
            return X
        if s.y == 0:
            return Y
        raise ValueError("volume has no zero-size axis")

    def nosize_direction(self, where):
        return self.normal_direction(where)

    # --- structure delegation ---
    def get_eps(self, pt):
        return self.s.get_eps(pt)

    def get_inveps(self, pt):
        return self.s.get_inveps(pt)

    def get_mu(self, pt):
        return self.s.get_mu(pt)

    def get_chi1inv(self, *a, **k):
        return self.s.get_chi1inv(*a, **k)

    def max_eps(self):
        return self.s.max_eps()

    def has_nonlinearities(self, *a):
        return self.s.has_chi()

    def remove_susceptibilities(self):
        self.s.remove_susceptibilities()

    def equal_layout(self, other):
        return self.s.equal_layout(other.s if isinstance(other, fields) else other)

    # --- trivial truths of this engine ---
    def is_aniso2d(self):
        return False

    def is_periodic(self, *a):
        return False

    def is_phasing(self):
        return False

    def use_real_fields(self):
        """No-op: the engine always runs real fields."""

    # --- output naming / HDF5 handles ---
    def set_output_directory(self, d):
        self._outdir = str(d)

    def h5file_name(self, name, prefix="", timestamp=False):
        base = f"{prefix}-{name}" if prefix else str(name)
        if timestamp:
            base = f"{base}-{int(time.time())}"
        return (f"{self._outdir}/" if self._outdir else "") + base + ".h5"

    def open_h5file(self, name, mode=None, prefix="", timestamp=False):
        return h5file(self.h5file_name(name, prefix, timestamp),
                      mode if mode is not None else h5file.WRITE)


_stub_raises(fields, "chunked domain decomposition / MPI comms are out of scope "
             "(whole-domain tensors)",
             "loop_in_chunks process_incoming_chunk_data "
             "process_dft_component get_dft_component_dims vec2diel_ceil "
             "vec2diel_floor locate_component_point do_get_array_slice "
             "add_srcdata lookup_src_time register_src_time get_source_slice")
_stub_raises(fields, "run timing/profiling is not tracked",
             "am_now_working_on finished_working get_timing_data time_spent_on "
             "mean_time_spent_on reset_timers print_times output_times log "
             "max_decimation")
_stub_raises(fields, "HDF5 field output/dump is not implemented "
             "(use get_array_slice / get_dft_array)",
             "output_hdf5 output_dft output_dft_components dump load")
_stub_raises(fields, "eigenmode/CW solves live in meep_torch.ports "
             "(compute_smatrix) and meep_torch.mpb",
             "add_eigenmode_source get_eigenmode get_eigenmode_coefficients "
             "get_mode_flux_overlap get_mode_mode_overlap get_overlap solve_cw "
             "set_solve_cw_omega unset_solve_cw_omega initialize_with_n_te "
             "initialize_with_n_tm initialize_with_nth_te initialize_with_nth_tm "
             "populate_maxwell_dielectric")
_stub_raises(fields, "box/weighted field integrals are not implemented "
             "(field_energy/flux_in_box cover the whole-domain cases)",
             "electric_energy_in_box electric_energy_max_in_box "
             "electric_energy_weighted_integral electric_sqr_weighted_integral "
             "energy_in_box field_energy_in_box magnetic_energy_in_box "
             "modal_volume_in_box thermo_energy_in_box flux_in_box_wrongH "
             "integrate integrate2 get_array_metadata get_complex_array_slice")
_stub_raises(fields, "engine limitation (2D cartesian, real fields, fixed "
             "PEC/PML walls)",
             "change_m use_bloch set_boundary casimir_stress_dct_integral "
             "phase_in_material lattice_vector dft_norm restore_magnetic_fields "
             "synchronize_magnetic_fields add_dft_energy "
             "add_dft_flux_box add_dft_force add_dft_near2far add_mode_monitor")


class h5file:
    """meep::h5file over h5py (imported lazily: h5py is NOT a dependency --
    a clear ImportError with an install hint is raised when it is absent)."""

    READONLY, READWRITE, WRITE = 0, 1, 2
    _MODES = {READONLY: "r", READWRITE: "a", WRITE: "w",
              "r": "r", "a": "a", "rw": "a", "w": "w"}

    def __init__(self, filename, mode=READWRITE, parallel_=False, local=False):
        if parallel_:
            raise NotImplementedError("parallel HDF5 is out of scope (no MPI)")
        self.filename = str(filename)
        self.mode = self._MODES.get(mode)
        if self.mode is None:
            raise ValueError(f"bad h5file mode {mode!r}")
        self._f = None

    def _h5(self):
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "mp.h5file needs the optional h5py package: uv pip install h5py") from e
        if self._f is None:
            self._f = h5py.File(self.filename, self.mode)
        return self._f

    def file_name(self):
        return self.filename

    def ok(self):
        try:
            self._h5()
            return True
        except OSError:
            return False

    def dataset_exists(self, name):
        return name in self._h5()

    def read(self, name):
        return np.asarray(self._h5()[name])

    def read_size(self, name):
        return tuple(self._h5()[name].shape)

    def write(self, name, data):
        f = self._h5()
        if name in f:
            del f[name]
        f.create_dataset(name, data=np.asarray(data))
        f.flush()

    def create_data(self, name, dims):
        f = self._h5()
        if name in f:
            del f[name]
        return f.create_dataset(name, shape=tuple(int(d) for d in dims), dtype="f8")

    def remove_data(self, name):
        f = self._h5()
        if name in f:
            del f[name]

    def remove(self):
        import os
        if self._f is not None:
            self._f.close()
            self._f = None
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


_stub_raises(h5file, "chunked/parallel HDF5 I/O is out of scope (no MPI; use "
             "read/write)",
             "read_chunk write_chunk extend_data create_or_extend_data "
             "get_extending done_writing_chunks prevent_deadlock")


class initialize:
    """meep::initialize -- RAII MPI init + wall clock. There is no MPI here:
    construction records the wall clock; elapsed_time() is seconds since."""

    def __init__(self, argc=0, argv=None):
        self._t0 = time.monotonic()

    def elapsed_time(self):
        return time.monotonic() - self._t0
