# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""mp.Simulation -- assembles and runs a meep-torch 2D FDTD from meep syntax.

Adapter only (no physics). Wraps core/coeffs, core/run, geometry/scene,
core/monitors, ports. dimensions=1 maps the meep z axis to the meep-torch x
axis with y_pmc=True and a thin ny. dtype= kwarg passes through to Grid
(float64 default; float32 opt-in, validated at scale in tests/autodiff/test_float32.py).
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from ..core.coeffs import build_te_coeffs, build_tm_coeffs
from ..core.grid import Grid
from ..core.materials import Material
from ..core.monitors import DFTFieldsMonitor, FluxMonitorTE, FluxMonitorTM, PointProbe
from ..core.monitors_3d import DFTFieldsMonitor3D, FluxMonitor3D, PlaneSource, PointSource3D, run_3d
from ..core.run import run_te, run_tm
from ..core.stepper import init_tm_state
from ..core.stepper_3d import build_3d_coeffs, init_3d_state
from ..core.stepper_te import init_te_state
from ..geometry.scene import rasterize_scene
from .coords import (GridSpec, box3d, cell_index, cell_index_y, cell_index_z, grid_spec,
                     pml_npml, rect_corners, span_index_range, to_corner)
from .geometry import Block, build_scene_items
from .material import Absorber, Medium, PML
from .run_control import (DftFields, DftFlux, DftLdos, DftNear2Far, DftObj, chunk_size,
                          dft_name, fire_per_step, fire_scheduled, plan_hooks)
from .sources import (Dielectric, EigenModeSource, Source, build_line_source,
                      build_mode_source, pol_for)
from .vec import Vector3
from ..ports.mode_monitor import decompose


# direction constants (meep swig enum values; visualization.py uses the same)
X, Y, Z, R, P = 0, 1, 2, 3, 4

_SYM_TODO = "see wiki/tasks/port_symmetry-system.md"


class Symmetry:
    """Symmetry of the structure *and* sources for the `symmetries` input
    variable (meep base class). `direction` is the normal to a mirror plane or
    the rotation axis; `phase` multiplies the fields under the symmetry op
    (-1 = odd mirror)."""

    def __init__(self, direction: int = None, phase: complex = 1.0 + 0j):
        self.direction = direction
        self.phase = complex(phase)
        self.swigobj = None


class Rotate2(Symmetry):
    """A 180-degree (twofold, C2) rotational symmetry about `direction`."""


class Rotate4(Symmetry):
    """A 90-degree (fourfold, C4) rotational symmetry about `direction`."""


class Mirror(Symmetry):
    """A mirror symmetry plane; `direction` is the normal to the plane.

    Engine support: even (phase=+1) y-mirror on a TM 2D run (half-domain,
    core y_mirror=+1). Anything else raises NotImplementedError when handed
    to Simulation (%s).""" % _SYM_TODO


class Identity(Symmetry):
    """The identity symmetry (a no-op)."""


def _grid_for(cell: Vector3, resolution: float, dimensions: int, dtype) -> tuple[Grid, GridSpec, bool]:
    if dimensions == 1:
        sx = float(cell.z)
        dx = 1.0 / float(resolution)
        nx = max(3, int(round(sx / dx)))
        ny = 3
        g = Grid(nx=nx, ny=ny, dx=dx, dtype=dtype)
        gs = GridSpec(nx, ny, dx, sx, ny * dx)
        return g, gs, True
    gs = grid_spec(cell, resolution)
    g = Grid(nx=gs.nx, ny=gs.ny, dx=gs.dx, dtype=dtype)
    return g, gs, False


@dataclass
class FluxRegion:
    center: Vector3 = None
    size: Vector3 = None
    direction: int = 1
    volume: object = None      # mp.Volume; supplies center/size when given

    def __post_init__(self):
        if self.volume is not None:
            self.center = self.volume.center
            self.size = self.volume.size


@dataclass
class Near2FarRegion:
    center: Vector3 = None
    size: Vector3 = None
    weight: int = 1
    volume: object = None      # mp.Volume; supplies center/size when given

    def __post_init__(self):
        if self.volume is not None:
            self.center = self.volume.center
            self.size = self.volume.size


class _Amp:
    """Scale a 3D source's time waveform by the meep ``Source.amplitude``
    (run_3d drives ``.amplitude(t)`` + ``.src``)."""
    def __init__(self, src, amp):
        self._src = src
        self.src = src.src
        self._amp = amp

    def amplitude(self, t_idx):
        return self._amp * self._src.amplitude(t_idx)


class Simulation:
    def __init__(self, *, cell_size, resolution, geometry=None, sources=None,
                 default_material=Medium(epsilon=1.0), boundary_layers=None,
                 dimensions=2, symmetries=None, dtype=torch.float64):
        self.dimensions = dimensions
        self.dtype = dtype
        self._sym_cy = 0     # y-mirror: full-cell rows below the cut plane (0 = off)
        sym_y = False
        for s in symmetries or []:
            if isinstance(s, Identity):
                continue
            if isinstance(s, Mirror) and s.direction == Y and s.phase == 1:
                sym_y = True
                continue
            raise NotImplementedError(
                f"{type(s).__name__}(direction={getattr(s, 'direction', None)}, "
                f"phase={getattr(s, 'phase', None)}) not supported: even "
                f"Mirror(Y) on a TM 2D run only ({_SYM_TODO})")
        if sym_y and dimensions != 2:
            raise NotImplementedError(f"Mirror(Y) is 2D-only ({_SYM_TODO})")
        if dimensions == 3:
            self._init_3d(cell_size, resolution, geometry, sources,
                          default_material, boundary_layers, dtype)
            return
        self.grid, self.gs, self._y_pmc = _grid_for(cell_size, resolution, dimensions, dtype)
        self._npml = 0
        self._nabs = 0
        for bl in boundary_layers or []:
            if isinstance(bl, PML):
                self._npml = max(self._npml, pml_npml(bl.thickness, self.gs))
            elif isinstance(bl, Absorber):
                self._nabs = max(self._nabs, pml_npml(bl.thickness, self.gs))
        self._default = default_material if isinstance(default_material, Medium) else Medium(epsilon=float(default_material))
        geom = list(geometry or [])
        eps_items, chi3_items, cond_items = build_scene_items(geom, self.gs, dimensions,
                                                              with_cond=True)
        bg_val, bg_chi3 = self._default.resolve()
        if bg_chi3 != 0.0:
            full = np.array([[0, 0], [self.gs.sx, 0], [self.gs.sx, self.gs.sy], [0, self.gs.sy]])
            from ..geometry.polygons import Polygon
            chi3_items.append((Polygon(full), bg_chi3))
        self._rm = rasterize_scene(eps_items, self.grid, background=bg_val, chi3_items=chi3_items)
        self._sources = list(sources or [])
        self._pol = pol_for(self._sources[0].component) if self._sources else "tm"
        # W5: conductivity / mu maps + Absorber sigma ramp
        bg_sd = float(getattr(self._default, "D_conductivity", 0.0) or 0.0)
        bg_sb = float(getattr(self._default, "B_conductivity", 0.0) or 0.0)
        bg_mu = float(getattr(self._default, "mu", 1.0) or 1.0)
        self._cond = None
        if cond_items or bg_sd or bg_sb or bg_mu != 1.0 or self._nabs:
            sd = rasterize_scene([(p, v[0]) for p, v in cond_items], self.grid,
                                 background=bg_sd).eps_tm
            sb = rasterize_scene([(p, v[1]) for p, v in cond_items], self.grid,
                                 background=bg_sb).eps_tm
            mu = rasterize_scene([(p, v[2]) for p, v in cond_items], self.grid,
                                 background=bg_mu).eps_tm
            if self._nabs:
                ramp = self._absorber_ramp()
                sd = sd + ramp
                sb = sb + ramp
            self._cond = (sd, sb, mu)
        if sym_y:
            # Even y-mirror: rasterize the full cell (above), then run the
            # engine on the upper half with a PMC wall at the cut plane
            # (build_tm_coeffs y_mirror=+1; tests/examples/test_y_mirror.py).
            # self.gs stays full-cell for all coordinate conversion; y spans
            # are folded via _ymap/_ypoint and outputs de-mirrored.
            if self._pol != "tm":
                raise NotImplementedError(f"Mirror(Y) is TM-only ({_SYM_TODO})")
            if self.gs.ny % 2:
                raise NotImplementedError(
                    f"Mirror(Y) needs an even cell row count ({_SYM_TODO})")
            cy = self._sym_cy = self.gs.ny // 2
            self.grid = Grid(nx=self.gs.nx, ny=self.gs.ny - cy, dx=self.gs.dx,
                             dtype=dtype)
            half = lambda t: None if t is None else t[:, cy:].contiguous()
            self._rm = self._rm._replace(
                eps_tm=half(self._rm.eps_tm), eps_te_xx=half(self._rm.eps_te_xx),
                eps_te_yy=half(self._rm.eps_te_yy),
                pole_masks=[half(m) for m in self._rm.pole_masks],
                chi3=half(self._rm.chi3))
            if self._cond is not None:
                self._cond = tuple(half(t) for t in self._cond)
        self._coeffs = self._build_coeffs()
        self._state = None
        self._t_step = 0
        self._monitors = []          # list of (name, kind, mon, meta)
        self._last_acc = {}
        # S-param path: an EigenModeSource + mode monitors route to compute_smatrix.
        self._eig_source = next((s for s in self._sources if isinstance(s, EigenModeSource)),
                               None)
        self._mode_ports = []        # list of (name, axis, idx, lo, hi, direction)
        self._S = None
        self._port_index = {}       # monitor name -> port index in compute_smatrix

    def _absorber_ramp(self):
        """Quadratic sigma ramp on every absorbing edge (R_asymptotic 1e-15)."""
        from ..core.pml import absorber_sigma
        nx, ny, dx = self.grid.nx, self.grid.ny, self.grid.dx
        kw = dict(dtype=self.grid.dtype, device=self.grid.device)
        ramp = absorber_sigma(nx, self._nabs, dx, **kw)[:, None].expand(nx, ny).clone()
        if not self._y_pmc:
            ramp = torch.maximum(
                ramp, absorber_sigma(ny, self._nabs, dx, **kw)[None, :].expand(nx, ny))
        return ramp

    # ----------------------------------------------------------------------
    # 3D path (W9 wave 3). Adapter only: builds the eps/conductivity volumes,
    # calls build_3d_coeffs/init_3d_state, and drives run_3d. Axis-aligned
    # Block only (voxel fill, no subpixel); scalar eps + D/B-conductivity only
    # (no Lorentz/Kerr in 3D this wave). See docs/specs/...-w9-wave3-3d-shim.
    # ----------------------------------------------------------------------
    def _init_3d(self, cell_size, resolution, geometry, sources, default_material,
                 boundary_layers, dtype):
        self.gs = grid_spec(cell_size, resolution, dimensions=3)
        g = self.gs
        self.grid = None                      # no core/Grid in 3D; use gs + coeffs
        self._y_pmc = False
        self._npml = 0
        self._nabs = 0
        for bl in boundary_layers or []:
            if isinstance(bl, PML):
                self._npml = max(self._npml, pml_npml(bl.thickness, g))
            elif isinstance(bl, Absorber):
                self._nabs = max(self._nabs, pml_npml(bl.thickness, g))
        self._default = (default_material if isinstance(default_material, Medium)
                         else Medium(epsilon=float(default_material)))
        nx, ny, nz = g.nx, g.ny, g.nz
        bg = self._default
        bg_eps, bg_chi3 = bg.resolve()
        if isinstance(bg_eps, Material) or bg_chi3 != 0.0:
            raise NotImplementedError("3D dispersive/Kerr default material not supported (W9 wave 3)")
        eps = torch.full((nx, ny, nz), float(bg_eps), dtype=dtype)
        bg_sd = float(getattr(bg, "D_conductivity", 0.0) or 0.0)
        bg_sb = float(getattr(bg, "B_conductivity", 0.0) or 0.0)
        sd = torch.full((nx, ny, nz), bg_sd, dtype=dtype)
        sb = torch.full((nx, ny, nz), bg_sb, dtype=dtype)
        for obj in (geometry or []):
            if not isinstance(obj, Block):
                raise NotImplementedError(
                    f"3D geometry {type(obj).__name__} not supported (W9 wave 3; "
                    f"axis-aligned Block only)")
            c = obj.center or Vector3()
            s = obj.size or Vector3(0, 0, 0)
            i0, i1, j0, j1, k0, k1 = box3d(c, s, g)
            m = obj.material
            if m is None:
                mv, sdv, sbv = float(bg_eps), bg_sd, bg_sb
            else:
                mv, mchi3 = (m.resolve() if isinstance(m, Medium) else (float(m), 0.0))
                if isinstance(mv, Material) or mchi3 != 0.0:
                    raise NotImplementedError("3D dispersive/Kerr Block not supported (W9 wave 3)")
                mv = float(mv)
                sdv = float(getattr(m, "D_conductivity", 0.0) or 0.0)
                sbv = float(getattr(m, "B_conductivity", 0.0) or 0.0)
            eps[i0:i1, j0:j1, k0:k1] = mv
            sd[i0:i1, j0:j1, k0:k1] = sdv
            sb[i0:i1, j0:j1, k0:k1] = sbv
        has_cond = bool((sd.abs().sum() + sb.abs().sum()).item())
        self._eps3d = eps
        self._coeffs = build_3d_coeffs(
            nx, ny, nz, g.dx, eps=eps, npml=self._npml, n_abs=self._nabs,
            sigma_d=(sd if has_cond else None), sigma_b=(sb if has_cond else None),
            dtype=dtype)
        self._state = None
        self._t_step = 0
        self._sources = list(sources or [])
        self._monitors = []
        self._last_acc = {}
        self._eig_source = None              # 3D has no S-param path this wave
        self._mode_ports = []
        self._S = None
        self._port_index = {}

    def _make_sources_3d(self):
        specs = list(self._sources)
        if not specs:
            return None
        spec = specs[0]
        prof = spec.src.profile()
        comp = spec.component
        c = spec.center or Vector3()
        s = spec.size
        g = self.gs
        dt = self._coeffs.dt
        amp = float(getattr(spec, "amplitude", 1.0) or 1.0)
        if s is None or (s.x == 0 and s.y == 0 and s.z == 0):
            i = cell_index(c.x, g); j = cell_index_y(c.y, g); k = cell_index_z(c.z, g)
            src = PointSource3D(dt, comp, i, j, k, prof)
        else:
            sizes = [abs(s.x or 0.0), abs(s.y or 0.0), abs(s.z or 0.0)]
            axis = sizes.index(min(sizes))             # propagation axis (zero/min extent)
            idx = [cell_index(c.x, g), cell_index_y(c.y, g), cell_index_z(c.z, g)][axis]
            # A full-plane planewave (mp.inf on a transverse axis) launches a
            # uniform transverse field -- the x/y CPML is inert (zero transverse
            # curl, no psi accumulation, same trick as the wave-2 gate-a mask),
            # so NO transverse aperture: clipping it would create a finite
            # aperture, ring the field, engage the side CPML and drain the beam
            # (vacuum |T| collapses to ~0.13). Only a finite-aperture source
            # (both transverse sizes bounded) takes the lo/hi window.
            raw = [s.x or 0.0, s.y or 0.0, s.z or 0.0]
            full_plane = any(math.isinf(t) for k, t in enumerate(raw) if k != axis)
            if full_plane:
                src = PlaneSource(g.nx, g.ny, g.nz, dt, comp, axis, idx, prof)
            else:
                ap = self._npml
                src = PlaneSource(g.nx, g.ny, g.nz, dt, comp, axis, idx, prof,
                                  lo=ap, hi=min(g.nx, g.ny, g.nz) - ap)
        return _Amp(src, amp)

    def _run_3d(self, n_steps):
        self._state = init_3d_state(self.gs.nx, self.gs.ny, self.gs.nz,
                                    npml=self._npml, dtype=self.dtype)
        src = self._make_sources_3d()
        monitors = [m for (_, _, m, _) in self._monitors]
        self._state, outs = run_3d(self._state, self._coeffs, n_steps,
                                   source=src, monitors=monitors)
        for (name, kind, mon, meta), acc in zip(self._monitors, outs):
            self._last_acc[name] = (acc, mon, meta)

    def _plane_spec_3d(self, region):
        c = region.center or Vector3()
        s = region.size or Vector3(0, 0, 0)
        g = self.gs
        sizes = [abs(s.x or 0.0), abs(s.y or 0.0), abs(s.z or 0.0)]
        axis = sizes.index(min(sizes))                 # plane normal axis (meep convention)
        normal = ("x", "y", "z")[axis]
        idx = [cell_index(c.x, g), cell_index_y(c.y, g), cell_index_z(c.z, g)][axis]
        return normal, idx, self._npml + 1

    def _add_flux_3d(self, fcen, df, nfreq, region, name):
        freqs = np.linspace(fcen - df / 2.0, fcen + df / 2.0, nfreq).tolist()
        name = name or f"flux{len(self._monitors)}"
        normal, idx, ap = self._plane_spec_3d(region)
        g = self.gs
        # A finite transverse extent (a closed-box face around a scatterer)
        # -> explicit rectangular window on the plane; ap=0 (no PML margin
        # for an interior face). Full-cell (mp.inf transverse, a port) ->
        # centered [ap : m-ap] aperture excluding the side PML (unchanged).
        s = region.size or Vector3(0, 0, 0)
        trans = [s.x or 0.0, s.y or 0.0, s.z or 0.0]
        if normal == "z":
            finite = math.isfinite(trans[0]) and math.isfinite(trans[1])
        elif normal == "x":
            finite = math.isfinite(trans[1]) and math.isfinite(trans[2])
        else:
            finite = math.isfinite(trans[0]) and math.isfinite(trans[2])
        win = None
        if finite:
            i0, i1, j0, j1, k0, k1 = box3d(region.center or Vector3(), s, g)
            if normal == "z":
                win = (i0, i1, j0, j1)
            elif normal == "x":
                win = (j0, j1, k0, k1)
            else:
                win = (i0, i1, k0, k1)
            ap = 0
        mon = FluxMonitor3D(self._coeffs.dt, freqs, normal=normal, idx=idx, ap=ap,
                            win=win, dtype=self.dtype)
        self._monitors.append((name, "flux", mon, {"normal": normal, "idx": idx}))
        return name

    def _build_coeffs(self):
        extra = {}
        if self._cond is not None:
            extra = dict(sigma_d=self._cond[0], sigma_b=self._cond[1], mu=self._cond[2])
        if self._pol == "tm":
            return build_tm_coeffs(self.grid, eps_inf=self._rm.eps_tm, poles=self._rm.poles,
                                    pole_masks=self._rm.pole_masks, chi3=self._rm.chi3,
                                    npml=self._npml, y_pmc=self._y_pmc,
                                    y_mirror=+1 if self._sym_cy else 0, **extra)
        if self._y_pmc:
            raise NotImplementedError("1D (y-PMC) TE Simulation is not supported")
        return build_te_coeffs(self.grid, eps_inf=self._rm.eps_te_xx, poles=self._rm.poles,
                               pole_masks=self._rm.pole_masks, chi3=self._rm.chi3,
                               npml=self._npml, **extra)

    def _make_sources(self):
        out = []
        for spec in self._sources:
            if isinstance(spec, EigenModeSource):
                raise NotImplementedError("EigenModeSource via Simulation.run is W1-limited; "
                                          "use compute_smatrix via meep_torch.ports for S-params")
            if self.dimensions == 1 and spec.center is not None:
                # 1D: source coords live in z -> map to x; default to a full-y plane.
                sz = spec.size if spec.size is not None else Vector3(0, math.inf, 0)
                spec = Source(spec.src, spec.component,
                             center=Vector3(spec.center.z, 0, 0), size=sz,
                             amplitude=spec.amplitude, amp_func=spec.amp_func)
            _, _, ix, j0, j1, prof, amp, amp_func, center, gs = build_line_source(spec, self.gs, self._pol)
            if self._sym_cy:
                if amp_func is not None:
                    raise NotImplementedError(
                        f"amp_func with Mirror(Y) not supported ({_SYM_TODO})")
                j0, j1, _ = self._ymap(j0, j1)
            from ..core.sources import LineSource
            src = LineSource(self.grid, ix, j0, j1, profile=prof)
            if amp_func is not None:
                # overwrite the uniform span mask with the spatial profile across y
                c = center or Vector3()
                ys = [c.y - gs.sy / 2 + (j + 0.5) * gs.dx for j in range(j0, j1)]
                w = [float(amp_func(Vector3(c.x, y, 0.0))) for y in ys]
                src.mask = self.grid.zeros(self.grid.nx, self.grid.ny)
                row = torch.tensor(w, dtype=self.grid.dtype)
                src.mask[ix, j0:j1] = row
            src.mask = src.mask * amp
            if self._pol == "te":
                # step_te injects into Dy (Ey grid, (nx, ny-1))
                src.mask = src.mask[:, :-1]
            out.append(src)
        return out

    def _init_state(self):
        n_poles = len(self._rm.poles)
        if self._pol == "tm":
            return init_tm_state(self.grid, n_poles=n_poles, y_pmc=self._y_pmc,
                                 y_mirror=+1 if self._sym_cy else 0)
        return init_te_state(self.grid, n_poles=n_poles)

    # --- y-mirror folding helpers (all no-ops when _sym_cy == 0) ---
    def _ymap(self, lo, hi):
        """Full-cell y span -> half-grid span; crossed=True when the span
        reached below the mirror plane (its flux then doubles back)."""
        cy = self._sym_cy
        if not cy:
            return lo, hi, False
        return max(0, lo - cy), max(1, hi - cy), lo < cy

    def _ypoint(self, j):
        """Full-cell y index -> half-grid row (mirrored below the plane)."""
        cy = self._sym_cy
        if not cy:
            return j
        return j - cy if j >= cy else cy - 1 - j

    def _demirror(self, a):
        """Half-grid (nx, ny/2) array -> full-cell even-mirrored (nx, ny)."""
        return np.concatenate([a[:, ::-1], a], axis=1) if self._sym_cy else a

    def _source_cutoff(self):
        """Latest source turn-off time (GaussianPulse.t_end; CW never ends -> 0)."""
        return max((float(getattr(s.src.profile(), "t_end", 0.0))
                    for s in self._sources), default=0.0)

    def round_time(self):
        return self._t_step * (self._coeffs.dt if self.dimensions == 3 else self.grid.dt)

    def get_field_point(self, c, pt):
        """Field value at the cell containing meep point ``pt`` (component per
        the run polarization: Ez for TM, Hz for TE)."""
        if self._state is None:
            return 0.0
        gs = self.gs
        if self.dimensions == 1:
            ix, iy = cell_index(pt.z, gs), gs.ny // 2
        else:
            ix = max(0, min(gs.nx - 1, cell_index(pt.x, gs)))
            iy = self._ypoint(max(0, min(gs.ny - 1, cell_index_y(pt.y, gs))))
        f = self._state.Hz if self._pol == "te" else self._state.Ez
        return float(f[ix, iy])

    def run(self, *step_funcs, until=None, until_after_sources=None, n_steps=None):
        if self._sym_cy and self._eig_source is not None:
            raise NotImplementedError(
                f"EigenModeSource with Mirror(Y) not supported: compute_smatrix "
                f"has no y_mirror path yet ({_SYM_TODO})")
        stepped = bool(step_funcs) or callable(until) or callable(until_after_sources) \
            or isinstance(until, (list, tuple)) or isinstance(until_after_sources, (list, tuple))
        if stepped:
            if self.dimensions == 3 or self._eig_source is not None:
                raise NotImplementedError(
                    "step functions / stop conditions: 2D non-eigenmode runs only")
            self._run_stepped(step_funcs, until, until_after_sources)
            return
        if n_steps is None:
            if until is not None:
                T = until
            elif until_after_sources is not None:
                # meep semantics: duration counted after the sources turn off
                T = self._source_cutoff() + until_after_sources
            else:
                T = 0
            dt = self._coeffs.dt if self.dimensions == 3 else self.grid.dt
            n_steps = max(1, int(round(float(T) / dt)))
        self._t_step = n_steps
        if self.dimensions == 3:
            self._run_3d(n_steps)
            return
        if self._eig_source is not None:
            self._run_modes(n_steps)
            return
        run = run_tm if self._pol == "tm" else run_te
        self._state = self._init_state()
        srcs = self._make_sources()
        monitors = [m for (_, _, m, _) in self._monitors]
        src = srcs[0] if srcs else None
        self._state, outs = run(self._state, self._coeffs, n_steps, source=src, monitors=monitors)
        for (name, kind, mon, meta), acc in zip(self._monitors, outs):
            self._last_acc[name] = (acc, mon, meta)

    def _run_stepped(self, step_funcs, until, until_after_sources):
        """meep run protocol: drive the engine chunk-by-chunk (_run_chunk keeps
        the source time index continuous), firing step functions between chunks
        and feeding stop conditions every step."""
        from ..core.run import _run_chunk
        from ..core.stepper import step_tm
        from ..core.stepper_te import step_te
        dt = self.grid.dt
        cutoff = self._source_cutoff()
        begin, end, per_step, scheduled = plan_hooks(step_funcs, cutoff)
        conds, gate, n_total = None, 0.0, None
        stop = until if until is not None else until_after_sources
        if callable(stop) or isinstance(stop, (list, tuple)):
            conds = list(stop) if isinstance(stop, (list, tuple)) else [stop]
            if until is None:
                gate = cutoff        # until_after_sources: condition applies post-cutoff
        elif until is not None:
            n_total = max(1, int(round(float(until) / dt)))
        else:
            n_total = max(1, int(round((cutoff + (until_after_sources or 0)) / dt)))
        step_fn = step_tm if self._pol == "tm" else step_te
        self._state = self._init_state()
        srcs = self._make_sources()
        src = srcs[0] if srcs else None
        monitors = [m for (_, _, m, _) in self._monitors]
        acc = [None] * len(monitors)
        self._t_step = 0
        for f in begin:
            f(self)
        fire_scheduled(self, scheduled, 0.0)
        while n_total is None or self._t_step < n_total:
            t = self._t_step * dt
            left = (n_total - self._t_step) if n_total is not None else 10 ** 9
            k = chunk_size(t, dt, left, per_step, scheduled, conds is not None)
            self._state, outs = _run_chunk(step_fn, self._state, self._coeffs,
                                           self._t_step, k, src, monitors)
            for i, (mon, out) in enumerate(zip(monitors, outs)):
                acc[i] = out if acc[i] is None else \
                    (acc[i] + out if mon.reduce == "sum" else torch.cat([acc[i], out]))
            self._t_step += k
            t = self._t_step * dt
            fire_per_step(self, per_step, t, dt)
            fire_scheduled(self, scheduled, t)
            if conds is not None:
                flags = [bool(g(self)) for g in conds]   # feed every condition
                if t >= gate - 1e-9 and any(flags):
                    break
        for (name, kind, mon, meta), a in zip(self._monitors, acc):
            self._last_acc[name] = (a, mon, meta)
        for f in end:
            f(self)

    def _run_modes(self, n_steps):
        """S-param path: EigenModeSource + mode monitors -> compute_smatrix.

        Normalization: meep's get_eigenmode_coefficients returns the mode
        coefficient alpha at a monitor; the S-parameter is alpha_out/alpha_in
        with the source port's injected alpha as the reference (the 'measured
        incident' a_in). meep-torch's compute_smatrix builds the per-port
        S-matrix in this same power-normalized convention (|S|<=1, validated vs
        meep 1.34), so get_eigenmode_coefficients(name) returns S[:, port, 0]
        -- the coefficient at monitor `name` when the EigenModeSource port
        (port 0) is excited. The monitor colocated with the source maps to
        port 0 (reflection S[0,0]); downstream monitors map to ports 1,2,...
        (transmission S[k,0]).
        """
        from ..ports.port import Port
        from ..ports.smatrix import compute_smatrix
        es = self._eig_source
        _, axis, idx, lo, hi, direction, band, prof, amp = build_mode_source(es, self.gs)
        src_port = Port(axis=axis, index=idx, lo=lo, hi=hi, direction=es.direction,
                        n_modes=es.eig_band)   # eig_band is 1-based (meep)
        ports = [src_port]
        self._port_index = {}
        for (name, ax, i, l, h, d) in self._mode_ports:
            if ax == axis and i == idx and l == lo and h == hi:
                self._port_index[name] = 0      # colocated with source -> reflection
                continue
            self._port_index[name] = len(ports)
            ports.append(Port(axis=ax, index=i, lo=l, hi=h, direction=d))
        fcen = float(getattr(es.src, "_fcen", None) or es.src.frequency or 0.0)
        fwidth = float(getattr(es.src, "_fw", None) or es.src.fwidth or 0.1)
        freqs = self._mode_freqs or [fcen]
        sm = compute_smatrix(self._rm.eps_tm if self._pol == "tm" else self._rm.eps_te_xx,
                             self.grid, ports, pol=self._pol, fcen=fcen, fwidth=fwidth,
                             freqs=freqs, n_steps=n_steps, npml=self._npml)
        self._S = sm.s

    # --- flux monitors ---
    def add_flux(self, fcen, df, nfreq, *fluxes, name=None):
        """meep add_flux(fcen, df, nfreq, *fluxregions) -> DftFlux (fluxes over
        multiple regions are summed, meep style)."""
        freqs = (np.linspace(fcen - df / 2.0, fcen + df / 2.0, nfreq)).tolist()
        if self.dimensions == 3:
            if len(fluxes) != 1:
                raise NotImplementedError("3D add_flux takes exactly one region (W9 wave 3)")
            return DftFlux(self, self._add_flux_3d(fcen, df, nfreq, fluxes[0], name), freqs)
        name = name or f"flux{len(self._monitors)}"
        parts = []
        for r, region in enumerate(fluxes):
            mon, meta = self._line_monitor(region, freqs, kind="flux")
            key = name if len(fluxes) == 1 else f"{name}:r{r}"
            self._monitors.append((key, "flux", mon, meta))
            parts.append(key)
        if len(fluxes) > 1:
            self._flux_parts = getattr(self, "_flux_parts", {})
            self._flux_parts[name] = parts
        return DftFlux(self, name, freqs)

    def add_near2far(self, fcen, df, nfreq, *regions, name=None):
        if self.dimensions == 3:
            raise NotImplementedError("3D add_near2far not supported (W9 wave 3; 2D only)")
        if self._sym_cy:
            raise NotImplementedError(
                f"add_near2far with Mirror(Y) not supported ({_SYM_TODO})")
        """2D near-to-far contour (TM + TE). Regions with outward normal =
        region axis * weight; a closed box = 4 regions, meep style. Trapezoid
        endpoint weights close the loop integral when the regions share
        corners."""
        from ..core.monitors import FluxMonitorTEY
        freqs = np.linspace(fcen - df / 2.0, fcen + df / 2.0, max(nfreq, 1)).tolist()
        name = name or f"n2f{len(self._monitors)}"
        dx = self.grid.dx
        specs = []
        for r, region in enumerate(regions):
            axis, idx, lo, hi = self._port_spec(FluxRegion(region.center, region.size))
            hi_cap = (self.grid.ny if axis == "x" else self.grid.nx)
            if self._pol == "te":
                hi_cap -= 1                     # Ey/Ex grids are one shorter
            hi = min(hi + 1, hi_cap)
            normal = (int(region.weight), 0) if axis == "x" else (0, int(region.weight))
            idxs = np.arange(lo, hi)
            if self._pol == "tm":
                mon = FluxMonitorTM(self.grid, freqs, x_normal=(axis == "x"),
                                    idx=idx, lo=lo, hi=hi)
                off = 0.0
            elif axis == "x":
                mon = FluxMonitorTE(self.grid, freqs, ix=idx, lo=lo, hi=hi)
                off = 0.5                       # Ey nodes at (idx, (j+1/2) dx)
            else:
                mon = FluxMonitorTEY(self.grid, freqs, iy=idx, lo=lo, hi=hi)
                off = 0.5                       # Ex nodes at ((i+1/2) dx, idx)
            if axis == "x":
                pts = np.stack([np.full(len(idxs), idx * dx), (idxs + off) * dx], axis=1)
            else:
                pts = np.stack([(idxs + off) * dx, np.full(len(idxs), idx * dx)], axis=1)
            self._monitors.append((f"{name}:f{r}", "n2f", mon,
                                   {"normal": normal, "pts": pts}))
            specs.append(f"{name}:f{r}")
        self._n2f_specs = getattr(self, "_n2f_specs", {})
        self._n2f_specs[name] = (specs, freqs)
        return DftNear2Far(self, name, freqs)

    def get_farfield(self, name, point):
        """Complex [Ex, Ey, Ez, Hx, Hy, Hz] at a far point (meep layout);
        (nfreq, 6) when nfreq > 1."""
        from ..core.near2far import faces_from_accs, farfield_te, farfield_tm
        specs, freqs = self._n2f_specs[dft_name(name)]
        mons, accs = [], []
        for key in specs:
            acc, mon, meta = self._last_acc[key]
            mons.append((mon, meta["normal"], meta["pts"]))
            accs.append(acc)
        faces = faces_from_accs(self.grid, mons, accs)
        obs = np.array([[point.x + self.gs.sx / 2.0, point.y + self.gs.sy / 2.0]])
        ks = [2 * math.pi * f for f in freqs]
        E, H = (farfield_tm if self._pol == "tm" else farfield_te)(faces, ks, obs)
        out = np.concatenate([E[:, 0, :], H[:, 0, :]], axis=1)   # (nfreq, 6)
        return out[0] if len(freqs) == 1 else out

    def dft_ldos(self, fcen, df, nfreq, name=None):
        if self.dimensions == 3:
            raise NotImplementedError("3D dft_ldos not supported (W9 wave 3; deferred)")
        """meep dft_ldos-lite: record Ez at the (point) source and return
        LDOS via core.ldos.ldos_spectrum after run(). TM only."""
        c = (self._sources[0].center or Vector3())
        ix = cell_index(c.x, self.gs)
        iy = self._ypoint(cell_index_y(c.y, self.gs))
        mon = PointProbe(ix, iy)
        name = name or f"ldos{len(self._monitors)}"
        freqs = np.linspace(fcen - df / 2.0, fcen + df / 2.0, max(nfreq, 1))
        self._monitors.append((name, "ldos", mon, {"freqs": freqs}))
        return DftLdos(self, name, freqs.tolist())

    def get_ldos(self, name):
        from ..core.ldos import ldos_spectrum
        acc, mon, meta = self._last_acc[dft_name(name)]
        e = acc.detach().cpu().numpy()
        prof = self._sources[0].src.profile()
        n = len(e)
        j = np.array([prof.amplitude((t + 0.5) * self.grid.dt) for t in range(n)])
        return meta["freqs"], ldos_spectrum(e, j, self.grid.dt, meta["freqs"])

    def add_dft_fields(self, components, fcen, df, nfreq, name=None):
        """Full-cell area DFT of field components (meep add_dft_fields, no
        `where` region: whole cell only). Returns a handle for get_dft_array."""
        freqs = np.linspace(fcen - df / 2.0, fcen + df / 2.0, nfreq).tolist()
        name = name or f"dft{len(self._monitors)}"
        for c in components:
            if self.dimensions == 3:
                mon = DFTFieldsMonitor3D(self._coeffs.dt, freqs, component=c, dtype=self.dtype)
            else:
                mon = DFTFieldsMonitor(self.grid, freqs, component=c)
            self._monitors.append((f"{name}:{c}", "dft", mon, {"freqs": freqs}))
        return DftFields(self, name, freqs)

    def get_dft_array(self, name, component, num_freq):
        """Complex (nx, ny) DFT plane of `component` at frequency index."""
        acc, mon, meta = self._last_acc[f"{dft_name(name)}:{component}"]
        a = acc[num_freq].detach().cpu().numpy()
        return self._demirror(a[0] + 1j * a[1])

    def add_mode_monitor(self, fcen, df, nfreq, *fluxes, name=None):
        if self.dimensions == 3:
            raise NotImplementedError("3D EigenModeSource/S-params not supported (W9 wave 3)")
        if len(fluxes) != 1:
            raise NotImplementedError("add_mode_monitor takes exactly one region here")
        region = fluxes[0]
        # For the S-param path: store the port spec; run() routes an
        # EigenModeSource + these monitors to compute_smatrix. freqs are kept
        # for get_flux_freqs parity.
        axis, idx, lo, hi = self._port_spec(region)
        name = name or f"mode{len(self._mode_ports)}"
        self._mode_ports.append((name, axis, idx, lo, hi, region.direction))
        self._mode_freqs = np.linspace(fcen - df / 2.0, fcen + df / 2.0, nfreq).tolist()
        return DftFlux(self, name, self._mode_freqs)

    def _port_spec(self, region: FluxRegion):
        c = region.center or Vector3()
        s = region.size or Vector3(0, 0, 0)
        gs = self.gs
        if self.dimensions == 1:
            return "x", cell_index(c.z, gs), 0, gs.ny
        if abs(s.x) <= abs(s.y):
            idx = cell_index(c.x, gs)
            y0 = c.y - s.y / 2 + gs.sy / 2
            y1 = c.y + s.y / 2 + gs.sy / 2
            lo, hi = span_index_range(max(0.0, y0), min(gs.sy, y1), gs, "y")
            return "x", idx, lo, hi
        idx = max(0, min(gs.ny, cell_index_y(c.y, gs)))
        x0 = c.x - s.x / 2 + gs.sx / 2
        x1 = c.x + s.x / 2 + gs.sx / 2
        lo, hi = span_index_range(max(0.0, x0), min(gs.sx, x1), gs, "x")
        return "y", idx, lo, hi

    def _line_monitor(self, region: FluxRegion, freqs, kind):
        c = region.center or Vector3()
        s = region.size or Vector3(0, 0, 0)
        gs = self.gs
        if self.dimensions == 1:
            idx = cell_index(c.z, gs)
            lo, hi = 0, gs.ny
        elif abs(s.x) <= abs(s.y):
            idx = cell_index(c.x, gs)
            y0 = c.y - s.y / 2 + gs.sy / 2
            y1 = c.y + s.y / 2 + gs.sy / 2
            lo, hi = span_index_range(max(0.0, y0), min(gs.sy, y1), gs, "y")
        else:
            idx = max(0, min(gs.ny, cell_index_y(c.y, gs)))
            x0 = c.x - s.x / 2 + gs.sx / 2
            x1 = c.x + s.x / 2 + gs.sx / 2
            lo, hi = span_index_range(max(0.0, x0), min(gs.sx, x1), gs, "x")
        meta = {"idx": idx, "lo": lo, "hi": hi}
        if self._sym_cy:
            if not (self.dimensions == 2 and abs(s.x) <= abs(s.y)):
                raise NotImplementedError(
                    f"only x-normal monitors with Mirror(Y) ({_SYM_TODO})")
            lo, hi, meta["mir2"] = self._ymap(lo, hi)
            meta.update(lo=lo, hi=hi)
        if self._pol == "tm":
            mon = FluxMonitorTM(self.grid, freqs, x_normal=True, idx=idx, lo=lo, hi=hi)
        else:
            mon = FluxMonitorTE(self.grid, freqs, ix=idx, lo=lo, hi=hi)
        return mon, meta

    def get_fluxes(self, name):
        name = dft_name(name)
        parts = getattr(self, "_flux_parts", {}).get(name)
        if parts:
            return [sum(v) for v in zip(*(self.get_fluxes(p) for p in parts))]
        acc, mon, meta = self._last_acc[name]
        dx = self.gs.dx if self.dimensions == 3 else self.grid.dx
        vals = mon.flux(acc, dx).detach().cpu().numpy()
        if meta.get("mir2"):
            vals = 2.0 * vals    # span crossed the mirror plane: half-cell flux
        return vals.tolist()

    def get_flux_freqs(self, name):
        name = dft_name(name)
        parts = getattr(self, "_flux_parts", {}).get(name)
        return mon_freqs(self, parts[0] if parts else name)

    def get_eigenmode_coefficients(self, name, n_modes=1):
        """S[:, port_index(name), 0] -- the coefficient at monitor `name` when
        the EigenModeSource (port 0) is excited (power-normalized S-param; see
        the _run_modes docstring for the normalization mapping)."""
        if self._S is None:
            raise RuntimeError("run() with an EigenModeSource before get_eigenmode_coefficients")
        j = self._port_index[dft_name(name)]
        return self._S[:, j, 0].detach().cpu().numpy()

    def get_array(self, component=Dielectric, center=None, size=None):
        if self.dimensions == 3:
            if component == Dielectric:
                return self._eps3d.detach().cpu().numpy()
            if self._state is None:
                raise RuntimeError("run() before get_array(fields)")
            f = getattr(self._state, component, None)
            if f is None:
                raise ValueError(f"unsupported component {component!r}")
            return f.detach().cpu().numpy()
        if component == Dielectric:
            return self._demirror(self._rm.eps_tm.detach().cpu().numpy() if self._pol == "tm"
                                  else self._rm.eps_te_xx.detach().cpu().numpy())
        if self._state is None:
            raise RuntimeError("run() before get_array(fields)")
        if component in ("Ez", "Ex"):
            return self._demirror(self._state.Ez.detach().cpu().numpy())
        if component in ("Hz", "Hy"):
            return self._demirror(self._state.Hz.detach().cpu().numpy()) if self._pol == "te" \
                else self._state.Hz if hasattr(self._state, "Hz") else None
        raise ValueError(f"unsupported component {component!r}")

    # --- meep Simulation utility methods ---
    def get_epsilon(self, frequency=0, snap=False):
        """Full-cell permittivity array (eps_inf; `frequency` must be 0)."""
        if frequency:
            raise NotImplementedError("get_epsilon(frequency != 0): eps_inf only")
        return self.get_array(component=Dielectric)

    def _cell_ij(self, pt):
        gs = self.gs
        if self.dimensions == 1:
            return max(0, min(gs.nx - 1, cell_index(pt.z, gs))), gs.ny // 2
        return (max(0, min(gs.nx - 1, cell_index(pt.x, gs))),
                max(0, min(gs.ny - 1, cell_index_y(pt.y, gs))))

    def get_epsilon_point(self, pt=None, frequency=0.0):
        """Permittivity at the cell containing meep point `pt`."""
        eps = self.get_epsilon(frequency)
        if self.dimensions == 3:
            g = self.gs
            return float(eps[max(0, min(g.nx - 1, cell_index(pt.x, g))),
                             max(0, min(g.ny - 1, cell_index_y(pt.y, g))),
                             max(0, min(g.nz - 1, cell_index_z(pt.z, g)))])
        ix, iy = self._cell_ij(pt)
        return float(eps[ix, iy])

    def get_mu(self, frequency=0, snap=False):
        """Full-cell permeability array (ones when no material sets mu)."""
        if self.dimensions == 3:
            raise NotImplementedError("3D get_mu not supported (W9 wave 3)")
        if self._cond is None:
            return np.ones((self.gs.nx, self.gs.ny))
        return self._demirror(self._cond[2].detach().cpu().numpy())

    def get_mu_point(self, pt=None, frequency=0.0):
        """Permeability at the cell containing meep point `pt`."""
        ix, iy = self._cell_ij(pt)
        return float(self.get_mu(frequency)[ix, iy])

    def get_array_metadata(self, vol=None, center=None, size=None, dft_cell=None,
                           return_pw=False):
        """(x, y, z, w): meep-centered grid-slice coordinates + cubature
        weights matching get_array/get_dft_array (2D; z is the collapsed dim).
        With return_pw=True, returns (points, w) with points as Vector3s."""
        if self.dimensions == 3:
            raise NotImplementedError("3D get_array_metadata not supported (W9 wave 3)")
        if dft_cell is not None:
            raise NotImplementedError("dft_cell regions are not tracked; pass vol/center/size")
        gs = self.gs
        if vol is not None:
            center, size = vol.center, vol.size
        if center is None and size is None:
            i0, i1, j0, j1 = 0, gs.nx, 0, gs.ny
        else:
            x0, y0, x1, y1 = rect_corners(center or Vector3(), size or Vector3(), gs)
            i0, i1 = span_index_range(x0, x1, gs, "x")
            j0, j1 = span_index_range(y0, y1, gs, "y")
        x = (np.arange(i0, i1) + 0.5) * gs.dx - gs.sx / 2.0
        y = (np.arange(j0, j1) + 0.5) * gs.dx - gs.sy / 2.0
        z = np.zeros(1)
        wshape = [n for n in (len(x), len(y)) if n > 1]
        w = np.full(wshape or [1], gs.dx ** max(1, len(wshape)))
        if return_pw:
            return [Vector3(xv, yv) for xv in x for yv in y], w
        return x, y, z, w

    def meep_time(self):
        """Current simulation time (meep units); alias of round_time()."""
        return self.round_time()

    def init_sim(self):
        """Allocate the field state (the structure is built at construction)."""
        if self._state is None:
            self.restart_fields()

    def restart_fields(self):
        """Restart the fields at time zero, with zero fields. DFT monitors are
        kept and continue to accumulate on the next run (meep semantics)."""
        if self.dimensions == 3:
            self._state = init_3d_state(self.gs.nx, self.gs.ny, self.gs.nz,
                                        npml=self._npml, dtype=self.dtype)
        else:
            self._state = self._init_state()
        self._t_step = 0

    def reset_meep(self):
        """Delete the fields and all accumulated monitor data, as if no
        computation had been run (the rasterized structure is kept: it is
        rebuilt deterministically from the constructor inputs)."""
        self._state = None
        self._t_step = 0
        self._monitors = []
        self._last_acc = {}
        self._mode_ports = []
        self._S = None
        self._port_index = {}
        for a in ("_flux_parts", "_n2f_specs", "_mode_freqs"):
            if hasattr(self, a):
                delattr(self, a)

    def has_mu(self):
        """True when any material (geometry or default) has mu != 1."""
        if self.dimensions == 3 or self._cond is None:
            return False
        return bool((self._cond[2] != 1.0).any().item())

    def using_real_fields(self):
        """Always True: this engine has no k_point/complex-field path."""
        return True

    def plot2D(self, ax=None, output_plane=None, fields=None, **kwargs):
        """Delegate to mp.visualization.plot2D (meep signature)."""
        from .visualization import plot2D
        return plot2D(self, ax=ax, output_plane=output_plane, fields=fields,
                      **kwargs)


def mon_freqs(sim, name):
    name = dft_name(name)
    for (n, _, mon, _) in sim._monitors:
        if n == name:
            return mon.freqs.detach().cpu().numpy().tolist()
    raise KeyError(name)