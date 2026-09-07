# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""mp.Simulation -- assembles and runs a meep-torch 2D FDTD from meep syntax.

Adapter only (no physics), following meep's object flow: __init__ stores the
spec; init_sim (called lazily by run() and the monitor adders) builds
``self.structure`` (mp.structure, single owner of the rasterized materials and
stepping coefficients, via _init_structure) and ``self.fields``
(mp.fields(structure), which owns the field state and steps the engine);
run()/_run_until drive fields.step / fields.step_n. The 3D and S-param paths
keep their engines (core/stepper_3d, ports/smatrix) behind the same facade.
dimensions=1 maps the meep z axis to the meep-torch x axis with y_pmc=True and
a thin ny. dtype= kwarg passes through to Grid (float64 default; float32
opt-in, validated at scale in tests/autodiff/test_float32.py).
"""
import math
import os
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from ..core.grid import Grid, default_device
from ..core.materials import Material
from ..core.monitors import DFTFieldsMonitor, FluxMonitorTE, FluxMonitorTM, PointProbe
from ..core.monitors_3d import DFTFieldsMonitor3D, FluxMonitor3D, PlaneSource, PointSource3D, run_3d
from ..core.stepper_3d import build_3d_coeffs, init_3d_state
from ..geometry.scene import rasterize_scene
from .cyl_shim import CYLINDRICAL
from .coords import (GridSpec, box3d, cell_index, cell_index_y, cell_index_z, grid_spec,
                     pml_npml, rect_corners, span_index_range, to_corner)
from .geometry import Block, build_scene_items
from .material import ALL, Absorber, Low, Medium, PML
from .run_control import (DftEnergy, DftFields, DftFlux, DftForce, DftLdos,
                          DftNear2Far, DftObj, EigCoeffsResult, FluxData,
                          _eval_step_func, chunk_size, dft_name, fire_per_step,
                          fire_scheduled, plan_hooks)
from .sources import (AUTOMATIC, EVEN_Z, NO_PARITY, ODD_Z, Dielectric,
                      DiffractedPlanewave, EigenModeSource, Ez,
                      GaussianBeam3DSource, Hz, IndexedSource, Source,
                      _Quadrature, _point_or_span, build_line_source,
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

    Engine support: Mirror(X) and Mirror(Y), phase +1 (even) or -1 (odd), on
    a TM 2D run -- each halves the domain (core x_mirror/y_mirror; both at
    once run the quarter cell). Off-plane sources imply their mirror image
    (place each physical source once, on the kept side or on the plane).
    TE runs accept mirrors as a consistency hint but keep the full cell.
    Rotations raise NotImplementedError (%s).""" % _SYM_TODO


class Identity(Symmetry):
    """The identity symmetry (a no-op)."""


def _layer_axes(direction, dimensions, cell_size=None) -> tuple[int, ...]:
    """Grid axes a boundary layer's meep ``direction`` covers. A 1D cell
    lies along z but the engine grid puts it on x (see _grid_for).

    ``direction=ALL`` covers only axes the cell actually spans: meep's
    grid_volume has no boundary on a direction of zero extent, so a
    degenerate axis stays PEC instead of taking a PML thicker than it.
    """
    if dimensions == 1:
        if direction in (ALL, Z):
            return (0,)
        raise ValueError(
            f"boundary layer direction {direction}: a 1D cell extends along "
            f"z only (use direction=mp.Z or mp.ALL)")
    naxes = 3 if dimensions == 3 else 2
    if direction == ALL:
        axes = tuple(range(naxes))
        if cell_size is not None:
            extent = (cell_size.x, cell_size.y, cell_size.z)
            spanned = tuple(ax for ax in axes if extent[ax])
            if spanned:
                return spanned
        return axes
    if 0 <= direction < naxes:                    # mp.X/Y(/Z)
        return (direction,)
    raise ValueError(
        f"boundary layer direction {direction} has no extent in a "
        f"{dimensions}D cell")


def _grid_for(cell: Vector3, resolution: float, dimensions: int, dtype,
              courant: float = 0.5,
              device: str | None = None) -> tuple[Grid, GridSpec, bool]:
    if dimensions == 1:
        sx = float(cell.z)
        dx = 1.0 / float(resolution)
        nx = max(3, int(round(sx / dx)))
        ny = 3
        g = Grid(nx=nx, ny=ny, dx=dx, dtype=dtype, courant=courant,
                 device=device or default_device())
        gs = GridSpec(nx, ny, dx, sx, ny * dx)
        return g, gs, True
    gs = grid_spec(cell, resolution)
    g = Grid(nx=gs.nx, ny=gs.ny, dx=gs.dx, dtype=dtype, courant=courant,
             device=device or default_device())
    return g, gs, False


@dataclass
class FluxRegion:
    center: Vector3 = None
    size: Vector3 = None
    direction: int = 1
    weight: complex = 1.0      # multiplies the retrieved flux (meep)
    volume: object = None      # mp.Volume; supplies center/size when given

    def __post_init__(self):
        if self.volume is not None:
            self.center = self.volume.center
            self.size = self.volume.size


# meep: ModeRegion is an alias of FluxRegion for add_mode_monitor
ModeRegion = FluxRegion


class ForceRegion(FluxRegion):
    """meep ForceRegion: a face of the closed contour handed to add_force;
    ``direction`` is the force component to compute (mp.X / mp.Y), ``weight``
    multiplies it. The union of the regions should enclose the object in
    vacuum (2D TM here; see Simulation.add_force)."""


class EnergyRegion(FluxRegion):
    """meep EnergyRegion: region handed to add_energy over which the DFT
    field-energy density is integrated (``weight`` multiplies it)."""


@dataclass
class Near2FarRegion:
    center: Vector3 = None
    size: Vector3 = None
    direction: int = AUTOMATIC  # meep default: the region normal (weight signs it here)
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
    """meep mp.Simulation with the full meep 1.34 constructor surface.

    Parameters with real engine backing beyond the W1 set: ``Courant``
    (passes through to the engine Grid; dt = Courant * dx) and
    ``geometry_center`` (an object at coordinate c lands at c -
    geometry_center in the cell -- geometry only, like meep; verified
    against pymeep 1.34). Accepted-and-stored (no single-process / no-file
    meaning, or numerical knobs the engine fixes): eps_averaging (the
    rasterizer always area-averages; False warns), kz_2d, extra_materials,
    progress_interval, subpixel_tol/subpixel_maxeval, loop_tile_base_db/eh,
    ensure_periodicity, num_chunks / split_chunks_evenly / chunk_layout /
    collect_stats (whole-domain tensors, no chunking), filename_prefix,
    output_single_precision, accurate_fields_near_cylorigin. k_point runs the
    Bloch-periodic path (mp/bloch_shim.py; unsupported combos raise at run
    time). Raise on non-default:
    force_complex_fields (real-field engine; cylindrical runs are exempt),
    force_all_components, bfast_scaled_k, material_function / epsilon_func /
    epsilon_input_file (no pointwise material sampling), output_volume."""

    def __init__(self, *, cell_size, resolution, geometry=None, sources=None,
                 eps_averaging=True, dimensions=2, boundary_layers=None,
                 symmetries=None, force_complex_fields=False,
                 default_material=Medium(epsilon=1.0), m=0, bfast_scaled_k=None,
                 k_point=False, kz_2d="complex", extra_materials=None,
                 material_function=None, epsilon_func=None,
                 epsilon_input_file="", progress_interval=4, subpixel_tol=1e-4,
                 subpixel_maxeval=100000, loop_tile_base_db=0,
                 loop_tile_base_eh=0, ensure_periodicity=True, num_chunks=0,
                 Courant=0.5, accurate_fields_near_cylorigin=False,
                 filename_prefix=None, output_volume=None,
                 output_single_precision=False, geometry_center=Vector3(),
                 force_all_components=False, split_chunks_evenly=True,
                 chunk_layout=None, collect_stats=False, dtype=torch.float64,
                 device=None):
        cs = Vector3(*cell_size)
        if dimensions in (2, 3) and cs.z and not cs.x and not cs.y:
            # meep's grid_volume infers a z-only cell as 1D whatever the
            # dimensions argument says (faraday-rotation runs it with the
            # default 2, refl-angular passes 3 for its oblique k)
            dimensions = 1
        self.dimensions = dimensions
        self.dtype = dtype
        # meep-torch extension: engine device. None -> default_device()
        # (MEEP_TORCH_DEVICE, else CUDA when present).
        self.device = device or default_device()
        self.cell_size = cell_size
        self.resolution = resolution
        self.run_index = 0           # meep: bumped after each run()
        if force_complex_fields and dimensions != CYLINDRICAL:
            raise NotImplementedError(
                "force_complex_fields: the engine steps real fields only "
                "(using_real_fields() is always True). Cylindrical runs are "
                "exempt: the BOR engine carries complex F(r, z) e^{i m phi} "
                "fields by construction")
        # k_point: Bloch-periodic run (mp/bloch_shim.py drives
        # core/stepper_bloch); unsupported combos raise there at run time
        self._bloch = k_point not in (False, None)
        self._bloch_state = None
        self.m = m
        if m and dimensions != CYLINDRICAL:
            raise NotImplementedError(
                f"m = {m} is the azimuthal number of a cylindrical run; pass "
                "dimensions=mp.CYLINDRICAL with it (cell_size = "
                "Vector3(sr, 0, sz))")
        if bfast_scaled_k is not None:
            raise NotImplementedError(
                "bfast_scaled_k (BFAST oblique planewaves) is not supported")
        if material_function is not None or epsilon_func is not None:
            raise NotImplementedError(
                "material_function/epsilon_func: no pointwise material "
                "sampling; describe the structure with geometry + Medium")
        if epsilon_input_file:
            raise NotImplementedError(
                "epsilon_input_file (HDF5 structure import) is not supported")
        if output_volume is not None:
            raise NotImplementedError(
                "output_volume: the output functions write the full cell")
        if force_all_components:
            raise NotImplementedError(
                "force_all_components: the 2D TM/TE component sets are fixed")
        if not eps_averaging:
            warnings.warn(
                "eps_averaging=False has no staircase path: the engine "
                "rasterizer always area-averages polygon coverage "
                "(meep_torch/geometry/scene.py)")
        self.eps_averaging = eps_averaging
        self.m = m
        self.k_point = k_point
        self.kz_2d = kz_2d
        self.extra_materials = list(extra_materials or [])
        self.progress_interval = progress_interval
        self.subpixel_tol = subpixel_tol
        self.subpixel_maxeval = subpixel_maxeval
        self.loop_tile_base_db = loop_tile_base_db
        self.loop_tile_base_eh = loop_tile_base_eh
        self.ensure_periodicity = ensure_periodicity
        self.num_chunks = num_chunks
        self.split_chunks_evenly = split_chunks_evenly
        self.chunk_layout = chunk_layout
        self.collect_stats = collect_stats
        self.Courant = float(Courant)
        self.accurate_fields_near_cylorigin = accurate_fields_near_cylorigin
        self.filename_prefix = filename_prefix
        self.output_single_precision = output_single_precision
        gc = Vector3(*geometry_center)
        self.geometry_center = gc
        geometry = list(geometry or [])
        if gc.x or gc.y or gc.z:
            # meep: an object at coordinate c lands at c - geometry_center in
            # the cell (geometry_center names the coordinate of the cell
            # center); sources/monitors are NOT shifted, matching meep
            geometry = [obj.shift(-gc) for obj in geometry]
        self.symmetries = list(symmetries or [])   # meep-mutable, like sources
        # meep flow: __init__ stores the spec only; init_sim (or the lazy
        # accessors) builds self.structure and self.fields.
        self._sources = list(sources or [])
        self._geometry = list(geometry)        # already geometry_center-shifted
        self._monitors = []          # list of (name, kind, mon, meta)
        self._last_acc = {}
        self._mode_ports = []        # list of (name, axis, idx, lo, hi, direction)
        self._S = None
        self._port_index = {}       # monitor name -> port index in compute_smatrix
        self._t_step = 0
        self.structure = None        # built by init_sim -> _init_structure
        self.fields = None           # built by init_sim
        if dimensions == CYLINDRICAL:
            # mp/cyl_shim.py owns the (r, z) grid and the BOR stepper state;
            # _grid_for/pol_for are cartesian-only and pol_for raises on the
            # cylindrical components (Er/Ep/Hr/Hp) outright.
            self.grid = self.gs = None
            self._y_pmc = False
            self._pol = "cyl"
            self._eig_source = None
            self._state3 = None
            self._cyl = None                  # CylRun, built on first run()
        elif dimensions == 3:
            if self.Courant != 0.5:
                raise NotImplementedError(
                    "3D Courant is fixed at 0.5 (core/stepper_3d builds its "
                    "own dt); 2D/1D runs pass Courant through to the Grid")
            self.gs = grid_spec(cell_size, resolution, dimensions=3)
            self.grid = None                  # no core/Grid in 3D; use gs + coeffs
            self._y_pmc = False
            self._pol = "tm"
            self._eig_source = None           # 3D has no S-param path this wave
            self._state3 = None
        else:
            self.grid, self.gs, self._y_pmc = _grid_for(cell_size, resolution,
                                                        dimensions, dtype,
                                                        courant=self.Courant,
                                                        device=self.device)
            self._pol = pol_for(self._sources[0].component) if self._sources else "tm"
            # S-param path: an EigenModeSource + mode monitors -> compute_smatrix.
            self._eig_source = next((s for s in self._sources
                                     if isinstance(s, EigenModeSource)), None)
        # per-grid-axis [lo, hi] cell counts (meep direction/side semantics;
        # edges with no layer stay PEC). _npml/_nabs keep the max for the
        # placement margins and 3D psi allocation.
        self.boundary_layers = list(boundary_layers or [])
        self._npml_sides = [[0, 0], [0, 0], [0, 0]]
        self._nabs_sides = [[0, 0], [0, 0], [0, 0]]
        # Cylindrical keeps the layer list (cyl_shim._absorber_cells reads it
        # straight off the Simulation) but not the cartesian cell counts:
        # there is no GridSpec to measure a thickness against, and the BOR
        # stepper places its own matched absorber ramp on outer r and both z.
        for bl in (() if dimensions == CYLINDRICAL else self.boundary_layers):
            if isinstance(bl, PML):
                sides = self._npml_sides
            elif isinstance(bl, Absorber):
                sides = self._nabs_sides
            else:
                continue
            n = pml_npml(bl.thickness, self.gs)
            for ax in _layer_axes(bl.direction, dimensions, cs):
                for s in ((0, 1) if bl.side == ALL else
                          ((0,) if bl.side == Low else (1,))):
                    sides[ax][s] = max(sides[ax][s], n)
        self._npml = max(max(p) for p in self._npml_sides)
        self._nabs = max(max(p) for p in self._nabs_sides)
        self._default = default_material if isinstance(default_material, Medium) else Medium(epsilon=float(default_material))
        self._apply_symmetries()

    def _apply_symmetries(self):
        """Derive the mirror-halving state (_sym_cx/_sym_cy = full-cell
        cols/rows below the cut plane, 0 = off; _sym_px/_sym_py = mirror
        phase +1 even / -1 odd) from ``self.symmetries``. Re-run by init_sim
        while the fields are unbuilt: meep's sim.symmetries is mutable."""
        self._sym_cx = self._sym_cy = 0
        self._sym_px = self._sym_py = 1
        self._sym_rot2 = 0
        self._sym_rot4 = 0
        mirrors = {}
        rot2 = rot4 = 0
        for s in self.symmetries:
            if isinstance(s, Identity):
                continue
            if (isinstance(s, Rotate4) and s.direction == Z
                    and s.phase in (1 + 0j, -1 + 0j)):
                rot4 = int(s.phase.real)
                continue
            if (isinstance(s, Rotate2) and s.direction == Z
                    and s.phase in (1 + 0j, -1 + 0j)):
                rot2 = int(s.phase.real)
                continue
            if (isinstance(s, Mirror) and not isinstance(s, (Rotate2, Rotate4))
                    and s.direction in (X, Y) and s.phase in (1 + 0j, -1 + 0j)):
                mirrors[s.direction] = int(s.phase.real)
                continue
            raise NotImplementedError(
                f"{type(s).__name__}(direction={getattr(s, 'direction', None)}, "
                f"phase={getattr(s, 'phase', None)}) not supported: "
                f"Mirror(X/Y, phase=+-1), Rotate2(Z, phase=+-1) and "
                f"Rotate4(Z, phase=+-1) only ({_SYM_TODO})")
        if (rot2 or rot4) and mirrors:
            raise NotImplementedError(
                f"Rotate2/Rotate4 combined with Mirror not supported: a "
                f"rotation fold and a mirror cut do not compose in the "
                f"half-domain engine ({_SYM_TODO})")
        if rot2 and rot4:
            raise NotImplementedError(
                f"Rotate2 with Rotate4: C4 already contains C2, list Rotate4 "
                f"alone ({_SYM_TODO})")
        if (mirrors or rot2 or rot4) and self.dimensions != 2:
            raise NotImplementedError(
                f"Mirror(X/Y) / Rotate2(Z) / Rotate4(Z) is 2D-only "
                f"({_SYM_TODO})")
        if (mirrors or rot2 or rot4) and self._bloch:
            # meep treats symmetries as an optimization hint; the Bloch
            # (k_point) path has no half-domain machinery, so keep the full
            # cell (same physics)
            warnings.warn("Symmetries on a k_point run do not reduce the "
                          f"domain (full-cell run; {_SYM_TODO})")
            mirrors, rot2, rot4 = {}, 0, 0
        if (rot2 or rot4) and self._pol != "tm":
            # Rotate2/Rotate4 fold the TM engine only; the TE engine has no
            # rotation path, so keep the full cell (same physics)
            warnings.warn("Rotate2/Rotate4 on a TE run keep the full cell "
                          f"({_SYM_TODO})")
            rot2, rot4 = 0, 0
        if Y in mirrors:
            if self.gs.ny % 2:
                raise NotImplementedError(
                    f"Mirror(Y) needs an even cell row count ({_SYM_TODO})")
            self._sym_cy = self.gs.ny // 2
            self._sym_py = mirrors[Y]
        if X in mirrors:
            if self.gs.nx % 2:
                raise NotImplementedError(
                    f"Mirror(X) needs an even cell column count ({_SYM_TODO})")
            self._sym_cx = self.gs.nx // 2
            self._sym_px = mirrors[X]
        if rot2:
            # C2 about z folds (i, j) -> (nx-1-i, ny-1-j). Halve in y (the
            # kept half keeps the FULL x extent, since the fold reverses x
            # rather than cutting it), so the y rules match a mirror's and
            # the x extent must be reversal-symmetric: even column count and
            # an x PML that is the same on both edges.
            if self.gs.ny % 2:
                raise NotImplementedError(
                    f"Rotate2(Z) needs an even cell row count ({_SYM_TODO})")
            if self.gs.nx % 2:
                raise NotImplementedError(
                    f"Rotate2(Z) needs an even cell column count ({_SYM_TODO})")
            if self._npml_sides[0][0] != self._npml_sides[0][1] or \
                    self._nabs_sides[0][0] != self._nabs_sides[0][1]:
                raise NotImplementedError(
                    f"Rotate2(Z) needs an x-symmetric absorber: the C2 fold "
                    f"maps the low-x edge onto the high-x edge, so unequal "
                    f"PML/Absorber thicknesses break the symmetry "
                    f"({_SYM_TODO})")
            self._sym_cy = self.gs.ny // 2
            self._sym_py = rot2
            self._sym_rot2 = rot2
        if rot4:
            # C4 about z folds the cell onto one quadrant and carries the y
            # cut onto the x cut, so the kept quadrant must be SQUARE and the
            # absorber identical on all four edges.
            if self.gs.nx % 2 or self.gs.ny % 2:
                raise NotImplementedError(
                    f"Rotate4(Z) needs even cell counts ({_SYM_TODO})")
            if self.gs.nx != self.gs.ny:
                raise NotImplementedError(
                    f"Rotate4(Z) needs a square cell, got "
                    f"{self.gs.nx}x{self.gs.ny} ({_SYM_TODO})")
            four = lambda t: set(tuple(t[0]) + tuple(t[1]))
            if len(four(self._npml_sides)) > 1 or len(four(self._nabs_sides)) > 1:
                raise NotImplementedError(
                    f"Rotate4(Z) needs the same absorber on all four edges: "
                    f"the C4 fold carries every edge onto the next "
                    f"({_SYM_TODO})")
            self._sym_cx = self.gs.nx // 2
            self._sym_cy = self.gs.ny // 2
            self._sym_px = self._sym_py = rot4
            self._sym_rot4 = rot4

    def __getattr__(self, name):
        # deferred-raster attributes: meep builds the structure in init_sim,
        # not __init__ -- first touch triggers _init_structure
        if name in ("_rm", "_cond", "_eps3d", "_coeffs3d"):
            self._init_structure()
            if name in self.__dict__:
                return self.__dict__[name]
        raise AttributeError(name)

    def _component_set(self):
        """Required field components, inferred from the source spec (meep
        require_component flow); the structure holds materials for any set."""
        from .cpp.core import _TE_COMPONENTS, _TM_COMPONENTS
        return _TE_COMPONENTS if self._pol == "te" else _TM_COMPONENTS

    def _init_structure(self):
        """meep Simulation._init_structure: rasterize the geometry and build
        ``self.structure`` -- the mp.structure owning the material tensors and
        the compiled stepping coefficients. Idempotent; init_sim, the monitor
        adders and the lazy accessors call it exactly where meep does."""
        if self.dimensions == 3:
            if self.__dict__.get("_coeffs3d") is None:
                self._init_3d()
            return
        if self.structure is not None:
            return
        if self._sym_cy or self._sym_cx:
            # rebuild the full-cell grid for rasterization (halved again below)
            self.grid = Grid(nx=self.gs.nx, ny=self.gs.ny, dx=self.gs.dx,
                             dtype=self.dtype, courant=self.Courant)
        eps_items, chi3_items, cond_items = build_scene_items(
            self._geometry, self.gs, self.dimensions, with_cond=True)
        bg_val, bg_chi3 = self._default.resolve()
        if bg_chi3 != 0.0:
            full = np.array([[0, 0], [self.gs.sx, 0], [self.gs.sx, self.gs.sy], [0, self.gs.sy]])
            from ..geometry.polygons import Polygon
            chi3_items.append((Polygon(full), bg_chi3))
        self._rm = rasterize_scene(eps_items, self.grid, background=bg_val, chi3_items=chi3_items)
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
        if self._sym_cy or self._sym_cx:
            # Mirror halving: rasterize the full cell (above), then run the
            # engine on the kept half/quarter with a mirror wall at each cut
            # plane (build_tm_coeffs x_mirror/y_mirror; the parity picks the
            # even/odd wall). self.gs stays full-cell for all coordinate
            # conversion; spans are folded via _fold/_foldpt and outputs
            # de-mirrored (mirrored half = phase * flip).
            cy, cx = self._sym_cy, self._sym_cx
            self.grid = Grid(nx=self.gs.nx - cx, ny=self.gs.ny - cy,
                             dx=self.gs.dx, dtype=self.dtype,
                             courant=self.Courant)
            half = lambda t: None if t is None else t[cx:, cy:].contiguous()
            self._rm = self._rm._replace(
                eps_tm=half(self._rm.eps_tm), eps_te_xx=half(self._rm.eps_te_xx),
                eps_te_yy=half(self._rm.eps_te_yy),
                pole_masks=[half(m) for m in self._rm.pole_masks],
                chi3=half(self._rm.chi3))
            if self._cond is not None:
                self._cond = tuple(half(t) for t in self._cond)
        if self._pol == "te" and self._y_pmc:
            raise NotImplementedError("1D (y-PMC) TE Simulation is not supported")
        from .cpp.core import structure
        self.structure = structure._from_sim(self)

    @property
    def _coeffs(self):
        """Compiled stepping coefficients; ``self.structure`` is their single
        owner (built lazily by _init_structure, meep style)."""
        if self.dimensions == 3:
            return self._coeffs3d            # lazy via __getattr__
        if self.structure is None:
            self._init_structure()
        return self.structure.coeffs_for(self._component_set())

    @property
    def _state(self):
        """Live engine field state: ``self.fields`` owns it (meep); the 3D
        and Bloch paths keep their tensor state behind the same facade
        (Bloch quadratures surface as complex tensors, meep complex fields)."""
        if self.dimensions == 3:
            return self._state3
        st = self._bloch_state
        if st is not None:
            from types import SimpleNamespace
            if hasattr(st, "Ez_c"):
                return SimpleNamespace(Ez=st.Ez_c + 1j * st.Ez_s,
                                       Hx=st.Hx_c + 1j * st.Hx_s,
                                       Hy=st.Hy_c + 1j * st.Hy_s)
            return SimpleNamespace(Hz=st.Hz_c + 1j * st.Hz_s,
                                   Ex=st.Ex_c + 1j * st.Ex_s,
                                   Ey=st.Ey_c + 1j * st.Ey_s)
        return None if self.fields is None else self.fields._state

    def _absorber_ramp(self):
        """Quadratic sigma ramp on every absorbing edge (R_asymptotic 1e-15)."""
        from ..core.pml import absorber_sigma
        nx, ny, dx = self.grid.nx, self.grid.ny, self.grid.dx
        kw = dict(dtype=self.grid.dtype, device=self.grid.device)
        ramp = absorber_sigma(nx, tuple(self._nabs_sides[0]), dx,
                              **kw)[:, None].expand(nx, ny).clone()
        if not self._y_pmc:
            ramp = torch.maximum(
                ramp, absorber_sigma(ny, tuple(self._nabs_sides[1]), dx,
                                     **kw)[None, :].expand(nx, ny))
        return ramp

    # ----------------------------------------------------------------------
    # 3D path (W9 wave 3). Adapter only: builds the eps/conductivity volumes,
    # calls build_3d_coeffs/init_3d_state, and drives run_3d. Axis-aligned
    # Block only (voxel fill, no subpixel); scalar eps + D/B-conductivity only
    # (no Lorentz/Kerr in 3D this wave). See docs/specs/...-w9-wave3-3d-shim.
    # ----------------------------------------------------------------------
    def _init_3d(self):
        g = self.gs
        dtype = self.dtype
        nx, ny, nz = g.nx, g.ny, g.nz
        bg = self._default
        bg_eps, bg_chi3 = bg.resolve()
        if isinstance(bg_eps, Material) or bg_chi3 != 0.0:
            raise NotImplementedError("3D dispersive/Kerr default material not supported (W9 wave 3)")
        eps = torch.full((nx, ny, nz), float(bg_eps), dtype=dtype,
                         device=self.device)
        bg_sd = float(getattr(bg, "D_conductivity", 0.0) or 0.0)
        bg_sb = float(getattr(bg, "B_conductivity", 0.0) or 0.0)
        sd = torch.full((nx, ny, nz), bg_sd, dtype=dtype, device=self.device)
        sb = torch.full((nx, ny, nz), bg_sb, dtype=dtype, device=self.device)
        for obj in self._geometry:
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
        self._coeffs3d = build_3d_coeffs(
            nx, ny, nz, g.dx, eps=eps,
            npml=tuple(tuple(p) for p in self._npml_sides),
            n_abs=tuple(tuple(p) for p in self._nabs_sides),
            sigma_d=(sd if has_cond else None), sigma_b=(sb if has_cond else None),
            dtype=dtype, device=self.device)

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
                src = PlaneSource(g.nx, g.ny, g.nz, dt, comp, axis, idx, prof,
                                  device=self.device)
            else:
                ap = self._npml
                src = PlaneSource(g.nx, g.ny, g.nz, dt, comp, axis, idx, prof,
                                  lo=ap, hi=min(g.nx, g.ny, g.nz) - ap,
                                  device=self.device)
        return _Amp(src, amp)

    def _run_3d(self, n_steps):
        self._state3 = init_3d_state(self.gs.nx, self.gs.ny, self.gs.nz,
                                     npml=self._npml, dtype=self.dtype,
                                     device=self.device)
        src = self._make_sources_3d()
        monitors = [m for (_, _, m, _) in self._monitors]
        self._state3, outs = run_3d(self._state3, self._coeffs, n_steps,
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

    def _add_flux_3d(self, freqs, region, name):
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

    def _build_indexed(self, spec):
        """IndexedSource -> LineSource with the per-point amp array as mask."""
        if self._sym_cy or self._sym_cx:
            raise NotImplementedError(
                f"IndexedSource with Mirror symmetry not supported ({_SYM_TODO})")
        ix, j0, j1 = _point_or_span(spec.center, spec.size, self.gs)
        if j1 - j0 != len(spec.amp_arr):
            raise ValueError(
                f"IndexedSource amp_arr has {len(spec.amp_arr)} points but "
                f"the srcdata region spans {j1 - j0} cells")
        amps = np.asarray(spec.amp_arr)
        if np.iscomplexobj(amps):
            if np.abs(amps.imag).max() > 1e-12 * max(np.abs(amps).max(), 1e-300):
                raise NotImplementedError(
                    "IndexedSource with complex amplitudes: the engine injects "
                    "a real spatial mask times a real time profile (no "
                    "per-point phase)")
            amps = amps.real
        from ..core.sources import LineSource
        # Pass the per-cell weights in rather than overwriting src.mask
        # afterwards: overwriting discarded LineSource's normalization, so an
        # IndexedSource and an equivalent uniform line source disagreed.
        src = LineSource(self.grid, ix, j0, j1, profile=spec.src.profile(),
                         amps=amps)
        if self._pol == "te":
            src.mask = src.mask[:, :-1]
        return src

    def _make_sources(self):
        out = []
        for spec in self._sources:
            if isinstance(spec, IndexedSource):
                out.append(self._build_indexed(spec))
                continue
            if type(spec) is GaussianBeam3DSource:
                raise NotImplementedError(
                    "GaussianBeam3DSource is the exact 3D complex-point-source "
                    "beam, which the 2D real-field engine cannot inject; use "
                    "GaussianBeam2DSource/GaussianBeamSource (amplitude-envelope "
                    "launch)")
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
            if self._sym_cy or self._sym_cx:
                if amp_func is not None:
                    raise NotImplementedError(
                        f"amp_func with Mirror symmetry not supported ({_SYM_TODO})")
                # a source below a cut plane folds onto its image with the
                # mirror phase (Ez transforms with the user phase); a span
                # crossing the plane keeps its upper part -- the wall implies
                # the mirrored image, so each physical source is listed once
                ix, mirx = self._foldpt(ix, "x")
                if self._sym_rot4 and mirx:
                    raise NotImplementedError(
                        f"a source outside the Rotate4(Z) quadrant is not "
                        f"folded: which power of the quarter turn carries it "
                        f"back depends on its quadrant. Place the source in "
                        f"the kept quadrant ({_SYM_TODO})")
                if mirx:
                    amp = amp * self._sym_px
                j0, j1, crossed_y, miry = self._fold(j0, j1, "y")
                if self._sym_rot4 and (miry or crossed_y):
                    raise NotImplementedError(
                        f"a source outside the Rotate4(Z) quadrant is not "
                        f"folded: which power of the quarter turn carries it "
                        f"back depends on its quadrant. Place the source in "
                        f"the kept quadrant ({_SYM_TODO})")
                if miry:
                    amp = amp * self._sym_py
                    if self._sym_rot2:
                        # C2 image: the y fold above placed the span, the
                        # rotation also reverses x
                        ix = self.grid.nx - 1 - ix
            from ..core.sources import LineSource
            src = LineSource(self.grid, ix, j0, j1, profile=prof)
            if amp_func is not None:
                # overwrite the uniform span mask with the spatial profile across y
                c = center or Vector3()
                ys = [c.y - gs.sy / 2 + (j + 0.5) * gs.dx for j in range(j0, j1)]
                w = np.array([complex(amp_func(Vector3(c.x, y, 0.0))) for y in ys])
                src.mask = self.grid.zeros(self.grid.nx, self.grid.ny)
                src.mask[ix, j0:j1] = torch.tensor(w.real, dtype=self.grid.dtype,
                                                device=self.grid.device)
                if np.abs(w.imag).max() > 0.0:
                    # complex amplitude on real fields (meep: inject
                    # Re[a * s(t)]): the imaginary part rides the
                    # quadrature-shifted carrier as a second soft source
                    fcen = getattr(spec.src, "frequency", None)
                    if not fcen:
                        raise NotImplementedError(
                            "complex amp_func needs a source with a carrier "
                            "frequency (GaussianSource/ContinuousSource)")
                    imsrc = LineSource(self.grid, ix, j0, j1,
                                       profile=_Quadrature(prof, fcen))
                    imsrc.mask = self.grid.zeros(self.grid.nx, self.grid.ny)
                    imsrc.mask[ix, j0:j1] = torch.tensor(w.imag,
                                                         dtype=self.grid.dtype)
                    imsrc.mask = imsrc.mask * amp
                    if self._pol == "te":
                        imsrc.mask = imsrc.mask[:, :-1]
                    out.append(imsrc)
            src.mask = src.mask * amp
            if self._pol == "te":
                # step_te injects into Dy (Ey grid, (nx, ny-1))
                src.mask = src.mask[:, :-1]
            out.append(src)
        return out

    # --- mirror folding helpers (all no-ops with no Mirror symmetry) ---
    def _fold(self, lo, hi, axis="y"):
        """Full-cell index span -> half-grid span along ``axis``. Returns
        (lo, hi, crossed, mirrored): crossed = the span reached below the cut
        plane (assumed symmetric about it; its flux doubles back), mirrored =
        the span lies entirely below the plane (mapped onto its image)."""
        c = self._sym_cy if axis == "y" else self._sym_cx
        if not c:
            return lo, hi, False, False
        if hi <= c:
            return c - hi, c - lo, False, True
        return max(0, lo - c), max(1, hi - c), lo < c, False

    def _foldpt(self, j, axis="y"):
        """Full-cell index -> (half-grid index, mirrored) along ``axis``."""
        c = self._sym_cy if axis == "y" else self._sym_cx
        if not c:
            return j, False
        return (j - c, False) if j >= c else (c - 1 - j, True)

    def _rot2_guard(self, what, *below_plane):
        """Reject a monitor that reaches below the C2 cut plane. A mirror
        image sits at the same x, so the folded half plus a phase is the
        whole answer; a C2 image sits at nx-1-i, i.e. on a DIFFERENT line of
        the kept half, so folding a crossing region would need the partner
        line summed in as well. Regions wholly in the kept half are exact."""
        if (self._sym_rot2 or self._sym_rot4) and any(below_plane):
            raise NotImplementedError(
                f"{what} reaching beyond the rotation cut plane is not "
                f"supported: the C2 image of a point is at (nx-1-i, ny-1-j), "
                f"so the region maps onto a different line of the kept "
                f"half. Place the region inside the kept half/quadrant "
                f"({_SYM_TODO})")

    def _ypoint(self, j):
        """Full-cell y index -> half-grid row (mirrored below the plane)."""
        return self._foldpt(j, "y")[0]

    def _demirror(self, a, parity=(1.0, 1.0)):
        """Half-grid array -> full-cell (nx, ny): mirrored half = phase *
        flip. ``parity`` = (px, py) of the stored component under the X/Y
        mirror (Dielectric/mu are even; Ez carries the mirror phases)."""
        if self._sym_rot4:
            # Quadrant -> full cell. With Ez(r) = p Ez(R r) and R = C4, the
            # three missing quadrants are the kept one rotated: a quarter turn
            # is a transpose plus a flip, and the half turn (p^2 = 1) is the
            # plain double flip.
            p = float(self._sym_py)
            q = a
            hi_x = np.concatenate([p * q[::-1, :].T, q], axis=1)   # j < cy | j >= cy
            lo_x = np.concatenate([q[::-1, ::-1], p * q[:, ::-1].T], axis=1)
            return np.concatenate([lo_x, hi_x], axis=0)
        if self._sym_cy:
            lower = a[:, ::-1] if not self._sym_rot2 else a[::-1, ::-1]
            a = np.concatenate([parity[1] * lower, a], axis=1)
        if self._sym_cx:
            a = np.concatenate([parity[0] * a[::-1, :], a], axis=0)
        return a

    def _comp_parity(self, comp):
        """(px, py) mirror parity of a TM component (E vector / H pseudo-
        vector transform times the user phase)."""
        px, py = float(self._sym_px), float(self._sym_py)
        if self._sym_rot4:
            # C4 rotates Hx into Hy, so no per-component parity exists for
            # the staggered pair; Ez/Dz are scalars and de-rotate cleanly.
            if comp not in ("Ez", "Dz", "Dielectric", "Permittivity", None):
                raise NotImplementedError(
                    f"get {comp} under Rotate4(Z): a quarter turn maps Hx "
                    f"onto Hy, so the component has no scalar parity to "
                    f"de-rotate with ({_SYM_TODO})")
            return (1.0, 1.0)
        if self._sym_rot2:
            # C2 about z: Ez is a scalar (unchanged), the in-plane H vector
            # reverses -- both components pick up -1 on top of the user phase
            return {"Ez": (1.0, py), "Dz": (1.0, py),
                    "Hx": (1.0, -py), "Hy": (1.0, -py)}.get(comp, (1.0, 1.0))
        return {"Ez": (px, py), "Dz": (px, py),
                "Hx": (px, -py), "Hy": (-px, py)}.get(comp, (1.0, 1.0))

    def _source_cutoff(self):
        """Latest source turn-off time (GaussianPulse.t_end; CW never ends -> 0)."""
        return max((float(getattr(s.src.profile(), "t_end", 0.0))
                    for s in self._sources), default=0.0)

    def round_time(self):
        if self.dimensions == CYLINDRICAL:
            # the BOR run owns its own clock; step functions (Harminv,
            # stop_when_fields_decayed) call this from inside run_cyl
            run = getattr(self, "_cyl", None)
            return 0.0 if run is None else run.t * run.dt
        return self._t_step * (self._coeffs.dt if self.dimensions == 3 else self.grid.dt)

    def get_field_point(self, c, pt):
        if self.dimensions == CYLINDRICAL:
            from .cyl_shim import get_field_point as _cyl_field_point
            return _cyl_field_point(self, c, pt)
        """Field value of component ``c`` at the cell containing meep point
        ``pt``. D components map to eps_inf * E at that cell; a component the
        state does not carry falls back to the run polarization's primary
        field (Ez for TM, Hz for TE -- the old behavior)."""
        if self._bloch_state is None and self._state is None:
            return 0.0
        gs = self.gs
        if self.dimensions == 1:
            ix, iy = cell_index(pt.z, gs), gs.ny // 2
            sgn = 1.0
        else:
            ix, mirx = self._foldpt(max(0, min(gs.nx - 1, cell_index(pt.x, gs))), "x")
            iy, miry = self._foldpt(max(0, min(gs.ny - 1, cell_index_y(pt.y, gs))), "y")
            px, py = self._comp_parity("Ez")
            sgn = (px if mirx else 1.0) * (py if miry else 1.0)
        name, scale = (c if isinstance(c, str) else ""), 1.0
        if name.startswith("D") and name != "Dielectric":
            name = "E" + name[1:]
            scale = float(np.asarray(self.get_epsilon())[ix, iy])
        elif name.startswith("B"):
            name = "H" + name[1:]        # mu = 1 engine: B == H
        st = self._bloch_state
        if st is not None:
            # per-point probe on the quadrature pair; decay stop conditions
            # call this every step, so skip the complex _state facade
            fc = getattr(st, name + "_c", None)
            if fc is None:
                fc, fs = ((st.Hz_c, st.Hz_s) if self._pol == "te"
                          else (st.Ez_c, st.Ez_s))
            else:
                fs = getattr(st, name + "_s")
            i, j = min(ix, fc.shape[0] - 1), min(iy, fc.shape[1] - 1)
            return sgn * scale * complex(float(fc[i, j]), float(fs[i, j]))
        f = getattr(self._state, name, None)
        if f is None:
            f = self._state.Hz if self._pol == "te" else self._state.Ez
        v = f[min(ix, f.shape[0] - 1), min(iy, f.shape[1] - 1)]
        return sgn * scale * (complex(v) if v.is_complex() else float(v))

    def _box_indices(self, box, center, size):
        """(i0, i1, j0, j1) cell-index bounds of a Volume/center+size box
        (whole cell when none given); shared by the *_in_box integrals."""
        if box is not None:
            center, size = box.center, box.size
        gs = self.gs
        if center is None and size is None:
            return 0, gs.nx, 0, gs.ny
        x0, y0, x1, y1 = rect_corners(center or Vector3(), size or Vector3(), gs)
        i0, i1 = span_index_range(x0, x1, gs, "x")
        j0, j1 = span_index_range(y0, y1, gs, "y")
        return i0, max(i1, i0 + 1), j0, max(j1, j0 + 1)

    def electric_energy_in_box(self, box=None, center=None, size=None):
        """meep electric_energy_in_box: integral of eps |E|^2 / 2 over the
        box (2D; whole cell when no box is given)."""
        from .output import _epwr
        i0, i1, j0, j1 = self._box_indices(box, center, size)
        return float(np.sum(_epwr(self)[i0:i1, j0:j1]) * self.gs.dx ** 2)

    def magnetic_energy_in_box(self, box=None, center=None, size=None):
        """meep magnetic_energy_in_box: integral of |H|^2 / 2 over the box."""
        from .output import _hpwr
        i0, i1, j0, j1 = self._box_indices(box, center, size)
        return float(np.sum(_hpwr(self)[i0:i1, j0:j1]) * self.gs.dx ** 2)

    def field_energy_in_box(self, box=None, center=None, size=None):
        """meep field_energy_in_box: total (electric + magnetic) energy."""
        return (self.electric_energy_in_box(box, center, size)
                + self.magnetic_energy_in_box(box, center, size))

    def integrate_field_function(self, cs, func, where=None, center=None,
                                 size=None):
        """meep integrate_field_function: integral over the cell (or box) of
        func(r, *component values) sampled per Yee cell (midpoint rule)."""
        return self.integrate2_field_function(self.fields, cs, [], func,
                                              where, center, size)

    def integrate2_field_function(self, fields2, cs1, cs2, func, where=None,
                                  center=None, size=None):
        """meep integrate2_field_function: as integrate_field_function, with
        the ``cs2`` components read from another fields object ``fields2``
        (the two-run mode-overlap workflow)."""
        # ponytail: per-point Python loop (N^2 func calls); vectorize over
        # numpy arrays if overlap integrals on big cells ever get hot
        self.init_sim()
        i0, i1, j0, j1 = self._box_indices(where, center, size)
        dx, gs = self.gs.dx, self.gs

        def comps(fobj, cs):
            out = []
            for c in cs:
                a = getattr(fobj._state, c, None)
                if a is None:
                    raise NotImplementedError(
                        f"component {c!r} is not in the field state")
                out.append(a.detach().cpu().numpy())
            return out
        a1 = comps(self.fields, cs1)
        a2 = comps(fields2, cs2)
        total = 0.0
        for i in range(i0, i1):
            for j in range(j0, j1):
                r = Vector3((i + 0.5) * dx - gs.sx / 2.0,
                            (j + 0.5) * dx - gs.sy / 2.0)
                vals = [a[min(i, a.shape[0] - 1), min(j, a.shape[1] - 1)]
                        for a in a1 + a2]
                total += func(r, *vals)
        return total * dx * dx

    def run(self, *step_funcs, until=None, until_after_sources=None, n_steps=None):
        self.run_index += 1          # meep numbers runs for display_* prefixes
        if (self._sym_cy or self._sym_cx) and self._eig_source is not None:
            raise NotImplementedError(
                f"EigenModeSource with Mirror symmetry not supported: "
                f"compute_smatrix has no mirror path yet ({_SYM_TODO})")
        if self.dimensions == CYLINDRICAL:
            from .cyl_shim import run_cyl
            return run_cyl(self, step_funcs, until, until_after_sources,
                           n_steps)
        if self._bloch and self.dimensions != 3:
            from .bloch_shim import needs_bloch, run_bloch
            if needs_bloch(self):
                return run_bloch(self, step_funcs, until, until_after_sources,
                                 n_steps)
            # every axis carries boundary layers: no periodic wrap -- meep
            # only switches to complex fields, fluxes match the real run
        stepped = bool(step_funcs) or callable(until) or callable(until_after_sources) \
            or isinstance(until, (list, tuple)) or isinstance(until_after_sources, (list, tuple))
        if stepped:
            if self.dimensions == 3 or self._eig_source is not None:
                raise NotImplementedError(
                    "step functions / stop conditions: 2D non-eigenmode runs only")
            self._run_until(step_funcs, until, until_after_sources)
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
        # _run_until owns the counter (it advances it from the fields'
        # own clock); the 3D and S-param backends do not, so they get it
        # bumped here. Doing both double-counts the time.
        if self.dimensions == 3:
            self._t_step += n_steps
            self._run_3d(n_steps)
            return
        if self._eig_source is not None:
            self._t_step += n_steps
            self._run_modes(n_steps)
            return
        self._run_until((), None, None, n_steps)

    def run_k_points(self, t=None, k_points=None):
        """meep run_k_points: run one Bloch-periodic simulation per k in
        ``k_points`` (until the sources turn off plus ``t``), extract the
        eigenfrequencies via Harminv, return a list (one per k) of complex
        frequencies (freq, decay). Prints meep's freqs:/freqs-im: rows."""
        from .verbosity_mgr import verbosity
        all_freqs = []
        for k_index, k in enumerate(k_points, 1):
            h = self.run_k_point(t, k)
            freqs = [complex(m.freq, m.decay) for m in h.modes]
            if verbosity.meep > 0:
                print(f"freqs:, {k_index}, {k.x}, {k.y}, {k.z}, "
                      + ", ".join(str(f.real) for f in freqs))
                print(f"freqs-im:, {k_index}, {k.x}, {k.y}, {k.z}, "
                      + ", ".join(str(f.imag) for f in freqs))
            all_freqs.append(freqs)
        return all_freqs

    def run_k_point(self, t=None, k=None):
        """meep run_k_point: one Bloch run at ``k``; returns the Harminv.

        Engine scope: the homogeneous 0D case (zero cell_size, no geometry) --
        a uniform (possibly Lorentz-dispersive) medium at Bloch wavevector k,
        stepped as a single Yee cell with the Bloch phase folded into the
        curls (the material-dispersion tutorial). Spatially-structured Bloch
        runs stay with the core machinery (meep_torch/core/stepper_bloch*)."""
        from .material import DrudeSusceptibility, LorentzianSusceptibility
        from .run_control import Harminv
        from .sources import GaussianSource
        cs = self.cell_size or Vector3()
        if self._geometry or cs.x or cs.y or cs.z:
            raise NotImplementedError(
                "run_k_points: only the homogeneous zero-size-cell case is "
                "supported (Bloch runs with geometry: core/stepper_bloch*)")
        gaussians = [s for s in self._sources
                     if isinstance(s.src, GaussianSource)]
        if not gaussians:
            raise ValueError(
                "Running with k_points requires a 'GaussianSource' source")
        fmin = max(0.0, min(s.src.frequency - 0.5 / s.src.width for s in gaussians))
        fmax = max(s.src.frequency + 0.5 / s.src.width for s in gaussians)
        spec = gaussians[0]
        prof = spec.src.profile()

        # medium: eps_inf + Lorentz poles of the (uniform) default material
        eps_inf = float(self._default.epsilon_diag.x)
        poles = []
        for su in (self._default.E_susceptibilities or []):
            if isinstance(su, DrudeSusceptibility) or not isinstance(
                    su, LorentzianSusceptibility):
                raise NotImplementedError(
                    "run_k_points: LorentzianSusceptibility media only")
            poles.append((float(su.frequency), float(su.gamma),
                          float(su.sigma_diag.x)))

        # single Yee cell, complex phasors, Bloch phase in the curls:
        # dHy/dt = dEz/dx -> Hy += (dt/dx)(bp - 1) Ez ; dD/dt = dHy/dx ->
        # D += (dt/dx)(1 - 1/bp) Hy ; E = (D - sum P) / eps_inf
        dx = 1.0 / self.resolution if getattr(self, "resolution", 0) else 0.05
        dt = self.Courant * dx
        bp = complex(math.cos(2 * math.pi * k.x * dx),
                     math.sin(2 * math.pi * k.x * dx))
        pc = []
        for (fn, gn, sn) in poles:      # meep Lorentz ADE coefficients
            wn, gam = 2 * math.pi * fn, 2 * math.pi * gn
            den = 1.0 + gam * dt / 2.0
            pc.append(((2.0 - wn * wn * dt * dt) / den,
                       -(1.0 - gam * dt / 2.0) / den,
                       sn * wn * wn * dt * dt / den))
        cutoff = prof.t_end
        n_steps = int(round((cutoff + float(t or 0.0)) / dt))
        h = Harminv(spec.component, spec.center or Vector3(),
                    0.5 * (fmin + fmax), fmax - fmin)
        # ponytail: record every m-th sample (Nyquist of the decimated series
        # stays >= 2 fmax) so the Harminv matrix pencil stays small; drop the
        # decimation if closely-spaced modes ever need the full series
        m = max(1, int(1.0 / (4.0 * fmax * dt)))
        h._dt = m * dt
        Ez = Hy = D = 0j
        P = [0j] * len(poles)
        Pp = [0j] * len(poles)
        for n in range(n_steps):
            Hy += (dt / dx) * (bp - 1.0) * Ez
            for i, (c1, c2, c3) in enumerate(pc):
                P[i], Pp[i] = c1 * P[i] + c2 * Pp[i] + c3 * Ez, P[i]
            tt = n * dt
            D += (dt / dx) * (1.0 - 1.0 / bp) * Hy
            if tt < cutoff:
                D += dt * prof.amplitude(tt)
            Ez = (D - sum(P)) / eps_inf
            if tt >= cutoff and n % m == 0:
                h.data.append(Ez)
        h._modes = None
        return h

    def _run_until(self, step_funcs, until, until_after_sources, n_steps=None):
        """meep run protocol (Simulation._run_until): fire step functions,
        advance ``self.fields``, and feed stop conditions every step.

        fields.step_n(k) covers the hook-free stretches in single chunked
        calls (the GPU/autograd analog of meep's chunked stepping; see its
        docstring); an active per-step hook or stop condition forces k=1,
        i.e. plain fields.step(). plan_hooks/chunk_size (mp.run_control)
        size k to the finest at_every/at_time interval."""
        self.init_sim()
        f = self.fields
        f.reset()                # each run() starts at t=0 with zero fields
        self._register_sources()
        # loaded DFT data (load_flux_data / load_minus_flux_data): seed the
        # monitor accumulators so this run accumulates on top (meep semantics)
        for name, seed in getattr(self, "_acc_seed", {}).items():
            if name in f._monitors:
                f._monitors[name][1] = seed.clone()
        dt = self.grid.dt
        cutoff = self._source_cutoff()
        begin, per_step, scheduled = plan_hooks(step_funcs, cutoff)
        conds, gate, n_total = None, 0.0, n_steps
        stop = until if until is not None else until_after_sources
        if callable(stop) or isinstance(stop, (list, tuple)):
            conds = list(stop) if isinstance(stop, (list, tuple)) else [stop]
            if until is None:
                gate = cutoff        # until_after_sources: condition applies post-cutoff
        elif n_total is None:
            if until is not None:
                n_total = max(1, int(round(float(until) / dt)))
            else:
                n_total = max(1, int(round((cutoff + (until_after_sources or 0)) / dt)))
        # meep's clock accumulates across run() calls: three run(until=10)
        # leave round_time() at 30, not 10. This used to reset to 0 while
        # still reading the fields' absolute counter back below, so the
        # loop compared an absolute step against a relative target and a
        # second run() exited immediately -- silently doing nothing. The
        # tutorial pattern "run to steady state, then run again to dump
        # fields" produced one frame instead of twenty.
        start = self._t_step
        stop_at = None if n_total is None else start + n_total
        for g in begin:
            _eval_step_func(self, g, "step")
        fire_scheduled(self, scheduled, start * dt)
        while stop_at is None or self._t_step < stop_at:
            t = self._t_step * dt
            left = (stop_at - self._t_step) if stop_at is not None else 10 ** 9
            k = chunk_size(t, dt, left, per_step, scheduled, conds is not None)
            if k == 1:
                f.step()
            else:
                f.step_n(k)
            self._t_step = f._t
            t = self._t_step * dt
            fire_per_step(self, per_step, t, dt)
            fire_scheduled(self, scheduled, t)
            if conds is not None:
                flags = [bool(g(self)) for g in conds]   # feed every condition
                if t >= gate - 1e-9 and any(flags):
                    break
        for (name, kind, mon, meta) in self._monitors:
            self._last_acc[name] = (f._monitors[name][1], mon, meta)
        for func in step_funcs:      # meep run-end protocol: todo="finish"
            _eval_step_func(self, func, "finish")

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
        # AUTOMATIC (meep default, -1) = launch along the region normal, forward
        src_dir = +1 if es.direction == AUTOMATIC else es.direction
        src_port = Port(axis=axis, index=idx, lo=lo, hi=hi, direction=src_dir,
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
    def _register_monitor(self, name, kind, mon, meta):
        """meep-level bookkeeping plus registration on the fields, which
        accumulates the DFT inside step() (meep's add_flux ->
        fields.add_dft_flux pattern; init_sim runs lazily here). The 3D and
        S-param paths keep list-only bookkeeping (no fields object)."""
        self._monitors.append((name, kind, mon, meta))
        if self.dimensions != 3 and self._eig_source is None:
            if self._bloch:
                from .bloch_shim import needs_bloch
                if needs_bloch(self):
                    return       # the Bloch driver samples from its own loop
            self.init_sim()
            self.fields._register(name, mon)

    def add_flux(self, fcen, df, nfreq, *fluxes, name=None):
        """meep add_flux(fcen, df, nfreq, *fluxregions) -> DftFlux (fluxes over
        multiple regions are summed, meep style)."""
        freqs = (np.linspace(fcen - df / 2.0, fcen + df / 2.0, nfreq)).tolist()
        return DftFlux(self._add_flux, [freqs, list(fluxes), name])

    def _add_flux(self, freqs, fluxes, name):
        if self.dimensions == CYLINDRICAL:
            raise NotImplementedError(
                "flux monitors in a cylindrical run are not supported: the "
                "surface element on an (r, z) cell is 2*pi*r dr dz, not dr "
                "dz, and that weighting was deliberately left unguessed -- a "
                "wrong constant silently scales a cross-section instead of "
                "failing. Needs a meep.mpb/meep oracle to pin. Point probes "
                "(get_field_point), near2far and dft_ldos do work")
        if self.dimensions == 3:
            if len(fluxes) != 1:
                raise NotImplementedError("3D add_flux takes exactly one region (W9 wave 3)")
            return self, self._add_flux_3d(freqs, fluxes[0], name), freqs
        name = name or f"flux{len(self._monitors)}"
        parts = []
        for r, region in enumerate(fluxes):
            mon, meta = self._line_monitor(region, freqs, kind="flux")
            key = name if len(fluxes) == 1 else f"{name}:r{r}"
            self._register_monitor(key, "flux", mon, meta)
            parts.append(key)
        if len(fluxes) > 1:
            self._flux_parts = getattr(self, "_flux_parts", {})
            self._flux_parts[name] = parts
        return self, name, freqs

    def add_near2far(self, fcen, df, nfreq, *regions, name=None):
        """2D near-to-far contour (TM + TE). Regions with outward normal =
        region axis * weight; a closed box = 4 regions, meep style. Trapezoid
        endpoint weights close the loop integral when the regions share
        corners."""
        if self.dimensions == CYLINDRICAL:
            from .cyl_shim import add_near2far as _cyl_add_near2far
            return _cyl_add_near2far(self, fcen, df, nfreq, *regions)
        freqs = np.linspace(fcen - df / 2.0, fcen + df / 2.0, max(nfreq, 1)).tolist()
        return DftNear2Far(self._add_near2far, [freqs, list(regions), name])

    def _add_near2far(self, freqs, regions, name):
        if self.dimensions == 3:
            return self._add_near2far_3d(freqs, regions, name)
        from ..core.monitors import FluxMonitorTEY
        name = name or f"n2f{len(self._monitors)}"
        dx = self.gs.dx
        specs = []
        for r, region in enumerate(regions):
            axis, idx, lo, hi = self._port_spec(FluxRegion(region.center, region.size))
            hi_cap = (self.gs.ny if axis == "x" else self.gs.nx)
            if self._pol == "te":
                hi_cap -= 1                     # Ey/Ex grids are one shorter
            hi = min(hi + 1, hi_cap)
            normal = (int(region.weight), 0) if axis == "x" else (0, int(region.weight))
            idxs = np.arange(lo, hi)            # full-cell span (meta pts)
            # mirror halving (TM only): register on the folded half grid,
            # keep full-cell pts; get_farfield rebuilds the full-span fields
            meta = {"normal": normal}
            idx2, lo2, hi2 = idx, lo, hi
            if self._sym_cy or self._sym_cx:
                span = "y" if axis == "x" else "x"
                idx2, pos_mir = self._foldpt(idx, "x" if axis == "x" else "y")
                lo2, hi2, n2f_cross, n2f_mir = self._fold(lo, hi, span)
                self._rot2_guard("a near2far face", pos_mir, n2f_cross,
                                 n2f_mir)
                meta.update(span=(span, lo, hi), lo_half=lo2, pos_mir=pos_mir)
            if self._pol == "tm":
                mon = FluxMonitorTM(self.grid, freqs, x_normal=(axis == "x"),
                                    idx=idx2, lo=lo2, hi=hi2)
                off = 0.0
            elif axis == "x":
                mon = FluxMonitorTE(self.grid, freqs, ix=idx2, lo=lo2, hi=hi2)
                off = 0.5                       # Ey nodes at (idx, (j+1/2) dx)
            else:
                mon = FluxMonitorTEY(self.grid, freqs, iy=idx2, lo=lo2, hi=hi2)
                off = 0.5                       # Ex nodes at ((i+1/2) dx, idx)
            if axis == "x":
                pts = np.stack([np.full(len(idxs), idx * dx), (idxs + off) * dx], axis=1)
            else:
                pts = np.stack([(idxs + off) * dx, np.full(len(idxs), idx * dx)], axis=1)
            meta["pts"] = pts
            self._register_monitor(f"{name}:f{r}", "n2f", mon, meta)
            specs.append(f"{name}:f{r}")
        self._n2f_specs = getattr(self, "_n2f_specs", {})
        self._n2f_specs[name] = (specs, freqs)
        return self, name, freqs

    def _add_near2far_3d(self, freqs, regions, name):
        """3D near2far: the regions' bounding box becomes the six DFT faces
        of one core.near2far_3d.Near2FarBox3D monitor (closed-box surface
        equivalence -- meep users pass the box faces as regions)."""
        from ..core.near2far_3d import Near2FarBox3D
        self.init_sim()
        name = name or f"n2f{len(self._monitors)}"
        gs = self.gs
        lo = [math.inf] * 3
        hi = [-math.inf] * 3
        for region in regions:
            c = region.center or Vector3()
            s = region.size or Vector3()
            for a, (cv, sv) in enumerate(((c.x, s.x), (c.y, s.y), (c.z, s.z))):
                lo[a] = min(lo[a], cv - abs(sv) / 2.0)
                hi[a] = max(hi[a], cv + abs(sv) / 2.0)
        box = (cell_index(lo[0], gs), cell_index(hi[0], gs),
               cell_index_y(lo[1], gs), cell_index_y(hi[1], gs),
               cell_index_z(lo[2], gs), cell_index_z(hi[2], gs))
        if any(box[2 * a + 1] - box[2 * a] < 2 for a in range(3)):
            raise NotImplementedError(
                "3D add_near2far: regions must span a closed box with "
                "positive extent on all three axes")
        mon = Near2FarBox3D(self._coeffs.dt, list(freqs), box, gs.dx,
                            dtype=self.dtype)
        self._register_monitor(name, "n2f3d", mon, {"freqs": list(freqs)})
        self._n2f_specs = getattr(self, "_n2f_specs", {})
        self._n2f_specs[name] = ([name], list(freqs))
        return self, name, freqs

    def _farfields_3d(self, name, obs_pts):
        """Far (E, H), each (nfreq, nobs, 3), at meep-coordinate points."""
        from ..core.near2far_3d import farfield_3d
        specs, freqs = self._n2f_specs[dft_name(name)]
        acc, mon, _meta = self._last_acc[specs[0]]
        faces = mon.faces(acc)
        gs = self.gs
        obs = (np.asarray(obs_pts, dtype=float)
               + np.array([gs.sx / 2.0, gs.sy / 2.0, gs.sz / 2.0]))
        ks = [2 * math.pi * f for f in freqs]
        E, H = farfield_3d(faces, ks, obs)
        return E, H, freqs

    def _defold_face(self, acc, meta):
        """Rebuild a near2far face's full-span (nfreq, 4, N) accumulator from
        its folded half-grid monitor: mirrored span rows carry the mirror
        phase (both slots), a position-folded face gets (p, -p) on its (E, H)
        slots -- the mirrored tangential H flips relative to E."""
        if "span" not in meta:
            return acc
        a = acc.detach().cpu().numpy() if hasattr(acc, "detach") else np.asarray(acc)
        span, lo, hi = meta["span"]
        c = self._sym_cy if span == "y" else self._sym_cx
        p = float(self._sym_py if span == "y" else self._sym_px)
        if c:
            j = np.arange(lo, hi)
            fold = np.where(j >= c, j - c, c - 1 - j) - meta["lo_half"]
            fold = np.clip(fold, 0, a.shape[-1] - 1)
            a = a[..., fold] * np.where(j >= c, 1.0, p)
        if meta.get("pos_mir"):
            q = float(self._sym_px if span == "y" else self._sym_py)
            a = a * np.array([q, q, -q, -q])[None, :, None]
        return a

    def get_farfield(self, name, point, tol=None):
        """Complex [Ex, Ey, Ez, Hx, Hy, Hz] at a far point (meep layout);
        (nfreq, 6) when nfreq > 1. ``tol`` is accepted for meep signature
        parity (the exact Hankel kernels have no iteration to tolerate)."""
        if self.dimensions == CYLINDRICAL:
            from .cyl_shim import get_farfield as _cyl_get_farfield
            return _cyl_get_farfield(self, name, point, tol)
        if self.dimensions == 3:
            E, H, freqs = self._farfields_3d(name, [[point.x, point.y, point.z]])
            out = np.concatenate([E[:, 0, :], H[:, 0, :]], axis=1)
            return out[0] if len(freqs) == 1 else out
        from ..core.near2far import faces_from_accs, farfield_te, farfield_tm
        specs, freqs = self._n2f_specs[dft_name(name)]
        mons, accs = [], []
        for key in specs:
            acc, mon, meta = self._last_acc[key]
            mons.append((mon, meta["normal"], meta["pts"]))
            accs.append(self._defold_face(acc, meta))
        faces = faces_from_accs(self.grid, mons, accs)
        obs = np.array([[point.x + self.gs.sx / 2.0, point.y + self.gs.sy / 2.0]])
        ks = [2 * math.pi * f for f in freqs]
        E, H = (farfield_tm if self._pol == "tm" else farfield_te)(faces, ks, obs)
        out = np.concatenate([E[:, 0, :], H[:, 0, :]], axis=1)   # (nfreq, 6)
        return out[0] if len(freqs) == 1 else out

    def get_farfields(self, near2far, resolution=None, where=None,
                      center=None, size=None, greencyl_tol=1e-3):
        """meep get_farfields: far fields on a grid over the volume
        (``resolution`` points per unit length). Returns {"Ex": ..., "Hz"}
        arrays shaped (grid dims with size > 0 ..., nfreq)."""
        if where is not None:
            center, size = where.center, where.size
        c = center or Vector3()
        s = size or Vector3()
        axes = []
        for cv, sv in ((c.x, s.x), (c.y, s.y), (c.z, s.z)):
            L = abs(sv or 0.0)
            n = max(1, int(round(L * resolution)) + 1) if L else 1
            axes.append(np.linspace(cv - L / 2.0, cv + L / 2.0, n))
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        if self.dimensions == 3:
            E, H, freqs = self._farfields_3d(near2far, pts)
            vals = np.concatenate([E, H], axis=2)             # (nfreq, npts, 6)
        else:
            freqs = self._n2f_specs[dft_name(near2far)][1]
            rows = [np.atleast_2d(self.get_farfield(near2far, Vector3(*p)))
                    for p in pts]                             # each (nfreq, 6)
            vals = np.stack(rows, axis=1)
        shape = tuple(len(a) for a in axes if len(a) > 1) + (len(freqs),)
        return {comp: vals[:, :, i].T.reshape(shape)
                for i, comp in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))}

    def dft_ldos(self, fcen, df, nfreq, name=None):
        """meep dft_ldos-lite: record Ez at the (point) source and return
        LDOS via core.ldos.ldos_spectrum after run(). TM only."""
        if self.dimensions == CYLINDRICAL:
            from .cyl_shim import dft_ldos as _cyl_dft_ldos
            return _cyl_dft_ldos(self, fcen, df, nfreq)
        freqs = np.linspace(fcen - df / 2.0, fcen + df / 2.0, max(nfreq, 1)).tolist()
        return DftLdos(self._add_ldos, [freqs, name])

    def _add_ldos(self, freqs, name):
        if self.dimensions == 3:
            raise NotImplementedError("3D dft_ldos not supported (W9 wave 3; deferred)")
        c = (self._sources[0].center or Vector3())
        ix, _ = self._foldpt(cell_index(c.x, self.gs), "x")
        iy = self._ypoint(cell_index_y(c.y, self.gs))
        mon = PointProbe(ix, iy)
        name = name or f"ldos{len(self._monitors)}"
        self._register_monitor(name, "ldos", mon, {"freqs": np.asarray(freqs)})
        return self, name, freqs

    def get_ldos(self, name):
        from ..core.ldos import ldos_spectrum
        acc, mon, meta = self._last_acc[dft_name(name)]
        e = acc.detach().cpu().numpy()
        prof = self._sources[0].src.profile()
        n = len(e)
        j = np.array([prof.amplitude((t + 0.5) * self.grid.dt) for t in range(n)])
        return meta["freqs"], ldos_spectrum(e, j, self.grid.dt, meta["freqs"])

    def add_dft_fields(self, components, fcen, df, nfreq, where=None,
                       center=None, size=None, yee_grid=False,
                       decimation_factor=0, name=None):
        """Area DFT of field components (meep add_dft_fields). The monitor
        accumulates the whole cell; a where/center/size region restricts
        get_dft_array / get_array_metadata(dft_cell=...) to that sub-volume."""
        freqs = np.linspace(fcen - df / 2.0, fcen + df / 2.0, nfreq).tolist()
        if where is not None:
            center, size = where.center, where.size
        return DftFields(self._add_dft_fields,
                         [freqs, list(components), name, (center, size)])

    def _add_dft_fields(self, freqs, components, name, region=(None, None)):
        name = name or f"dft{len(self._monitors)}"
        center, size = region
        box = None
        if center is not None or size is not None:
            if self.dimensions == 3:
                raise NotImplementedError(
                    "add_dft_fields with a region: 2D runs only")
            gs = self.gs
            x0, y0, x1, y1 = rect_corners(center or Vector3(),
                                          size or Vector3(), gs)
            i0, i1 = span_index_range(x0, x1, gs, "x")
            j0, j1 = span_index_range(y0, y1, gs, "y")
            box = (i0, i1, j0, j1)
        for c in components:
            if (self._sym_cy or self._sym_cx) and c not in ("Ez", "Dz"):
                raise NotImplementedError(
                    f"add_dft_fields({c}) with Mirror symmetry: staggered "
                    f"components do not de-mirror onto the full cell ({_SYM_TODO})")
            if self.dimensions == 3:
                mon = DFTFieldsMonitor3D(self._coeffs.dt, freqs, component=c, dtype=self.dtype)
                self._monitors.append((f"{name}:{c}", "dft", mon, {"freqs": freqs}))
            else:
                mon = DFTFieldsMonitor(self.grid, freqs, component=c)
                self._register_monitor(f"{name}:{c}", "dft", mon,
                                       {"freqs": freqs, "box": box})
        return self, name, freqs

    def get_dft_array(self, name, component, num_freq):
        """Complex (nx, ny) DFT plane of `component` at frequency index."""
        acc, mon, meta = self._last_acc[f"{dft_name(name)}:{component}"]
        a = acc[num_freq].detach().cpu().numpy()
        out = self._demirror(a[0] + 1j * a[1], self._comp_parity(component))
        box = meta.get("box")
        if box is not None:
            i0, i1, j0, j1 = box
            out = out[i0:i1, j0:j1]
        return out

    # --- MST force + field-energy monitors (2D TM; core.forces machinery) ---
    def _region_box(self, region):
        """Region -> Ez-node index box (i0, i1, j0, j1) (half-open), clamped
        one cell inside the grid so the staggered Hx/Hy samples the MST and
        energy sums need stay in range."""
        gs = self.gs
        x0, y0, x1, y1 = rect_corners(region.center or Vector3(),
                                      region.size or Vector3(), gs)
        i0, i1 = span_index_range(x0, x1, gs, "x")
        j0, j1 = span_index_range(y0, y1, gs, "y")
        i0, j0 = max(1, i0), max(1, j0)
        return i0, min(gs.nx - 1, max(i1, i0 + 1)), j0, min(gs.ny - 1, max(j1, j0 + 1))

    def _add_mst_monitors(self, freqs, name, kind):
        """Register full-cell Ez/Hx/Hy DFT monitors backing add_force/add_energy."""
        if self.dimensions != 2 or self._pol != "tm":
            raise NotImplementedError(f"add_{kind}: 2D TM runs only")
        if self._sym_cy or self._sym_cx:
            raise NotImplementedError(
                f"add_{kind} with Mirror symmetry not supported ({_SYM_TODO})")
        for c in ("Ez", "Hx", "Hy"):
            self._register_monitor(f"{name}:{c}", kind,
                                   DFTFieldsMonitor(self.grid, freqs, c), {})

    def _dft_planes(self, name, comps):
        """Complex (nfreq, ...) planes from the name:comp DFT accumulators."""
        out = {}
        for c in comps:
            acc, _, _ = self._last_acc[f"{name}:{c}"]
            a = acc.detach().cpu().numpy()
            out[c] = a[:, 0] + 1j * a[:, 1]
        return out

    def add_energy(self, fcen, df, nfreq, *energys, name=None):
        """meep add_energy(fcen, df, nfreq, *energyregions) -> DftEnergy.
        2D TM: accumulates full-cell DFTs of Ez/Hx/Hy; the get_*_energy
        methods integrate the time-averaged energy density over the regions."""
        freqs = np.linspace(fcen - df / 2.0, fcen + df / 2.0, max(nfreq, 1)).tolist()
        return DftEnergy(self._add_energy, [freqs, list(energys), name])

    def _add_energy(self, freqs, regions, name):
        name = name or f"energy{len(self._monitors)}"
        self._add_mst_monitors(freqs, name, "energy")
        self._energy_specs = getattr(self, "_energy_specs", {})
        self._energy_specs[name] = [
            (self._region_box(r), complex(getattr(r, "weight", 1.0) or 1.0).real)
            for r in regions]
        return self, name, freqs

    def get_electric_energy(self, f):
        """Per-frequency integral of <u_E> = eps |Ez|^2 / 4 over the regions."""
        return self._energy(f, electric=True)

    def get_magnetic_energy(self, f):
        """Per-frequency integral of <u_H> = mu (|Hx|^2 + |Hy|^2) / 4."""
        return self._energy(f, electric=False)

    def get_total_energy(self, f):
        """Electric + magnetic energy per frequency."""
        return [a + b for a, b in zip(self.get_electric_energy(f),
                                      self.get_magnetic_energy(f))]

    def _energy(self, f, *, electric):
        name = dft_name(f)
        dx = self.grid.dx
        eps = self._rm.eps_tm.detach().cpu().numpy()
        mu = (self._cond[2].detach().cpu().numpy() if self._cond is not None
              else np.ones_like(eps))
        planes = self._dft_planes(name, ("Ez",) if electric else ("Hx", "Hy"))
        out = 0.0
        for (i0, i1, j0, j1), w in self._energy_specs[name]:
            # integration measure: dx per axis the box actually extends over
            # (area for a 2D region, length for a line, density at a point)
            d = (i1 - i0 > 1) + (j1 - j0 > 1)
            if electric:
                u = eps[None, i0:i1, j0:j1] * np.abs(planes["Ez"][:, i0:i1, j0:j1]) ** 2
            else:
                # ponytail: Hx/Hy summed at their own staggered nodes (half a
                # cell off the Ez box) -- fine for smooth fields; average to
                # Ez nodes like core.forces if sub-cell accuracy matters
                u = mu[None, i0:i1, j0:j1] * (
                    np.abs(planes["Hx"][:, i0:i1, j0:j1]) ** 2
                    + np.abs(planes["Hy"][:, i0:i1, j0:j1]) ** 2)
            out = out + 0.25 * w * u.sum(axis=(1, 2)) * dx ** d
        return out.tolist()

    def add_force(self, fcen, df, nfreq, *forces, name=None):
        """meep add_force(fcen, df, nfreq, *forceregions) -> DftForce.
        2D TM: accumulates full-cell DFTs of Ez/Hx/Hy; get_forces integrates
        the Maxwell stress tensor over the closed box bounding the regions
        (core.forces.mst_force_tm). All regions must share one direction
        (mp.X or mp.Y -- the force component); their union should enclose
        the object in vacuum, meep style."""
        freqs = np.linspace(fcen - df / 2.0, fcen + df / 2.0, max(nfreq, 1)).tolist()
        return DftForce(self._add_force, [freqs, list(forces), name])

    def _add_force(self, freqs, regions, name):
        dirs = {r.direction for r in regions}
        if not regions or len(dirs) != 1 or not dirs <= {X, Y}:
            raise NotImplementedError(
                "add_force needs one or more ForceRegions sharing a single "
                "direction (mp.X or mp.Y); the MST box is the regions' union")
        name = name or f"force{len(self._monitors)}"
        self._add_mst_monitors(freqs, name, "force")
        boxes = [self._region_box(r) for r in regions]
        box = (min(b[0] for b in boxes), max(b[1] for b in boxes),
               min(b[2] for b in boxes), max(b[3] for b in boxes))
        self._force_specs = getattr(self, "_force_specs", {})
        self._force_specs[name] = (box, dirs.pop(),
                                   complex(getattr(regions[0], "weight", 1.0)
                                           or 1.0).real)
        return self, name, freqs

    def get_forces(self, f):
        """Per-frequency force along the regions' shared direction."""
        from ..core.forces import mst_force_tm
        name = dft_name(f)
        (i0, i1, j0, j1), direction, w = self._force_specs[name]
        planes = self._dft_planes(name, ("Ez", "Hx", "Hy"))
        fx, fy = mst_force_tm(planes, self.grid, ix0=i0, ix1=i1 - 1,
                              iy0=j0, iy1=j1 - 1)
        return (w * (fx if direction == X else fy)).tolist()

    def add_mode_monitor(self, fcen, df, nfreq, *fluxes, name=None):
        freqs = np.linspace(fcen - df / 2.0, fcen + df / 2.0, nfreq).tolist()
        return DftFlux(self._add_mode_monitor, [freqs, list(fluxes), name])

    def _add_mode_monitor(self, freqs, fluxes, name):
        if self.dimensions == 3:
            raise NotImplementedError("3D EigenModeSource/S-params not supported (W9 wave 3)")
        if len(fluxes) != 1:
            raise NotImplementedError("add_mode_monitor takes exactly one region here")
        if self._eig_source is None:
            # meep's add_mode_monitor is add_flux plus mode decomposition. With
            # no EigenModeSource there is no S-param path to feed, so register
            # a real DFT flux monitor -- get_eigenmode_coefficients decomposes
            # it -- instead of recording a port spec nothing ever fills.
            return self._add_flux(freqs, fluxes, name or f"mode{len(self._monitors)}")
        region = fluxes[0]
        # For the S-param path: store the port spec; run() routes an
        # EigenModeSource + these monitors to compute_smatrix. freqs are kept
        # for get_flux_freqs parity.
        axis, idx, lo, hi = self._port_spec(region)
        name = name or f"mode{len(self._mode_ports)}"
        self._mode_ports.append((name, axis, idx, lo, hi, region.direction))
        self._mode_freqs = freqs
        return self, name, freqs

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
        if self.dimensions != 3 and self._eig_source is None:
            self.init_sim()   # meep add_flux flow: polarization settles first
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
        meta = {"idx": idx, "lo": lo, "hi": hi,
                "wt": complex(getattr(region, "weight", 1.0) or 1.0).real}
        x_normal = self.dimensions == 1 or abs(s.x) <= abs(s.y)
        if self._sym_cy or self._sym_cx:
            # fold the line into the half grid: a plane-crossing span keeps
            # its (assumed symmetric) upper half and doubles back (mir2); a
            # face below a cut plane maps onto its image, where the mirrored
            # tangential H flips relative to E -> flux sign flips (msign)
            idx, pos_mir = self._foldpt(idx, "x" if x_normal else "y")
            lo, hi, meta["mir2"], span_mir = self._fold(
                lo, hi, "y" if x_normal else "x")
            self._rot2_guard("a flux/mode region", pos_mir, meta["mir2"],
                             span_mir)
            if pos_mir:
                meta["msign"] = -1.0
            meta.update(lo=lo, hi=hi)
        if self._pol == "tm":
            mon = FluxMonitorTM(self.grid, freqs, x_normal=x_normal,
                                idx=idx, lo=lo, hi=hi)
        elif x_normal:
            mon = FluxMonitorTE(self.grid, freqs, ix=idx, lo=lo, hi=hi)
        else:
            from ..core.monitors import FluxMonitorTEY
            mon = FluxMonitorTEY(self.grid, freqs, iy=idx, lo=lo, hi=hi)
        return mon, meta

    def get_fluxes(self, name):
        name = dft_name(name)
        parts = getattr(self, "_flux_parts", {}).get(name)
        if parts:
            return [sum(v) for v in zip(*(self.get_fluxes(p) for p in parts))]
        acc, mon, meta = self._last_acc[name]
        dx = self.gs.dx if self.dimensions == 3 else self.grid.dx
        vals = (mon.flux(acc, dx).detach().cpu().numpy()
                * meta.get("wt", 1.0) * meta.get("msign", 1.0))
        if meta.get("mir2"):
            vals = 2.0 * vals    # span crossed the mirror plane: half-cell flux
        return vals.tolist()

    def get_flux_freqs(self, name):
        name = dft_name(name)
        parts = getattr(self, "_flux_parts", {}).get(name)
        return mon_freqs(self, parts[0] if parts else name)

    def display_fluxes(self, *fluxes):
        """meep display_fluxes: comma-separated rows 'flux<run_index>:,
        freq, flux...' for the given flux objects."""
        from .verbosity_mgr import verbosity
        if verbosity.meep > 0:
            for row in zip(self.get_flux_freqs(fluxes[0]),
                           *[self.get_fluxes(f) for f in fluxes]):
                print(f"flux{self.run_index}:, "
                      + ", ".join(str(v) for v in row))

    def get_flux_data(self, flux):
        """Accumulated Fourier-transformed fields of a flux monitor as an
        opaque FluxData(E, H) (meep get_flux_data): complex (nfreq, npts)
        arrays, multi-region monitors concatenated along points. Only useful
        for load_flux_data / load_minus_flux_data on a same-shaped monitor."""
        name = dft_name(flux)
        parts = getattr(self, "_flux_parts", {}).get(name) or [name]
        Es, Hs = [], []
        for p in parts:
            if p not in self._last_acc:
                raise RuntimeError(f"get_flux_data({name!r}): no accumulated "
                                   "fields -- run() first")
            acc = self._last_acc[p][0].detach().cpu().numpy()
            Es.append(acc[:, 0] + 1j * acc[:, 1])
            Hs.append(acc[:, 2] + 1j * acc[:, 3])
        return FluxData(E=np.concatenate(Es, axis=-1),
                        H=np.concatenate(Hs, axis=-1))

    def load_flux_data(self, flux, fdata, _scale=1.0):
        """Seed the monitor's DFT accumulator with ``fdata`` (a FluxData from
        get_flux_data on a monitor of the same shape); the next run()
        accumulates on top of it (meep load_flux_data)."""
        if self.dimensions == 3:
            raise NotImplementedError("load_flux_data: 2D runs only")
        name = dft_name(flux)
        parts = getattr(self, "_flux_parts", {}).get(name) or [name]
        mons = {n: mon for (n, _, mon, _) in self._monitors}
        E, H = np.asarray(fdata.E), np.asarray(fdata.H)
        seeds, off = {}, 0
        for p in parts:
            npts = mons[p].hi - mons[p].lo
            e, h = E[:, off:off + npts], H[:, off:off + npts]
            off += npts
            acc = _scale * np.stack([e.real, e.imag, h.real, h.imag], axis=1)
            seeds[p] = torch.as_tensor(acc, dtype=self.grid.dtype,
                                       device=self.grid.device)
        if off != E.shape[-1]:
            raise ValueError(f"load_flux_data({name!r}): data has "
                             f"{E.shape[-1]} points, monitor has {off}")
        self._acc_seed = {**getattr(self, "_acc_seed", {}), **seeds}

    def load_minus_flux_data(self, flux, fdata):
        """load_flux_data with negated fields, so subsequent accumulation
        *subtracts* them (meep's reflection-normalization workflow)."""
        self.load_flux_data(flux, fdata, _scale=-1.0)

    def get_eigenmode_coefficients(self, flux, bands, eig_parity=NO_PARITY,
                                   eig_vol=None, eig_resolution=0,
                                   eig_tolerance=1e-12, kpoint_func=None,
                                   direction=AUTOMATIC):
        """meep get_eigenmode_coefficients -> EigCoeffsResult; ``alpha`` is
        (len(bands), nfreq, 2) with [..., 0] forward and [..., 1] backward.

        Two paths, picked by the monitor: a mode port of an EigenModeSource
        run reports the S-parameter column (see the _run_modes docstring); a
        flux monitor on a Bloch-periodic plane with uniform transverse eps is
        decomposed into the 0th diffraction order (mp/eig_coeffs.py).
        `eig_resolution`/`eig_tolerance` are accepted and ignored (there is no
        iterative mode solver on this path)."""
        if isinstance(bands, DiffractedPlanewave):
            raise NotImplementedError(
                "get_eigenmode_coefficients(bands=DiffractedPlanewave): only "
                "the 0th order is wired (bands=[1] on a flux monitor); "
                "selecting an arbitrary order m needs the full "
                "diffraction-order projection at k_y != 0 "
                "(meep_torch/ports/orders.py)")
        if not isinstance(bands, (list, tuple, range)):
            raise TypeError("get_eigenmode_coefficients: bands must be either "
                            "a list or DiffractedPlanewave object")
        if kpoint_func is not None:
            raise NotImplementedError(
                "get_eigenmode_coefficients kpoint_func: modes are solved "
                "along the monitor normal only (no oblique eigenmode solver)")
        if direction not in (AUTOMATIC, None):
            raise NotImplementedError(
                "get_eigenmode_coefficients direction: the mode is decomposed "
                "along the monitor normal only (direction=AUTOMATIC)")
        if eig_vol is not None:
            raise NotImplementedError(
                "get_eigenmode_coefficients eig_vol: the mode is solved on "
                "the monitor line itself")
        bands = list(bands)
        name = dft_name(flux)
        if self._S is not None and name in self._port_index:
            alpha = self._mode_port_alpha(name, bands)
        else:
            alpha = self._planewave_alpha(flux, name, bands, eig_parity)
        nb, nf, _ = alpha.shape
        return EigCoeffsResult(alpha, np.zeros(nb * nf), [], [],
                               np.zeros(nb * nf))

    def _mode_port_alpha(self, name, bands):
        """S-parameter path: S[:, port_index(name), 0] as the forward
        coefficient of band 1. The S-matrix carries one mode per port, so the
        backward column stays zero (the reflection port already reports the
        backward wave as its forward coefficient)."""
        if bands != [1]:
            raise NotImplementedError(
                f"get_eigenmode_coefficients(bands={bands}) on a mode port: "
                "compute_smatrix carries one mode per port (band 1)")
        col = self._S[:, self._port_index[name], 0].detach().cpu().numpy()
        alpha = np.zeros((1, col.shape[0], 2), dtype=np.complex128)
        alpha[0, :, 0] = col
        return alpha

    def _planewave_alpha(self, flux, name, bands, eig_parity):
        """Flux-monitor path: 0th diffraction order on a Bloch-periodic plane
        with uniform transverse eps (mp/eig_coeffs.py)."""
        from .eig_coeffs import planewave_alpha
        if self._S is not None or self._eig_source is not None:
            raise RuntimeError(
                f"get_eigenmode_coefficients({name!r}): not a mode port of "
                "this EigenModeSource run (use add_mode_monitor)")
        if self.dimensions == 3:
            raise NotImplementedError(
                "get_eigenmode_coefficients: 3D flux monitors have no "
                "decomposition path here (meep_torch/ports/mode_monitor.py "
                "decompose_3d)")
        want = "tm" if eig_parity & ODD_Z else ("te" if eig_parity & EVEN_Z else None)
        if want is not None and want != self._pol:
            raise ValueError(
                f"eig_parity implies a {want.upper()} mode but the run is "
                f"{self._pol.upper()} (ODD_Z = Ez/TM, EVEN_Z = Hz/TE)")
        if bands != [1]:
            raise NotImplementedError(
                f"get_eigenmode_coefficients(bands={bands}) on a flux "
                "monitor: only band 1 (the 0th/specular diffraction order) is "
                "implemented; the band -> order map for higher bands is "
                "parity- and degeneracy-dependent (meep_torch/ports/orders.py)")
        k = self.k_point
        if k not in (False, None) and (abs(k.y) > 1e-12 or abs(k.x) > 1e-12):
            raise NotImplementedError(
                f"get_eigenmode_coefficients at oblique incidence (k_point="
                f"{k}): the planewave basis is built at k_transverse = 0")
        if name not in self._last_acc:
            raise RuntimeError(
                f"get_eigenmode_coefficients({name!r}): no accumulated fields "
                "-- run() first")
        acc, mon, meta = self._last_acc[name]
        if not getattr(mon, "x_normal", True):
            raise NotImplementedError(
                "get_eigenmode_coefficients on a y-normal flux monitor is not "
                "implemented (x-normal monitor planes only)")
        eps = (self._rm.eps_tm if self._pol == "tm" else self._rm.eps_te_yy)
        eps_line = eps[meta["idx"], meta["lo"]:meta["hi"]].detach().cpu().numpy()
        a_p, a_m = planewave_alpha(acc, eps_line, self.gs.dx, self._pol)
        alpha = np.zeros((1, a_p.shape[0], 2), dtype=np.complex128)
        alpha[0, :, 0], alpha[0, :, 1] = a_p, a_m
        return alpha

    def _subvolume(self, arr, center, size):
        """Restrict a full-cell array to meep's (center, size) box.

        meep collapses an axis whose ``size`` component is 0, so a
        ``size=Vector3(16,0)`` cut returns a 1D line, not a 2D patch. Both
        arguments used to be accepted and ignored here, which returned the
        whole field and silently changed the shape a caller got -- the 90
        degree bend tutorial collects 334 such slices and hands them to
        imshow, getting (334, 160, 160) instead of (334, 160).
        """
        if center is None and size is None:
            return arr
        if self.dimensions == 3:
            raise NotImplementedError(
                "get_array(center=, size=) sub-volumes are 1D/2D only")
        gs = self.gs
        c = center if center is not None else Vector3()
        sz = size if size is not None else Vector3(gs.sx, gs.sy, 0)
        out, drop = arr, []
        for ax, (lo_c, ext, n, idx) in enumerate((
                (float(c.x), float(sz.x), gs.nx, lambda v: cell_index(v, gs)),
                (float(c.y), float(sz.y), gs.ny, lambda v: cell_index_y(v, gs)))):
            if ext == 0:                      # meep collapses a zero extent
                i0 = max(0, min(n - 1, idx(lo_c)))
                out = out[i0:i0 + 1] if ax == 0 else out[:, i0:i0 + 1]
                drop.append(ax)
                continue
            # meep counts both endpoints and one more: a size-4 box at
            # resolution 10 comes back as 42 samples, not 41. Clamped to the
            # cell, which is why a full-width cut matches either way.
            npts = min(n, int(round(ext / gs.dx)) + 2)
            i0 = max(0, min(n - npts, idx(lo_c - ext / 2.0)))
            out = out[i0:i0 + npts] if ax == 0 else out[:, i0:i0 + npts]
        for ax in reversed(drop):
            out = np.squeeze(out, axis=ax)
        return out

    def _bloch_array(self, component):
        """Physical field of a Bloch run, from the stored envelope.

        The Bloch steppers evolve a periodic envelope e(r) with
        F(r) = e(r) * exp(i k.r), split into two real quadratures e_c, e_s.
        The observable real field is therefore

            Re[(e_c + i e_s) exp(i k_y y)] = e_c cos(k_y y) - e_s sin(k_y y)

        Returning e_c alone would look plausible and be wrong by the Bloch
        phase, which is exactly the kind of silently-wrong array this
        codebase has been burned by, so reconstruct it properly.
        """
        from .bloch_shim import _bloch_ky

        st = self._bloch_state
        fc = getattr(st, component + "_c", None)
        if fc is None:                       # fall back to the run's primary
            fc, fs = ((st.Hz_c, st.Hz_s) if self._pol == "te"
                      else (st.Ez_c, st.Ez_s))
        else:
            fs = getattr(st, component + "_s")
        ky = _bloch_ky(self)
        if not ky:
            return fc.detach().cpu().numpy()
        y = (torch.arange(fc.shape[1], dtype=fc.dtype, device=fc.device)
             + 0.5) * self.gs.dx - self.gs.sy / 2.0
        phase_c, phase_s = torch.cos(ky * y), torch.sin(ky * y)
        return (fc * phase_c - fs * phase_s).detach().cpu().numpy()

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
            return self._subvolume(self._demirror(
                self._rm.eps_tm.detach().cpu().numpy() if self._pol == "tm"
                else self._rm.eps_te_xx.detach().cpu().numpy()), center, size)
        if self._bloch_state is None and self._state is None:
            raise RuntimeError("run() before get_array(fields)")
        if self._bloch_state is not None:
            return self._subvolume(self._bloch_array(component), center, size)
        if component in ("Ez", "Ex"):
            return self._subvolume(
                self._demirror(self._state.Ez.detach().cpu().numpy(),
                               self._comp_parity("Ez")), center, size)
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
            center, size = dft_cell.args[3] if len(dft_cell.args) > 3 else (None, None)
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
        """meep init_sim: build the structure (_init_structure), then the
        fields that step it, and register the sources on the fields.
        Idempotent; run() and the monitor adders call it lazily, meep style.
        The 3D path keeps its tensor state behind this facade (no fields
        object; the 3D engine is driven by core.monitors_3d.run_3d)."""
        if self.dimensions == CYLINDRICAL:
            # cyl_shim builds the BOR state lazily on the first run(); there
            # is no cartesian structure to compile and pol_for has no name
            # for the cylindrical components
            return
        if self.dimensions != 3 and self.fields is None:
            # meep's sim.sources / sim.symmetries are mutable (append after
            # construction): re-derive polarization, eigenmode source, and
            # mirror state before compiling the structure
            self._eig_source = next((s for s in self._sources
                                     if isinstance(s, EigenModeSource)), None)
            pol = pol_for(self._sources[0].component) if self._sources else "tm"
            if pol != self._pol:
                self._pol = pol
                self.structure = None
            sym = (self._sym_cx, self._sym_cy, self._sym_px, self._sym_py,
                   self._sym_rot2, self._sym_rot4)
            self._apply_symmetries()
            if sym != (self._sym_cx, self._sym_cy, self._sym_px, self._sym_py,
                       self._sym_rot2, self._sym_rot4):
                self.structure = None
        self._init_structure()
        if self.dimensions == 3:
            if self._state3 is None:
                self.restart_fields()
            return
        if self.fields is None:
            from .cpp.core import fields
            self.fields = fields(self.structure,
                                 components=self._component_set())
            if self._eig_source is None:
                self._register_sources()

    def _register_sources(self):
        """(Re)build the engine sources from the meep specs and register them
        on the fields (meep's fields.add_source pattern)."""
        self.fields.remove_sources()
        for src in self._make_sources():
            self.fields.add_source(src)

    def restart_fields(self):
        """Restart the fields at time zero, with zero fields. DFT monitors are
        kept and continue to accumulate on the next run (meep semantics)."""
        if self.dimensions == 3:
            self._state3 = init_3d_state(self.gs.nx, self.gs.ny, self.gs.nz,
                                         npml=self._npml, dtype=self.dtype,
                                         device=self.device)
        else:
            self.init_sim()
            self.fields.zero_fields()
            self.fields._t = 0
        self._t_step = 0

    def reset_meep(self):
        """Delete the fields and all accumulated monitor data, as if no
        computation had been run (the rasterized structure is kept: it is
        rebuilt deterministically from the constructor inputs)."""
        self.fields = None
        if self.dimensions == 3:
            self._state3 = None
        if self.dimensions == CYLINDRICAL:
            self._cyl = None              # CylRun: stepper state + monitors
        self._bloch_state = None
        self._t_step = 0
        self._monitors = []
        self._last_acc = {}
        self._mode_ports = []
        self._S = None
        self._port_index = {}
        for a in ("_flux_parts", "_n2f_specs", "_mode_freqs", "_energy_specs",
                  "_force_specs", "_acc_seed"):
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

    def use_output_directory(self, dname: str = ""):
        """meep use_output_directory: send every output_* file to ``dname``.

        Created if missing. With no argument meep uses
        ``get_filename_prefix() + "-out"`` and clears the prefix, so a
        script `foo.py` writes into `foo-out/`; that is reproduced here.
        meep also trashes the directory's previous contents on first use --
        not done, since silently deleting a directory the caller named is a
        worse failure than leaving a stale file behind.
        """
        if not dname:
            dname = self.get_filename_prefix() + "-out"
            self.filename_prefix = None
        os.makedirs(dname, exist_ok=True)
        self._output_directory = dname
        from .verbosity_mgr import verbosity
        if verbosity.meep > 0:
            print(f"Meep: using output directory '{dname}'")
        return dname

    def get_filename_prefix(self):
        """meep get_filename_prefix: filename_prefix, or the running script's
        basename without .py (meep's default output prefix)."""
        import os
        import re
        import sys
        if isinstance(self.filename_prefix, str):
            return self.filename_prefix
        if self.filename_prefix is None:
            fn = os.path.split(sys.argv[0])[1]
            if fn in ("ipykernel_launcher.py", "__main__.py", ""):
                return ""
            return re.sub(r"\.py$", "", fn)
        raise TypeError(
            f"Expected a string for filename_prefix, or None for the default: "
            f"{self.filename_prefix!r}")

    def change_sources(self, new_sources):
        """meep change_sources: replace the source list. Sources are built
        lazily at run(); a polarization change (TM <-> TE) rebuilds the
        coefficients and drops the field state."""
        self._sources = (list(new_sources)
                         if isinstance(new_sources, (list, tuple))
                         else [new_sources])
        if self.dimensions in (3, CYLINDRICAL):
            return
        self._eig_source = next((s for s in self._sources
                                 if isinstance(s, EigenModeSource)), None)
        pol = pol_for(self._sources[0].component) if self._sources else "tm"
        if pol != self._pol:
            # a polarization flip changes the compiled coefficients: drop the
            # structure and fields; init_sim rebuilds them from the spec
            self._pol = pol
            self.structure = None
            self.fields = None

    @property
    def sources(self):
        """meep Simulation.sources: the live source list (mutable; appends
        are picked up when the fields are next built)."""
        return self._sources

    @sources.setter
    def sources(self, new_sources):
        self.change_sources(new_sources)

    def _run_sources_until(self, cond, step_funcs):
        """meep _run_sources_until: run with step_funcs until the sources
        have turned off plus ``cond`` (a duration or stop condition)."""
        self.run(*step_funcs, until_after_sources=cond)

    def add_sources(self):
        """meep add_sources: register self.sources with the fields (init_sim
        builds the fields and registers the sources, meep style)."""
        self.init_sim()

    def clear_dft_monitors(self):
        """meep clear_dft_monitors: remove all DFT monitors and their
        accumulated data (the field state is kept)."""
        self._monitors = []
        self._last_acc = {}
        self._mode_ports = []
        if self.fields is not None:
            self.fields.clear_dft_monitors()
        for a in ("_flux_parts", "_n2f_specs", "_energy_specs", "_force_specs"):
            if hasattr(self, a):
                delattr(self, a)

    # --- meep whole-cell field getters (get_efield_z family) ---
    def _state_array(self, name):
        """Raw field plane at its own staggered nodes (2D state fields)."""
        if self.dimensions == 3:
            return self.get_array(component=name)
        if self._state is None:
            raise RuntimeError(f"run() before get field {name!r}")
        f = getattr(self._state, name, None)
        if f is None:
            raise NotImplementedError(
                f"component {name} does not exist on a {self._pol.upper()} run")
        if self._sym_cy or self._sym_cx:
            raise NotImplementedError(
                f"staggered {name} with Mirror symmetry not supported ({_SYM_TODO})")
        return f.detach().cpu().numpy()

    def get_efield_x(self):
        """meep get_efield_x: full-cell Ex array (TE runs; staggered nodes)."""
        return self._state_array("Ex")

    def get_efield_y(self):
        """meep get_efield_y: full-cell Ey array (TE runs; staggered nodes)."""
        return self._state_array("Ey")

    def get_efield_z(self):
        """meep get_efield_z: full-cell Ez array (TM runs)."""
        return self.get_array(component=Ez)

    def get_hfield_x(self):
        """meep get_hfield_x: full-cell Hx array (TM runs; staggered nodes)."""
        return self._state_array("Hx")

    def get_hfield_y(self):
        """meep get_hfield_y: full-cell Hy array (TM runs; staggered nodes)."""
        return self._state_array("Hy")

    def get_hfield_z(self):
        """meep get_hfield_z: full-cell Hz array (TE runs)."""
        return self.get_array(component=Hz)

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
