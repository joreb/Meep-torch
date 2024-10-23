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
from ..core.monitors import FluxMonitorTE, FluxMonitorTM
from ..core.run import run_te, run_tm
from ..core.stepper import init_tm_state
from ..core.stepper_te import init_te_state
from ..geometry.scene import rasterize_scene
from .coords import GridSpec, cell_index, cell_index_y, grid_spec, pml_npml, span_index_range, to_corner
from .geometry import build_scene_items
from .material import Medium, PML
from .sources import (Dielectric, EigenModeSource, Source, build_line_source,
                      build_mode_source, pol_for)
from .vec import Vector3
from ..ports.mode_monitor import decompose


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


class Simulation:
    def __init__(self, *, cell_size, resolution, geometry=None, sources=None,
                 default_material=Medium(epsilon=1.0), boundary_layers=None,
                 dimensions=2, symmetries=None, dtype=torch.float64):
        self.dimensions = dimensions
        self.dtype = dtype
        self.grid, self.gs, self._y_pmc = _grid_for(cell_size, resolution, dimensions, dtype)
        self._npml = 0
        for bl in boundary_layers or []:
            if isinstance(bl, PML):
                self._npml = max(self._npml, pml_npml(bl.thickness, self.gs))
        self._default = default_material if isinstance(default_material, Medium) else Medium(epsilon=float(default_material))
        geom = list(geometry or [])
        eps_items, chi3_items = build_scene_items(geom, self.gs, dimensions)
        bg_val, bg_chi3 = self._default.resolve()
        if bg_chi3 != 0.0:
            full = np.array([[0, 0], [self.gs.sx, 0], [self.gs.sx, self.gs.sy], [0, self.gs.sy]])
            from ..geometry.polygons import Polygon
            chi3_items.append((Polygon(full), bg_chi3))
        self._rm = rasterize_scene(eps_items, self.grid, background=bg_val, chi3_items=chi3_items)
        self._sources = list(sources or [])
        self._pol = pol_for(self._sources[0].component) if self._sources else "tm"
        self._coeffs = self._build_coeffs()
        self._state = None
        self._monitors = []          # list of (name, kind, mon, meta)
        self._last_acc = {}

    def _build_coeffs(self):
        if self._pol == "tm":
            return build_tm_coeffs(self.grid, eps_inf=self._rm.eps_tm, poles=self._rm.poles,
                                    pole_masks=self._rm.pole_masks, chi3=self._rm.chi3,
                                    npml=self._npml, y_pmc=self._y_pmc)
        return build_te_coeffs(self.grid, eps_inf=self._rm.eps_te_xx, poles=self._rm.poles,
                               pole_masks=self._rm.pole_masks, chi3=self._rm.chi3,
                               npml=self._npml, y_pmc=self._y_pmc)

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
            out.append(src)
        return out

    def _init_state(self):
        n_poles = len(self._rm.poles)
        if self._pol == "tm":
            return init_tm_state(self.grid, n_poles=n_poles, y_pmc=self._y_pmc)
        return init_te_state(self.grid, n_poles=n_poles)

    def run(self, until=None, until_after_sources=None, n_steps=None):
        run = run_tm if self._pol == "tm" else run_te
        if n_steps is None:
            T = until if until is not None else (until_after_sources or 0)
            n_steps = max(1, int(round(float(T) / self.grid.dt)))
        self._state = self._init_state()
        srcs = self._make_sources()
        monitors = [m for (_, _, m, _) in self._monitors]
        # W1: single source supported (the 34 examples use one).
        src = srcs[0] if srcs else None
        self._state, outs = run(self._state, self._coeffs, n_steps, source=src, monitors=monitors)
        for (name, kind, mon, meta), acc in zip(self._monitors, outs):
            self._last_acc[name] = (acc, mon, meta)

    # --- flux monitors ---
    def add_flux(self, fcen, df, nfreq, region: FluxRegion, name=None):
        freqs = (np.linspace(fcen - df / 2.0, fcen + df / 2.0, nfreq)).tolist()
        mon, meta = self._line_monitor(region, freqs, kind="flux")
        name = name or f"flux{len(self._monitors)}"
        self._monitors.append((name, "flux", mon, meta))
        return name

    def add_mode_monitor(self, fcen, df, nfreq, region: FluxRegion, name=None):
        freqs = (np.linspace(fcen - df / 2.0, fcen + df / 2.0, nfreq)).tolist()
        mon, meta = self._line_monitor(region, freqs, kind="mode")
        name = name or f"mode{len(self._monitors)}"
        self._monitors.append((name, "mode", mon, meta))
        return name

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
        if self._pol == "tm":
            mon = FluxMonitorTM(self.grid, freqs, x_normal=True, idx=idx, lo=lo, hi=hi)
        else:
            mon = FluxMonitorTE(self.grid, freqs, ix=idx, lo=lo, hi=hi)
        return mon, meta

    def get_fluxes(self, name):
        acc, mon, meta = self._last_acc[name]
        return mon.flux(acc, self.grid.dx).detach().cpu().numpy().tolist()

    def get_flux_freqs(self, name):
        return mon_freqs(self, name)

    def get_eigenmode_coefficients(self, name, n_modes=1):
        acc, mon, meta = self._last_acc[name]
        idx, lo, hi = meta["idx"], meta["lo"], meta["hi"]
        if self._pol == "tm":
            eps_line = self._coeffs.eps_inf[idx, lo:hi].detach()
        else:
            eps_line = 0.5 * (self._coeffs.eps_inf[idx, lo:hi].detach()
                             + self._coeffs.eps_inf[idx, lo + 1:hi + 1].detach())
        freqs = mon.freqs.detach().cpu().numpy().tolist()
        a_p, a_m = decompose(acc, eps_line, self.grid.dx, freqs, self._pol, n_modes,
                             axis_sign=-1.0)
        return a_p.detach().cpu().numpy(), a_m.detach().cpu().numpy()

    def get_array(self, component=Dielectric, center=None, size=None):
        if component == Dielectric:
            return self._rm.eps_tm.detach().cpu().numpy() if self._pol == "tm" \
                else self._rm.eps_te_xx.detach().cpu().numpy()
        if self._state is None:
            raise RuntimeError("run() before get_array(fields)")
        if component in ("Ez", "Ex"):
            return self._state.Ez.detach().cpu().numpy()
        if component in ("Hz", "Hy"):
            return self._state.Hz.detach().cpu().numpy() if self._pol == "te" \
                else self._state.Hz if hasattr(self._state, "Hz") else None
        raise ValueError(f"unsupported component {component!r}")


def mon_freqs(sim, name):
    for (n, _, mon, _) in sim._monitors:
        if n == name:
            return mon.freqs.detach().cpu().numpy().tolist()
    raise KeyError(name)