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
from ..core.monitors import DFTFieldsMonitor, FluxMonitorTE, FluxMonitorTM, PointProbe
from ..core.run import run_te, run_tm
from ..core.stepper import init_tm_state
from ..core.stepper_te import init_te_state
from ..geometry.scene import rasterize_scene
from .coords import GridSpec, cell_index, cell_index_y, grid_spec, pml_npml, span_index_range, to_corner
from .geometry import build_scene_items
from .material import Absorber, Medium, PML
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


@dataclass
class Near2FarRegion:
    center: Vector3 = None
    size: Vector3 = None
    weight: int = 1


class Simulation:
    def __init__(self, *, cell_size, resolution, geometry=None, sources=None,
                 default_material=Medium(epsilon=1.0), boundary_layers=None,
                 dimensions=2, symmetries=None, dtype=torch.float64):
        self.dimensions = dimensions
        self.dtype = dtype
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
        self._coeffs = self._build_coeffs()
        self._state = None
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

    def _build_coeffs(self):
        extra = {}
        if self._cond is not None:
            extra = dict(sigma_d=self._cond[0], sigma_b=self._cond[1], mu=self._cond[2])
        if self._pol == "tm":
            return build_tm_coeffs(self.grid, eps_inf=self._rm.eps_tm, poles=self._rm.poles,
                                    pole_masks=self._rm.pole_masks, chi3=self._rm.chi3,
                                    npml=self._npml, y_pmc=self._y_pmc, **extra)
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
            return init_tm_state(self.grid, n_poles=n_poles, y_pmc=self._y_pmc)
        return init_te_state(self.grid, n_poles=n_poles)

    def run(self, until=None, until_after_sources=None, n_steps=None):
        if n_steps is None:
            T = until if until is not None else (until_after_sources or 0)
            n_steps = max(1, int(round(float(T) / self.grid.dt)))
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
                        n_modes=es.eig_band + 1)
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
    def add_flux(self, fcen, df, nfreq, region: FluxRegion, name=None):
        freqs = (np.linspace(fcen - df / 2.0, fcen + df / 2.0, nfreq)).tolist()
        mon, meta = self._line_monitor(region, freqs, kind="flux")
        name = name or f"flux{len(self._monitors)}"
        self._monitors.append((name, "flux", mon, meta))
        return name

    def add_near2far(self, fcen, df, nfreq, *regions, name=None):
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
        return name

    def get_farfield(self, name, point):
        """Complex [Ex, Ey, Ez, Hx, Hy, Hz] at a far point (meep layout);
        (nfreq, 6) when nfreq > 1."""
        from ..core.near2far import faces_from_accs, farfield_te, farfield_tm
        specs, freqs = self._n2f_specs[name]
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
        """meep dft_ldos-lite: record Ez at the (point) source and return
        LDOS via core.ldos.ldos_spectrum after run(). TM only."""
        c = (self._sources[0].center or Vector3())
        ix = cell_index(c.x, self.gs)
        iy = cell_index_y(c.y, self.gs)
        mon = PointProbe(ix, iy)
        name = name or f"ldos{len(self._monitors)}"
        freqs = np.linspace(fcen - df / 2.0, fcen + df / 2.0, max(nfreq, 1))
        self._monitors.append((name, "ldos", mon, {"freqs": freqs}))
        return name

    def get_ldos(self, name):
        from ..core.ldos import ldos_spectrum
        acc, mon, meta = self._last_acc[name]
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
            mon = DFTFieldsMonitor(self.grid, freqs, component=c)
            self._monitors.append((f"{name}:{c}", "dft", mon, {"freqs": freqs}))
        return name

    def get_dft_array(self, name, component, num_freq):
        """Complex (nx, ny) DFT plane of `component` at frequency index."""
        acc, mon, meta = self._last_acc[f"{name}:{component}"]
        a = acc[num_freq].detach().cpu().numpy()
        return a[0] + 1j * a[1]

    def add_mode_monitor(self, fcen, df, nfreq, region: FluxRegion, name=None):
        # For the S-param path: store the port spec; run() routes an
        # EigenModeSource + these monitors to compute_smatrix. freqs are kept
        # for get_flux_freqs parity.
        axis, idx, lo, hi = self._port_spec(region)
        name = name or f"mode{len(self._mode_ports)}"
        self._mode_ports.append((name, axis, idx, lo, hi, region.direction))
        self._mode_freqs = np.linspace(fcen - df / 2.0, fcen + df / 2.0, nfreq).tolist()
        return name

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
        """S[:, port_index(name), 0] -- the coefficient at monitor `name` when
        the EigenModeSource (port 0) is excited (power-normalized S-param; see
        the _run_modes docstring for the normalization mapping)."""
        if self._S is None:
            raise RuntimeError("run() with an EigenModeSource before get_eigenmode_coefficients")
        j = self._port_index[name]
        return self._S[:, j, 0].detach().cpu().numpy()

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