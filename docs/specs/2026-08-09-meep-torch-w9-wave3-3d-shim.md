# W9 wave 3 — 3D meep-shaped shim + example (spec)

Date: 2026-08-09
Workstream: W9 wave 3 (master spec). Wave 2 (`649d63f`) shipped the 3D
Yee stepper (CFS-PML/CPML, material D/B-conductivity, `monitors_3d`:
`PlaneSource`, `DFTFieldsMonitor3D`, `FluxMonitor3D`, `run_3d`) but it is
only reachable through the low-level `step_3d` API. Wave 3 wires it into
the meep-shaped `mp.Simulation` adapter so `dimensions=3` runs from meep
syntax, and ports one 3D example to prove the path.

## Physics (pinned)

- 3D Yee layout, CPML, and Courant `dt = 0.5 dx / sqrt(3)` as pinned in
  the wave-2 spec (`docs/specs/2026-08-08-meep-torch-w9-wave2-3d-hardening.md`).
  The shim is an adapter only — no physics; it calls `build_3d_coeffs` /
  `init_3d_state` / `step_3d` / `run_3d` from `core/stepper_3d.py` and
  `core/monitors_3d.py`.
- **Coordinate map**: meep centers the cell at the origin with the cell
  spanning `[-sx/2, sx/2) x [-sy/2, sy/2) x [-sz/2, sz/2)`; meep-torch is
  corner-origin, cell `(i,j,k)` center at `((i+0.5)dx, (j+0.5)dx,
  (k+0.5)dx)`. The 2D map in `mp/coords.py` extends to z by the same
  recipe (single place that maps world coords to indices).
- **Geometry rasterization (3D)**: the 2D scene uses subpixel polygon
  coverage (`geometry/scene.py`). 3D needs only **axis-aligned `Block`
  → eps voxel fill** in the shim (the gate and the example use only such
  blocks): for each `Block`, compute the integer index box
  `[i0:i1, j0:j1, k0:k1]` from world center/size (clipping `mp.inf` to the
  cell) and paint the resolved eps value into the `(nx,ny,nz)` volume in
  painter's order (later overrides earlier). `Cylinder`/`Prism` in 3D
  are out of scope (deferred) — only axis-aligned `Block` is supported
  this wave. No subpixel smoothing in 3D this wave (clean stair-stepped
  eps; the gate keeps eps uniform per region so there is no contrast
  staircasing at a graded interface to bias the budget).
- **Source (3D)**: a `Source` with a zero-size component on one axis is a
  plane source on that axis-normal plane (a z-normal `Ex` planewave has
  `size=(inf,inf,0)`); a fully-zero `size` is a soft point current. The
  plane case wraps `monitors_3d.PlaneSource` (axis = the propagation
  axis, idx = the plane index, transverse aperture clipped to the
  interior, excluding the side PML); the point case wraps a `step_3d`
  point-source tuple `("Ex"|"Ey"|"Ez", i, j, k)`. Both expose
  `.amplitude(t_idx)` + `.src` so `run_3d` drives them uniformly.
- **Flux (3D)**: `add_flux` with a `FluxRegion` whose non-zero size
  component picks the plane normal (the axis of the *smallest* extent,
  meep convention); `FluxMonitor3D(normal=..., idx=..., ap=...)` from
  `monitors_3d` accumulates the time-averaged Poynting flux through the
  plane. `get_fluxes` returns `mon.flux(acc, dx)`.

## In scope

1. `meep_torch/mp/coords.py`: extend `GridSpec` with `nz`/`sz` (default
   0, 2D callers unaffected); `grid_spec` honors `dimensions=3`
   (fills `nz`/`sz` from `cell.z`); add `cell_index_z` and a 3D index-box
   helper for axis-aligned blocks.
2. `meep_torch/mp/simulation.py`: `dimensions=3` branch through
   `__init__` (3D grid, axis-aligned-Block eps voxel fill, PML→`npml`,
   optional `sigma_d`/`sigma_b` voxel fill + absorber, `build_3d_coeffs`,
   `init_3d_state`), `_make_sources` (plane/point), `run` (→ `run_3d`),
   `add_flux` (→ `FluxMonitor3D`), `get_fluxes`, `get_array` (3D state
   component or eps volume), `add_dft_fields` (→ `DFTFieldsMonitor3D`).
   No Lorentz/Kerr/chi3 in 3D this wave (the 3D stepper is scalar eps +
   conductivity only). No `add_near2far`/`dft_ldos`/`EigenModeSource` in
   3D this wave (deferred).
3. `meep_torch/core/monitors_3d.py`: add a `PointSource3D` wrapper (soft
   point current: `.amplitude(t_idx)` + `.src = (comp, i, j, k)`) so the
   point-source path uses the same `run_3d` source interface as
   `PlaneSource`.
4. `tests/examples/test_w9_wave3.py` — the gates below.
5. `examples/edge_emitter_3D.py` — port the 3D stub with substitutions
   (noted in the file).

## Gates

- (a) **3D vacuum planewave `|T|=1`**: a z-normal `Ex` planewave in vacuum
  (PML on all six faces), two z-normal `FluxMonitor3D` planes — incident
  just after the source, transmitted near the `+z` PML —
  `T = flux_trans / flux_inc`. Assert `|T - 1| <= 0.02` (z-PML throughput;
  vacuum, so any loss is boundary spurious).
- (b) **real-eps slab `|T|` vs 1D analytic**: z-normal `Ex` planewave through
  a finite real-eps slab (full transverse extent), transmitted flux
  normalized by a vacuum reference run (same source/monitors, no slab),
  `T = flux_slab / flux_vac`, vs the analytic normal-incidence slab
  transmission `T = |(1-r^2) e^{iδ} / (1-r^2 e^{2iδ})|^2` with
  `n=sqrt(eps)`, `r=(1-n)/(1+n)`, `δ=ω n d`. Assert relative error
  **<=3%**. This exercises the 3D eps voxel fill + the stepper's eps
  handling through the shim path (a real permittivity contrast, not a
  conductivity substitute).
- (c) **suite runtime <10 min** (small 3D grids; one drive frequency,
  short Gaussian pulse).

### As built (gate parameterization)

- **(a) full-plane source, not a clipped aperture.** The first shim cut
  clipped the planewave transverse aperture to the interior (`lo=npml,
  hi=n-npml`), which gives a *finite* aperture: the launched beam has
  transverse curl, the side x/y CPML engages and drains it, and vacuum
  `|T|` collapses to ~0.13. Fix: when a source size carries `mp.inf` on a
  transverse axis the `PlaneSource` is built with no aperture (full plane),
  so the transverse field is uniform and the x/y CPML is inert (zero
  transverse curl, no psi accumulation) — the wave-2 inert-CPML trick. With
  `npml=8` (PML 0.4 at R=20) this gives `|T-1| ~ 1.7e-3`.
- **(b) `eps=2.25` (n=1.5), `f=0.7`, `R=40`, `d=1.2` — not `eps=12`.** The
  delegation's literal `eps=12` at `f=1, dx=0.05` is 5.8 cells/wavelength
  in the slab (`k*dx=1.09`): numerical dispersion + interface staircasing
  dominate the 3% budget, and `eps=12` needs `R>=100` (1D: R=100 -> 2.5%,
  R=20 -> 12%), which at 3D is ~6-8 min — out of the <10 min suite budget.
  The wave-2 gate-(a) comment records the same physics ("a hard eps step
  at a Yee cell face shifts the slab Fabry-Perot phase by ~0.5 rad/cell and
  would dominate the 3% budget"). `eps=2.25 / f=0.7 / R=40` is
  well-resolved (`k*dx=0.165`, ~38 cells/λ), off-resonance (`T~=0.85`, so
  the gate exercises reflection + Fabry-Perot phase, not a trivial `T=1`),
  matches the validated 1D precedent (`test_dispersion.py`, `eps=2.25`),
  and is the smallest grid. Peer-approved (W9 wave 3); the gate intent
  ("3D shim path reproduces the 1D analytic slab") is unchanged.
- The two-run `flux_slab / flux_vac` ratio (not a single-run inc/trans) is
  required: `FluxMonitor3D` measures the *net* (signed) Poynting flux, so a
  single-run inc/trans ratio is 1 by lossless energy conservation and tests
  nothing. The vacuum reference has no reflection, so its transmitted flux
  is the forward flux, and the slab run's transmitted flux (no back-reflection
  beyond the slab) is the forward transmitted flux — the ratio is `T`.

## Tests / acceptance

- `tests/examples/test_w9_wave3.py` (a)+(b). MIT, <=500 lines/file, TDD.
- W9 wave-2 `test_w9_wave2.py` + wave-1 `test_w9_3d.py` still green (the
  shim calls the same `stepper_3d`/`monitors_3d` unchanged).
- `examples/edge_emitter_3D.py` runs (the stub is replaced by a working
  port with documented substitutions).
- No meep import. Oracle not needed (analytic slab + vacuum `|T|=1`).

## Deferred (not this wave)

- 3D `Cylinder`/`Prism` rasterization, 3D subpixel smoothing.
- 3D `add_near2far`, `dft_ldos`, `EigenModeSource`/S-params.
- 3D Lorentz/Kerr dispersive materials (the 3D stepper is scalar eps +
  conductivity only until a later wave adds ADE poles).
- 3D symmetries (`mp.Mirror`) — no-op in the shim (full cell), as in W1.