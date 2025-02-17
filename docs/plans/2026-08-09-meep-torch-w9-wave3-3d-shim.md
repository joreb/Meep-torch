# W9 wave 3 — 3D meep-shaped shim + example (plan)

Date: 2026-08-09
Spec: `docs/specs/2026-08-09-meep-torch-w9-wave3-3d-shim.md`.
TDD, commit per task, <=500 lines/file, MIT headers. Suite addition <10 min.

## Task count: 5

- **T1** `docs/specs/2026-08-09-meep-torch-w9-wave3-3d-shim.md` +
  `docs/plans/2026-08-09-meep-torch-w9-wave3-3d-shim.md` (this pair,
  house style per w6/w8/w9w2).
- **T2** `meep_torch/mp/coords.py` 3D extension:
  - `GridSpec`: add `nz: int = 0`, `sz: float = 0.0` (frozen, defaults
    keep 2D positional construction `GridSpec(nx, ny, dx, sx, sy)` working).
  - `grid_spec(cell, resolution, dimensions=2)`: when `dimensions==3`,
    fill `nz`/`sz` from `cell.z` (same `round(s/dx)` recipe).
  - `cell_index_z(z_world, gs)`: world z -> z cell index (corner-origin).
  - `_box3d(center, size, gs)`: world center+size -> integer index box
    `[i0:i1, j0:j1, k0:k1]`, clipping `mp.inf` to the cell on each axis.
- **T3** `meep_torch/core/monitors_3d.py`: add `PointSource3D(dt, comp,
  i, j, k, profile)` — `.amplitude(t_idx)` returns
  `profile.amplitude((t_idx+0.5)*dt)`, `.src = (comp, i, j, k)`. Lets
  `run_3d` drive a soft point current through the same source interface
  as `PlaneSource`.
- **T4** `meep_torch/mp/simulation.py` 3D branch (adapter only):
  - `_grid_for(dimensions==3)`: `grid_spec(cell, resolution, 3)`; store
    `gs` with `nz`/`sz`; no `core/Grid` (3D stepper takes `nx,ny,nz,dx`
    directly). `_y_pmc = False`.
  - `__init__` 3D: build eps volume `(nx,ny,nz)` via axis-aligned-Block
    voxel fill (painter order; resolve each `Medium` to eps via
    `Medium.resolve()` — scalar eps only; dispersive `Material` in 3D ->
    raise `NotImplementedError`). PML -> `npml`; `Absorber` -> `n_abs`.
    Optional `sigma_d`/`sigma_b` voxel fill (background `D_conductivity`/
    `B_conductivity` + per-block + absorber ramp on `n_abs`). Call
    `build_3d_coeffs(nx,ny,nz,dx, eps=..., npml=..., n_abs=...,
    sigma_d=..., sigma_b=...)` and `init_3d_state(nx,ny,nz,npml=...)`.
    Skip `chi3`/poles (3D scalar). `_pol` from `pol_for` is unused in 3D.
  - `_make_sources` 3D: for each `Source`, if any size component is `inf`
    or the size is non-zero on a plane -> `PlaneSource` (axis = the
    zero/min-size axis = propagation; idx = `cell_index_*` of the source
    center on that axis; comp = the `E*` component; transverse aperture
    `lo/hi` from the other two size components, clipped to the interior
    minus side PML); else `PointSource3D` (soft point current at the
    center index). `GaussianSource`/`ContinuousSource`/`CustomSource`
    `.profile()` -> core source profile (reuse the 2D `Source.profile()`
    path). Return a single source (`run_3d` takes one source; first
    source wins, matching the 2D `srcs[0]` convention).
  - `run` 3D: `run_3d(self._state, self._coeffs, n_steps, source=src,
    monitors=[...])`; stash `acc` per monitor.
  - `add_flux` 3D: `FluxRegion` -> normal axis (axis of smallest size),
    idx, aperture `ap = npml+1`; `FluxMonitor3D(dt, freqs, normal=...,
    idx=..., ap=...)`; append `(name,"flux",mon,meta)`.
  - `get_fluxes` 3D: `mon.flux(acc, dx)`.
  - `get_array` 3D: `Dielectric` -> eps volume; `E*`/`H*` -> state
    component.
  - `add_dft_fields` 3D: `DFTFieldsMonitor3D(dt, freqs, component=c)`.
  - `_port_spec`/`_line_monitor`/`add_near2far`/`dft_ldos`/
    `add_mode_monitor`/`_run_modes`: raise `NotImplementedError` for
    `dimensions==3` (deferred; clear message pointing at the spec).
- **T5** `tests/examples/test_w9_wave3.py`:
  - gate (a): small 3D vacuum cell, z-normal `Ex` `PlaneSource` via the
    shim (`mp.Simulation(dimensions=3, ...)`, `add_flux` incident +
    transmitted), `T = trans/inc`, assert `|T-1| <= 0.02`.
  - gate (b): real-eps slab (axis-aligned `Block`, full transverse),
    transmitted flux vs a vacuum reference run (same shim, no slab),
    `T = flux_slab/flux_vac`, vs analytic normal-incidence slab
    transmission, assert rel err <=3%. **As built: `eps=2.25` (n=1.5),
    `f=0.7`, `R=40`, `d=1.2`** — not the delegated `eps=12` (which needs
    `R>=100`, out of suite budget; see spec "As built"). Peer-approved.
    Two-run ratio (not single-run inc/trans): `FluxMonitor3D` is net flux,
    so single-run inc/trans is 1 by energy conservation. Separate config
    from gate (a): `B_*` constants (NX=NY=44, NZ=200, PML 0.5, until 60).
  - assert wave-1/wave-2 W9 tests still green (absorber/CPML unchanged).
- **T6** `examples/edge_emitter_3D.py`: replace the stub with a working
  port. Substitutions (noted in the file header): `mp.Mirror` symmetries
    -> no-op (full cell, W1 limitation); `stop_when_fields_decayed` ->
    fixed `n_steps`; `dft_ldos` + the `ldos_Fdata`/`ldos_Jdata`
    denominator -> deferred (3D LDOS is not this wave; report the air
    flux only, note the extraction-efficiency denominator is pending).
    Keeps the 3D axis-aligned Blocks (substrate/cladding/active), the
    point dipole source, PML, and the 3-region flux box (extraction into
    air). Reduced resolution/sizes for a fast smoke run.

## Standing rules

- Ground truth = analytic slab + vacuum `|T|=1` (no meep oracle).
- `status:` to peer before each git commit (shared working tree); commit
  ONLY docs/T2/T3/T4/T5/T6 files (docs w9-wave3, `meep_torch/mp/*` 3D
  branches, `monitors_3d.py`, `tests/examples/test_w9_wave3.py`, the
  example); `done:` with hashes + gate numbers.
- Keep 3D grids small — 3D is O(n^3); one drive frequency, short pulse.
- 3D shim is adapter-only: no physics in `simulation.py`; all numerics in
  `stepper_3d`/`monitors_3d` (already shipped wave 2).