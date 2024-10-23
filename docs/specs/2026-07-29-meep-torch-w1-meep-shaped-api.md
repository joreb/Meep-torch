# W1 — meep-shaped Python API (spec)

Date: 2026-07-29
Workstream: W1 (master spec `2026-07-29-meep-torch-complete-port-master-spec.md`,
execution plan `2026-07-29-complete-port-execution-plan.md`). Wiki banked
decisions: `wiki/summaries/phase3-phase4-outlook.md` §Phase 3.

## Goal

`import meep_torch.mp as mp` — an adapter that mirrors the meep Python API
surface the 34 ported examples use, so meep example scripts run with an import
swap. Adapter only: wraps `geometry/scene.py`, `core/coeffs.py`, `core/run.py`,
`modes/effindex.py`, `ports/`. **No physics changes** (wiki banked decision).

## In scope (W1 surface)

- `mp.Vector3(x,y,z)` — lightweight 3D-capable dataclass (2D uses x,y; z carried for W9).
- `mp.Medium(epsilon=, index=, E_susceptibilities=, chi3=)` → `Material`/`LorentzPole`; `chi3` → Kerr.
- `mp.PML(thickness)` → `npml = round(thickness/dx)`.
- `mp.Block`, `mp.Cylinder`, `mp.Prism` → `Polygon` scenes; Cylinder = N-gon (N scaled by resolution), painter's order.
- `mp.GaussianSource(fcen, fwidth, cutoff=)`, `mp.ContinuousSource(frequency, width=, slowness=, cutoff=)` → `GaussianPulse`/`CWRamp` (basic W1 semantics; faithful CW width/slowness lands in W2).
- `mp.Source(src, component, center, size, amplitude=)`, `mp.EigenModeSource(src, center, size, direction, eig_band, eig_parity)` → `LineSource`/`ModeSourceTM/TE`.
- `mp.Simulation(cell_size, resolution, geometry, sources, default_material, boundary_layers, dimensions, symmetries=)` → `Grid`+`build_*_coeffs`; `resolution` → `dx`; `cell_size` → `nx,ny`.
- `sim.run(until=)` / `run(until_after_sources=)` → `run_tm/te` with `n_steps = round(T/dt)`.
- `sim.add_flux(...)`, `get_fluxes`, `get_flux_freqs` → `FluxMonitorTM/TE` (minus-flux via two-run, as in `examples/refl-quartz.py`).
- `sim.add_mode_monitor(...)` + `get_eigenmode_coefficients` → `compute_smatrix` / `decompose`.
- `sim.get_array(component=mp.Dielectric|mp.Ez|...)` → field/eps snapshot.
- **Coordinate converter** (center-origin → corner-origin): single tested module. meep centers cells at origin with the cell spanning `[-L/2, L/2)`; meep-torch grids are corner-origin `[0, L)`. The converter maps meep `Vector3` centers/sizes to grid indices and eps tensor placement. This is the bug farm — tested hard.

## Out of scope (deferred)

- Faithful `ContinuousSource` width/slowness/cutoff ramp → W2.
- `Harminv` / resonant extraction → W3 (examples keep FFT substitution).
- Oblique/Bloch sources, `stop_when_fields_decayed` → W2.
- `mp.Mirror` symmetries (halving) — not emulated; full cell run (examples already do this).
- 3D Vector3 usage, Cylindrical, near2far, LDOS, adjoint, materials beyond eps/Lorentz/Kerr.

## Gate (master spec)

≥5 of the 34 ported examples rewritten to run via `import meep_torch.mp as mp`
with **unchanged meep syntax**, outputs matching their raw-port twins
(numeric comparison, not image). Gate examples chosen to exercise the whole
surface:

| # | example | exercises |
|---|---------|-----------|
| 1 | `straight-waveguide.py` | Block, PML, ContinuousSource Ez, get_array (eps+Ez) |
| 2 | `bent-waveguide.py` | Block bend geometry, source, get_array field |
| 3 | `refl-quartz.py` | 1D, FluxRegion, add_flux/get_fluxes, minus-flux, Material |
| 4 | `mode-decomposition.py` | EigenModeSource, add_flux, taper R |
| 5 | `bend-flux.py` | EigenModeSource, get_eigenmode_coefficients/S-params |

Each gets a twin `examples/<name>_mp.py` written in meep syntax against
`meep_torch.mp`; a test runs both and asserts the numeric outputs match
(field L2 / flux per-freq / S-param deltas within a tolerance).

## Open decisions (for orchestrator)

- `question:` dtype at the user-facing layer: keep float64 (engine default) or expose float32 opt-in here (wiki deferred this to Phase 3)?
- `question:` `ContinuousSource` in W1: ship the basic `CWRamp` ramp (matches `examples/straight-waveguide.py` raw port which already uses CWRamp), defer width/slowness to W2 — confirm OK.

## Tests / acceptance

- `tests/examples/test_w1_coord_converter.py` — center→corner mapping unit tests (the bug farm): object placement, source index, PML band, symmetry-mirrored coordinates. Hard gate.
- `tests/examples/test_w1_shim_examples.py` — the 5 twin-example numeric-match tests.
- Suite addition < 10 min. Example scripts excluded from pytest (runnable standalone). MIT headers, ≤500 lines, TDD, commit per task.