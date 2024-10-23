# W1 — meep-shaped Python API (plan)

Date: 2026-07-29
Spec: `docs/specs/2026-07-29-meep-torch-w1-meep-shaped-api.md`.
Adapter only (no physics). TDD, commit per task, ≤500 lines/file, MIT headers.

## Task count: 8

### Wave 1 — core adapter (T1–T4)

- **T1 `mp.Vector3` + coordinate converter.** `meep_torch/mp/vec.py`: Vector3 dataclass (x,y,z). `meep_torch/mp/coords.py`: `cell_to_grid(cell, resolution)` → `(nx, ny, dx)`; `world_to_index(center, cell, resolution, corner_origin)` → grid indices for sources/monitors/geometry; PML band bookkeeping. TDD: `tests/examples/test_w1_coord_converter.py` — placement of centered objects, edge cells, PML thickness→npml, off-center sources. The bug farm; tested hard first.
- **T2 `mp.Medium` + `mp.PML` + geometry.** `meep_torch/mp/material.py`: Medium → Material/LorentzPole (epsilon vs index, E_susceptibilities, chi3). `meep_torch/mp/geometry.py`: Block/Cylinder/Prism → `Polygon` (Cylinder N-gon, N=resolution-ish; Prism passes through), painter's-order list → `rasterize_scene`. `mp.PML` dataclass. TDD: cylinder N-gon area vs πr² within 1%; Block eps placement vs raw raster.
- **T3 `mp.GaussianSource`/`ContinuousSource`/`Source`/`EigenModeSource`.** `meep_torch/mp/sources.py`: wrap `GaussianPulse`/`CWRamp`/`LineSource`/`ModeSourceTM/TE`; component→pol mapping (Ez→TM, Hz→TE for 2D Ex/Hy), center/size→LineSource span, amplitude scaling. TDD: source mask placement matches raw LineSource; EigenModeSource neff matches `solve_1d`.
- **T4 `mp.Simulation` + run + flux + get_array.** `meep_torch/mp/simulation.py`: assemble Grid+coeffs from cell/resolution/geometry/sources/default_material/PML; `run(until=)`→run_tm/te n_steps; `add_flux`/`get_fluxes`/`get_flux_freqs` via FluxMonitorTM/TE; `add_mode_monitor`+`get_eigenmode_coefficients` via compute_smatrix/decompose; `get_array(component=)` for Dielectric/Ez/Hz. TDD: Simulation of a straight guide matches raw `examples/straight-waveguide.py` Ez field (L2 < 1e-3 rel).

### Wave 2 — gate examples (T5–T8)

- **T5** twin `examples/straight-waveguide_mp.py` + `bent-waveguide_mp.py` (meep syntax via `meep_torch.mp`). Test: field L2 vs raw twins.
- **T6** twin `examples/refl-quartz_mp.py`. Test: R(λ) per-freq vs raw twin within 2%.
- **T7** twin `examples/mode-decomposition_mp.py` + `bend-flux_mp.py`. Test: taper R and S-params vs raw twins within 2%.
- **T8** gate verification + commit. Run all 5 twins vs raw, record deltas in `done:`. Ensure `import meep_torch.mp as mp` is the only meep symbol surface used.

## Wave gates

- After Wave 1: coord-converter unit tests green; straight-guide Simulation field matches raw.
- After Wave 2 (W1 gate): ≥5 examples run verbatim via the shim, outputs match raw twins (numeric). `done:` with per-example deltas.

## Standing rules

- Numeric verification only (no image reads). Oracle (meep-oracle) not needed for W1 — raw-port twins are the reference.
- Suite addition < 10 min; example scripts excluded from pytest.
- `question:` only for the two open decisions in the spec (dtype, ContinuousSource scope).