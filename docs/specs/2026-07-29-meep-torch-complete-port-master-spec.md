# meep-torch complete-port master spec

Date: 2026-07-29
Status: approved scope (user 2026-07-29): **2D-complete first, 3D last; forward-only
autodiff** (gradients where free, guaranteed only when a workstream says so).
Execution: inter-session agent, one workstream at a time, each with its own
spec → plan → implement → verify cycle (specs in `docs/specs/`, plans in
`docs/plans/`, per project convention).

## Goal

Every meep capability needed to run `meep/python/examples/*.py` (82 scripts)
exists in meep-torch, 2D first. The Phase B triage (inter-session task, pending)
maps examples → workstreams; each workstream closes the gap blocking its group
and lands the corresponding example ports in `tests/examples/`.

## Ground rules (all workstreams)

- Wheel stays 100% MIT. meep itself is GPL2: **no code ports from `meep/`** —
  blueprints from docs/papers/fdtdx(MIT)/tidy3d(MIT parts) or original code.
- Oracle: pymeep 1.34 in conda env `meep-oracle` (`mamba run -n meep-oracle`) —
  generate committed reference JSON like `tests/ports/generate_meep_reference.py`;
  tests never import meep.
- Physics-gated tests over golden numbers where possible. Suite addition per
  workstream target < 10 min.
- Known measurement rules: |S11|/R below 1e-3 needs npml>=60 + decayed window;
  |S21| robust at npml=10; Kottke Polygon scenes over staircase rasters.
- Files ≤ 500 lines; SPDX MIT headers; TDD; commit per task.

## Workstreams (execution order)

### W1 — meep-shaped Python API (= wiki roadmap Phase 3)
`mp.Simulation`, `Medium`, `Source`/`GaussianSource`/`ContinuousSource`,
`EigenModeSource` (wraps ports/ModeSource), `PML`, `Block`/`Cylinder`/`Prism`
(rasterize via Polygon scenes), `Vector3`, `FluxRegion`/`add_flux`/`get_fluxes`,
`add_mode_monitor`/`get_eigenmode_coefficients` (wraps ports/decompose),
center-origin → corner-origin coordinate converter (single converter, tested
hard — the bug farm), `resolution` → dx, `run(until=…)`.
Value: meep example scripts run near-verbatim (import swap) — every later
workstream's examples get cheaper. Banked decisions in wiki
`phase3-phase4-outlook.md`. Oracle: run ported examples under both meep
(conda env) and meep-torch shim, compare outputs.

### W2 — sources completion
Custom source time-profiles (callable), CW proper (`ContinuousSource` semantics
incl. slowness/cutoff), amplitude-function spatial profiles, line/area sources
both polarizations, oblique planewave source (Bloch-phased line source; 2D),
`until_after_sources` / field-decay `stop_when_fields_decayed` run control
(kills the fixed-n_steps `ponytail:` mark in smatrix too).

### W3 — frequency-domain analysis
Harminv equivalent: filter-diagonalization or ESPRIT/Prony on time series
(scipy only) for resonant f + Q — unlocks cavity/resonance examples
(`Resonant_Modes_…`, ring examples). Area DFT field monitors (full-region
`add_dft_fields` equivalent, 2D). Optional (flag if costly): `solve_cw`
frequency-domain solver — sparse direct solve on the 2D curl-curl operator.

### W4 — near-to-far transform (2D)
Green's-function near2far for 2D (TM+TE): far-field radiation patterns from
DFT fields on a closed contour. Unlocks `antenna-radiation`,
`binary_grating_n2f`, `finite_grating`, `Near_to_Far_Field_Spectra` group.
Oracle: analytic dipole radiation pattern + meep reference JSON.

### W5 — materials completion
Electric conductivity (D-conductivity), magnetic permeability mu + magnetic
Lorentz poles, full anisotropic eps tensor in the FDTD stepper (wiki open
thread; tensor eps exists only in mode solver today), gyrotropic media
(magnetized-plasma / gyrotropic-Lorentz), saturable gain / multilevel atomic
susceptibility (rate-equation ADE). Split into sub-tasks; gyrotropic +
multilevel may land as a second wave.

### W6 — LDOS + forces
`dft_ldos` equivalent (point LDOS from source-field overlap), Purcell-factor
example gate; Maxwell-stress-tensor optical forces on a contour
(`Optical_Forces` example).

### W7 — cylindrical coordinates
Body-of-revolution FDTD (r,z grid, azimuthal number m): stepper pair, axis
conditions, radial CPML, cylindrical eigenmode ports. Unlocks
`Cylindrical_Coordinates` group + ring-resonator examples that are truly
axisymmetric. Largest 2D-family workstream — its spec may split it.

### W8 — inverse design (= wiki roadmap Phase 5)
Density → filters → SSP projection, ClosestIndex STE, fabrication constraints
(fdtdx MIT blueprint), optimization loop driving |S|^2 objectives through the
Phase 1 checkpointed autodiff. Here the forward-only stance is revisited:
eigengrad non-Hermitian backward + mode-field IFT gradients land here (wiki
banked decision).

### W9 — 3D engine (last)
Full 3D Yee stepper (6 fields), 3D CPML, 3D Lorentz/Kerr, 3D rasterization
(extend Kottke), 3D eigenmode ports (fd2d cross-section modes feed 3D TFSF),
checkpointed autodiff memory strategy (wiki Q2 reversible-gradient engine is
the enabler — schedule together). Unlocks the remaining 3D examples.

## Out of scope (declared, not silent)

- MPB plane-wave solver port (wiki Q1, GPL2 reimplementation) — separate track.
- MPI/distributed runs; GPU is torch-native already (device= plumbing only).
- Special functions meep exposes but no example uses (flag during triage).

## Reporting protocol (inter-session)

Per workstream: `status:` after spec+plan (with task count), `status:` per
implementation wave, `done:` with key numbers. `question:` for any decision a
spec leaves open — decisions this master spec or the wiki already made are not
re-litigated. Orchestrator (this session) reviews diffs via code-review-graph
`detect_changes_tool` per wave and gates workstream transitions.
