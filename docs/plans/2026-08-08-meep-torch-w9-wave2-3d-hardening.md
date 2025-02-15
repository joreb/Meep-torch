# W9 wave 2 — 3D engine hardening (plan)

Date: 2026-08-08
Spec: `docs/specs/2026-08-08-meep-torch-w9-wave2-3d-hardening.md`.
TDD, commit per task, <=500 lines/file, MIT headers. Forward-only (no
autograd/checkpoint in 3D — master spec). Suite addition <10 min.

## Task count: 4

- **T1** `docs/specs/2026-08-08-meep-torch-w9-wave2-3d-hardening.md` +
  `docs/plans/2026-08-08-meep-torch-w9-wave2-3d-hardening.md` (this pair,
  house style per w6/w8).
- **T2** `meep_torch/core/stepper_3d.py` hardening:
  - `State3D`: add 12 psi fields (6 H-update: `psi_Ez_y, psi_Ey_z,
    psi_Ex_z, psi_Ez_x, psi_Ey_x, psi_Ex_y`; 6 E-update: `psi_Hz_y,
    psi_Hy_z, psi_Hx_z, psi_Hz_x, psi_Hy_x, psi_Hx_y`). Optional (None
    when CPML off) to keep the absorber path zero-overhead.
  - `Coeffs3D`: add per-axis CPML `(b,a,k)_x/y/z` (from `cpml_profiles`,
    broadcast shapes), optional `sigma_d`/`sigma_b` (d1/d2 pairs at E/H
    nodes), and a boundary-mode selector (`npml` CPML vs `n_abs`
    absorber). `build_3d_coeffs(..., npml=0, n_abs=0, sigma_d=None,
    sigma_b=None)`.
  - `step_3d`: CPML recursion in each curl term when psi present
    (`psi' = b*psi + a*d`; `curl = d/k + psi`); material-conductivity
    d1/d2 fast-path; plane/area source via a spatial mask (extend `src`
    to `(comp, mask)` where mask is the E-node field of the component, or
    keep the point form).
  - Absorber path (wave 1) untouched as the `npml=0` fallback.
- **T3** `meep_torch/core/monitors_3d.py`:
  - `DFTFieldsMonitor3D(grid3d-ish, freqs, component, plane=...)`:
    `reduce="sum"`, `sample` returns `(nfreq,2,...)` `dt*[cos, -sin]F`.
  - `FluxMonitor3D(..., normal="x"|"y"|"z", idx, lo, hi)`:
    `reduce="sum"`, samples transverse E + averaged H on the plane;
    `flux(acc, dx)` returns the signed Poynting integral.
  - `run_3d(state, coeffs, n_steps, *, source=None, monitors=())`:
    forward-only loop, monitor accumulation (same absorb/stack protocol
    as `run_tm`), no checkpoint.
- **T4** `tests/examples/test_w9_wave2.py`:
  - gate (a): small 3D cell, z-normal planewave (area source on a y-z
    plane), lossy slab (eps, sigma_d), far `FluxMonitor3D` → `|T|` vs
    analytic lossy-slab transmission <=3%.
  - gate (b): same geometry, vacuum, measure back-boundary spurious
    reflection for CPML vs absorber (same depth); assert CPML residual
    <= absorber and CPML <= few-% of incident. **As built** (see spec
    "Gate (b) resolution"): z-uniform planewave + mixed mode (identity
    CPML profile on the absorber axis, `absorber_m`/`absorber_R0` matched
    to the CPML), observable = phase-robust per-bounce decay rate
    (`mean(E^2)` in two late windows); the point-source reflected-flux
    ratio was a near-field artifact (size-independent) and was dropped.
  - assert wave-1 `test_w9_3d.py` still green (absorber fallback).

## Standing rules

- Ground truth = analytic slab + relative PML reflection (no meep oracle).
- `status:` to peer before each git commit (shared working tree); commit
  ONLY docs/T2/T3/T4 files; `done:` with hashes + gate numbers.
- Keep grids small — 3D is O(n^3); single drive frequency, short pulse.