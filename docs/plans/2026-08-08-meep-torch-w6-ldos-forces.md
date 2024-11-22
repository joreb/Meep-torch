# W6 — LDOS + forces (plan)

Date: 2026-08-08
Spec: `docs/specs/2026-08-08-meep-torch-w6-ldos-forces.md`.
TDD, commit per task, <=500 lines/file, MIT headers. Numeric verification only.

## Task count: 4 (2 waves)

### Wave 1 — LDOS (T1-T2)
- **T1** `core/ldos.py` `ldos_spectrum(e_series, j_series, dt, freqs)` =
  -Re[E_hat conj(J_hat)]/|J_hat|^2 per freq (trapezoid DFT). TDD
  (`tests/examples/test_w6_ldos.py`): vacuum 2D Ez dipole -- band-flat
  <=3%, translation-invariant <=2% (gate a).
- **T2** oracle `generate_metal_cavity_ldos.py` (meep, w=0.2 cavity,
  sigma_d metal substitute is NOT needed oracle-side -- meep uses mp.metal;
  dump ldos spectrum + harminv f/Q) -> `metal_cavity_ldos.json`; port
  `examples/metal-cavity-ldos.py` (sigma_d=1e4 metal, harminv + LDOS);
  gate (b).

### Wave 2 — MST forces (T3-T4)
- **T3** `core/forces.py` `mst_force_tm(planes, grid, ix0, ix1, iy0, iy1,
  freqs)` from add_dft_fields planes (Ez, Hx, Hy; H averaged to contour
  points, trapezoid corners like near2far). TDD
  (`tests/examples/test_w6_force.py`): radiation pressure on a sigma_d
  mirror vs (1+R) S_inc <5% (gate c analytic half).
- **T4** oracle `generate_radiation_force.py` (meep add_force, same mirror)
  -> `radiation_force.json`, gate (c) oracle half; re-scope
  parallel-wvgs-force stub (blocker: EigenModeSource in Simulation.run);
  mark W6 done; suite green; `done:` with numbers.

## Standing rules
- Reference data via `mamba run -n meep-oracle`; tests never import meep.
- `question:` only if a spec decision is blocked (none currently).
