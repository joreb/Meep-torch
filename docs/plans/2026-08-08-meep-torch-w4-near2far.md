# W4 — near-to-far transform, 2D (plan)

Date: 2026-08-08
Spec: `docs/specs/2026-08-08-meep-torch-w4-near2far.md`.
TDD, commit per task, <=500 lines/file, MIT headers. Numeric verification only.

## Task count: 6 (3 waves)

### Wave 1 — Green kernels + analytic self-test (T1-T2)
- **T1** `core/near2far.py` part 1: `line_source_fields(k, pts, x0=0, y0=0)`
  analytic TM z line current reference (Ez = -(omega/4) H0^(1)(kR),
  Hx = -dEz/dy / (i omega), Hy = dEz/dx / (i omega); H1^(1) closed forms) +
  `farfield_tm(Ez, Ht, faces, k, obs_pts)` contour sum with exact Hankel
  kernels. faces = list of (pts (N,2), n_hat (2,), dl, Ez (nf,N), Ht (nf,N)).
  TDD (`tests/examples/test_w4_green2d.py`): synthesize near fields on a
  1x1 box contour from `line_source_fields`, farfield at r=500..2000 over
  16 angles matches `line_source_fields` there rel <1e-3; pattern invariant
  under contour translation (source fixed).
- **T2** TE dual `farfield_te(Hz, Et, faces, k, obs_pts)` via the same kernels
  with (E<->H, J<->-M) swap; TDD: magnetic line source (Hz = -(omega/4) H0)
  dual self-test, same tolerances.

### Wave 2 — FDTD monitor + shim + oracle (T3-T4)
- **T3** `Near2FarMonitor(grid, freqs, faces, pol)` — per-face DFT
  accumulation (reuse the FluxMonitorTM/TE sampling pattern: x-normal face
  samples Ez + avg Hy [TM]; y-normal Ez + avg Hx; TE dual), `reduce="sum"`;
  `mp` shim `Near2FarRegion`, `add_near2far(fcen, df, nfreq, *regions)`,
  `get_farfield(handle, Vector3)` -> 6 complex meep-layout components.
  TDD: FDTD Ez point source in vacuum, box contour, farfield magnitude at
  r=1000 isotropic to <=2% rms over 100 angles (gate a).
- **T4** meep-oracle reference: `mamba run -n meep-oracle` antenna-radiation
  (resolution 25, Ez + Ey) -> `tests/examples/data/antenna_radiation_n2f.json`
  (radial flux vs 100 polar angles, both polarizations);
  `tests/examples/test_w4_antenna_radiation.py` gate (c): normalized patterns
  match oracle <=5% rms.

### Wave 3 — TE gate + example + close (T5-T6)
- **T5** TE path end-to-end: FDTD Ey dipole (TE stepper), Near2FarMonitor TE,
  gate (b) cos^2 lobe <=3% rms normalized + oracle Ey row of gate (c).
- **T6** Port `examples/antenna-radiation.py` (Ez + Ey polar plots vs
  analytic, png); suite green; mark W4 done in execution plan; `done:` with
  rms numbers.

## Wave gates
- W1: analytic line-source self-test <1e-3 (TM+TE).
- W2: FDTD Ez isotropy <=2% rms; oracle Ez match <=5% rms.
- W3: Ey cos^2 <=3% rms; example ported; suite green.

## Standing rules
- Reference data via `mamba run -n meep-oracle`; tests never import meep.
- Numeric verification only. Suite addition <10 min. `question:` only if a
  spec decision is blocked (none currently).
