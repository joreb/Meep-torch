# W5 — materials completion, wave 1 scope (plan)

Date: 2026-08-08
Spec: `docs/specs/2026-08-08-meep-torch-w5-materials.md`.
TDD, commit per task, <=500 lines/file, MIT headers. Numeric verification only.

## Task count: 6 (3 waves)

### Wave 1 — conductivity + mu, TM+TE core (T1-T2)
- **T1** TM: `build_tm_coeffs(..., sigma_d=None, sigma_b=None, mu=None)` ->
  TMCoeffs gains d1/d2 (nx, ny) and h1x/h2x (nx, ny-1), h1y/h2y (nx-1, ny)
  (None when all absent -> stepper fast-path unchanged). `step_tm`:
  Dz = d1*Dz + d2*curl; Hx = h1x*Hx - h2x*(...); Hy = h1y*Hy + h2y*(...).
  TDD (`tests/examples/test_w5_conductivity.py`): (i) sigma=0 tensors give
  bit-identical fields to the fast path; (ii) gate (a): 1D-style TM slab
  (y_pmc), sigma_d slab L=2, f=1, |T| vs analytic exp(-alpha L) < 2%.
- **T2** TE dual (same coeff names on the Ex/Ey/Hz grids) + oracle:
  `generate_conductive_slab.py` (mamba meep-oracle, 2D TM slab transmission)
  -> `conductive_slab.json`; torch vs oracle < 3% rms over the pulse band.

### Wave 2 — Absorber + shim + absorber-1d (T3-T4)
- **T3** absorber profile helper `absorber_sigma(n, n_layer, dt)` (quadratic,
  R=1e-15, applied at both edges of an axis) + shim: `Medium(D_conductivity,
  B_conductivity, mu)` rasterized via build_scene_items -> sigma/mu maps
  fed to build_*_coeffs; `mp.Absorber(thickness)` accepted in
  boundary_layers (adds the sigma ramp, npml=0 for that edge).
  TDD (`tests/examples/test_w5_absorber.py`): 2D vacuum pulse into an
  Absorber edge -> residual probe amplitude <= 5e-3 of peak.
- **T4** oracle `generate_absorber_1d.py` (meep absorber-1d, Al, ringdown
  |R| proxy) -> `absorber_1d.json`; port `examples/absorber-1d.py` (1D Al,
  Absorber vs PML ringdown overlay png); gate (b).
- Al Drude-Lorentz poles: dump `meep.materials.Al` eps model params in the
  oracle JSON, rebuild as LorentzPole list in the port (no meep import).

### Wave 3 — TE tensor eps + close (T5-T6)
- **T5** `build_te_coeffs(eps_tensor=(exx, exy, eyy))`: per-step 2x2 E-from-D
  with 4-point Yee averaging of the cross terms. TDD
  (`tests/examples/test_w5_aniso_te.py`): exy=0 == isotropic bit-for-bit;
  gate (c) rotated uniaxial slab n_eff(45deg) phase < 1%.
- **T6** Re-scope `examples/polarization_grating.py` stub blocker to
  "full-vector 2D stepper (ez-coupled eps tensor)"; mark W5 wave-1 scope
  done in the execution plan (gyrotropic/multilevel/noise -> W5.5);
  suite green; `done:` with gate numbers.

## Wave gates
- W1: sigma=0 bit-parity; analytic attenuation <2%; oracle slab <3% rms.
- W2: Absorber residual <=5e-3; absorber-1d ringdown vs oracle (gate b).
- W3: tensor reduces to isotropic; n_eff(45deg) <1%; suite green.

## Standing rules
- Reference data via `mamba run -n meep-oracle`; tests never import meep.
- Numeric verification only. Suite addition <10 min. `question:` only if a
  spec decision is blocked (none currently).
