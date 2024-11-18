# W5.5 — materials completion, wave 2 (plan)

Date: 2026-08-08
Spec: `docs/specs/2026-08-08-meep-torch-w5.5-materials-2.md`.
TDD, commit per task, <=500 lines/file, MIT headers. Numeric verification only.

## Task count: 6 (4 waves)

### Wave 1 — magnetic Lorentz poles (T1)
- **T1** `build_tm_coeffs(m_poles=, m_pole_masks=)`: c1/c2/c3 duals on the
  Hx/Hy grids (position-averaged masks); `step_tm` M-ADE: B update then
  H = (B - sum M)/mu. Same for TE (Hz grid). TDD
  (`tests/examples/test_w55_magnetic_poles.py`): mu-Lorentz slab R/T vs
  transfer-matrix analytic <3% (TM); no-pole path untouched (bit-parity).

### Wave 2 — gyrotropic 1D stepper + faraday (T2-T3)
- **T2** `core/stepper_1d.py`: Vec1DState (Ey, Ez, Hy, Hz, Dy, Dz, P pairs,
  psi), `build_1d_coeffs(grid_x, eps, mu, npml, gyro=(f0, gamma, sigma, b0))`,
  `step_1d`; gyrotropic central-difference 2x2 ADE. TDD
  (`tests/examples/test_w55_faraday.py`): CW Ey line launch, rotation angle
  at L=10 and L=15 vs analytic k_gyro <2%.
- **T3** oracle `generate_faraday.py` (meep tutorial setup, res 50) ->
  `faraday.json` (Ex/Ey profiles at tmax); port
  `examples/faraday-rotation.py` (profiles overlay png); profiles rms <=5%.

### Wave 3 — noise sources (T4)
- **T4** `core/sources.py` `RandomSource(grid, region_mask, amplitude,
  seed)`: fresh N(0,1) per step. TDD (`tests/examples/test_w55_noise.py`):
  band-averaged spectrum flat +-3 dB. Re-check `stochastic_emitter.py`
  blockers; port only if random sources were the sole gap (it also needs
  LDOS-style normalization -> note if still blocked).

### Wave 4 — multilevel atom, best-effort (T5-T6)
- **T5** MultilevelAtom ADE (transitions: Lorentzian P + populations via
  rate equations) in the 1D stepper; oracle lasing run.
- **T6** Gate (d) or documented deferral to W5.75; mark W5.5 in the
  execution plan; suite green; `done:`.

## Standing rules
- Reference data via `mamba run -n meep-oracle`; tests never import meep.
- `question:` only if a spec decision is blocked (none currently).
