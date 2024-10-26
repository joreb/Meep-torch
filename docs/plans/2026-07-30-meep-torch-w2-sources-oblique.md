# W2 — sources completion + oblique/Bloch (plan)

Date: 2026-07-30
Spec: `docs/specs/2026-07-30-meep-torch-w2-sources-oblique.md`.
TDD, commit per task, <=500 lines/file, MIT headers. Numeric verification only.

## Task count: 11 (4 waves)

### Wave 1 — source semantics (T1–T3)
- **T1** `core/sources.py`: `CustomSource(callable)` profile; faithful
  `ContinuousSource(frequency, width, slowness, cutoff)` is-1-cosine ramp.
  TDD: ramp shape vs analytic is-1-cos over slowness*width; CustomSource passes
  a known f(t).
- **T2** `core/sources.py`: `LineSource.amp_func` spatial profile (both pol) +
  `AreaSource(grid, mask, profile, amp_func)` 2D area source. TDD: gaussian /
  oblique-cos spatial mask matches the expected weights; AreaSource launches a
  tilted field.
- **T3** `core/sources.py`: `ObliqueLineSource(grid, ix, j0, j1, profile,
  theta, fcen)` phased real line, `cos(k_y y - 2π f t)`, `k_y=2π f sin(theta)`.
  TDD: launched field's k-spectrum peak at k_y within 1 deg.

### Wave 2 — Bloch two-quadrature (T4–T5)
- **T4** `core/stepper.py` + `core/coeffs.py`: a Bloch-periodic y boundary mode
  (`y_bloch=True`, `k_y`) implemented as a real rotation on the (cos, sin)
  quadrature pair, OR a single-run real Bloch BC that links the two y edges with
  a cos/sin phase via a 2-field buffer. Decide the cleanest real implementation
  in T4; document. TDD: a standing wave in a periodic cell stays periodic with
  the right phase.
- **T5** `core/run.py` + `meep_torch/mp/simulation.py`: a `BlochSimulation`
  wrapper (or a `bloch_k` kwarg) running the cos + sin quadrature and combining
  `E = E_cos + i E_sin`. TDD: reconstruct `exp(i k_y y)` from the two runs on a
  unit cell.

### Wave 3 — run control (T6)
- **T6** `core/run.py`: `run_until_decayed` (decay at a probe over `n_decay`
  steps). Plumb an optional `stop_decay`/`probe` into `compute_smatrix` to
  replace the fixed-`n_steps` `ponytail:` mark. TDD: a pulse run stops within a
  few steps of the analytic decay time; S-params unchanged vs a long fixed run.

### Wave 4 — port the 6 stubs (T7–T11)
- **T7** `examples/oblique-source.py` (overwrite stub) via ObliqueLineSource
  (PML box, no Bloch). Gate: launched angle within 1 deg.
- **T8** `examples/oblique-planewave.py` via Bloch. Gate: propagation angle
  within 1 deg at res.
- **T9** `examples/refl-angular.py` + `refl-angular-kz2d.py` via Bloch. Gate:
  Fresnel R(theta) vs analytic <1%.
- **T10** `examples/diffracted_planewave.py` via Bloch. Gate: diffraction order
  angles vs grating equation.
- **T11** `examples/binary_grating_oblique.py` via Bloch. Gate: diffraction
  orders vs meep-oracle JSON (`tests/examples/data/binary_grating_oblique.json`
  generated via `mamba run -n meep-oracle`).
- `tests/examples/test_w2_examples.py` runs the gates.

## Wave gates
- W1: source-semantics unit tests green; oblique line launches at theta.
- W2: Bloch two-quadrature reconstructs exp(i k_y y).
- W3: run_until_decayed stops on time; S-params unchanged.
- W4: the 6 stubs ported, all 3 numeric gates pass.

## Standing rules
- Reference data via `mamba run -n meep-oracle` into committed JSON; tests
  never import meep.
- Numeric verification only (no image reads). Suite addition <10 min.
- `question:` only if a spec decision is blocked (none currently).