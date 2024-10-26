# W2 — sources completion + oblique/Bloch (spec)

Date: 2026-07-30
Workstream: W2 (master spec `2026-07-29-meep-torch-complete-port-master-spec.md`,
execution plan `2026-07-29-complete-port-execution-plan.md`). Unblocks ~6 stubs
(oblique-planewave, oblique-source, refl-angular, refl-angular-kz2d,
diffracted_planewave, binary_grating_oblique).

## In scope

1. **Custom callable time profiles.** `core/sources.py`: a `CustomSource(callable)`
   profile wrapping `f(t) -> float` into the `.amplitude(t)` interface, so meep
   `Source(src=callable)` / `amp_func` time envelopes work.
2. **ContinuousSource semantics.** `core/sources.py`: faithful CW turn-on --
   `ContinuousSource(frequency, width, slowness, cutoff)` with an is-1-cosine
   ramp over `slowness*width` (meep convention); W1's basic CWRamp becomes the
   `width=None` default.
3. **Spatial amplitude functions.** `core/sources.py` LineSource (both pol) +
   a 2D `AreaSource`: `amp_func(Vector3)->float` weights the source mask. W1's
   shim `Source.amp_func` already does the line case; lift the real spatial
   profile into core and add the 2D area path.
4. **Oblique/Bloch planewave source.**
   - Oblique source in a PML box (`oblique-source.py`): a phased real line
     source, amplitude `cos(k_y y - 2π f t)`, `k_y = k0 sin(theta)`, launches a
     planewave at angle theta. ONE real run.
   - Bloch-periodic boundaries (`oblique-planewave.py`, `refl-angular.py`,
     `refl-angular-kz2d.py`, `diffracted_planewave.py`,
     `binary_grating_oblique.py`): the meep-torch engine is real-valued, so a
     nonzero Bloch phase `exp(i k_y L)` is realized by the **two-quadrature
     real-run trick**: run the sim twice (cos and sin spatial quadrature) and
     combine `E = E_cos + i E_sin`. The y boundary becomes a phase-shifted
     periodic link implemented as a real 2x2 rotation on the (cos, sin) pair
     (no complex fields). Decision: two-quadrature (not time-delay) for Bloch,
     because the periodic boundary itself carries the phase; tradeoff is 2x
     runtime + a wrapper, documented. (A single real phased line is used only
     for the PML-box oblique source, where no Bloch boundary is needed.)
5. **stop_when_fields_decayed.** `core/run.py`: `run_until_decayed(state,
   coeffs, source, monitors, point, decay_ratio=1e-6, n_decay=50)` stops when
   `|E|^2` at `point` decays by `decay_ratio` over `n_decay` steps. Plumb an
   optional decay stop into `compute_smatrix` (replacing the fixed-`n_steps`
   `ponytail:` mark).
6. **Port the 6 stubs** to real `examples/<name>.py` (overwriting stubs) +
   `tests/examples/test_w2_*.py` physics gates.

## Gates (master spec)

- `oblique-planewave`: propagation angle matches set theta within 1 deg at res.
- `refl-angular`: Fresnel R(theta) vs analytic <1%.
- `binary_grating_oblique`: diffraction orders vs meep-oracle JSON
  (`mamba run -n meep-oracle`, committed JSON).

## Out of scope

- Complex-valued FDTD (two-quadrature avoids it). Harminv (W3), near2far (W4).
- 3D Bloch (W9). MPB (Q1 track).

## Open decisions (decided here, per the queued brief)

- Oblique phasing: **two-quadrature real runs** for Bloch boundaries; single
  real phased line for the PML-box oblique source. (Peer flagged: "complex
  phasing via two-quadrature real runs OR time-delay phasing -- decide in W2
  spec, note tradeoff." Two-quadrature chosen: keeps the engine real, the
  periodic BC carries the phase as a 2x2 rotation. Cost: 2x runtime per Bloch
  sim + a combining wrapper.)

## Tests / acceptance

- `tests/examples/test_w2_sources.py` -- CustomSource/ContinuousSource/spatial
  amp/oblique-phased-line unit tests.
- `tests/examples/test_w2_bloch.py` -- two-quadrature Bloch: a planewave at
  theta exits the y-periodic boundary with the right phase (field periodicity
  matches `exp(i k_y L)` reconstructed from cos/sin).
- `tests/examples/test_w2_examples.py` -- the 6 ported-stub gates.
- Suite addition <10 min. MIT, <=500 lines, TDD, commit per task.