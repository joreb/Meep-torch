# W3 — frequency-domain analysis (plan)

Date: 2026-07-30
Spec: `docs/specs/2026-07-30-meep-torch-w3-frequency-domain.md`.
TDD, commit per task, <=500 lines, MIT. Numeric verification only.

## Task count: 6 (3 waves)

### Wave 1 -- harminv (T1-T2)
- **T1** `modes/harminv.py`: `harminv(series, dt, fmin, fmax)` -- decimate to
  band, build the FDM generalized Hankel eigenproblem, scipy SVD + eig, return
  (f, Q, amp) sorted by |amp| in-band. TDD: synthetic 1- and 2-mode decaying
  sinusoids recovered within 0.1% f, 5% Q; noise floor robust.
- **T2** `tests/examples/test_w3_harminv.py` -- the synthetic-mode tests above;
  plus a ringdown from the ported ring example -> FDM f matches the W1 FFT peak
  but with a real (non-bin-floored) Q.

### Wave 2 -- area DFT + cavity gate (T3-T4)
- **T3** `core/monitors.py` `DFTFieldsMonitor(grid, freqs, component)` (full
  2D Re/Im accumulation) + `mp` shim `add_dft_fields`. TDD: a standing wave's
  area DFT at its frequency matches the analytic pattern.
- **T4** meep-oracle JSON for holey-wvg-cavity harminv (`mamba run -n meep-oracle`
  -> `tests/examples/data/holey_wvg_cavity_harminv.json`); meep-torch
  holey-wvg-cavity probe -> FDM -> Q within 5% of oracle.
  `tests/examples/test_w3_holey_cavity_q.py`.

### Wave 3 -- upgrades + close (T5-T6)
- **T5** Replace the FFT-peak Q in `examples/ring.py`, `examples/holey-wvg-cavity.py`,
  `examples/solve-cw.py` with `modes.harminv`; drop the bin-floored caveats.
- **T6** Gate verification + commit; `done:` with Q vs oracle. Optional
  solve-cw sparse-direct flagged not-built (W3 ships time-domain + FDM).

## Wave gates
- W1: FDM recovers synthetic f/Q.
- W2: holey-wvg-cavity Q within 5% of meep-oracle; area DFT of a standing wave.
- W3: ring/holey/solve-cw use FDM; suite green.

## Standing rules
- Reference data via `mamba run -n meep-oracle`; tests never import meep.
- Numeric verification only. Suite addition <10 min. `question:` only if a
  spec decision is blocked (none currently).