# W3 — frequency-domain analysis (spec)

Date: 2026-07-30
Workstream: W3 (master spec `2026-07-29-meep-torch-complete-port-master-spec.md`,
execution plan `2026-07-29-complete-port-execution-plan.md`). Upgrades the FFT
substitutions in ported cavity/ring examples with a real resonant-mode solver.

## In scope

1. **Harminv replacement (resonant f + Q).** `modes/harminv.py`: filter-
   diagonalization (FDM, Mandelshtam) on a real time series -- decimate to the
   band of interest, build the generalized Hankel eigenproblem, solve with
   scipy SVD/eig, return (f, Q, amplitude) per mode. Pure scipy (no meep).
   Decision: FDM (meep's harminv uses it; robust for Q; ESPRIT is the fallback
   if FDM is noisy on a given series). Replaces the FFT-peak Q in
   `examples/ring.py`, `examples/holey-wvg-cavity.py`, `examples/solve-cw.py`.
2. **Area DFT field monitors.** `core/monitors.py` + `mp` shim: a
   `DFTFieldsMonitor(grid, freqs)` that accumulates Re/Im of Ez (and Hy) over
   the full 2D region per frequency (meep `add_dft_fields` equivalent, 2D).
   Feeds near2far (W4) and field visualizations.
3. **solve-cw honest.** `examples/solve-cw.py`: replace the W1 FFT-substitute
   with the FDM harminv extraction (resonant freq + Q from the probe series), so
   the port is honest about being time-domain (no frequency-domain solver). The
   true sparse-direct CW solver is OPTIONAL and flagged costly -- deferred
   unless a W3 gate needs it.

## Out of scope

- True frequency-domain (sparse direct) CW solver -- optional, flagged costly;
  W3 ships the time-domain + FDM route. 3D/TE-Bloch/2.5D (later workstreams).

## Gate (master spec)

`holey-wvg-cavity` Q within 5% of meep-oracle harminv. Reference: generate the
meep `Harminv` Q for the holey-wvg-cavity via `mamba run -n meep-oracle` into
`tests/examples/data/holey_wvg_cavity_harminv.json`; tests never import meep.

## Tests / acceptance

- `tests/examples/test_w3_harminv.py` -- FDM on synthetic decaying sinusoids
  (known f, Q) recovers them; multi-mode separation; noise robustness.
- `tests/examples/test_w3_holey_cavity_q.py` -- meep-torch holey-wvg-cavity
  probe series -> FDM -> Q within 5% of meep-oracle JSON. Also re-check
  `examples/ring.py` resonant f via FDM.
- `tests/examples/test_w3_dft_fields.py` -- area DFT of a known standing wave.
- Suite addition <10 min. MIT, <=500 lines, TDD, commit per task.

## Open decisions (decided here)

- Harminv algorithm: FDM (filter-diagonalization). ESPRIT kept as a fallback
  flag if FDM is noisy. No new deps (scipy only).