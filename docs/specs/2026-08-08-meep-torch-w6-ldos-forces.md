# W6 — LDOS + forces (spec)

Date: 2026-08-08
Workstream: W6 (master spec). Point LDOS from source-field overlap and
Maxwell-stress-tensor optical forces from DFT contour fields.

## Physics (pinned)

- **LDOS** (meep dft_ldos): with a point current J(t) at r0 and the field
  it drives, LDOS(w) ∝ -Re[E_hat(w) conj(J_hat(w))] / |J_hat(w)|^2 (the
  meep normalization constant drops out of every gate below — all gates are
  ratios: cavity/vacuum Purcell, or Q/Vmode cross-checks).
- **MST force** (2D TM, time-averaged from DFT phasors):
  <T_ij> = 1/2 Re[ E_i conj(E_j) - delta_ij |E|^2 / 2
                 + H_i conj(H_j) - delta_ij |H|^2 / 2 ],
  F = closed-contour sum of T . n_hat dl. TM has E = Ez z_hat only:
  T_xx = 1/2 Re[ -|Ez|^2/2 + |Hx|^2 - (|Hx|^2+|Hy|^2)/2 ], etc.
  Contour fields from full-grid DFTFieldsMonitor planes (Ez, Hx, Hy),
  staggered H averaged to the contour points.
- **PEC substitute**: mp.metal -> large D_conductivity (sigma_d = 1e4;
  skin depth << cell at the gate frequencies). Used by metal-cavity-ldos.

## In scope

1. `core/ldos.py`: `ldos_spectrum(e_series, j_series, dt, freqs)`;
   shim `Simulation.dft_ldos(fcen, df, nfreq)` -> run records the source
   point Ez series alongside monitors.
2. `core/forces.py`: `mst_force_tm(dft_planes, grid, box_indices, freqs)`
   -> (Fx, Fy) per freq; shim `add_force`-lite deferred until an example
   needs meep's exact API (the gate uses the core function).
3. `examples/metal-cavity-ldos.py` port (sigma_d metal, harminv Q, LDOS
   vs Q/Vmode cross-check like the meep original).
4. parallel-wvgs-force: re-scope stub -> blocked on EigenModeSource in
   Simulation.run (S-param path only today), NOT on MST (which lands here).

## Gates

- (a) vacuum LDOS: flat across the band to <=3% and equal at two probe
  locations (translation invariance) to <=2%.
- (b) metal-cavity-ldos: LDOS resonance f within 2% of the harminv f of
  the same cavity; LDOS-derived Q (peak fit) within 15% of harminv Q
  (meep's own example demonstrates ~10% agreement of ldos0 vs Q/Vmode).
  Oracle: meep JSON of the w=0.2 cavity ldos spectrum, peak f within 2%.
- (c) radiation pressure: plane wave onto a sigma_d mirror, contour force
  <F_x>/A vs (1 + R) S_inc within 5% (R from the measured reflection);
  Oracle: meep add_force JSON for the same setup within 10%.

## Tests / acceptance

- `tests/examples/test_w6_ldos.py` (a) + (b); oracle JSON
  `metal_cavity_ldos.json`.
- `tests/examples/test_w6_force.py` (c); oracle JSON `radiation_force.json`.
- Oracle scripts via `mamba run -n meep-oracle`; tests never import meep.
  Suite addition <10 min. MIT, <=500 lines/file, TDD, commit per task.
