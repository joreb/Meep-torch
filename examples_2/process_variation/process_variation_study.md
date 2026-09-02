# Process-variation re-optimization study

Device: inverse-designed MMI 1x2 splitter (DEFAULT gdsfactory mmi1x2_with_sbend seed), SiN 150 nm / SiO2, 674 nm, 2D lateral effective index, exact-adjoint FDFD (pardiso). Baseline design: `examples_2/c_y_splitter_fdfd_rho.npy` (24-iteration FDFD optimization).

Perturbation: core (film) index n_SiN shifted by Δn = -0.04 (PECVD-like deficit vs stoichiometric LPCVD) and Δn = -0.10; the slab effective index and the straight-guide normalization reference are rebuilt at each index, so all transmissions are like-for-like mode-power ratios.

| state | n_SiN | n_eff | Tu [%] | Td [%] | total [%] | IL [dB] | imbalance [%] |
|---|---|---|---|---|---|---|---|
| baseline@n1 | 2.0347 | 1.6321 | 39.68 | 41.99 | 81.67 | 0.88 | 2.82 |
| frozen@n2 | 1.9947 | 1.6145 | 41.57 | 43.23 | 84.80 | 0.72 | 1.97 |
| reopt@n2 | 1.9947 | 1.6145 | 44.61 | 44.39 | 89.00 | 0.51 | 0.24 |
| frozen@n3 | 1.9347 | 1.5896 | 41.07 | 41.31 | 82.38 | 0.84 | 0.30 |
| reopt@n3 | 1.9347 | 1.5896 | 49.71 | 49.85 | 99.56 | 0.02 | 0.14 |
| frozen@n1+0.04 | 2.0747 | 1.6504 | 38.50 | 41.38 | 79.88 | 0.98 | 3.61 |
| reopt-design@n1(xcheck) | 2.0347 | 1.6321 | 40.51 | 40.51 | 81.02 | 0.91 | 0.00 |

- Re-optimization: Adam (lr=0.03, beta 32→64), seeded from baseline rho, early-stop on plateau; 48 iterations / best it47 / 1.3 min (Δn=-0.04), 64 iterations / best it63 / 1.7 min (Δn=-0.10).
- Re-optimized designs saved: `c_y_splitter_fdfd_rho_reopt_n2.npy` (Δn=-0.04), `c_y_splitter_fdfd_rho_reopt_n3.npy` (Δn=-0.10).
- All values from actual FDFD solves; nothing extrapolated.
