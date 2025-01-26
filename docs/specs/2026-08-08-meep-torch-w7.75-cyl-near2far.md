# W7.75 — cylindrical near2far, m = 0 (spec)

Date: 2026-08-08
Workstream: W7.75. Far-field radiation patterns from a BOR (r, z) contour,
m = 0 (axisymmetric currents — on-axis dipole class of examples).
|m| >= 1 far fields deferred to W7.9.

## Physics (pinned)

Surface equivalence on an (r, z) box around the sources: J = n_hat x H,
M = -n_hat x E per contour point, azimuthally symmetric (m = 0). Far
field in direction theta from z (phi = 0 WLOG), beta = k r' sin theta:

  Nx = 2 pi i INT dl r' Jr J1(beta) e^{i k z' cos theta}
  Ny = 2 pi i INT dl r' Jphi J1(beta) e^{...}
  Nz = 2 pi   INT dl r' Jz  J0(beta) e^{...}
  N_theta = Nx cos theta - Nz sin theta,  N_phi = Ny  (same for L from M)
  E_theta ~ -(N_theta + L_phi),  E_phi ~ -(N_phi - L_theta)   (eta = 1)

Overall prefactor/sign pinned by the analytic gate (a); the DFT phasors
are e^{+i w t} (project the z' phase accordingly). Radial Poynting
pattern = |E_theta|^2 + |E_phi|^2 (normalized — the 1/R falloff and
absolute constant drop out of every gate).

Face currents (BOR components): outer-r face (n = +r): J = (0, -Hz,
Hphi), M = (0, Ez, -Ephi); z faces (n = +-z): J = -+ (Hphi, -Hr·?, 0)
— exact signs pinned by gate (a), the spec fixes only the structure.

## In scope

`core/near2far_cyl.py`: `bor_faces_from_state` (DFT contour monitors over
run_bor — accumulated by a provided callback or post-hoc from stored DFT
planes), `farfield_bor_m0(faces, k, thetas)`; gates; one example port
(`plot_radiation_pattern_dipole`, on-axis Ez dipole).

## Gates

- (a) on-axis Ez point dipole (m = 0) in vacuum: normalized radial
  Poynting pattern vs sin^2 theta, rms <= 3% over theta in (10, 170) deg.
- (b) same pattern vs the meep-oracle cylindrical near2far JSON
  (reduced res), rms <= 5% normalized.

## Tests / acceptance

`tests/examples/test_w775_cyl_n2f.py` vs `dipole_cyl_n2f.json`. Oracle
via `mamba run -n meep-oracle`. Suite addition <10 min. MIT, <=500
lines/file, TDD.
