# W7.5 — cylindrical engine, wave 2: (r, z) 2D BOR (spec)

Date: 2026-08-08
Workstream: W7.5. Full 2D body-of-revolution FDTD for azimuthal number m:
all six field components F(r, z) e^{i m phi}, complex.

## Physics (pinned)

Yee staggering on the (r, z) grid (i = r index, j = z index):
  Er(i+1/2, j)   Ephi(i, j)     Ez(i, j+1/2)
  Hr(i, j+1/2)   Hphi(i+1/2, j+1/2)   Hz(i+1/2, j)

Updates (eps0 = mu0 = 1; e^{i m phi}):
  dBr/dt   = -(i m / r) Ez + dEphi/dz          at (i, j+1/2), r = i dr
  dBphi/dt = -dEr/dz + dEz/dr                  at (i+1/2, j+1/2)
  dBz/dt   = -(1/r) d(r Ephi)/dr + (i m / r) Er   at (i+1/2, j), r = (i+1/2) dr
  dDr/dt   = (i m / r) Hz - dHphi/dz           at (i+1/2, j)
  dDphi/dt = dHr/dz - dHz/dr                   at (i, j)
  dDz/dt   = (1/r) d(r Hphi)/dr - (i m / r) Hr at (i, j+1/2), r = i dr

Axis i = 0:
- m = 0: Ephi(0) = Hr(0) = 0 (regularity); Ez(0, j+1/2) via the ampere
  loop: dDz/dt(0) = (4/dr) Hphi(1/2, j+1/2). Dphi(0) row pinned.
- m != 0: all i = 0 rows pinned; additionally rows r < m dr / 4 pinned
  (the i m / r rotation vs the leapfrog limit, W7 wave-1 result).

Boundaries: matched sigma_d = sigma_b absorber on outer r and both z ends
(cylindrical CPML deferred). Courant dt = 0.5 dr / sqrt(2) (2D).
eps(r, z) cell map averaged to each E position (like build_te_coeffs).

## In scope

`core/stepper_bor.py` (coeffs/state/step/run), gates (a) + (b), the
`disc` resonance path that most cylindrical examples build on.

## Out of scope (W7.75+)

Cylindrical near2far, eigenmode ports, LDOS-in-cyl shim wiring, the ~9
example ports (they follow the same pattern once this core lands),
dispersive materials in BOR.

## Gates

- (a) m = 0 on-axis dipole in vacuum = physical 3D point dipole: LDOS
  (core.ldos on the axis probe) proportional to f^2 within 4% over the
  band — distinguishes 3D BOR from 2D (which is linear in f).
- (b) disc resonator (radius 1.2, height 0.29, n = 2.4, m = 3 — the
  disc_radiation_pattern geometry): dominant harminv f within 1% and Q
  within 25% of the meep-oracle cylindrical reference.

## Tests / acceptance

`tests/examples/test_w75_bor.py` vs `disc_bor.json` (oracle via
`mamba run -n meep-oracle`). Suite addition <10 min. MIT, <=500
lines/file, TDD.
