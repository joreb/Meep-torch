# W7 — cylindrical engine, wave 1: radial BOR (spec)

Date: 2026-08-08
Workstream: W7 (master spec allows the split: "largest; its spec may
split"). Wave 1 delivers the 1D radial body-of-revolution stepper (the
`ring-cyl` class of problems: fields F(r) e^{i m phi}, no z dependence).
Full (r, z) 2D BOR, the {Er, Ephi, Hz} dual set, and cylindrical eigenmode
ports go to W7.5.

## Physics (pinned)

Fields ~ F(r) e^{i m phi}, d/dz = 0. The Ez-polarized set closes on
{Ez, Hr, Hphi} (complex, torch complex128):

  dBr/dt   = -(i m / r) Ez              (Hr collocated with Ez at r_i = i dr)
  dBphi/dt = + dEz/dr                   (Hphi at r_{i+1/2})
  dDz/dt   = (1/r) d(r Hphi)/dr - (i m / r) Hr

Discrete Dz curl at r_i (i >= 1):
  (r_{i+1/2} Hphi_i - r_{i-1/2} Hphi_{i-1}) / (r_i dr) - (i m / r_i) Hr_i.
Axis r = 0: for m != 0, regularity forces Ez(0) = 0 (pinned, update
skipped; Hr(0) = 0). m = 0 axis (ampere loop 4 Hphi_{1/2}/dr) ships with
W7.5 when an m=0 example needs it -- wave-1 gate uses m >= 1.

Outer boundary: matched sigma_d = sigma_b quadratic absorber ramp
(absorber_sigma, W5) on the outer radius -- the meep tutorial itself uses
a huge dpml = 32 with pad 4, so plain absorption at that depth is
adequate; CPML-in-r deferred to W7.5. Courant dt = 0.5 dr.

## In scope

`core/stepper_cyl.py`: CylCoeffs/CylState, `build_cyl_coeffs(grid_r, m,
eps, n_absorber)`, `step_cyl`, `run_cyl(probe_ix)`. `examples/ring-cyl.py`
port (harminv resonances per m). Oracle + gate.

## Out of scope (W7.5)

(r, z) 2D BOR, {Er, Ephi, Hz} set, m = 0 axis, cylindrical CPML,
eigenmode ports, the other ~9 cylindrical examples.

## Gate

ring-cyl (n = 3.4, w = 1, r = 1, pad = 4, dpml = 32, res 20, m = 3,
fcen = 0.15, df = 0.1): dominant harminv resonance f within 1% of the
meep-oracle, Q within 20% (radiative Q is boundary-sensitive; f is the
physics gate). Second m (m = 4) f within 1%.

## Tests / acceptance

`tests/examples/test_w7_ring_cyl.py` vs `ring_cyl.json` (oracle:
`mamba run -n meep-oracle`, harminv modes for m = 3 and m = 4).
Suite addition <10 min. MIT, <=500 lines/file, TDD.
