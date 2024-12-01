# W5.75 — multilevel-atom gain medium (spec)

Date: 2026-08-08
Workstream: W5.75 (deferred from W5.5). meep MultilevelAtom subset needed by
`examples/multilevel-atom.py`: a two-level gain medium (radiative transition
+ pump + non-radiative decay) in the 1D vector stepper.

## Physics (pinned; Cerjan et al., Opt. Express 20, 474 (2012))

Two-level atom per cell, populations N1 (ground), N2 (excited), radiative
transition 2->1 at omega_a with FWHM gamma and coupling sigma:

- Polarization ADE (transition-driven Lorentzian, inversion-scaled):
    d2P/dt2 + gamma dP/dt + omega_a^2 P = sigma * (N1 - N2) * E
  discretized like the Lorentz pole with the c3 term evaluated with the
  CURRENT inversion: P+ = c1 P + c2 P- + (sigma dt^2/den)(N1 - N2) E.
  The inversion sign is pinned by the lasing gate (inverted medium must
  amplify).
- Rate equations (ħ = 1, meep units; stimulated term = E dP/dt / omega_a):
    dN2/dt = +Rp N1 - rate_21 N2 + (E (P+ - P-)/(2 dt)) / omega_a
    dN1/dt = -Rp N1 + rate_21 N2 - (E (P+ - P-)/(2 dt)) / omega_a
  (N1 + N2 conserved up to the stimulated exchange; forward-Euler in dt.)
- Lives in `core/stepper_1d.py` (Ey polarization only — the meep example is
  1D Ex). Scalar eps host (ncav^2 inside the cavity block).

## In scope

`build_1d_coeffs(..., mla=(omega_a, gamma, sigma, Rp, rate_21, N0, mask))`;
State1D gains N1, N2 (allocated when mla set); the example port
(`examples/multilevel-atom.py`, one-sided Fabry-Perot laser, reduced
resolution) and the oracle gate.

## Out of scope

>2 levels, sigma_diag anisotropy, multiple radiative transitions, 2D. The
meep API shim (mp.MultilevelAtom/mp.Transition) — the port drives the core
stepper directly (documented substitute), the shim lands if a second
example needs it.

## Gate

(d) reduced meep-oracle run (resolution 100, endt 2000, same laser): steady
-state lasing spectrum peak f within 5% of the oracle; both engines lase
(tail field amplitude non-decaying).

## Tests / acceptance

`tests/examples/test_w575_multilevel.py`: (i) below-threshold pump (Rp
tiny) -> field decays; (ii) example parameters -> lases, peak f within 5%
of `multilevel_lasing.json`. Oracle via `mamba run -n meep-oracle`.
Suite addition <10 min. MIT, <=500 lines/file, TDD.
