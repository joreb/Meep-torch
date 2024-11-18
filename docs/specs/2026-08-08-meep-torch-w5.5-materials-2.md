# W5.5 — materials completion, wave 2 (spec)

Date: 2026-08-08
Workstream: W5.5 (second materials wave allowed by the master spec).
Delivers: magnetic Lorentz poles, gyrotropic media (1D vector stepper +
faraday-rotation), noise sources. Multilevel-atom ADE is LAST and may
defer to W5.75 if its oracle gate does not close in-session.

## Physics (pinned)

- **Magnetic Lorentz poles** (mu dispersion): dual of the electric ADE.
  M pole per H component: M(n+1/2) = c1 M(n-1/2) + c2 M(n-3/2) + c3 H;
  B = mu_inf H + sum M; H update solves H = (B - sum M)/mu_inf. Discretely
  mirrors the electric c1/c2/c3 with (fn, gamma, sigma) on H grids.
- **Gyrotropic Lorentzian** (meep GyrotropicLorentzianSusceptibility, bias
  along the propagation axis): d2P/dt2 + gamma dP/dt - 2 pi b x dP/dt
  + w0^2 P = sigma w0^2 E, P in the transverse plane. Central-difference
  discretization gives a constant 2x2 system for (Py+, Pz+); precomputed
  inverse. Lives in a NEW 1D full-vector stepper `core/stepper_1d.py`
  (fields Ey, Ez, Hy, Hz along x; the TM/TE split cannot host the Ey<->Ez
  coupling). Faraday rotation analytic reference (from the meep tutorial):
  k_+- = 2 pi f sqrt(eperp -+ eta), rotation angle = (k_- - k_+)/2 * L.
- **Noise sources**: white-noise current per cell in a region
  (`RandomSource`: fresh N(0,1) amplitude per step, fixed spatial mask,
  seeded torch generator). Spectral gate only checks band-averaged
  flatness — statistical, loose.
- **Multilevel atom** (meep MultilevelAtom): transitions with Lorentzian
  polarization + rate-equation populations, saturable gain. 1D lasing
  example. Attempted last; hard gate = meep-oracle steady-state field
  spectrum peak within 5%.

## In scope

1. `core/coeffs.py` + steppers: magnetic poles (`m_poles=` on both TM/TE).
2. `core/stepper_1d.py`: 1D vector stepper (Ey, Ez, Hy, Hz), scalar eps/mu,
   gyrotropic ADE, x-PML. `examples/faraday-rotation.py` port.
3. `core/sources.py` `RandomSource`; `examples/stochastic_emitter.py`
   re-check (port if only random sources were missing).
4. Multilevel-atom ADE (best-effort, may defer).

## Out of scope

Complex-field mode (the two-quadrature Bloch stepper already covers the
ported need; full force_complex_fields deferred until an example demands
it). 3x3 gyrotropic in 2D. Gyrotropic-Drude (meep has it; no example gate).

## Gates

- (a) magnetic-mu slab, TM: R/T vs transfer-matrix analytic with
  mu(f) = 1 + sigma f0^2/(f0^2 - f^2 - i gamma f) within 3%.
- (b) faraday-rotation: polarization rotation angle over L=20 within 2% of
  the tutorial analytic k_gyro; meep-oracle Ex/Ey profiles rms <= 5%.
- (c) RandomSource: band-averaged |E(f)|^2 flat to within +-3 dB over the
  source band (statistical, N-average).
- (d) multilevel (if landed): oracle lasing spectrum peak f within 5%.

## Tests / acceptance

- `tests/examples/test_w55_magnetic_poles.py` (a)
- `tests/examples/test_w55_faraday.py` (b)
- `tests/examples/test_w55_noise.py` (c)
- Oracle scripts via `mamba run -n meep-oracle`; tests never import meep.
  Suite addition <10 min. MIT, <=500 lines/file, TDD, commit per task.
