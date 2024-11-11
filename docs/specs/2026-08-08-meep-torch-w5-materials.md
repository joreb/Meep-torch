# W5 — materials completion, wave 1 scope (spec)

Date: 2026-08-08
Workstream: W5 (master spec `2026-07-29-meep-torch-complete-port-master-spec.md`,
execution plan `2026-07-29-complete-port-execution-plan.md`). Materials the
FDTD steppers are missing: electric/magnetic conductivity, scalar mu,
Absorber boundary, in-plane anisotropic eps (TE). Gyrotropic, multilevel-atom,
noise sources, magnetic Lorentz poles, complex-field mode split off to a
W5.5 spec (master spec allows the second wave).

## Physics (pinned)

- **D-conductivity** (meep `D_conductivity`): dD/dt = curl H - sigma_d D.
  Discrete: D^{n+1} = d1 D^n + d2 curl, d1 = (1 - sigma_d dt/2)/(1 + sigma_d
  dt/2), d2 = dt/(1 + sigma_d dt/2). Effective eps(w) = eps (1 + i sigma_d/w)
  — meep's convention: sigma_d multiplies D, so the absorption is
  eps-independent in the update.
- **B-conductivity + scalar mu**: dB/dt = -curl E - sigma_b B, B = mu H.
  H^{n+1/2} = h1 H^{n-1/2} -+ h2 curl, h1 as d1 with sigma_b, h2 =
  dt/(mu (1 + sigma_b dt/2)), mu averaged to H positions.
- **Absorber** (meep `mp.Absorber`): no coordinate stretching; a quadratic
  sigma_d = sigma_b ramp over the layer (matched impedance in meep units),
  peak set like meep's pml_profile with R_asymptotic = 1e-15:
  sigma_max = -(pml_order+1) ln(R) / (2 thickness), pml_order = 2, applied
  to BOTH sigma_d and sigma_b so the layer works where PML coordinate
  stretching fails (Bloch/dispersive edge cases; here it just needs to
  absorb comparably to PML).
- **In-plane anisotropic eps (TE)**: symmetric tensor (exx, exy, eyy). E from
  D per step: invert the 2x2 at each point with the off-diagonal component
  Yee-averaged (Dy 4-point-averaged to the Ex grid and vice versa; simple
  averaging, no subpixel smoothing). TM ezz stays scalar. Full 3x3 with
  ez-coupling (twisted-nematic `polarization_grating`) requires a
  full-vector 2D stepper -> re-scope that stub's blocker, stays blocked.

## In scope

1. `core/coeffs.py` + `core/stepper.py` (TM) and `core/stepper_te.py` (TE):
   sigma_d, sigma_b, mu fields (None fast-path when absent).
2. `core/pml.py` or `core/coeffs.py`: absorber sigma profile helper.
3. `mp` shim: `Medium(D_conductivity=, B_conductivity=, mu=)` rasterized to
   maps; `mp.Absorber(thickness)` boundary layer (all-direction 2D + 1D).
4. `examples/absorber-1d.py` port: Al slab (Drude-Lorentz poles from the
   meep materials library, dumped by the oracle), Absorber vs PML ringdown.
5. TE tensor eps: `build_te_coeffs(eps_tensor=(exx, exy, eyy))`.

## Out of scope (W5.5)

Gyrotropic media, multilevel-atom ADE, noise/stochastic sources, magnetic
Lorentz poles, complex-field mode, 3x3 eps with z-coupling.

## Gates

- (a) conductive slab, TM: analytic attenuation. Plane wave through sigma_d
  slab; |T| matches exp(-alpha L) from the complex index of
  eps_eff = eps (1 + i sigma_d / (2 pi f)) within 2%.
  Oracle: meep JSON transmission for the same slab within 3% rms.
- (b) absorber-1d: meep-torch Absorber ringdown envelope decays with
  residual reflection comparable to PML — |R_absorber| <= 5e-3 (field probe
  after ringdown, normalized to peak), and oracle-JSON |R| ratio within 10x.
- (c) TE tensor: rotated uniaxial slab (no=1.5, ne=2.0, 45 deg in-plane)
  transmission phase vs the analytic index ellipse n_eff(theta) within 1%.

## Tests / acceptance

- `tests/examples/test_w5_conductivity.py` — gates (a) TM + the TE dual.
- `tests/examples/test_w5_absorber.py` — gate (b) + Absorber-in-2D smoke.
- `tests/examples/test_w5_aniso_te.py` — gate (c) + exy=0 reduces to the
  isotropic stepper bit-for-bit.
- Oracle scripts under tests/examples/data/ via `mamba run -n meep-oracle`;
  tests never import meep. Suite addition <10 min. MIT, <=500 lines/file,
  TDD, commit per task.

## Open decisions (decided here)

- meep sigma convention: D_conductivity acts on D (not J = sigma E); the
  analytic gate uses meep's own eps_eff formula so oracle and torch agree.
- Absorber profile: quadratic, R_asymptotic 1e-15, matched sigma_d=sigma_b.
- Tensor averaging: plain 4-point Yee averaging (no subpixel); documented.
