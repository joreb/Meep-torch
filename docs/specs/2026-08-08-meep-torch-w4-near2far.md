# W4 — near-to-far transform, 2D (spec)

Date: 2026-08-08
Workstream: W4 (master spec `2026-07-29-meep-torch-complete-port-master-spec.md`,
execution plan `2026-07-29-complete-port-execution-plan.md`). Green's-function
near2far for 2D TM+TE: far-field radiation patterns from DFT fields on a
closed contour. Unlocks `antenna-radiation` first; `binary_grating_n2f`,
`finite_grating`, `cavity-farfield` et al. follow on this machinery.

## Physics (pinned)

Surface equivalence on a closed box contour C around the sources: equivalent
currents J = n_hat x H, M = -n_hat x E from the frequency-domain (DFT) near
fields. Far fields at point r are contour sums of exact 2D Green's functions
(no large-argument asymptotics; matches meep `green2d`, works at finite r):

- Time convention e^{-i omega t}; outgoing wave = Hankel H0^(1)(kR),
  R = |r - r'|; meep units eps0 = mu0 = 1, vacuum k = omega = 2 pi f.
- TM (Ez, Hx, Hy): currents Jz = n_x Hy - n_y Hx (electric, z-directed) and
  (Mx, My) = (-n_y Ez, n_x Ez)... i.e. M = -n_hat x (Ez z_hat). Far fields
  (Ez, Hx, Hy) from:
    Ez(r) += (i k / 4) * [ -Jz * H0^(1)(kR)
                           + (Mx * R_hat_y - My * R_hat_x) * H1^(1)(kR) * (1/ ?)]
  Exact prefactors are pinned by the T2 analytic gate (line-source
  self-test), not by this spec: the implementation must reproduce the
  analytic cylindrical wave of a z line current, Ez = -(omega/4) H0^(1)(kR),
  H = curl E / (i omega), to <1e-3 rel at kR >~ 50.
- TE (Hz, Ex, Ey): exact dual (E <-> H, eps <-> mu, J <-> -M). One
  implementation parameterized by polarization; TE reuses the TM kernels with
  swapped roles and signs.
- Vacuum-only far region (background eps=1), 2D in-plane points. kz = 0.

## In scope

1. **`core/near2far.py`**: `green2d_tm(J, M, pts_src, pts_obs, k)` exact
   Hankel kernels (scipy.special.hankel1); `Near2FarMonitor` — 4 box faces,
   each DFT-accumulating the tangential fields FluxMonitor-style (x-normal
   face: Ez+Hy [TM] / Ey+Hz [TE]; y-normal: Ez+Hx / Ex+Hz); `farfield(...)`
   contour sum -> complex (E, H) 3-vectors per far point per freq.
2. **`mp` shim**: `Near2FarRegion(center, size, weight=+-1)`,
   `add_near2far(fcen, df, nfreq, *regions)` -> handle,
   `get_farfield(handle, Vector3)` -> 6 complex [Ex,Ey,Ez,Hx,Hy,Hz] (meep
   layout). Box helper: 4 regions = closed contour, outward normals from
   weights, meep convention.
3. **Gate example port**: `examples/antenna-radiation.py` — Ez and Ey point
   dipoles in vacuum, radial Poynting pattern over 2 pi, plotted vs analytic.

## Out of scope

- 3D and cylindrical near2far; periodic near2far (`binary_grating_n2f`'s
  `nperiods` tiling — ships as a follow-on once the flat transform lands, or
  W4.5); far fields inside non-vacuum backgrounds; `save_farfields` HDF5.

## Gate (master spec)

`antenna-radiation`: (a) Ez (2D line) dipole pattern isotropic to <=2% rms
over 100 polar angles at r = 1000 lambda; (b) Ey dipole pattern matches
cos^2-type analytic lobe to <=3% rms (normalized); (c) both patterns match
meep-oracle JSON (`mamba run -n meep-oracle`, reduced resolution 25) to <=5%
rms normalized. Reference: `tests/examples/data/antenna_radiation_n2f.json`;
tests never import meep.

## Tests / acceptance

- `tests/examples/test_w4_green2d.py` — analytic line-source self-test: fields
  on a synthetic contour from the exact H0^(1) wave of a z line current ->
  farfield() reproduces the same analytic wave at r=500..2000 (rel <1e-3);
  reciprocity: translating the contour does not change the far pattern.
- `tests/examples/test_w4_antenna_radiation.py` — FDTD Ez dipole -> gate (a)
  + (c); FDTD Ey dipole -> gate (b) + (c).
- Suite addition <10 min (reduced resolution). MIT, <=500 lines/file, TDD,
  commit per task.

## Open decisions (decided here)

- Exact Green (Hankel) evaluation, not asymptotic: matches meep, one code
  path for any r. numpy/scipy eval (no autograd through far fields in W4).
- Contour = axis-aligned box of 4 line faces on grid lines (same index
  conventions as FluxMonitorTM). Corners: half-weight not needed at gate
  accuracy; revisit only if the T2 self-test misses 1e-3.
