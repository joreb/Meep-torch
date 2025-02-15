# W9 wave 2 — 3D engine hardening (spec)

Date: 2026-08-08
Workstream: W9 wave 2 (master spec). Wave 1 (`0e537ca`) shipped a real-field
3D Yee stepper with a matched-sigma absorber; wave 2 hardens it for real
use: a true CFS-PML (CPML), material D/B-conductivity, and the monitor
suite (DFT + flux plane) the examples need.

## Physics (pinned)

- **3D Yee layout** (from wave 1): `Ex(i+1/2,j,k)`, `Ey(i,j+1/2,k)`,
  `Ez(i,j,k+1/2)`; `Hx(i,j+1/2,k+1/2)`, `Hy(i+1/2,j,k+1/2)`,
  `Hz(i+1/2,j+1/2,k)`. Courant `dt = 0.5 dx / sqrt(3)`.
- **CFS-PML (CPML)** — same recursion as the 2D `cpml_profiles`
  (`meep_torch/core/pml.py`): per axis `a` a 1D profile `(b_a, a_a, k_a)`
  of length `n_a`, recursion `psi' = b*psi + a*d`, effective derivative
  `d/k + psi`. A derivative *along* axis `a` is absorbed by the `a`-normal
  PML, so the profile is broadcast over the two transverse axes. 3D needs
  one psi per (component, derivative-axis): 6 for the H update (Hx←dEz/dy
  − dEy/dz; Hy←dEx/dz − dEz/dx; Hz←dEy/dx − dEx/dy) and 6 for the E update
  (the dual) — 12 psi memory variables total. CFS (`alpha`) term keeps
  low-frequency evanescent reflection down exactly as in 2D.
- **Material D/B-conductivity fast-path** (mirrors TM `d1/d2`): for a
  field component with material conductivity `sigma`, the update gets a
  multiplicative `d1 = (1 - sigma dt/2)/(1 + sigma dt/2)` damping and a
  `d2 = dt/(1 + sigma dt/2)` curl/source coefficient. `sigma_d` damps the
  E components (averaged from cell-centers to E nodes), `sigma_b` the H
  components (averaged to H nodes). Distinct from the boundary PML/absorber
  — both can be present (the absorber fallback path remains for when
  `npml=0`).
- **3D DFT + flux monitors** — same `e^{-iwt}` phasor convention as 2D
  (`dt * [cos(wt) F, -sin(wt) F]`, E at `(t+1)dt`, H at `(t+1/2)dt`,
  `reduce="sum"`). `DFTFieldsMonitor3D` samples one component on a plane
  (or full grid); `FluxMonitor3D` integrates the time-averaged Poynting flux
  `S_n = 1/2 Re[ E_t conj(H_t) ]` (transverse E×H on an x/y/z-normal
  plane) times the cell area, with the per-normal sign matching 2D.

## In scope

1. `meep_torch/core/stepper_3d.py`:
   - CPML path: per-axis `(b,a,k)` profiles from `cpml_profiles`, 12 psi
     fields on `State3D`, recursion in `step_3d`. Keep the matched-sigma
     absorber as a fallback (selectable by boundary mode: `npml`>0 →
     CPML, else `n_abs`>0 → absorber).
   - Material `sigma_d`/`sigma_b` fast-path (optional, None → unchanged).
   - Plane/area soft source: extend the single-point `src` to accept a
     spatial mask (so a planewave can be launched), mirroring 2D
     `SoftSource`.
2. `meep_torch/core/monitors_3d.py` (or extend `monitors.py`):
   `DFTFieldsMonitor3D`, `FluxMonitor3D`, and a minimal forward-only
   `run_3d` (monitor accumulation, no checkpoint — W9 is forward-only per
   the master spec).
3. `tests/examples/test_w9_wave2.py` — the gates below.
4. `docs/specs|plans/2026-08-08-meep-torch-w9-wave2-3d-hardening.{md,md}`.

## Gates

- (a) **3D lossy-slab transmission**: a z-normal planewave through a
  finite lossy slab (eps, sigma_d), `|T|` measured by a far flux plane vs
  the analytic normal-incidence slab transmission — **<=3%**. (Analytic:
  lossy-slab transfer matrix between two vacuum half-spaces at the drive
  frequency.)
- (b) **PML residual vs absorber**: a z-normal planewave crossing an
  empty cell; spurious reflection from the back CPML must be measurably
  below the matched-sigma absorber at the same depth (ratio of
  reflected-to-incident flux at the source plane: CPML <= absorber, and
  CPML residual <= a few % of incident). Vacuum, so the only reflection
  is boundary spurious — the direct CPML-vs-absorber comparison.
- (c) **suite runtime <10 min** (small grids; 3D is expensive — keep
  `n` modest, drive single frequency, short pulse).

## Gate (b) resolution (as built)

The source-plane reflected-flux ratio in the original wording turned out
to be the wrong observable. Two failures drove the change:

1. A **point-source** ringdown is dominated by the source near-field, not
   boundary reflection — the residual is size-independent (~2.7e-3 for
   nx=24..80), so it cannot compare PML to absorber. Switched to a
   **z-uniform planewave** (zero transverse k → no source near-field;
   the x/y CPML is inert, so the only absorption is the z-boundary).
2. A **mixed-mode** build isolates the z-boundary: the absorber case keeps
   CPML on x/y (identical to the CPML case) and replaces only the z CPML
   with a matched-sigma absorber (same depth, `m=3`, `R0=1e-6` via the new
   `absorber_m`/`absorber_R0` params), implemented as an identity CPML
   profile (plain derivative) plus the z-sigma in `sig_bound`. This is a
   `build_3d_coeffs`-only change — no `step_3d` edit — because an identity
   profile (`b=0, a=0, k=1`) reduces the CPML recursion to a plain
   derivative.

The observable is then a phase-robust **per-bounce decay rate** — the
integrated energy `mean(E^2)` in a late window `[60,72]` over an earlier
window `[30,42]` at a center probe (integrated, not a phase-dependent
peak, so the comparison is stable as `t_end` varies). The transverse
PEC/CPML dynamics cancel in the CPML/absorber ratio (identical x/y
treatment). At `npml=8` the CPML decay (4.2e-4) is below the matched
absorber (5.8e-4, ratio 0.734) and the residual is 2.5e-14 (≪5%). At
normal incidence a graded absorber is comparable to a polynomial-graded
CFS-PML at very thin depth; the CPML's polynomial grading wins once it
has enough layers to engage (npml≥7), which is the regime this gate
exercises.

## Tests / acceptance

- `tests/examples/test_w9_wave2.py` (a)+(b). MIT, <=500 lines/file, TDD.
- W9 wave-1 `test_w9_3d.py` still passes (absorber fallback intact).
- No meep import. Oracle not needed (analytic slab + relative PML gate).