# Complete-port execution plan

> Master spec: `docs/specs/2026-07-29-meep-torch-complete-port-master-spec.md`.
> Ground truth for gaps: commit `4b9fac0` — 85 examples processed, 34 ported
> runnable, 51 stubbed with the blocking feature named in each stub docstring.
> Executor: inter-session agent (`meep-photonics`), one workstream at a time,
> each with its own spec → plan → TDD implementation cycle. Orchestrator
> (`meep-torch-phase4`) reviews per wave via code-review-graph and gates
> transitions. Forward-only autodiff. MIT-only wheel; no meep (GPL2) code.

## Scoreboard (from 4b9fac0 stub census)

| Blocker | Stubbed examples | Workstream |
|---|---|---|
| MPB band solver | ~15 | out of scope here (wiki Q1 track) |
| cylindrical coords | ~10 | W7 |
| bloch/oblique planewave | ~6 | W2 |
| near2far/farfield | ~6 | W4 |
| LDOS / absorbed power | ~4 | W6 |
| adjoint | 3 | W8 |
| stochastic noise sources | 3 | W5 |
| 3D | remainder | W9 |
| multilevel atom, gyrotropic, aniso tensor, complex fields, Absorber, force | 1 each | W5/W6 |

## Order and gates

1. **W1 — meep-shaped API.** First (wiki Phase 3 recommendation): the 34
   ported examples currently use raw meep-torch calls; the shim makes the
   remaining 51 land near-verbatim as each blocker falls, and opens gplugins.
   Gate: ≥5 of the 34 ported examples rewritten to run via `import meep_torch.mp
   as mp` with unchanged meep syntax, outputs matching their raw-port twins.
2. **W2 — sources completion** (bloch/oblique, custom/CW profiles,
   `stop_when_fields_decayed`). Unblocks ~6 stubs. Gate: `oblique-planewave`
   + `refl-angular` ports pass vs meep-oracle reference JSON.
   **DONE 2026-07-30** (commits 9513219..51b39d4): all 3 gates pass (angle
   <1 deg, Fresnel <=2%, binary_grating_oblique orders vs meep-oracle JSON).
   Delivered: custom/CW/spatial/oblique sources, two-quadrature Bloch TM
   stepper (backward-curl Yee fix), run_until_decayed + smatrix decay_probe,
   ports/orders.py diffraction decomposer. 4/6 stubs ported; 2 re-scoped to
   accurate blockers: `diffracted_planewave` → **TE Bloch stepper** (add to
   W5 or a W2.5), `refl-angular-kz2d` → **2.5D kz_2d** (new small workstream,
   schedule with W5).
3. **W3 — frequency-domain analysis** (true harminv replacement:
   filter-diagonalization/ESPRIT; area DFT fields; keep `solve-cw` port
   honest). Upgrades the FFT substitutions in ported cavity examples.
   Gate: `holey-wvg-cavity` Q within 5% of meep-oracle harminv.
   **DONE 2026-08-08** (commits 762b5fd..92a6858): gate pass (Q=65.26 vs
   oracle 63.98, 2%). Delivered: `modes/harminv.py` (FDM/matrix-pencil,
   near-undamped modes kept as Q=inf for driven-CW series), ring/holey/
   solve-cw examples on FDM harminv (solve-cw f=0.11801 vs drive 0.118).
   solve-cw sparse-direct CW solver not built (time-domain + FDM ships).
4. **W4 — near2far (2D).** Unblocks ~6. Gate: `antenna-radiation` dipole
   pattern vs analytic + meep JSON.
   **DONE 2026-08-08** (commits d2da1c7..06ff78d + example): all gates pass
   (Ez isotropy rms 1.3% <= 2%, Ey cos^2 rms 2.4% <= 3%, both vs meep-oracle
   <= 5%). Delivered: `core/near2far.py` (exact Hankel-H^(2) kernels, TM+TE
   surface equivalence, trapezoid-closed box contour), FluxMonitorTEY,
   `mp` add_near2far/get_farfield/Near2FarRegion, antenna-radiation port.
   Unblocked on the way: 2D TE Simulation path (Ey source mapping, mask
   trim, stale y_pmc kwarg). Periodic near2far (binary_grating_n2f
   nperiods) deferred to a W4.5/follow-on.
5. **W5 — materials completion** (conductivity, mu, aniso eps tensor in
   stepper, gyrotropic, multilevel, noise sources, Absorber boundary,
   complex-field mode). Sub-waved; unblocks ~7.
   **Wave-1 scope DONE 2026-08-08** (commits 9672278..T6): D/B-conductivity
   + scalar mu in TM+TE steppers (analytic slab <2%, oracle slab <3% rms),
   mp.Absorber + Medium(D_conductivity/B_conductivity/mu) shim,
   absorber-1d ported (residuals within 2x of oracle floors; Al -> sigma_d
   substitute, Drude w0=0 unrepresentable), in-plane TE eps tensor
   (rotated-uniaxial n_eff <1%). Deferred to W5.5: gyrotropic, multilevel
   atom, noise sources, magnetic Lorentz poles, complex-field mode.
   polarization_grating re-scoped: needs full-vector 2D stepper (ez-coupled
   tensor), not just in-plane anisotropy.
6. **W6 — LDOS + forces.** Unblocks ~5. Gate: `metal-cavity-ldos` Purcell
   vs meep JSON; `parallel-wvgs-force` MST force sign+magnitude.
7. **W7 — cylindrical engine.** Unblocks ~10. Largest; its spec may split.
   Gate: `ring-cyl` resonances vs meep JSON.
8. **W8 — inverse design** (adjoint examples; eigengrad completion lands
   here). Gate: one adjoint_optimization example reproduced via torch autodiff.
9. **W9 — 3D engine.** Last. Everything remaining.

MPB-blocked examples stay stubbed until the Q1 track is picked up separately.

## Standing rules for the executor

- Per workstream: write `docs/specs/…-wX-….md` + `docs/plans/…-wX-….md` first,
  `status:` with task count, then implement in waves, `status:` per wave,
  `done:` with gate numbers. `question:` only for decisions the master spec
  and wiki left open.
- Convert the workstream's stub examples to real ports in the same workstream
  (they are the acceptance tests), plus physics-gated `tests/examples/` tests.
- Reference data via `mamba run -n meep-oracle` into committed JSON.
- Numeric verification only (executor model has no image input); orchestrator
  eyeballs PNGs.
- Suite runtime budget: +10 min per workstream max; example scripts excluded
  from pytest (runnable standalone).
