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
3. **W3 — frequency-domain analysis** (true harminv replacement:
   filter-diagonalization/ESPRIT; area DFT fields; keep `solve-cw` port
   honest). Upgrades the FFT substitutions in ported cavity examples.
   Gate: `holey-wvg-cavity` Q within 5% of meep-oracle harminv.
4. **W4 — near2far (2D).** Unblocks ~6. Gate: `antenna-radiation` dipole
   pattern vs analytic + meep JSON.
5. **W5 — materials completion** (conductivity, mu, aniso eps tensor in
   stepper, gyrotropic, multilevel, noise sources, Absorber boundary,
   complex-field mode). Sub-waved; unblocks ~7.
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
