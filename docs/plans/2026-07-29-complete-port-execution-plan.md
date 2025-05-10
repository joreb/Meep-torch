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
   **W5.5 (wave 2) DONE 2026-08-08** except multilevel: magnetic Lorentz
   poles (mu-slab R/T vs transfer matrix, R 1.6%), gyrotropic 1D vector
   stepper + faraday-rotation port (kappa 0.4% vs analytic, oracle profiles
   rms 3.0%), RandomSource noise (+-3 dB transfer-flatness gate).
   stochastic_emitter sources unblocked (port deferred, ensemble runtime).
   Multilevel-atom ADE deferred to W5.75 (in-session budget; spec allowed).
   **W5.75 DONE 2026-08-08**: two-level gain in stepper_1d (inversion-scaled
   Lorentz ADE + rate equations, per-end npml for the PEC mirror);
   multilevel-atom ported, lasing peak f=6.44 within 5% of the meep-oracle
   6.412 (res 100); below-threshold decay gate. >2 levels + mp shim deferred
   until an example needs them.
6. **W6 — LDOS + forces.** Unblocks ~5. Gate: `metal-cavity-ldos` Purcell
   vs meep JSON; `parallel-wvgs-force` MST force sign+magnitude.
   **DONE 2026-08-08**: core/ldos.py (vacuum LDOS linear-in-f <=3%,
   translation-invariant <=2%), shim dft_ldos/get_ldos, metal-cavity-ldos
   ported (sigma_d=1e4 PEC substitute; LDOS peak == harminv f <=2%, LDOS
   Q vs harminv <=15%, f vs PEC oracle 4.2% -- half-cell wall-position
   offset documented). core/forces.py mst_force_tm (radiation pressure
   (1+R) S_inc within 5% analytic, meep add_force pressure-ratio within
   10%). Fixed W5 painter bug (cond_items skipped default-sigma media).
   parallel-wvgs-force re-scoped: blocked on EigenModeSource in
   Simulation.run, not MST.
7. **W7 — cylindrical engine.** Unblocks ~10. Largest; its spec may split.
   Gate: `ring-cyl` resonances vs meep JSON.
   **Wave 1 DONE 2026-08-08**: radial BOR stepper (core/stepper_cyl.py,
   complex F(r) e^{i m phi}, Ez set, axis rows pinned for regularity +
   the m/r leapfrog stability limit, outer sigma absorber). ring-cyl
   ported: m=3 f=0.11821 Q=80, m=4 f=0.14746 Q=336 vs meep-oracle to
   <0.01% (edge-averaged node eps reproduces meep subpixel exactly);
   m=5 Q=1592 whispering-gallery ladder.
   **W7.5 DONE 2026-08-08** (5b0e5f3): (r,z) 2D BOR stepper (all six
   components, m=0 ampere-loop axis + m!=0 stability pinning, sigma
   absorber). Gates: m=0 dipole LDOS ~ f^2 (true 3D, <=4%); disc
   resonator m=3 f within 1.5% of oracle (isotropic vs anisotropic
   subpixel brackets it), Q within 25%. Still open (W7.75): cylindrical
   near2far/CPML/eigenmode ports + the ~9 example ports.
8. **W8 — inverse design** (adjoint examples; eigengrad completion lands
   here). Gate: one adjoint_optimization example reproduced via torch autodiff.
   **DONE 2026-08-08** (5673129, by the delegated session): autograd
   through run_tm (checkpoint_every) -- dJ/dtheta vs central FD rel err
   7.9e-9 (DFT objective) / 2.7e-9 (flux objective), gate 1e-3;
   adjoint_gradient.py demo: 5 Adam steps, flux x11.9. Also fixed the
   pre-existing checkpoint-equivalence break (a534172).
9. **W9 — 3D engine.** Last. Everything remaining.
   **Wave 1 DONE 2026-08-08** (0e537ca): full 3D Yee stepper (six real
   components, sigma absorber shells, dt = dx/(2 sqrt 3)); vacuum dipole
   LDOS ~ f^2 within 5%, agreeing with the independent W7.5 BOR result.
   **Wave 2 DONE 2026-08-09** (649d63f, delegated session): 3D CPML
   (mixed-mode axes), D/B-conductivity fast-path, monitors_3d.py; gates
   lossy-slab |T| <=3%, CPML decay 0.73x the absorber's, residual
   2.5e-14.
   **Wave 3 DONE 2026-08-09** (ceb4124): 3D shim path (vacuum
   |T-1|=1.7e-3, real-eps slab 0.94%). **Wave 4 DONE** (1a498e3,
   takeover): absorbed_power_density 3D port (D_conductivity dissipation
   = sigma_d*eps*|E|^2, energy balance 2.6%). kz-aware CPML landed
   (1a498e3): Fresnel p 0-80 deg max err 0.0030. Morning-2 threads
   closed (925fd80..ce235cf): per-m coherent sums, closed-surface cyl
   flux, fd1d supermode launch + parallel-wvgs-force, DiffractedPlanewave
   complex-mask launch, reciprocity 2.4%. Census 46 -> 19; remaining
   stubs: MPB pointer-stubs resolved, rest are 3D-heavy examples and
   pointer variants.
   ALSO CLOSED 2026-08-08: W2 deferrals -- TE Bloch stepper (85833b2,
   Fresnel p-pol <=0.02; diffracted_planewave re-scoped to
   DiffractedPlanewave order launch) and kz_2d 1D reduction (8087500,
   Fresnel p-pol <=0.02; refl-angular-kz2d core unblocked); W4.5
   periodic near2far tiling (cbfd8aa, grating-order lobe gate).

MPB-blocked examples stay stubbed until the Q1 track is picked up separately.
**MPB track waves 1-2 DONE 2026-08-09** (c6698e4..0a97927): bands via the
Bloch+harminv FDTD route (no PWE solver). Wave 1: holey-wvg-bands ported
(TE Bloch axes-swapped, guided bands within 2.5% of the meep oracle;
supersampled coverage eps == meep subpixel again, kx=0.4 band exact;
zone-edge modes excluded, scheme-sensitive at res 10; i k_y coupling now
midpoint-averaged). Wave 2: stepper_bloch2 (doubly periodic TM) --
square-lattice rod bands and 1D sine-Bragg bands within 2% of meep.mpb;
mpb_bragg_sine ported. Remaining mpb stubs re-scoped: oblique lattice
basis (tri/honeycomb), 3D slab bands, strip effective-index, supercell
defects, mode-profile tooling.
**MPB track COMPLETE 2026-08-09** (64eefc4..049059f): all remaining mpb
stubs closed. Oblique lattices via the rect sqrt(3) x 1 supercell (tri
Gamma/M/K within 2%, honeycomb band 1 within 2%, degenerate-pair capture
noted); line defect 7-row supercell (no-spurious containment gate);
strip via fd2d cross-section (neff within 2% after supersampling);
hole slab via NEW stepper_bloch3 (complex 3D, periodic x/y, absorber z,
midplane parity selection -- z-even K band within 2.5% of meep.mpb).
6 example ports (tri/honey/strip/line_defect/data_analysis/hole_slab).
Stub census 34 -> 25.

**Example-port pass DONE 2026-08-08** (ecaad6d..6888a03): 7 ports --
refl-angular-kz2d (Fresnel p max err 0.0039, 0-65 deg, grazing needs
kz-aware CPML), dipole_in_vacuum_cyl_on_axis (circular Er+iEphi source,
|m|=1 axis updates in stepper_bor; 12.8% rms at res 25,
source-discretization limited), point_dipole_cyl (m=0 dipole-in-slab
relative extraction), cavity-farfield (far peak centered on the cavity),
disc_radiation_pattern (m=1), binary_grating_n2f (Bloch unit cell +
tile_faces, lobes exactly at 0/+-30 deg orders), stochastic_emitter
(incoherent ensemble). W7.75/W7.9 cyl near2far (84d4bc1, 4979e4b) carried
the pass. Stub census 46 -> 34 blocked. Remaining stubs: MPB (~15), 3D
examples (W9 waves 3+), off-axis per-m sum, reciprocity variants,
DiffractedPlanewave order launch, eigenmode-source-in-run, cyl flux
normalization.

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
