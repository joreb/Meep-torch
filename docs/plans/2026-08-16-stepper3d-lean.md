# Memory-lean forward 3D stepper (plan)

**Trigger**: mmi_tapered-3D validation (212M cells, RES=40) cannot fit
16 GB VRAM; even splitter-3D (60M cells, f32) peaked 14.4 GB. Forward
runs cannot use autograd checkpointing -- the memory is structural.

**Measured memory anatomy of step_3d (functional, full-grid)**
- 18 state arrays double-buffered (old+new alive) ............ 2x state
- 12 CPML psi arrays stored FULL-GRID (nonzero only in shells)  ~45% of state
- 6 damp pairs materialized full-grid even when sigma == 0 .... ~10 GB @212M
- per-step temporaries (curls, eps averages, ce/ch) .......... ~2.5x static
  (peer-measured: static array count under-predicts peak by ~2.5x)

**Design: `meep_torch/core/stepper_3d_lean.py` (forward-only)**
1. `build_3d_coeffs_lean(...)`: CPML-only; asserts sigma_d = sigma_b = 0
   and stores damp as SCALARS (d1=1, d2=dt); precomputes
   `inv_eps_x/y/z` once (E-update becomes mul, no div, no per-step
   `_avg`); CPML kept as 1-D (b, a, k) vectors + shell slice objects
   (lo: [0, npml), hi: [n-npml, n)).
2. `init_3d_state_lean(...)`: 6 full-grid fields + 12 psi arrays stored
   as SHELL PAIRS only (each psi: lo+hi slabs of thickness npml along
   its CPML axis). Shell update:
       psi_lo.mul_(b_lo).add_(a_lo * D[shell_lo])
       D[shell_lo].div_(k_lo).add_(psi_lo)        # D becomes ce in place
   Outside shells k == 1, so the raw derivative IS the curl term.
3. `step_3d_lean(state, c, src_amp, src)`: fully in-place under
   torch.no_grad(); per-shape scratch pool (buffers reused across the
   6 component updates -- each H/E component's two curl terms share its
   shape). Update algebra unchanged:
       H.mul_(d1).sub_(d2 * (ce_a - ce_b))
       E.mul_(d1).add_(curl.mul_(inv_eps).mul_(d2))   # == (d1 E eps + d2 curl)/eps
   Soft-current injection: E[mask] += dt * amp * mask * inv_eps
   (same semantics as _inject_src).
4. `run_3d_lean(...)`: plain loop; monitors sample the live state (all
   3D monitors compute fresh tensors in sample(), no view-pinning).

**Memory budget after (mmi-3D, 212M cells, f32)**
fields 5.1 GB + inv_eps 2.5 GB + psi shells ~1.5 GB + scratch ~2.5 GB
+ profiles/mask ~0.3 GB  =>  ~12 GB peak. Fits.

**Gates**
- Parity: small grid (60^3, npml=10), pulsed point source, 500 steps:
  max |Ez_lean - Ez_ref| / max|Ez_ref| < 1e-12 (f64, CPU) and < 1e-5
  (f32, GPU vs f64 CPU reference).
- Memory: torch.cuda.max_memory_allocated on splitter-3D geometry
  <= 0.5x the functional stepper's peak.
- Then: run mmi_tapered-3D at RES=40 f32 on GPU.

**Out of scope**: autograd (lean is forward-only by contract; adjoint
3D keeps the functional stepper), absorber/mixed boundary modes,
dispersive media (assert against).
