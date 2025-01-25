# W8 — adjoint / inverse design (spec)

Date: 2026-08-08
Workstream: W8 (master spec). Differentiable FDTD objective -> gradient wrt
a material parameterization -> Adam update loop. Since meep-torch IS torch,
"adjoint" needs no hand-derived reverse code: `build_tm_coeffs` ->
`run_tm(checkpoint_every=...)` -> scalar objective is one autograd graph, and
`torch.utils.checkpoint` bounds the memory of the backward pass exactly as
the Phase-2 [fdtd-autodiff-memory-strategies](../concepts/fdtd-autodiff-memory-strategies.md)
concept prescribes for the checkpointed variant.

## Physics (pinned)

- **Adjoint gradient = autograd backward through the forward time loop.**
  The chain is `theta -> eps_inf(theta)` (dense tensor, `requires_grad`) ->
  `build_tm_coeffs` -> `run_tm(state, coeffs, n_steps, checkpoint_every=N)`
  -> monitor accumulation -> scalar J. Every step is a differentiable torch
  op (the source amplitude depends on time only, never on theta), so
  `J.backward()` yields dJ/dtheta directly.
- **Memory bound.** With `checkpoint_every=N` the time loop is chunked; each
  chunk's outputs (field state + monitor partials) are O(1)-sized relative
  to the chunk, and backward recomputes chunk interiors. Full-field
  history is never stored — the same bounded-memory contract the existing
  `run_tm` already honors (see `tests/autodiff/test_memory.py`).
- **Objective.** Transmitted flux/power through a monitor on the far side of
  a scatterer, maximized by moving material. Two valid forms:
  (i) `|DFT(probe region)|^2` at f0 from a `DFTFieldsMonitor` acc
  (Re/Im phasors, `e^{-iwt}` convention — `J = Re^2 + Im^2` summed);
  (ii) `FluxMonitorTM.flux(acc, dx)` summed over freqs. Both accs carry grad
  (`reduce="sum"`).
- **Parameterization.** Material control points -> smooth field, so
  gradients are dense in theta and Adam can take material-sized steps:
  bilinear-upsampled control grid times a support mask, `eps = 1 + alpha*S`.
  No projection/thresholding in W8 (SSP/ClosestIndex is Phase 5 scope;
  W8 is the gradient + loop gate).

## In scope

1. `docs/specs|plans/2026-08-08-meep-torch-w8-adjoint.{md,md}` (this pair).
2. `tests/examples/test_w8_adjoint.py` — the gradient gate (below).
3. `examples/adjoint_gradient.py` — tiny 2D lens/blob inverse design:
   control-grid eps parameterization, 5 Adam steps improving transmitted
   flux, objective printed per step, final eps+field png.
4. No new `meep_torch/` core code: the adjoint path already exists
   (Phase 1 checkpointed run + Phase 4 ports). W8 proves it end-to-end and
   locks the pattern with a test + example. (Full inverse-design
   fabrication filters/SSP land as Phase 5.)

## Gates

- (a) **Autograd-vs-FD, DFT objective**: parameterize a blob eps via a
  scalar theta; J = |DFT probe|^2 at f0 (DFTFieldsMonitor through
  `checkpoint_every=200`); `dJ/dtheta` from autograd matches central
  finite difference to **<=1e-3 relative**.
- (b) **Grad through FluxMonitorTM**: same chain but J2 = flux through a
  far-side x-normal `FluxMonitorTM` (summed over freqs); autograd vs
  central FD **<=1e-3 relative**.
- (c) **Loop moves flux up**: `examples/adjoint_gradient.py` prints an
  increasing transmitted-flux objective over >=5 Adam steps (demo gate,
  asserted loosely in the example itself).

## Tests / acceptance

- `tests/examples/test_w8_adjoint.py` (a)+(b). MIT, <=500 lines/file, TDD.
- Suite addition <2 min (tiny grids, ~200 steps).
- Example saves `examples/adjoint-gradient.png`, runtime <60s.
