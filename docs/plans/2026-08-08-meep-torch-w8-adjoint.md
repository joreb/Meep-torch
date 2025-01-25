# W8 — adjoint / inverse design (plan)

Date: 2026-08-08
Spec: `docs/specs/2026-08-08-meep-torch-w8-adjoint.md`.
TDD, commit per task, <=500 lines/file, MIT headers. Numeric verification only.
No new `meep_torch/` core code — the differentiable run path already exists;
W8 is a test + example + docs gate.

## Task count: 3

- **T1** `docs/specs/2026-08-08-meep-torch-w8-adjoint.md` +
  `docs/plans/2026-08-08-meep-torch-w8-adjoint.md` (this pair, house style
  per w6 docs).
- **T2** `tests/examples/test_w8_adjoint.py`:
  - helper `_run_blob(theta, monitor)` — theta-scaled smooth-disk eps on a
    small grid, LineSource GaussianPulse, `run_tm(checkpoint_every=200)`,
    returns accs. theta requires_grad, source fixed (time-only amplitude).
  - gate (a): `J = |DFT probe|^2` (DFTFieldsMonitor Re/Im at f0, summed over
    a shadow-region slice); autograd dJ/dtheta vs central FD `<=1e-3` rel.
  - gate (b): `J2 = FluxMonitorTM.flux(acc, dx).sum()` on a far-side
    x-normal line; autograd vs central FD `<=1e-3` rel.
  - float64 (grid default), FD step ~1e-4, small grids so both gates run in
    well under a minute.
- **T3** `examples/adjoint_gradient.py`:
  - 2D TM, PML, left line source, mid-cell scatterer parameterized by a
    coarse control grid (e.g. 4x4) bilinearly upsampled to eps, `alpha`
    via `torch.nn.Parameter`.
  - objective = transmitted `FluxMonitorTM` flux at f0; 5 Adam steps,
    `print` objective per step (must rise); final `adjoint-gradient.png`
    (eps field + |Ez| snapshot). Runtime <60s, no meep import.

## Standing rules

- Reference/regression gate for a *gradient* is central finite difference
  (no meep oracle needed — FD is the ground truth here).
- `status:` to peer before the git commit (shared working tree); commit
  ONLY docs/T2/T3 + png; `done:` with hash + gate numbers.
