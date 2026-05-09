# Tier-2: full-3D inverse design (checkpointed 3D adjoint) — plan

Status: approved (user "build tier 2"), implementing
Driver: plan-view-patterned 3D devices (the npj AAFGC class) need
gradients through the 3D stepper; 2.5D design + 3D validation cannot
shape lateral structure.

## Why not the existing paths

- `stepper_3d_lean`: forward-only by contract (in-place, no_grad).
- `run_tm`-style `torch.utils.checkpoint` on the functional `step_3d`:
  the known boundary-state accumulation makes it O(n_chunks x state) on
  the GPU (agent memory: checkpoint-boundary-state-accumulation) — the
  31 GB wall that killed RES=40 3D adjoints.

## Design: manual two-phase checkpointed adjoint, boundary states on CPU

`meep_torch/core/adjoint3d.py`, `run_3d_adjoint(c, n_steps, chunk,
source, monitors, loss_fn)`:

1. Forward under no_grad: step `step_3d`, save the FULL State3D (6
   fields + 12 psi) at each chunk boundary to CPU pinned memory;
   accumulate monitor DFTs as plain numbers.
2. Loss head: re-leaf the accumulators, `loss_fn(accs)`, autograd.grad
   -> per-monitor cotangents g_acc (tiny).
3. Reverse sweep, last chunk first: restore the chunk's start state as
   leaves, re-run the chunk WITH graph (monitor samples included),
   then one `torch.autograd.backward` seeded with (state cotangent
   from the next-younger chunk, g_acc). eps gradients accumulate on
   the eps leaf through `Coeffs3D.eps` (step_3d averages eps in-step,
   so the graph reaches eps without any coeffs rebuild). The start
   state's .grad becomes the next cotangent.

GPU peak = one chunk's graph; CPU holds n_chunks x state. Choose chunk
~ sqrt balance; for the demo scene (~3M cells) chunk 16-32.

## Gates

1. FD gradient: small grid (npml=6), eps voxels central-difference vs
   adjoint, rel < 1e-4 at float64, SINGLE-application objective (the
   even-power sign-blindness lesson).
2. Parity of the forward pass against a plain no_grad step_3d loop
   (bitwise -- same code path, same order).
3. Demo: plan-view 3D grating design (film-layer 2D pattern extruded
   through thickness, the litho-faithful parameterization), objective =
   DFT power in a CE-plane window; a few Adam iterations must increase
   it monotonically-ish from the analytic seed.

## Out of scope

Metals, multimode (TE10) objectives, revolve-style optimal
checkpointing (linear CPU schedule is enough at demo scale).
