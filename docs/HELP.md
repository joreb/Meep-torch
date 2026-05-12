# meep-torch — user help

Differentiable FDTD/FDFD photonics engine in PyTorch. This page is the
practical API map: what to call for each kind of job, with the
non-obvious rules that cost us debugging time already learned.

## Which engine for which job

| Job | Use | Why |
|---|---|---|
| Broadband spectra, transients, videos | FDTD (`run_tm`, 3D steppers) | one pulsed run = whole spectrum |
| Steady state / high-Q response at a frequency | FDFD (`solve_fdfd_tm`) | no charging wait; a resonance is one solve |
| Inverse design (2D) | FDFD adjoint (`backend="pardiso"`) | exact gradients, ~1 s/iteration |
| Undepleted chi2 (SHG) | two chained FDFD solves | steady state exact, minutes |
| Kerr (chi3) steady state | fixed-point FDFD loop | see `examples_2/nonlinear/fdfd_kerr_ring.py` |
| True-3D validation | `stepper_3d_lean` (forward-only) | in-place + pooled scratch; 139M cells on one GPU |

## 2D FDTD (TM)

```python
from meep_torch.core.grid import Grid
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.run import run_tm
from meep_torch.core.stepper import init_tm_state
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port
from meep_torch.ports.mode_monitor import decompose
```
House conventions (deviations have all been bugs before):
`GaussianPulse(fcen, 0.2)`; `T_RUN = 2*(5/0.2) + nx*dx*1.7 + margin`;
`decompose(..., axis_sign=-1.0)`; ports need `SPAN` beyond the core
half-width (a window inside the core has no guided mode); keep ports
and sources out of the PML (`validate_ports`). Differentiable via
checkpointing (`checkpoint_every=64..128` with gradients, `400` without).

## 2D FDFD (TM)

```python
from meep_torch.fdfd import (solve_fdfd_tm, make_mode_source,
                             mode_amplitude, port_mode_line,
                             solve_fdfd_tm_real_backend)
Ez = solve_fdfd_tm(eps, dx, freq, jz, npml, backend="scipy"|"pardiso"|"torch-sla")
```
- Differentiable w.r.t. `eps` AND `jz` (chained nonlinear pipelines).
  Every gradient is finite-difference gated; keep it that way.
- `backend="pardiso"` (MKL, multithreaded): ~12x scipy. Default scipy.
- `solve_fdfd_tm_real_backend(..., backend="cudss"|"strumpack", device=...)`:
  FORWARD-ONLY complex solves through GPU direct solvers via the 2N
  real embedding. cuDSS is the working GPU backend (~8M-real-unknown
  ceiling on the GPU); strumpack crashes at ring scale.
- S-parameters need a REFERENCE solve (a current sheet radiates both
  ways): ratio device/reference, same port. Never quote absolute FDFD
  amplitudes.

## Unified S-parameters

```python
from meep_torch.solvers import sparams
S = sparams(eps, dx, freqs, ports={"in": p1, "out": p2}, src="in",
            npml=NPML, method="fdtd"|"fdfd", ref_eps=...)  # ref for fdfd
```

## 3D

- `meep_torch.core.stepper_3d_lean`: `build_lean(...)` + `run_3d_lean(...,
  compile=True)` — forward-only, in-place, global 4-slot scratch pool,
  auto eager-fallback if the compiled step OOMs. Monitors:
  `FluxMonitor3D`, `DFTFieldsMonitor3D`.
- `meep_torch.ports.validate3d`: `extrude(eps2d, ...)` (real index, no
  effective medium in 3D) + `ModePlaneSourceX` (quasi-TE from the full
  cross-section). RES >= 40/um for a 150 nm film (6 cells across).
- Differentiable 3D exists (`step_3d` + checkpointing) but watch the
  checkpoint-memory rules (agent memory: checkpoint-chunk-sizing).

## Inverse design

```python
from meep_torch.adjoint import DesignRegion, MaterialGrid, rho_from_eps
from meep_torch.adjoint.filters import conic_filter, tanh_projection
r = tanh_projection(conic_filter(rho, 0.12, dx), beta)   # beta ramp 8->64
eps = region.paint(eps_base, r)                          # engine-agnostic
```
Rules that came from real failures:
- Best-iterate selection: re-evaluate post-step at final beta each
  iteration and keep the argmax; the pre-step J curve ranks wrong.
- Binary imported seeds saturate the projection (near-zero gradients):
  soften to e.g. 0.3/0.7 before optimizing.
- Every new gradient path gets a SINGLE-application FD gate (chained /
  even-power objectives are blind to sign errors) with python-float FD
  steps (`torch.tensor(1+h)` is float32).
- Cross-check the final design in the other engine; expect a constant
  normalization offset (optimization unaffected), quote the FDTD number.

## Geometry / materials / plugins

- `meep_torch.plugins.gdsfactory` (`component_to_eps`): rasterize any
  gdsfactory component to eps + ports. GDS exports on SiN layer (4,0);
  contour PROJECTED density, not raw rho.
- `meep_torch.core.materials`: `Si3N4.n(f)`, `SiO2.n(f)`, ... exact at
  any frequency (FDFD uses this directly; FDTD needs pole fits).
- Mode solvers: `modes.fd1d.solve_1d` (port lines, TE slab collapse),
  `modes.fd2d.solve_2d` (full-vector cross sections). Call with CPU
  tensors.

## Nonlinear (2D/2.5D by decision; 3D chi2 shelved)

- SHG: pump solve at f0, then SH solve at 2f0 with
  `jz = -1j*w_SH*chi2*Ep**2`. QPM poling must be SELF-CALIBRATED from
  the measured grid beat, per radius (numerical dispersion is
  band-asymmetric; analytic Lambda desyncs).
- Kerr: damped fixed point `eps + chi3|E|^2`, resonance re-hunt per
  drive; parabolic sub-grid peak refinement or df quantizes to the
  hunt step.
- An E-advancing stepper gaining chi2 must rebuild
  `rhs = eps*E_old + chi2*E_old^2 + dt*curl` — dropping the nonlinear
  history is a 3.4%-per-step error (docs/plans/2026-08-17-3d-chi2.md).

## Environment

Python 3.12 (pinned; nvmath tops out at cp312), torch cu128 via the
explicit index in pyproject (backend wheels are CUDA-12/torch-ABI
tied; see wiki/TODO.md for the full backend truth table). Tests:
`uv run python -m pytest` (parity suite ~15 min; FDFD gates seconds).
