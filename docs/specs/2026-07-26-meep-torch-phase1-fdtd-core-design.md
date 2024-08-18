# meep-torch — Phase 1: Torch FDTD Core Engine — Design

**Date:** 2026-07-26
**Status:** Approved (brainstorming complete)
**Repo:** New standalone repository `meep-torch` (Python package `meep_torch`). This meep checkout (D:\Github\meep) serves as read-only reference for the update equations (`src/update_eh.cpp`, `src/update_pols.cpp`, PML/boundary code).

## Project context

Goal of the overall project: a PyTorch-based reimplementation of the meep FDTD workflow so that:

1. It installs as a plain pip/uv wheel on Windows and Linux (no autotools, HDF5, MPI, Guile, SWIG, MPB).
2. gdsfactory/gplugins-style S-parameter extraction works against it with only an import swap (same class/method shapes as `meep`, but a separately versioned package — **not** a drop-in `import meep` replacement).
3. Inverse design works via torch autodiff with memory kept under control (checkpointed time-stepping; naive backprop-through-time over thousands of steps is the failure mode to avoid).

### Phase decomposition (overall roadmap)

| Phase | Content | Status |
|-------|---------|--------|
| 1 | Core torch FDTD engine (this spec) | Designed |
| 2 | Geometry & materials layer (polygon rasterization, Medium-equivalents) | Future |
| 3 | meep-shaped Python API (Simulation/Source/FluxRegion signatures) | Future |
| 4 | Eigenmode source + mode decomposition → S-parameters | Future |
| 5 | Inverse design / adjoint-or-checkpointed gradients wiring | Future |
| 6 | Packaging, CI, Windows+Linux wheels | Future |

Each phase gets its own spec → plan → implementation cycle.

## Phase 1 scope

**In:**
- 2D Yee-grid FDTD, both polarizations: TM (Ez, Hx, Hy) and TE (Hz, Ex, Ey).
- CFS-PML (stretched-coordinate, auxiliary-field formulation) on all boundaries.
- Linear dispersive materials via ADE (auxiliary differential equation) with Lorentz poles; built-in Sellmeier→Lorentz fits for Si, SiO2, SiN. No general-purpose fitting UI.
- Kerr χ³ nonlinearity as an additional nonlinear polarization term.
- DFT flux accumulation at chosen frequencies over line regions.
- Sources: point/line hard+soft sources and normal-incidence plane wave. (Eigenmode sources are Phase 4.)
- Checkpointed time-stepping (`torch.utils.checkpoint`) so autodiff through the run is memory-flat.
- CPU and CUDA via standard torch device placement; float32 default, float64 selectable.

**Out (explicitly):**
- 3D, MPI/chunking, metals/plasmonics, gain/saturable media, symmetry reduction, near2far, Casimir.
- GDS/polygon geometry rasterization (Phase 2). Phase 1 accepts raw per-cell material tensors.
- meep-compatible API classes (Phase 3).
- Mode solving / MPB replacement (Phase 4).
- The inverse-design optimization loop itself (Phase 5). Phase 1 only guarantees gradients *can* flow without memory blow-up.

## Approach decision

**Chosen: direct Yee-grid port ("Approach A").** Same update equations as meep's `update_eh.cpp`/`update_pols.cpp`, expressed as torch tensor slicing/padding ops for the staggered-grid finite differences, explicit Python time loop, chunks of `checkpoint_every` steps wrapped in `torch.utils.checkpoint.checkpoint()`.

Rejected alternatives:
- **Conv2d-based stencils:** better `torch.compile` fusion ceiling, but adds a translation layer between grid indexing and conv padding semantics (error-prone at PML edges) and diverges from meep's own formulation, weakening parity validation. Can be revisited later as a pure optimization behind the same stepper interface.
- **Pseudospectral (FFT derivatives):** different numerical method entirely; goal is meep parity, not method research.

## Architecture

Package layout (in the new repo):

```
meep_torch/
  core/
    grid.py        # Grid
    materials.py   # Material, Sellmeier→Lorentz fits
    pml.py         # PML profiles + auxiliary state
    sources.py     # point/line/plane-wave sources
    monitors.py    # FluxMonitor (DFT accumulator)
    stepper.py     # pure update function
    run.py         # checkpointed time loop
tests/
  analytic/        # tier 1
  parity/          # tier 2 (skipif no meep)
  autodiff/        # tier 3
```

### Components

- **`Grid`** — owns field tensors (per polarization), device/dtype, cell dimensions, dx, CFL-derived dt.
- **`Material`** — per-cell ε∞ tensor; list of Lorentz poles (ω₀, γ, Δε) each with per-cell amplitude mask; optional per-cell Kerr χ³ tensor. Constructors `Material.silicon()`, `.silica()`, `.silicon_nitride()` carry pre-fit Lorentz-pole sets valid over the telecom band (1.2–1.7 µm).
- **`PML`** — per-axis conductivity/stretching profiles (polynomial grading), auxiliary field tensors.
- **`Source`** — time-profile (Gaussian pulse / CW ramp) × spatial footprint (point, line, plane-wave line). Additive ("soft") injection by default.
- **`FluxMonitor`** — accumulates DFT of tangential fields over a line region at requested frequencies each step; yields Poynting flux per frequency after the run.
- **`stepper`** — pure function `(state, coeffs, t) → state'` where `state` bundles E, H, ADE polarization states, PML auxiliaries. No hidden mutable instance state — purity is what makes `torch.utils.checkpoint` recomputation correct.
- **`run(state, n_steps, checkpoint_every, monitors)`** — drives the loop; wraps each `checkpoint_every`-step chunk in `torch.utils.checkpoint.checkpoint()`; feeds monitor accumulation outside the checkpointed region (DFT running sums are O(1) memory per step, no need to recompute them on backward).

### Data flow

All mutable simulation state (E, H, per-pole ADE polarizations, PML auxiliaries) travels in one flat namedtuple/dict through `stepper`. Static per-run data (material coefficient tensors, PML profiles, source time series) is precomputed before the loop. Monitors consume field slices per step and accumulate DFT sums in leaf tensors with normal autograd tracking.

### Precision & device

- Default float32, on whatever device the tensors were created (CPU/CUDA).
- float64 mode for validation runs (real meep is double); correctness tests run float64 to separate "physics right?" from "float32 adequate?". A dedicated test confirms float32 stability at target 2D grid sizes.

## Validation & testing

**Tier 1 — analytic (no meep required, runs on native Windows):**
- Vacuum plane-wave propagation speed vs. Yee numerical-dispersion prediction.
- PML broadband reflection < −40 dB (measure reflected energy).
- Energy conservation in a lossless closed (PEC-bounded) box.
- Dispersive slab transmission vs. analytic transfer-matrix result (Lorentz material).
- Kerr effect: intensity-dependent phase shift (self-phase modulation) vs. known scaling, or third-harmonic presence.

**Tier 2 — meep parity (written now, `pytest.mark.skipif` when `import meep` fails; activates on the future Linux machine):**
- Tiny 2D scenes run in both engines, float64; compare field snapshots and DFT fluxes to tight tolerance.

**Tier 3 — autodiff:**
- `torch.autograd.gradcheck` on short runs (tiny grid, few steps, float64) through stepper, PML, ADE, and Kerr paths — no NaNs, no graph breaks.
- Memory test: checkpointed long run (≥10k steps) stays flat in memory; documents the uncheckpointed blow-up as the contrast case.

No WSL / no local meep during initial development: tier 1 + tier 3 are the acceptance gate for Phase 1; tier 2 runs when a Linux box is available.

## Error handling

Fail fast, thin layer:
- CFL violation (dt vs. dx) → raise at construction time.
- Grid/PML/source/monitor shape mismatches → raise at build time with shapes in the message.
- Optional NaN watchdog during `run` (off by default; costs a device sync per check on GPU).
- No silent clamping or coercion anywhere.

## Dependencies & packaging (Phase 1 slice)

- Pure Python. Deps: `torch`, `numpy`, `scipy` (Sellmeier fitting at material-constructor build time), `pytest` (dev).
- No SWIG/C++/autotools/HDF5/MPI/Guile.
- Managed with uv + `pyproject.toml` in the new repo.
- Dev platform now: native Windows. Linux machine later for parity tests + eventual CI.

## Success criteria (Phase 1 done means)

1. All tier-1 analytic tests pass in float64 on Windows CPU.
2. Tier-3 gradcheck passes through every physics path (vacuum, PML, ADE, Kerr).
3. Checkpointed 10k-step 2D run shows flat memory profile with gradients enabled.
4. float32 stability test passes at a representative photonic grid size (~2000×2000 cells).
5. Tier-2 parity tests exist and are skipped cleanly where meep is absent.
