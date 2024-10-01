# Phase 4 — Eigenmode sources, mode decomposition, S-parameters (design)

Date: 2026-07-28
Status: approved design, pre-plan
Predecessors: Phase 1 (`v0.1.0-phase1`), Phase 2 (`v0.2.0-phase2`)
Wiki inputs: `wiki/summaries/phase3-phase4-outlook.md`, fdtdx audit (MIT blueprints)

## Goal

`get_eigenmode_coefficients`-equivalent for the 2D torch FDTD engine: eigenmode
port sources, mode-overlap decomposition of DFT fields, and an S-matrix driver.
Consumes Phase 2 mode solvers (`modes/fd1d.py`). Phase 3 (meep-shaped API) wraps
this later.

## Decisions (banked during brainstorm)

- **Sequencing**: Phase 4 before Phase 3 (wiki recommendation).
- **License**: shipped code is original/MIT (fdtdx patterns are MIT — portable).
  femwell (GPL-3.0) is a **test-only** oracle behind a guarded import in dev
  extras; nothing GPL ships in the wheel.
- **Autodiff**: forward-only this phase. Mode profiles come from
  `fd1d.solve_1d` (already detached). FDTD interior remains differentiable via
  Phase 1 checkpointing, but no gradient guarantee through mode profiles or
  S-parameters. eigengrad non-Hermitian backward + eigenvector IFT path
  deferred to Phase 5 (inverse design), where gradients are consumed.
- **Ports**: axis-aligned only (x- or y-normal lines). Both polarizations
  (TM and TE). Angled ports deferred (needs tensorial autograd path).
- **Modes**: first N modes per port, N user-set, default 1. S-matrix indexed
  by (port, mode). Degenerate-gap hardening for k>1 gradients stays deferred
  (forward-only anyway).
- **Validation**: physics invariants (unitarity, reciprocity, straight-guide
  |S21|→1) as hard gates, plus meep parity on this Linux box (pymeep guarded
  import) against `get_eigenmode_coefficients` on 1–2 benchmark structures.
- **Injection approach**: fdtdx TFSF blueprint (approach A) — one-way
  equivalence-current launch. Rejected: soft source + calibration run (extra
  run per column, source self-contamination); reimplementing meep's GPL2 math
  from papers (no advantage over MIT fdtdx patterns).

## Architecture

New bounded subpackage `meep_torch/ports/`:

```
meep_torch/ports/
  __init__.py
  port.py          # Port spec: axis (x|y normal), position index, span [lo,hi),
                   # direction (+|-), n_modes, pol
  mode_source.py   # ModeSourceTM / ModeSourceTE — TFSF one-way injection
  mode_monitor.py  # DFTLineMonitor + overlap decomposition (post-processing)
  smatrix.py       # compute_smatrix() — per-port excitation loop, assembly
```

Data flow per S-column (excited port j):

1. fd1d solves mode profiles on port j's eps cross-section — at fcen for
   injection, at each monitor frequency for decomposition.
2. FDTD run with `ModeSource(j)` + `DFTLineMonitor` on every port.
3. Post: overlap integrals give complex a±(f) per (port, mode).
4. S[(i,m),(j,n)](f) = b_out[i,m](f) / a_in[j,n](f), where a_in is the
   **measured** forward coefficient at port j's own monitor. Self-normalizing:
   absorbs broadband injection imperfection; no calibration run.

## ModeSource (TFSF injection)

x-normal TM case; y-normal and TE are Yee duals.

- Setup: fd1d TM solve of the port line at fcen → Ez_m(y), neff.
  Derived H profile: Hy_m = neff · Ez_m (impedance relation), sampled at
  Yee half-nodes. (Hx_m = (i/k0)·dEz_m/dy exists but is unused: axis-aligned
  injection and overlap only need the transverse pair Ez/Hy carrying Sx.)
- Injection: equivalence-principle current pair on adjacent columns —
  electric current Jz(y) ∝ Hy_m(y) into Dz at column i; magnetic current
  My(y) ∝ Ez_m(y) into Hy at column i−½. Time signatures: same Gaussian
  pulse; H series offset −dt/2 (Yee time stagger) plus a spatial phase
  offset realized as a time shift ≈ neff·dx/(2c) (fields are real — no
  quadrature pair needed). Direction sign flips the relative sign of the
  pair → left or right launch.
- Broadband: single fcen mode profile injected. Per-frequency impedance
  error self-corrects through measured-incident normalization. fdtdx's
  impedance H-filter deliberately dropped (`ponytail:` ceiling — very wide
  bands may want per-frequency injection; upgrade path is the fdtdx filter).
- Quality gate: forward/backward power ratio ≥ 30 dB in a uniform waveguide.

### Stepper touch (only physics-layer change)

`step_tm`/`step_te` gain an optional second current term for the magnetic
TFSF component (TM: Hy or Hx injection alongside existing Dz; TE dual).
Existing `LineSource` path untouched (back-compat). `run.py` source protocol
widens from a single `(amp, mask)` to a source object contributing per-field
currents; checkpoint machinery unchanged.

## ModeMonitor

- `DFTLineMonitor` (reduce="sum", same protocol as `FluxMonitorTM`):
  accumulates Re/Im DFT of tangential E and H on the port line per frequency,
  with the existing tE/tH half-step time conventions and H spatial averaging
  to E positions (as `FluxMonitorTM` does).
- Post (`decompose()`): per frequency f, fd1d solve at f → mode profiles
  (Em, Hm) per mode n. Coefficients:

  a±ₙ(f) = [∫ Et·Hmₙ ± ∫ Emₙ·Ht] / (2·Nₙ),  Nₙ = ∫ Emₙ·Hmₙ

  Bi-orthogonal, **no conjugate** (lossy-correct; fd2d post convention).
  ± selects forward/backward.
- Complex arithmetic only in post (torch complex from Re/Im DFT pairs);
  the time loop stays real.

## smatrix.py

- `compute_smatrix(scene, ports, fcen, fwidth, freqs, *, pol, resolution,
  run_until, ...)`: builds grid + coeffs once, loops excited port j
  (run FDTD, decompose all monitors, fill column S[:, j]).
- Normalization: measured-incident (above). Mode power norm Nₙ from the
  overlap itself, per frequency — no effindex table needed for normalization.
- Geometry: monitor sits a few cells inside the port position; source a few
  cells further out. Phase reference plane = monitor plane (documented).
- Run length: fixed `run_until` steps supplied by the user. Field-decay
  auto-stop deferred (`ponytail:` mark in code).
- Returns `SMatrix` NamedTuple: `s` complex tensor (nf, P·N, P·N), `freqs`,
  and a (port, mode) index map.

## Error handling

- Port span must not touch PML (validated against pml cell counts).
- Port line must support ≥ n_modes guided modes at fcen
  (neff > cladding index check; ValueError otherwise).
- Overlap norm Nₙ near zero → raise (degenerate or leaky mode).

## Testing (tests/ports/)

1. fd1d-derived H profiles: overlap orthonormality δᵢⱼ across the mode bank.
2. TFSF one-way: uniform waveguide, forward/backward power ≥ 30 dB.
3. Straight guide: |S21|→1 (tol ~1e-2), |S11| small, arg(S21) matches
   neff·L per frequency.
4. Lossless unitarity: Σᵢ |Sᵢⱼ|² ≈ 1 per frequency.
5. Reciprocity: asymmetric lossless structure, S12 ≈ S21.
6. TE duals of tests 2–3.
7. femwell oracle (test-only dep, guarded import): overlap-integral parity.
8. meep parity (guarded `import meep`): straight waveguide + one scattering
   structure vs `get_eigenmode_coefficients`, few-percent tolerance.

## Deferred (tracked)

- End-to-end S-parameter gradients (eigengrad PML/non-Hermitian backward,
  eigenvector IFT) → Phase 5.
- Angled ports / tensorial path.
- Per-frequency injection (fdtdx impedance H-filter).
- Field-decay auto-stop for run length.
- k>1 degenerate-gap gradient hardening.
