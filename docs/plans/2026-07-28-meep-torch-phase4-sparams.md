# Phase 4 S-Parameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eigenmode port sources (TFSF one-way), mode-overlap decomposition of DFT line fields, and an S-matrix driver for the 2D torch FDTD engine.

**Architecture:** New bounded subpackage `meep_torch/ports/` (port spec, mode source, decomposition post-processing, S-matrix loop). Only physics-layer change: `step_tm`/`step_te` gain an optional magnetic-current term; `run.py` source protocol widens. DFT line accumulation reuses the existing `FluxMonitorTM`/`FluxMonitorTE` layout (ReE, ImE, ReH, ImH per freq) — decomposition is pure post-processing.

**Tech Stack:** torch (float64 CPU default), numpy/scipy (fd1d), pytest. Oracles: pymeep 1.34.0 in conda env `meep-oracle` (reference-data generation only, not a test-time dep), femwell (guarded test-only import).

**Spec:** `docs/specs/2026-07-28-meep-torch-phase4-sparams-design.md`

## Global Constraints

- Shipped code is original/MIT. fdtdx patterns (MIT) may be ported. Nothing GPL ships in the wheel; femwell (GPL-3.0) appears only behind guarded imports in tests.
- Forward-only autodiff this phase: mode profiles from `fd1d.solve_1d` stay detached. No gradient claims through S-parameters.
- Axis-aligned ports only (x- or y-normal). TM and TE.
- All new files ≤ 500 lines. SPDX header on every new file:
  `# SPDX-License-Identifier: MIT` / `# Copyright (C) 2026 meep-torch contributors (original code)`
- Run tests with `uv run pytest`. Existing suite must stay green after every task.
- Units: c = 1, lengths in µm, frequency in 1/µm (meep convention).
- Sign convention note (used across tasks): monitor DFT accumulates `∫E(t)e^{−iωt}dt`; a +x-propagating pulse reaching a monitor at distance L acquires phase `−2πf·neff·L`.

## File map

| File | Responsibility |
| --- | --- |
| `meep_torch/ports/__init__.py` | public re-exports |
| `meep_torch/ports/port.py` | `Port` dataclass + geometric validation |
| `meep_torch/ports/mode_source.py` | `ModeSourceTM`, `ModeSourceTE` — TFSF one-way injection |
| `meep_torch/ports/mode_monitor.py` | `dft_fields`, `solve_port_modes`, `decompose` post-processing |
| `meep_torch/ports/smatrix.py` | `SMatrix`, `compute_smatrix` |
| `meep_torch/core/stepper.py` | modify: optional Hy/Hx current in `step_tm` |
| `meep_torch/core/stepper_te.py` | modify: optional Hz current in `step_te` |
| `meep_torch/core/run.py` | modify: dual-current source protocol |
| `tests/ports/test_port.py` … `tests/ports/test_meep_parity.py` | per-task tests |
| `tests/ports/generate_meep_reference.py` + `tests/ports/data/meep_reference.json` | oracle data + regeneration script |

---

### Task 1: Port spec + validation

**Files:**
- Create: `meep_torch/ports/__init__.py`, `meep_torch/ports/port.py`
- Test: `tests/ports/test_port.py`

**Interfaces:**
- Consumes: `meep_torch.core.grid.Grid`
- Produces:
  ```python
  @dataclass(frozen=True)
  class Port:
      axis: str        # "x" | "y" — port-plane normal
      index: int       # grid index of the port plane (E-node column/row)
      lo: int          # span start (cells along the port line), inclusive
      hi: int          # span end, exclusive
      direction: int   # +1 | -1: launch direction INTO the structure along axis
      n_modes: int = 1

  def validate_ports(ports: Sequence[Port], grid: Grid, npml: int) -> None
  ```
  `validate_ports` raises `ValueError` when: axis not in ("x","y"); direction not ±1; n_modes < 1; span empty or outside grid; port plane or span inside PML (plane must satisfy `npml + 3 <= index < n_axis - npml - 3` to leave room for the source offset of 3 cells, span must satisfy `lo >= npml` and `hi <= n_perp - npml`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/ports/test_port.py
import pytest
from meep_torch.core.grid import Grid
from meep_torch.ports.port import Port, validate_ports


def _grid():
    return Grid(nx=100, ny=60, dx=0.1)


def test_valid_port_passes():
    p = Port(axis="x", index=20, lo=15, hi=45, direction=+1)
    validate_ports([p], _grid(), npml=10)


@pytest.mark.parametrize("kw,msg", [
    (dict(axis="z"), "axis"),
    (dict(direction=0), "direction"),
    (dict(n_modes=0), "n_modes"),
    (dict(lo=45, hi=15), "span"),
    (dict(index=11), "PML"),          # 11 < npml+3 = 13
    (dict(index=95), "PML"),          # 95 >= nx-npml-3 = 87
    (dict(lo=5), "PML"),              # span dips into PML
])
def test_invalid_ports_raise(kw, msg):
    base = dict(axis="x", index=20, lo=15, hi=45, direction=+1, n_modes=1)
    base.update(kw)
    with pytest.raises(ValueError, match=msg):
        validate_ports([Port(**base)], _grid(), npml=10)


def test_y_normal_axis_lengths():
    # y-normal: index checked against ny, span against nx
    p = Port(axis="y", index=20, lo=15, hi=85, direction=-1)
    validate_ports([p], _grid(), npml=10)
    bad = Port(axis="y", index=55, lo=15, hi=85, direction=-1)  # 55 >= ny-13
    with pytest.raises(ValueError, match="PML"):
        validate_ports([bad], _grid(), npml=10)
```

- [ ] **Step 2: Run tests, verify FAIL** — `uv run pytest tests/ports/test_port.py -v` — expect `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# meep_torch/ports/port.py
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
from dataclasses import dataclass
from typing import Sequence

from ..core.grid import Grid

SRC_OFFSET = 3  # cells between monitor plane (port.index) and TFSF source plane


@dataclass(frozen=True)
class Port:
    axis: str
    index: int
    lo: int
    hi: int
    direction: int
    n_modes: int = 1


def validate_ports(ports: Sequence[Port], grid: Grid, npml: int) -> None:
    for k, p in enumerate(ports):
        if p.axis not in ("x", "y"):
            raise ValueError(f"port {k}: axis must be 'x' or 'y', got {p.axis!r}")
        if p.direction not in (+1, -1):
            raise ValueError(f"port {k}: direction must be +1 or -1, got {p.direction}")
        if p.n_modes < 1:
            raise ValueError(f"port {k}: n_modes must be >= 1, got {p.n_modes}")
        n_axis = grid.nx if p.axis == "x" else grid.ny
        n_perp = grid.ny if p.axis == "x" else grid.nx
        if not p.lo < p.hi:
            raise ValueError(f"port {k}: empty span lo={p.lo}, hi={p.hi}")
        if p.lo < 0 or p.hi > n_perp:
            raise ValueError(f"port {k}: span [{p.lo},{p.hi}) outside grid (n={n_perp})")
        if not (npml + SRC_OFFSET <= p.index < n_axis - npml - SRC_OFFSET):
            raise ValueError(
                f"port {k}: plane index={p.index} inside/adjacent to PML "
                f"(need {npml + SRC_OFFSET} <= index < {n_axis - npml - SRC_OFFSET})"
            )
        if p.lo < npml or p.hi > n_perp - npml:
            raise ValueError(f"port {k}: span [{p.lo},{p.hi}) reaches into PML (npml={npml})")
```

```python
# meep_torch/ports/__init__.py
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
from .port import Port, validate_ports

__all__ = ["Port", "validate_ports"]
```

- [ ] **Step 4: Run tests, verify PASS** — `uv run pytest tests/ports/test_port.py -v`
- [ ] **Step 5: Commit** — `git add meep_torch/ports tests/ports && git commit -m "feat: port spec and validation for S-parameter ports"`

---

### Task 2: Stepper magnetic-current terms

**Files:**
- Modify: `meep_torch/core/stepper.py` (`step_tm`), `meep_torch/core/stepper_te.py` (`step_te`)
- Test: `tests/ports/test_stepper_hcurrent.py`

**Interfaces:**
- Produces (backward-compatible extensions — all new params optional):
  ```python
  def step_tm(state, m, src_amp=0.0, src_mask=None,
              src_h_amp=0.0, src_hy_mask=None, src_hx_mask=None) -> TMState
  def step_te(state, m, src_amp=0.0, src_mask_ey=None,
              src_h_amp=0.0, src_hz_mask=None) -> TEState
  ```
  Injection semantics: after the H update, `Hy += dt * src_h_amp * src_hy_mask` (shape `(nx-1, ny)`), `Hx += dt * src_h_amp * src_hx_mask` (shape `(nx, ny-1)`), `Hz += dt * src_h_amp * src_hz_mask` (shape `(nx-1, ny-1)`). The injected H is then seen by the same step's D update (magnetic current at time n drives H at n+½ — matches TFSF timing).

- [ ] **Step 1: Write the failing tests**

```python
# tests/ports/test_stepper_hcurrent.py
import torch
from meep_torch.core.grid import Grid
from meep_torch.core.coeffs import build_tm_coeffs, build_te_coeffs
from meep_torch.core.stepper import init_tm_state, step_tm
from meep_torch.core.stepper_te import init_te_state, step_te


def test_tm_hy_current_injects():
    grid = Grid(nx=20, ny=20, dx=0.1)
    m = build_tm_coeffs(grid, eps_inf=1.0, npml=0)
    state = init_tm_state(grid)
    mask = grid.zeros(grid.nx - 1, grid.ny)
    mask[10, 10] = 1.0
    out = step_tm(state, m, src_h_amp=2.0, src_hy_mask=mask)
    assert torch.isclose(out.Hy[10, 10], torch.tensor(2.0 * m.dt, dtype=torch.float64))
    # no-current call unchanged (back-compat)
    out2 = step_tm(state, m)
    assert out2.Hy.abs().max() == 0.0


def test_tm_hx_current_injects():
    grid = Grid(nx=20, ny=20, dx=0.1)
    m = build_tm_coeffs(grid, eps_inf=1.0, npml=0)
    state = init_tm_state(grid)
    mask = grid.zeros(grid.nx, grid.ny - 1)
    mask[10, 10] = 1.0
    out = step_tm(state, m, src_h_amp=-1.5, src_hx_mask=mask)
    assert torch.isclose(out.Hx[10, 10], torch.tensor(-1.5 * m.dt, dtype=torch.float64))


def test_te_hz_current_injects():
    grid = Grid(nx=20, ny=20, dx=0.1)
    m = build_te_coeffs(grid, eps_inf=1.0, npml=0)
    state = init_te_state(grid)
    mask = grid.zeros(grid.nx - 1, grid.ny - 1)
    mask[10, 10] = 1.0
    out = step_te(state, m, src_h_amp=3.0, src_hz_mask=mask)
    assert torch.isclose(out.Hz[10, 10], torch.tensor(3.0 * m.dt, dtype=torch.float64))


def test_h_current_seen_by_same_step_d_update():
    # injected Hy at column i must contribute to Dz curl in the SAME step
    grid = Grid(nx=20, ny=20, dx=0.1)
    m = build_tm_coeffs(grid, eps_inf=1.0, npml=0)
    state = init_tm_state(grid)
    mask = grid.zeros(grid.nx - 1, grid.ny)
    mask[10, 10] = 1.0
    out = step_tm(state, m, src_h_amp=1.0, src_hy_mask=mask)
    assert out.Dz.abs().max() > 0.0
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest tests/ports/test_stepper_hcurrent.py -v` — expect `TypeError: unexpected keyword argument`.

- [ ] **Step 3: Implement.** In `step_tm`, after the two H updates (`Hx = ...`, `Hy = ...`) and before the D update, insert:

```python
    if src_hy_mask is not None:
        Hy = Hy + dt * src_h_amp * src_hy_mask
    if src_hx_mask is not None:
        Hx = Hx + dt * src_h_amp * src_hx_mask
```

with the new signature `def step_tm(state, m, src_amp=0.0, src_mask=None, src_h_amp=0.0, src_hy_mask=None, src_hx_mask=None)`. In `step_te`, after the Hz update and before the E/D updates:

```python
    if src_hz_mask is not None:
        Hz = Hz + dt * src_h_amp * src_hz_mask
```

with signature `def step_te(state, m, src_amp=0.0, src_mask_ey=None, src_h_amp=0.0, src_hz_mask=None)`. (Read `stepper_te.py` first to place the insert after Hz is computed and before it feeds the D/E curl.)

- [ ] **Step 4: Run new tests AND full suite** — `uv run pytest tests/ -x -q`. Expected: all pass (back-compat).
- [ ] **Step 5: Commit** — `git commit -am "feat: optional magnetic-current injection in TM/TE steppers"`

---

### Task 3: run.py dual-current source protocol

**Files:**
- Modify: `meep_torch/core/run.py` (`_run_chunk`)
- Test: `tests/ports/test_run_protocol.py`

**Interfaces:**
- Consumes: Task 2 stepper signatures.
- Produces: `_run_chunk` dispatches on the source object. Protocol:
  - Legacy source (e.g. `LineSource`): has `.amplitude(t_idx)` and `.mask` → called exactly as today.
  - Dual-current source: additionally has attribute `h_masks` — a dict mapping stepper kwarg name to mask tensor, e.g. `{"src_hy_mask": tensor}` — and method `.h_amplitude(t_idx) -> float`. Then the step call is
    `step_fn(state, coeffs, source.amplitude(t_idx), source.mask, source.h_amplitude(t_idx), **source.h_masks_kw)` — implemented via keyword expansion so TM and TE names both work.

- [ ] **Step 1: Write the failing test**

```python
# tests/ports/test_run_protocol.py
import torch
from meep_torch.core.grid import Grid
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.run import run_tm
from meep_torch.core.stepper import init_tm_state


class DualSource:
    def __init__(self, grid):
        self.mask = grid.zeros(grid.nx, grid.ny)
        self.mask[8, 10] = 1.0
        hy = grid.zeros(grid.nx - 1, grid.ny)
        hy[7, 10] = 1.0
        self.h_masks = {"src_hy_mask": hy}

    def amplitude(self, t_idx):
        return 1.0

    def h_amplitude(self, t_idx):
        return -1.0


def test_dual_source_drives_both_fields():
    grid = Grid(nx=20, ny=20, dx=0.1)
    m = build_tm_coeffs(grid, eps_inf=1.0, npml=0)
    state, _ = run_tm(init_tm_state(grid), m, 3, source=DualSource(grid))
    assert state.Hy[7, 10].abs() > 0.0
    assert state.Dz[8, 10].abs() > 0.0


def test_dual_source_checkpointed_matches_plain():
    grid = Grid(nx=20, ny=20, dx=0.1)
    m = build_tm_coeffs(grid, eps_inf=1.0, npml=0)
    s1, _ = run_tm(init_tm_state(grid), m, 8, source=DualSource(grid))
    s2, _ = run_tm(init_tm_state(grid), m, 8, source=DualSource(grid), checkpoint_every=3)
    assert torch.allclose(s1.Ez, s2.Ez)
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest tests/ports/test_run_protocol.py -v` — expect `TypeError` (step called without h kwargs → Hy stays zero → assert fails).

- [ ] **Step 3: Implement.** In `run.py::_run_chunk` replace the source call:

```python
        if source is not None:
            h_masks = getattr(source, "h_masks", None)
            if h_masks:
                state = step_fn(
                    state, coeffs, source.amplitude(t_idx), source.mask,
                    source.h_amplitude(t_idx), **h_masks,
                )
            else:
                state = step_fn(state, coeffs, source.amplitude(t_idx), source.mask)
        else:
            state = step_fn(state, coeffs)
```

(`src_h_amp` is the 5th positional parameter of both steppers; `h_masks` keys are the mask kwarg names.)

- [ ] **Step 4: Run new tests AND full suite** — `uv run pytest tests/ -x -q`.
- [ ] **Step 5: Commit** — `git commit -am "feat: dual-current source protocol in run loop"`

---

### Task 4: ModeSourceTM — TFSF one-way injection

**Files:**
- Create: `meep_torch/ports/mode_source.py`
- Test: `tests/ports/test_mode_source_tm.py`

**Interfaces:**
- Consumes: `Port` (Task 1), stepper/run protocol (Tasks 2–3), `fd1d.solve_1d`, `GaussianPulse` (`core/sources.py`), `TMCoeffs.eps_inf`.
- Produces:
  ```python
  class ModeSourceTM:
      def __init__(self, grid: Grid, coeffs: TMCoeffs, port: Port,
                   profile: GaussianPulse, fcen: float, mode: int = 0): ...
      # attrs: mask (nx, ny) Dz current; h_masks dict; neff: float
      def amplitude(self, t_idx: int) -> float
      def h_amplitude(self, t_idx: int) -> float
  ```
  Physics (x-normal, direction d = ±1):
  - eps line = `coeffs.eps_inf[port.index, port.lo:port.hi]` (detached). `fd1d.solve_1d(eps_line, dx, fcen, "tm", k=mode+1)` → `Ez_m` (span,), `neff`.
  - Guided check: `neff > max(eps_line[0], eps_line[-1]).sqrt()` else `ValueError("no guided mode")`.
  - Dispersive check: if `coeffs.c3` is nonzero anywhere on the port line → `ValueError("dispersive media on port line unsupported")`.
  - `Hy_m = neff * Ez_m` (impedance relation, same y-nodes as Ez).
  - Electric current: `mask[index, lo:hi] = Hy_m / dx` (Jz ∝ Hy_m).
  - Magnetic current: Hy column `index - 1` if d = +1 else `index`; `hy_mask[col, lo:hi] = d * Ez_m / dx` (My ∝ Ez_m; sign carries direction).
  - Time series: `amplitude(t_idx) = g((t_idx + 0.5) * dt)` (existing E-current convention); `h_amplitude(t_idx) = -g(t_idx * dt - d * neff * dx / 2)` — Yee half-step earlier plus the half-cell propagation delay between the two sheets.
  - `_H_SIGN = -1.0` module constant carrying the overall relative sign; the one-way test is the arbiter — if the launch goes the wrong way, flip this one constant (single documented decision point, then frozen).
  - y-normal: transpose roles — profile along x from `eps_inf[lo:hi, index]`, electric current ∝ `-Hx_m` where `Hx_m = neff * Ez_m`, magnetic current on `hx_mask` rows `index-1`/`index`, same time series. (In TM the y-propagating transverse pair is Ez/Hx; the sign difference relative to x-normal follows from `S_y = -Ez*Hx`.)

- [ ] **Step 1: Write the failing test**

```python
# tests/ports/test_mode_source_tm.py
import torch
from meep_torch.core.grid import Grid
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper import init_tm_state
from meep_torch.ports.port import Port
from meep_torch.ports.mode_source import ModeSourceTM

FCEN, FWIDTH = 0.15, 0.05


def _waveguide():
    # straight waveguide eps=12, width 1um, along x; dx=0.1 -> 10 cells wide
    grid = Grid(nx=160, ny=80, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    eps[:, 35:45] = 12.0
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=10)
    return grid, coeffs


def test_one_way_launch_30db():
    grid, coeffs = _waveguide()
    port = Port(axis="x", index=80, lo=20, hi=60, direction=+1)
    src = ModeSourceTM(grid, coeffs, port, GaussianPulse(FCEN, FWIDTH), FCEN)
    state = init_tm_state(grid)
    # run long enough for the pulse to clear both probe planes
    n = 1200
    fwd_e, bwd_e = 0.0, 0.0
    for t in range(n):
        from meep_torch.core.stepper import step_tm
        state = step_tm(state, coeffs, src.amplitude(t), src.mask,
                        src.h_amplitude(t), **src.h_masks)
        fwd_e += float((state.Ez[110, 20:60] ** 2).sum())
        bwd_e += float((state.Ez[50, 20:60] ** 2).sum())
    ratio_db = 10.0 * torch.log10(torch.tensor(fwd_e / bwd_e))
    assert ratio_db > 30.0, f"one-way ratio {ratio_db:.1f} dB < 30 dB"


def test_guided_mode_required():
    grid = Grid(nx=160, ny=80, dx=0.1)
    coeffs = build_tm_coeffs(grid, eps_inf=1.0, npml=10)  # vacuum: nothing guided
    port = Port(axis="x", index=80, lo=20, hi=60, direction=+1)
    import pytest
    with pytest.raises(ValueError, match="guided"):
        ModeSourceTM(grid, coeffs, port, GaussianPulse(FCEN, FWIDTH), FCEN)
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest tests/ports/test_mode_source_tm.py -v` — expect `ImportError`.

- [ ] **Step 3: Implement `ModeSourceTM`** per the interface block above. Skeleton:

```python
# meep_torch/ports/mode_source.py
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""TFSF one-way eigenmode injection (fdtdx MIT blueprint, reimplemented).

Forward-only: mode profiles are detached (fd1d solves in numpy).
"""
import math

import torch

from ..core.grid import Grid
from ..modes.fd1d import solve_1d
from .port import Port

_H_SIGN = -1.0  # relative sign of the magnetic sheet; fixed by the one-way test


class ModeSourceTM:
    def __init__(self, grid, coeffs, port, profile, fcen, mode=0):
        self.profile, self.dt, self.d = profile, grid.dt, port.direction
        dx = grid.dx
        if port.axis == "x":
            eps_line = coeffs.eps_inf[port.index, port.lo:port.hi].detach()
        else:
            eps_line = coeffs.eps_inf[port.lo:port.hi, port.index].detach()
        if coeffs.c3.numel() and self._line_dispersive(coeffs.c3, port):
            raise ValueError("dispersive media on port line unsupported")
        modes = solve_1d(eps_line, dx, fcen, "tm", k=mode + 1)
        neff = float(modes.neff[mode])
        clad = float(torch.sqrt(torch.max(eps_line[0], eps_line[-1])))
        if neff <= clad + 1e-9:
            raise ValueError(f"no guided mode at port (neff={neff:.4f} <= cladding {clad:.4f})")
        prof = modes.profiles[mode].to(grid.dtype)
        self.neff = neff
        self.delay = self.d * neff * dx / 2.0

        mask = grid.zeros(grid.nx, grid.ny)
        if port.axis == "x":
            mask[port.index, port.lo:port.hi] = neff * prof / dx      # Jz ∝ Hy_m
            hy = grid.zeros(grid.nx - 1, grid.ny)
            col = port.index - 1 if self.d > 0 else port.index
            hy[col, port.lo:port.hi] = self.d * prof / dx             # My ∝ Ez_m
            self.h_masks = {"src_hy_mask": hy}
        else:
            mask[port.lo:port.hi, port.index] = -neff * prof / dx     # Jz ∝ -Hx_m
            hx = grid.zeros(grid.nx, grid.ny - 1)
            row = port.index - 1 if self.d > 0 else port.index
            hx[port.lo:port.hi, row] = self.d * prof / dx
            self.h_masks = {"src_hx_mask": hx}
        self.mask = mask

    @staticmethod
    def _line_dispersive(c3, port) -> bool:
        if port.axis == "x":
            return bool((c3[:, port.index, port.lo:port.hi] != 0).any())
        return bool((c3[:, port.lo:port.hi, port.index] != 0).any())

    def amplitude(self, t_idx: int) -> float:
        return self.profile.amplitude((t_idx + 0.5) * self.dt)

    def h_amplitude(self, t_idx: int) -> float:
        return _H_SIGN * self.profile.amplitude(t_idx * self.dt - self.delay)
```

- [ ] **Step 4: Run one-way test.** If `ratio_db` comes out strongly negative (launch went backward), flip `_H_SIGN` to `+1.0` — one constant, one decision, add a comment recording the outcome. Then: `uv run pytest tests/ports/test_mode_source_tm.py -v` → PASS.
- [ ] **Step 5: Run full suite** — `uv run pytest tests/ -x -q`.
- [ ] **Step 6: Commit** — `git commit -am "feat: TM eigenmode TFSF one-way source"`

---

### Task 5: Decomposition post-processing (TM)

**Files:**
- Create: `meep_torch/ports/mode_monitor.py`
- Test: `tests/ports/test_decompose_tm.py`

**Interfaces:**
- Consumes: `FluxMonitorTM` accumulator layout `(nf, 4, span)` = (ReE, ImE, ReH, ImH) — produced by `run_tm(..., monitors=[FluxMonitorTM(...)])`; `fd1d.solve_1d`.
- Produces:
  ```python
  def dft_fields(acc: Tensor) -> tuple[Tensor, Tensor]
      # (nf,4,span) -> complex (nf,span) E, H

  def decompose(acc: Tensor, eps_line: Tensor, dx: float,
                freqs: Sequence[float], pol: str, n_modes: int,
                ) -> tuple[Tensor, Tensor]
      # -> (a_plus, a_minus): complex (nf, n_modes) coefficients of
      #    +axis- and -axis-propagating waves at the monitor plane
  ```
  Math (TM): per frequency f, solve `fd1d.solve_1d(eps_line, dx, f, "tm", k=n_modes)` → `Ez_m`, `neff(f)`; `Hy_m = neff * Ez_m`. With complex DFT fields `Ec, Hc` on the line:
  - `N_m = Σ Ez_m * Hy_m * dx` (real; raise if `|N_m| < 1e-12` — degenerate/leaky)
  - `a±_m(f) = (Σ Ec·Hy_m*dx ± Σ Ez_m·Hc*dx) / (2 N_m)`
  Bi-orthogonal, **no conjugate** anywhere. The x-normal monitor samples `H = avg(Hy)` at Ez nodes (existing `FluxMonitorTM.sample`), so both integrals live on the same nodes. y-normal: the transverse H is Hx and `S_y = -Ez*Hx`, so pass `pol="tm"` the same way but the caller flips the ± meaning via the port direction logic in Task 6 — `decompose` itself is orientation-agnostic: "+" always means +axis propagation because `FluxMonitorTM.flux` sign conventions are already handled at the flux level, and the overlap uses the raw accumulated H (which for x-normal is Hy, for y-normal is Hx with the sign convention of the monitor). Concretely: for y-normal monitors negate `Hy_m -> -Hx_m = -neff*Ez_m` by passing `axis_sign=-1`:
  ```python
  def decompose(acc, eps_line, dx, freqs, pol, n_modes, axis_sign=+1.0)
  ```
  with `Hm = axis_sign * neff * Ez_m` (x-normal `+1`, y-normal `-1`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/ports/test_decompose_tm.py
import torch
from meep_torch.core.grid import Grid
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper import init_tm_state
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.port import Port
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.mode_monitor import decompose, dft_fields

FCEN, FWIDTH = 0.15, 0.05
FREQS = [0.13, 0.15, 0.17]


def _run_straight():
    grid = Grid(nx=160, ny=80, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    eps[:, 35:45] = 12.0
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=10)
    port = Port(axis="x", index=30, lo=20, hi=60, direction=+1)
    src = ModeSourceTM(grid, coeffs, port, GaussianPulse(FCEN, FWIDTH), FCEN)
    mon = FluxMonitorTM(grid, FREQS, x_normal=True, idx=120, lo=20, hi=60)
    _, (acc,) = run_tm(init_tm_state(grid), coeffs, 2500, source=src, monitors=[mon])
    eps_line = eps[120, 20:60]
    return acc, eps_line, grid.dx


def test_mode_orthonormality():
    # decompose's own overlap of mode m against mode n is delta_mn
    eps_line = torch.ones(40, dtype=torch.float64)
    eps_line[15:25] = 12.0
    dx = 0.1
    modes = solve_1d(eps_line, dx, FCEN, "tm", k=3)
    for m in range(3):
        Em = modes.profiles[m]
        for n in range(3):
            Hn = float(modes.neff[n]) * modes.profiles[n]
            ovl = float((Em * Hn).sum() * dx)
            norm = float(modes.neff[m]) * float((modes.profiles[m] ** 2).sum()) * dx
            if m == n:
                assert abs(ovl / norm - 1.0) < 1e-10
            else:
                assert abs(ovl) / norm < 1e-8


def test_forward_dominates_backward():
    acc, eps_line, dx = _run_straight()
    a_p, a_m = decompose(acc, eps_line, dx, FREQS, "tm", 1)
    for i in range(len(FREQS)):
        assert abs(a_p[i, 0]) > 100.0 * abs(a_m[i, 0])


def test_dft_fields_shapes():
    acc, eps_line, dx = _run_straight()
    E, H = dft_fields(acc)
    assert E.shape == (3, 40) and E.dtype == torch.complex128
    assert H.shape == (3, 40) and H.dtype == torch.complex128
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest tests/ports/test_decompose_tm.py -v` — expect `ImportError`.

- [ ] **Step 3: Implement**

```python
# meep_torch/ports/mode_monitor.py
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Mode-overlap decomposition of DFT line fields.

Bi-orthogonal <E1 x H2> convention, no conjugate (lossy-correct).
Forward-only: mode profiles detached.
"""
from typing import Sequence

import torch

from ..modes.fd1d import solve_1d

_NORM_FLOOR = 1e-12


def dft_fields(acc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ReE, ImE, ReH, ImH = acc.unbind(1)
    return torch.complex(ReE, ImE), torch.complex(ReH, ImH)


def decompose(acc, eps_line, dx, freqs: Sequence[float], pol: str,
              n_modes: int, axis_sign: float = 1.0):
    Ec, Hc = dft_fields(acc)
    nf = len(freqs)
    a_p = torch.zeros(nf, n_modes, dtype=torch.complex128)
    a_m = torch.zeros(nf, n_modes, dtype=torch.complex128)
    for i, f in enumerate(freqs):
        modes = solve_1d(eps_line.detach(), dx, f, pol, k=n_modes)
        for m in range(n_modes):
            Em = modes.profiles[m]
            Hm = axis_sign * float(modes.neff[m]) * Em
            N = float((Em * Hm).sum()) * dx
            if abs(N) < _NORM_FLOOR:
                raise ValueError(f"degenerate/leaky mode {m} at f={f}: norm {N:.3e}")
            ovl_EH = (Ec[i] * Hm.to(torch.complex128)).sum() * dx
            ovl_HE = (Em.to(torch.complex128) * Hc[i]).sum() * dx
            a_p[i, m] = (ovl_EH + ovl_HE) / (2.0 * N)
            a_m[i, m] = (ovl_EH - ovl_HE) / (2.0 * N)
    return a_p, a_m
```

- [ ] **Step 4: Run tests, verify PASS** — `uv run pytest tests/ports/test_decompose_tm.py -v` (runtime ~1 min for the FDTD run; reuse the module-scope fixture if slower).
- [ ] **Step 5: Commit** — `git commit -am "feat: bi-orthogonal mode-overlap decomposition of DFT line fields"`

---

### Task 6: S-matrix driver (TM)

**Files:**
- Create: `meep_torch/ports/smatrix.py`
- Modify: `meep_torch/ports/__init__.py` (export `SMatrix`, `compute_smatrix`, `ModeSourceTM`, decomposition helpers)
- Test: `tests/ports/test_smatrix_tm.py`

**Interfaces:**
- Consumes: everything above plus `FluxMonitorTM`, `run_tm`, `init_tm_state`, `effindex.group_index` (phase-slope test only).
- Produces:
  ```python
  class SMatrix(NamedTuple):
      s: torch.Tensor                  # complex128 (nf, P*N, P*N)
      freqs: torch.Tensor              # (nf,)
      index: list[tuple[int, int]]     # flat row/col -> (port_idx, mode_idx)

  def compute_smatrix(eps: torch.Tensor, grid: Grid, ports: Sequence[Port], *,
                      pol: str, fcen: float, fwidth: float,
                      freqs: Sequence[float], n_steps: int, npml: int,
                      checkpoint_every: int = 0) -> SMatrix
  ```
  Semantics: `eps` is the raster (`(nx, ny)` cell-center permittivity; caller may produce it via `rasterize_scene(...).eps_tm`). For each excited port j: build `ModeSourceTM` (direction = `ports[j].direction`), attach a `FluxMonitorTM` at every port plane (x-normal ports use `x_normal=True, idx=port.index`; y-normal `x_normal=False`), run `n_steps`, decompose each monitor with `axis_sign=+1` for x-normal / `-1` for y-normal. Then:
  - incident at j: `a_in = coeff in direction ports[j].direction` (i.e. `a_plus` if direction=+1 else `a_minus`), mode = source mode;
  - outgoing at any port i: `b_out = coeff in direction -ports[i].direction`;
  - `S[(i,m),(j,n)](f) = b_out[i,m](f) / a_in[j,n](f)`.
  Phase reference plane = monitor plane = `port.index`. Currently one source mode per column: columns are (j, n) for n in `range(ports[j].n_modes)` — run one FDTD per (j, n) pair (source built with `mode=n`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/ports/test_smatrix_tm.py
import math

import pytest
import torch
from meep_torch.core.grid import Grid
from meep_torch.ports.port import Port
from meep_torch.ports.smatrix import compute_smatrix

FCEN, FWIDTH = 0.15, 0.05
FREQS = [0.13, 0.15, 0.17]


@pytest.fixture(scope="module")
def straight():
    grid = Grid(nx=200, ny=80, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    eps[:, 35:45] = 12.0
    ports = [
        Port(axis="x", index=40, lo=20, hi=60, direction=+1),
        Port(axis="x", index=160, lo=20, hi=60, direction=-1),
    ]
    return compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                           freqs=FREQS, n_steps=3000, npml=10), grid


def test_straight_guide_transmission(straight):
    sm, _ = straight
    for i in range(len(FREQS)):
        assert abs(abs(sm.s[i, 1, 0]) - 1.0) < 1e-2
        assert abs(sm.s[i, 0, 0]) < 5e-2          # |S11| small


def test_straight_guide_unitarity(straight):
    sm, _ = straight
    for i in range(len(FREQS)):
        col = sm.s[i, :, 0]
        assert abs(float((col.abs() ** 2).sum()) - 1.0) < 1e-2


def test_phase_slope_matches_group_index(straight):
    sm, grid = straight
    L = (160 - 40) * grid.dx
    ph = torch.tensor([torch.angle(sm.s[i, 1, 0]) for i in range(3)])
    ph = torch.from_numpy(__import__("numpy").unwrap(ph.numpy()))
    slope = float((ph[2] - ph[0]) / (FREQS[2] - FREQS[0]))
    # arg S21 ~ -2*pi*f*neff(f)*L  =>  d(arg)/df ~ -2*pi*ng*L
    ng_est = -slope / (2 * math.pi * L)
    # slab ng for this profile is ~ sqrt(12) bounded; compare against fd1d-based value
    from meep_torch.modes.fd1d import solve_1d
    eps_line = torch.ones(40, dtype=torch.float64); eps_line[15:25] = 12.0
    n1 = float(solve_1d(eps_line, grid.dx, 0.13, "tm", k=1).neff[0])
    n2 = float(solve_1d(eps_line, grid.dx, 0.17, "tm", k=1).neff[0])
    ng_ref = n2 + 0.15 * (n2 - n1) / 0.04  # neff + f dneff/df (crude FD at fcen)
    assert abs(ng_est - ng_ref) / ng_ref < 0.05


def test_reciprocity_taper():
    grid = Grid(nx=240, ny=100, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    # width 1um -> 2um linear taper between x-cells 100..140
    for ix in range(grid.nx):
        w = 10 if ix < 100 else (20 if ix >= 140 else 10 + (ix - 100) * 10 // 40)
        eps[ix, 50 - w // 2 : 50 + (w + 1) // 2] = 12.0
    ports = [
        Port(axis="x", index=40, lo=25, hi=75, direction=+1),
        Port(axis="x", index=200, lo=25, hi=75, direction=-1),
    ]
    sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                         freqs=[FCEN], n_steps=3500, npml=10)
    s21, s12 = sm.s[0, 1, 0], sm.s[0, 0, 1]
    assert abs(abs(s21) - abs(s12)) < 5e-3
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest tests/ports/test_smatrix_tm.py -v` — expect `ImportError`.

- [ ] **Step 3: Implement**

```python
# meep_torch/ports/smatrix.py
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Sequential per-port S-matrix driver (fdtdx utils/sparams.py pattern).

Forward-only. Phase reference plane = each port's monitor plane.
"""
from typing import NamedTuple, Sequence

import torch

from ..core.coeffs import build_tm_coeffs
from ..core.grid import Grid
from ..core.monitors import FluxMonitorTM
from ..core.run import run_tm
from ..core.sources import GaussianPulse
from ..core.stepper import init_tm_state
from .mode_monitor import decompose
from .mode_source import ModeSourceTM
from .port import Port, validate_ports


class SMatrix(NamedTuple):
    s: torch.Tensor
    freqs: torch.Tensor
    index: list


def _eps_line(eps, p: Port):
    return eps[p.index, p.lo:p.hi] if p.axis == "x" else eps[p.lo:p.hi, p.index]


def _monitor(grid, freqs, p: Port):
    return FluxMonitorTM(grid, freqs, x_normal=(p.axis == "x"),
                         idx=p.index, lo=p.lo, hi=p.hi)


def compute_smatrix(eps, grid: Grid, ports: Sequence[Port], *, pol: str,
                    fcen: float, fwidth: float, freqs: Sequence[float],
                    n_steps: int, npml: int, checkpoint_every: int = 0) -> SMatrix:
    if pol != "tm":
        raise NotImplementedError("pol='te' lands in Task 8")
    validate_ports(ports, grid, npml)
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=npml)
    index = [(j, n) for j, p in enumerate(ports) for n in range(p.n_modes)]
    nf, dim = len(freqs), len(index)
    s = torch.zeros(nf, dim, dim, dtype=torch.complex128)

    # ponytail: fixed n_steps; field-decay auto-stop when someone needs it
    for col, (j, n) in enumerate(index):
        src = ModeSourceTM(grid, coeffs, ports[j], GaussianPulse(fcen, fwidth),
                           fcen, mode=n)
        monitors = [_monitor(grid, freqs, p) for p in ports]
        _, accs = run_tm(init_tm_state(grid), coeffs, n_steps, source=src,
                         monitors=monitors, checkpoint_every=checkpoint_every)
        coeff = []
        for p, acc in zip(ports, accs):
            sign = 1.0 if p.axis == "x" else -1.0
            coeff.append(decompose(acc, _eps_line(eps, p), grid.dx, freqs,
                                   "tm", p.n_modes, axis_sign=sign))
        a_p, a_m = coeff[j]
        a_in = a_p[:, n] if ports[j].direction > 0 else a_m[:, n]
        for row, (i, m) in enumerate(index):
            b_p, b_m = coeff[i]
            b_out = b_m[:, m] if ports[i].direction > 0 else b_p[:, m]
            s[:, row, col] = b_out / a_in
    return SMatrix(s=s, freqs=torch.tensor(list(freqs), dtype=torch.float64),
                   index=index)
```

  Note the incident/outgoing selection: for the excited port j the incident wave travels **in** `direction`; every outgoing wave at port i (including the reflection at j itself) travels **against** `ports[i].direction`.

- [ ] **Step 4: Run tests, verify PASS** — `uv run pytest tests/ports/test_smatrix_tm.py -v` (several FDTD runs; expect a few minutes).
- [ ] **Step 5: Run full suite** — `uv run pytest tests/ -q`.
- [ ] **Step 6: Commit** — `git commit -am "feat: TM S-matrix driver with measured-incident normalization"`

---

### Task 7: TE duals

**Files:**
- Modify: `meep_torch/ports/mode_source.py` (add `ModeSourceTE`), `meep_torch/ports/smatrix.py` (pol="te" branch), `meep_torch/ports/__init__.py`
- Test: `tests/ports/test_te_ports.py`

**Interfaces:**
- Consumes: `FluxMonitorTE` (x-normal only — see its `__init__`), `build_te_coeffs`, `run_te`, `init_te_state`, `step_te` H-current (Task 2).
- Produces:
  ```python
  class ModeSourceTE:
      def __init__(self, grid, coeffs, port, profile, fcen, mode=0): ...
      # attrs: mask (nx, ny-1) Dy current ("src_mask_ey" positional), h_masks {"src_hz_mask": (nx-1, ny-1)}
      # methods: amplitude(t_idx), h_amplitude(t_idx)
  ```
  Physics (x-normal only this phase — `FluxMonitorTE` supports only x-normal; TE y-normal raises `NotImplementedError` in both source and `compute_smatrix`):
  - eps line at Ey nodes: `eps_ey = 0.5*(eps[idx, lo:hi] + eps[idx, lo+1:hi+1])` conceptually — concretely average the cell-center raster to half-y nodes: `eps_line = 0.5 * (eps[p.index, p.lo:p.hi] + eps[p.index, p.lo+1:p.hi+1])` with span interpreted on Ey nodes `(ny-1)` indexing; require `hi <= ny - 1`.
  - `fd1d.solve_1d(eps_line, dx, fcen, "te", k=mode+1)` → `Hz_m` profile at half-y nodes, `neff`.
  - `Ey_m = neff * Hz_m / eps_line` (TE impedance relation).
  - Electric current: `mask[index, lo:hi] = Hz_m / dx` into Dy; magnetic current: `hz_mask[col, lo:hi] = d * Ey_m / dx`, col = `index-1` (d=+1) else `index` (Hz half-x column). Same `amplitude`/`h_amplitude` time series as TM but with the TE time conventions: E-current at `(t_idx)*dt + dt/2`... **read `stepper_te.py` and `FluxMonitorTE.sample` first**: TE advances E to `(t_idx+0.5)dt` and Hz to `(t_idx+1)dt`, so `amplitude(t_idx) = g(t_idx*dt)` (drives the E half-step) and `h_amplitude(t_idx) = _H_SIGN * g((t_idx+0.5)*dt - delay)` — the reversed stagger relative to TM. Verify against the one-way test; if the launch is weak or bidirectional, the two time series are the first suspect (single documented decision like Task 4's `_H_SIGN`).
  - `decompose` reuse: TE monitor accumulates (ReEy, ImEy, ReHz, ImHz); pass `pol="te"` and mode profiles `Em = Ey_m = neff*Hz_m/eps_line`, `Hm = Hz_m` — add a `pol` branch inside `decompose` computing `(Em, Hm)` from the fd1d profile accordingly (TM: `Em=prof, Hm=axis_sign*neff*prof`; TE: `Hm=prof, Em=neff*prof/eps_line`, x-normal only).
  - `compute_smatrix(pol="te")`: `build_te_coeffs(grid, eps_inf=eps, npml=npml)`, `ModeSourceTE`, `FluxMonitorTE`, `run_te`, `init_te_state`; source protocol: `source.mask` binds to `src_mask_ey` (positional slot 4 of `step_te` — already correct via Task 3's call shape).

- [ ] **Step 1: Write the failing tests**

```python
# tests/ports/test_te_ports.py
import pytest
import torch
from meep_torch.core.grid import Grid
from meep_torch.ports.port import Port
from meep_torch.ports.smatrix import compute_smatrix

FCEN, FWIDTH = 0.15, 0.05
FREQS = [0.13, 0.15, 0.17]


def test_te_straight_guide():
    grid = Grid(nx=200, ny=80, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    eps[:, 35:45] = 12.0
    ports = [
        Port(axis="x", index=40, lo=20, hi=59, direction=+1),
        Port(axis="x", index=160, lo=20, hi=59, direction=-1),
    ]
    sm = compute_smatrix(eps, grid, ports, pol="te", fcen=FCEN, fwidth=FWIDTH,
                         freqs=FREQS, n_steps=3000, npml=10)
    for i in range(len(FREQS)):
        assert abs(abs(sm.s[i, 1, 0]) - 1.0) < 2e-2
        assert abs(sm.s[i, 0, 0]) < 5e-2


def test_te_y_normal_raises():
    grid = Grid(nx=200, ny=80, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    ports = [Port(axis="y", index=40, lo=20, hi=59, direction=+1)]
    with pytest.raises(NotImplementedError):
        compute_smatrix(eps, grid, ports, pol="te", fcen=FCEN, fwidth=FWIDTH,
                        freqs=FREQS, n_steps=10, npml=10)
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest tests/ports/test_te_ports.py -v` — expect `NotImplementedError: pol='te' lands in Task 8` (note: fix that message string to "Task 7" or drop it when implementing).
- [ ] **Step 3: Implement** per interface block. Read `stepper_te.py` lines 58–105 and `FluxMonitorTE` before writing the time series.
- [ ] **Step 4: Run tests, verify PASS**, then full suite — `uv run pytest tests/ -q`.
- [ ] **Step 5: Commit** — `git commit -am "feat: TE eigenmode source and S-matrix path (x-normal)"`

---

### Task 8: meep reference data + parity gate

**Files:**
- Create: `tests/ports/generate_meep_reference.py`, `tests/ports/data/meep_reference.json`, `tests/ports/test_meep_parity.py`

**Interfaces:**
- Consumes: `compute_smatrix` (Task 6). pymeep via `mamba run -n meep-oracle` (generation time only; JSON is committed so tests never need meep).
- Produces: `meep_reference.json` schema:
  ```json
  {"straight": {"freqs": [...], "S21_mag": [...], "S11_mag": [...]},
   "taper":    {"freqs": [...], "S21_mag": [...], "S11_mag": [...]},
   "provenance": {"meep_version": "1.34.0", "generated": "2026-07-28",
                   "command": "mamba run -n meep-oracle python tests/ports/generate_meep_reference.py"}}
  ```

- [ ] **Step 1: Write the generation script.** Both structures mirror the Task 6 test geometries exactly (same eps, same physical dimensions in µm, same fcen/fwidth/freqs), built with meep's `EigenModeSource` + `add_mode_monitor` + `get_eigenmode_coefficients`:

```python
# tests/ports/generate_meep_reference.py
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Regenerate tests/ports/data/meep_reference.json.

Run: mamba run -n meep-oracle python tests/ports/generate_meep_reference.py
Requires pymeep (conda-forge, env meep-oracle). Not a test-time dependency.
"""
import json
import pathlib

import meep as mp

FCEN, FWIDTH = 0.15, 0.05
FREQS = [0.13, 0.15, 0.17]
NF = len(FREQS)
RES = 10  # cells/um == dx 0.1


def _sim(geom_fn, sx, sy):
    cell = mp.Vector3(sx, sy, 0)
    sim = mp.Simulation(
        cell_size=cell, resolution=RES, boundary_layers=[mp.PML(1.0)],
        geometry=geom_fn(),
        sources=[mp.EigenModeSource(
            mp.GaussianSource(FCEN, fwidth=FWIDTH),
            center=mp.Vector3(-sx / 2 + 4.0, 0), size=mp.Vector3(0, sy - 2),
            eig_parity=mp.ODD_Z,  # TM (Ez) in meep 2D
        )],
    )
    mon_in = sim.add_mode_monitor(
        FCEN, FWIDTH * 0.8, NF,
        mp.ModeRegion(center=mp.Vector3(-sx / 2 + 5.0, 0), size=mp.Vector3(0, sy - 2)))
    mon_out = sim.add_mode_monitor(
        FCEN, FWIDTH * 0.8, NF,
        mp.ModeRegion(center=mp.Vector3(sx / 2 - 4.0, 0), size=mp.Vector3(0, sy - 2)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(sx / 2 - 4.0, 0), 1e-8))
    ci = sim.get_eigenmode_coefficients(mon_in, [1], eig_parity=mp.ODD_Z).alpha
    co = sim.get_eigenmode_coefficients(mon_out, [1], eig_parity=mp.ODD_Z).alpha
    s21 = [abs(co[0, i, 0] / ci[0, i, 0]) for i in range(NF)]
    s11 = [abs(ci[0, i, 1] / ci[0, i, 0]) for i in range(NF)]
    return {"freqs": FREQS, "S21_mag": s21, "S11_mag": s11}


def straight_geom():
    return [mp.Block(mp.Vector3(mp.inf, 1, mp.inf), center=mp.Vector3(),
                     material=mp.Medium(epsilon=12))]


def taper_geom():
    # width 1 -> 2 um linear taper over 4 um centered at origin (matches Task 6 test)
    return [
        mp.Block(mp.Vector3(10, 1, mp.inf), center=mp.Vector3(-7, 0),
                 material=mp.Medium(epsilon=12)),
        mp.Prism([mp.Vector3(-2, -0.5), mp.Vector3(2, -1), mp.Vector3(2, 1),
                  mp.Vector3(-2, 0.5)], height=mp.inf,
                 material=mp.Medium(epsilon=12)),
        mp.Block(mp.Vector3(10, 2, mp.inf), center=mp.Vector3(7, 0),
                 material=mp.Medium(epsilon=12)),
    ]


if __name__ == "__main__":
    out = {
        "straight": _sim(straight_geom, 20, 8),
        "taper": _sim(taper_geom, 24, 10),
        "provenance": {"meep_version": mp.__version__, "generated": "2026-07-28",
                        "command": "mamba run -n meep-oracle python tests/ports/generate_meep_reference.py"},
    }
    p = pathlib.Path(__file__).parent / "data" / "meep_reference.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)
```

- [ ] **Step 2: Generate the data** — `mamba run -n meep-oracle python tests/ports/generate_meep_reference.py`. Inspect JSON: straight `S21_mag ≈ 1` (within ~1%), taper values plausible (S21 near 1, S11 ≪ 1). If meep API names differ (e.g. `alpha` indexing), fix the script against meep 1.34 docs — `alpha` has shape `(num_bands, nf, 2)` with `[:, :, 0]` forward, `[:, :, 1]` backward.
- [ ] **Step 3: Write the parity test** (meep-torch side mirrors the same physical geometry on its grid; hard gate, no skips — JSON is committed):

```python
# tests/ports/test_meep_parity.py
import json
import pathlib

import torch
from meep_torch.core.grid import Grid
from meep_torch.ports.port import Port
from meep_torch.ports.smatrix import compute_smatrix

DATA = json.loads((pathlib.Path(__file__).parent / "data" / "meep_reference.json").read_text())
FCEN, FWIDTH = 0.15, 0.05


def test_straight_matches_meep():
    ref = DATA["straight"]
    grid = Grid(nx=200, ny=80, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    eps[:, 35:45] = 12.0
    ports = [Port(axis="x", index=40, lo=20, hi=60, direction=+1),
             Port(axis="x", index=160, lo=20, hi=60, direction=-1)]
    sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                         freqs=ref["freqs"], n_steps=3000, npml=10)
    for i in range(len(ref["freqs"])):
        assert abs(abs(sm.s[i, 1, 0]) - ref["S21_mag"][i]) < 0.03


def test_taper_matches_meep():
    ref = DATA["taper"]
    grid = Grid(nx=240, ny=100, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    for ix in range(grid.nx):
        # straight w=10 cells for x<10um, taper 10->20 cells over [10,14]um, then 20
        if ix < 100:
            w = 10
        elif ix >= 140:
            w = 20
        else:
            w = 10 + (ix - 100) * 10 // 40
        eps[ix, 50 - w // 2 : 50 + (w + 1) // 2] = 12.0
    ports = [Port(axis="x", index=40, lo=25, hi=75, direction=+1),
             Port(axis="x", index=200, lo=25, hi=75, direction=-1)]
    sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                         freqs=ref["freqs"], n_steps=3500, npml=10)
    for i in range(len(ref["freqs"])):
        assert abs(abs(sm.s[i, 1, 0]) - ref["S21_mag"][i]) < 0.03
```

- [ ] **Step 4: Run parity tests** — `uv run pytest tests/ports/test_meep_parity.py -v`. Expected PASS within 3% absolute on |S21|. If the taper disagrees beyond tolerance while straight agrees, check geometry congruence first (taper cell widths vs meep Prism vertices) — geometry mismatch, not physics, is the usual cause.
- [ ] **Step 5: Commit** — `git add tests/ports && git commit -m "test: meep 1.34 parity gate for TM S-parameters (committed reference data)"`

---

### Task 9: femwell overlap oracle (guarded, test-only)

**Files:**
- Create: `tests/ports/test_femwell_overlap.py`
- Modify: `pyproject.toml` — add `femwell` to the dev/test extras (NOT runtime deps; GPL-3.0 stays out of the wheel)

**Interfaces:**
- Consumes: `fd1d.solve_1d`, `decompose`'s overlap convention. Mirrors the guarded-oracle pattern in the existing Phase 2 femwell parity test (find it: `grep -rl femwell tests/`) — same skip mechanism.

- [ ] **Step 1: Locate and read the Phase 2 femwell test** — `grep -rl femwell tests/` then read it; reuse its import guard and slab-structure helpers verbatim.
- [ ] **Step 2: Write the test** — same slab (eps 12 core, width 1 µm, cladding 1) at fcen 0.15: femwell solves the slab modes; compare (a) neff per mode (tolerance follows the Phase 2 test's), (b) the normalized bi-orthogonal overlap matrix `O_mn = ∫Em·Hn / sqrt(Nm*Nn)` computed from femwell profiles vs fd1d profiles — assert off-diagonals < 1e-6 in both and diagonal ratio within 1e-3. Guard: `pytest.importorskip("femwell")`.
- [ ] **Step 3: Run** — `uv run pytest tests/ports/test_femwell_overlap.py -v` (skips if femwell absent; run once with it installed: `uv pip install femwell && uv run pytest ...`).
- [ ] **Step 4: Commit** — `git commit -am "test: femwell overlap/neff oracle (guarded, GPL dep test-only)"`

---

### Task 10: Final gate

**Files:**
- Modify: `wiki/summaries/meep-torch-roadmap.md` (Phase 4 row → Done + key numbers), `wiki/log.md` (entry), `wiki/summaries/phase3-phase4-outlook.md` (mark Phase 4 done, Phase 3 next)

- [ ] **Step 1: Full suite** — `uv run pytest tests/ -q`. All green.
- [ ] **Step 2: Line-count check** — every `meep_torch/ports/*.py` ≤ 500 lines (`wc -l meep_torch/ports/*.py`).
- [ ] **Step 3: License scan** — `grep -rL "SPDX-License-Identifier: MIT" meep_torch/ports/` returns nothing; confirm no femwell/GPL import outside tests.
- [ ] **Step 4: Update wiki pages** (submodule: commit inside `wiki/`, then bump pointer in superproject). Record: one-way dB ratio, straight-guide |S21| deviation, meep parity deltas, reciprocity delta.
- [ ] **Step 5: Tag** — `git tag v0.3.0-phase4 && git log --oneline -5` to confirm.
- [ ] **Step 6: Commit wiki bump** — `git commit -am "chore: bump wiki submodule (Phase 4 close)"`

---

## Self-review notes (resolved inline)

- Spec coverage: injection (T4/T7), decomposition (T5/T7), S-matrix (T6), invariant gates (T6), meep parity (T8), femwell oracle (T9), error handling (T1 spans/PML, T4 guided+dispersive, T5 norm floor), deferred list untouched.
- `NotImplementedError` message in Task 6 says "Task 8" but TE lands in Task 7 — Task 7 Step 2 notes fixing the string.
- Type consistency: `decompose` signature identical in Tasks 5–7 (`axis_sign` kw, `pol` branch added in T7); `h_masks` dict protocol identical in Tasks 3, 4, 7.
- TE y-normal explicitly out (FluxMonitorTE x-normal only) — matches spec's "axis-aligned" with a documented narrowing; wiki close-out note records it.
