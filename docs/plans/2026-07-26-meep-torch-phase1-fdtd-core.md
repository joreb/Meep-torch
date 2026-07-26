# meep-torch Phase 1: Torch FDTD Core Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A differentiable 2D Yee-grid FDTD engine in PyTorch (TM+TE, CFS-PML, Lorentz-ADE dispersion, Kerr χ³, DFT flux monitors, checkpointed time-stepping) in a new standalone repo `meep-torch`, validated by analytic tests on native Windows.

**Architecture:** Pure-functional stepper (`state in → state out`, no in-place mutation of tracked tensors) driven by a loop that wraps k-step chunks in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. Static per-run coefficients (material, PML, pole) precomputed into a frozen coeffs object closed over by the chunk function. Monitors sample inside chunks and return small partial accumulators as extra chunk outputs.

**Tech Stack:** Python ≥3.10, torch ≥2.6 (first version with Windows cp313 wheels — host has Python 3.13.12), numpy; dev: pytest, psutil. Managed with uv (on PATH via chocolatey). No C++/SWIG/scipy.

**Spec:** `docs/superpowers/specs/2026-07-26-meep-torch-phase1-fdtd-core-design.md` (in D:\Github\meep). One approved deviation: no scipy Sellmeier fitting — we port meep's already-fitted Lorentz-pole constants from `D:\Github\meep\python\materials.py` verbatim (better parity, one less dep).

## Global Constraints

- New repo lives at `D:\Github\meep-torch`, package name `meep_torch`. All Files paths below are relative to that repo (except the parity-reference paths into D:\Github\meep).
- Units: meep natural units. c = 1, ε₀ = μ₀ = 1, lengths in µm, frequency f in units of 1/λ[µm] (so 1.55 µm ⇒ f = 0.6452). Angular quantities: ω = 2πf.
- Yee grid, square cells (dx = dy). Default Courant factor 0.5 (meep's default); constructor rejects courant > 1/√2 (2D CFL limit).
- Field staggering (TM): Ez(i,j) at integer nodes, shape (nx,ny); Hx(i,j+½) shape (nx,ny−1); Hy(i+½,j) shape (nx−1,ny). Outer x-boundary is always PEC (frozen Ez=0 boundary rows); y-boundary is PEC or PMC per `y_pmc` flag.
- NEVER mutate autograd-tracked state tensors in place. Every stepper output is a fresh tensor. Slice-assign only into freshly created zero tensors, never into state.
- All physics-correctness tests run in float64. float32 is default runtime dtype.
- Checkpointing always `use_reentrant=False` (required: supports closure-captured tensors getting gradients).
- Lorentz susceptibility convention (meep's): χ(f) = σ fₙ²/(fₙ² − f² − i f γ). ADE ODE: P̈ + ΓṖ + ω₀²P = ω₀²σE with ω₀ = 2πfₙ, Γ = 2πγ.
- Tests: `uv run pytest` from repo root. Slow tests marked `@pytest.mark.slow`, excluded by default via addopts.
- Commit style: plain conventional commits (`feat:`, `test:`, `chore:`). No Co-Authored-By trailers.

## File Structure (final state)

```
D:\Github\meep-torch\
  pyproject.toml
  README.md
  docs\specs\2026-07-26-meep-torch-phase1-fdtd-core-design.md   (copied from meep repo)
  docs\plans\2026-07-26-meep-torch-phase1-fdtd-core.md          (this file, copied)
  meep_torch\
    __init__.py
    core\
      __init__.py
      grid.py        # Grid dataclass, dt/CFL
      pml.py         # CPML 1D profile builder
      materials.py   # LorentzPole, Material, cSi/SiO2/Si3N4 constants, analytic_eps
      coeffs.py      # TMCoeffs/TECoeffs builders (static per-run tensors)
      stepper.py     # TMState, step_tm
      stepper_te.py  # TEState, step_te
      sources.py     # GaussianPulse, CWRamp time profiles + LineSource mask
      monitors.py    # FluxMonitorTM/TE, PointProbe
      run.py         # run_tm / run_te checkpointed loops
  tests\
    analytic\
      test_energy.py test_propagation.py test_pml.py test_flux.py
      test_dispersion.py test_materials.py test_kerr.py test_te.py
    autodiff\
      test_gradcheck.py test_memory.py test_float32.py
    parity\
      test_meep_parity.py
    test_smoke.py
```

---

### Task 1: Repo scaffold

**Files:**
- Create: `pyproject.toml`, `README.md`, `meep_torch/__init__.py`, `meep_torch/core/__init__.py`, `tests/test_smoke.py`
- Copy: spec + this plan into `docs/specs/` and `docs/plans/`

**Interfaces:**
- Produces: importable package `meep_torch`, `meep_torch.__version__ == "0.1.0"`, working `uv run pytest`.

- [ ] **Step 1: Create repo and scaffold**

```powershell
New-Item -ItemType Directory -Force D:\Github\meep-torch\meep_torch\core, D:\Github\meep-torch\tests, D:\Github\meep-torch\docs\specs, D:\Github\meep-torch\docs\plans
Set-Location D:\Github\meep-torch
git init
Copy-Item D:\Github\meep\docs\superpowers\specs\2026-07-26-meep-torch-phase1-fdtd-core-design.md docs\specs\
Copy-Item D:\Github\meep\docs\superpowers\plans\2026-07-26-meep-torch-phase1-fdtd-core.md docs\plans\
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "meep-torch"
version = "0.1.0"
description = "Differentiable 2D FDTD engine in PyTorch (meep-inspired)"
requires-python = ">=3.10"
dependencies = ["torch>=2.6", "numpy>=1.24"]

[dependency-groups]
dev = ["pytest>=8", "psutil>=5.9"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["meep_torch"]

[tool.pytest.ini_options]
markers = ["slow: long-running tests, excluded by default"]
addopts = "-m 'not slow'"
testpaths = ["tests"]
```

- [ ] **Step 3: Write package init and smoke test**

`meep_torch/__init__.py`:
```python
__version__ = "0.1.0"
```
`meep_torch/core/__init__.py`: empty file.

`tests/test_smoke.py`:
```python
import meep_torch


def test_import():
    assert meep_torch.__version__ == "0.1.0"


def test_torch_available():
    import torch

    assert torch.tensor([1.0]).sum().item() == 1.0
```

`README.md` (3 lines): name, one-sentence description, `uv sync && uv run pytest`.

- [ ] **Step 4: Install and run**

Run: `uv sync && uv run pytest`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add -A; git commit -m "chore: scaffold meep-torch package with uv/pytest"
```

---

### Task 2: Grid and TM state

**Files:**
- Create: `meep_torch/core/grid.py`, `meep_torch/core/stepper.py` (state only), `tests/analytic/test_grid.py`

**Interfaces:**
- Produces:
  - `Grid(nx: int, ny: int, dx: float, courant: float = 0.5, dtype=torch.float64, device="cpu")`, property `dt = courant * dx`.
  - `TMState(NamedTuple)` fields in order: `Ez, Hx, Hy, Dz, P_cur, P_prev, psi_Ezy, psi_Ezx, psi_Hyx, psi_Hxy` (all `torch.Tensor`).
  - `init_tm_state(grid: Grid, n_poles: int = 0, y_pmc: bool = False) -> TMState` (zeros, correct shapes).

- [ ] **Step 1: Write failing tests**

`tests/analytic/test_grid.py`:
```python
import pytest
import torch

from meep_torch.core.grid import Grid
from meep_torch.core.stepper import TMState, init_tm_state


def test_dt_from_courant():
    g = Grid(nx=10, ny=10, dx=0.02)
    assert g.dt == pytest.approx(0.01)


def test_cfl_rejected():
    with pytest.raises(ValueError, match="courant"):
        Grid(nx=10, ny=10, dx=0.02, courant=0.8)


def test_tiny_grid_rejected():
    with pytest.raises(ValueError, match="nx"):
        Grid(nx=2, ny=10, dx=0.02)


def test_state_shapes_pec():
    g = Grid(nx=8, ny=6, dx=0.02)
    s = init_tm_state(g, n_poles=2)
    assert s.Ez.shape == (8, 6)
    assert s.Hx.shape == (8, 5)
    assert s.Hy.shape == (7, 6)
    assert s.Dz.shape == (8, 6)
    assert s.P_cur.shape == (2, 8, 6)
    assert s.psi_Hyx.shape == (6, 4)   # (nx-2, ny-2) PEC-y
    assert s.psi_Hxy.shape == (6, 4)
    assert s.Ez.dtype == torch.float64


def test_state_shapes_pmc_y():
    g = Grid(nx=8, ny=6, dx=0.02)
    s = init_tm_state(g, y_pmc=True)
    assert s.psi_Hyx.shape == (6, 6)   # full ny when y is PMC
    assert s.psi_Hxy.shape == (6, 6)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/analytic/test_grid.py -v` — Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

`meep_torch/core/grid.py`:
```python
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    dx: float
    courant: float = 0.5
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self):
        if self.nx < 3:
            raise ValueError(f"nx must be >= 3, got {self.nx}")
        if self.ny < 3:
            raise ValueError(f"ny must be >= 3, got {self.ny}")
        if self.dx <= 0:
            raise ValueError(f"dx must be > 0, got {self.dx}")
        if not 0 < self.courant <= 1 / math.sqrt(2):
            raise ValueError(
                f"courant must be in (0, 1/sqrt(2)] for 2D stability, got {self.courant}"
            )

    @property
    def dt(self) -> float:
        return self.courant * self.dx

    def zeros(self, *shape) -> torch.Tensor:
        return torch.zeros(*shape, dtype=self.dtype, device=self.device)
```

`meep_torch/core/stepper.py` (state part; step_tm comes in Task 4):
```python
from typing import NamedTuple

import torch

from .grid import Grid


class TMState(NamedTuple):
    Ez: torch.Tensor       # (nx, ny)        E_z at time n
    Hx: torch.Tensor       # (nx, ny-1)      at n-1/2
    Hy: torch.Tensor       # (nx-1, ny)      at n-1/2
    Dz: torch.Tensor       # (nx, ny)
    P_cur: torch.Tensor    # (n_poles, nx, ny)
    P_prev: torch.Tensor   # (n_poles, nx, ny)
    psi_Ezy: torch.Tensor  # (nx, ny-1)      CPML memory, dEz/dy in Hx update
    psi_Ezx: torch.Tensor  # (nx-1, ny)      dEz/dx in Hy update
    psi_Hyx: torch.Tensor  # (nx-2, ny-2) or (nx-2, ny) if y_pmc
    psi_Hxy: torch.Tensor  # same shape as psi_Hyx


def init_tm_state(grid: Grid, n_poles: int = 0, y_pmc: bool = False) -> TMState:
    nx, ny = grid.nx, grid.ny
    ny_e = ny if y_pmc else ny - 2
    return TMState(
        Ez=grid.zeros(nx, ny),
        Hx=grid.zeros(nx, ny - 1),
        Hy=grid.zeros(nx - 1, ny),
        Dz=grid.zeros(nx, ny),
        P_cur=grid.zeros(n_poles, nx, ny),
        P_prev=grid.zeros(n_poles, nx, ny),
        psi_Ezy=grid.zeros(nx, ny - 1),
        psi_Ezx=grid.zeros(nx - 1, ny),
        psi_Hyx=grid.zeros(nx - 2, ny_e),
        psi_Hxy=grid.zeros(nx - 2, ny_e),
    )
```

- [ ] **Step 4: Run tests** — Expected: all PASS.

- [ ] **Step 5: Commit** — `git add -A; git commit -m "feat: Grid with CFL validation and TM field state"`

---

### Task 3: CPML profiles

**Files:**
- Create: `meep_torch/core/pml.py`, `tests/analytic/test_pml_profiles.py`

**Interfaces:**
- Produces: `cpml_profiles(n: int, npml: int, offset: float, dt: float, dx: float, m: int = 3, R0: float = 1e-6, alpha_max: float = 0.05, dtype=torch.float64, device="cpu") -> tuple[Tensor, Tensor, Tensor]` returning `(b, a, kappa)` 1D length-n tensors for field positions `x_i = i + offset` (offset 0.0 integer, 0.5 half-integer). PML occupies `npml` cells at both ends. κ = 1 everywhere in v1 (kept in the interface so stretching can be added without touching the stepper).
- CPML recursion used by steppers: `psi' = b*psi + a*deriv`, effective derivative `deriv/kappa + psi'`.

- [ ] **Step 1: Write failing tests**

`tests/analytic/test_pml_profiles.py`:
```python
import torch

from meep_torch.core.pml import cpml_profiles


def test_interior_is_identity():
    b, a, k = cpml_profiles(n=100, npml=10, offset=0.0, dt=0.01, dx=0.02)
    # interior: sigma=0, alpha=0 -> b=1, a=0, kappa=1
    assert torch.all(b[20:80] == 1.0)
    assert torch.all(a[20:80] == 0.0)
    assert torch.all(k == 1.0)


def test_pml_ranges():
    b, a, _ = cpml_profiles(n=100, npml=10, offset=0.0, dt=0.01, dx=0.02)
    assert torch.all(b[:10] < 1.0) and torch.all(b[:10] > 0.0)
    assert torch.all(a[:5] < 0.0)          # a = sigma*(b-1)/... < 0 where sigma>0
    assert torch.allclose(b[:10], b[-10:].flip(0))  # symmetric ends


def test_npml_zero_degenerates():
    b, a, k = cpml_profiles(n=50, npml=0, offset=0.5, dt=0.01, dx=0.02)
    assert torch.all(b == 1.0) and torch.all(a == 0.0) and torch.all(k == 1.0)
```

- [ ] **Step 2: Run to verify failure** — Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

`meep_torch/core/pml.py`:
```python
import math

import torch


def cpml_profiles(
    n: int,
    npml: int,
    offset: float,
    dt: float,
    dx: float,
    m: int = 3,
    R0: float = 1e-6,
    alpha_max: float = 0.05,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
):
    """CFS-PML (CPML) 1D coefficient profiles at positions i + offset.

    Returns (b, a, kappa), each shape (n,). Recursion: psi' = b*psi + a*d,
    effective derivative = d/kappa + psi'.
    """
    x = torch.arange(n, dtype=dtype, device=device) + offset
    if npml > 0:
        u = torch.clamp((npml - x) / npml, min=0.0) + torch.clamp(
            (x - (n - 1 - npml)) / npml, min=0.0
        )
        u = torch.clamp(u, max=1.0)
    else:
        u = torch.zeros(n, dtype=dtype, device=device)
    sigma_max = 0.0
    if npml > 0:
        sigma_max = -(m + 1) * math.log(R0) / (2.0 * npml * dx)
    sigma = sigma_max * u**m
    alpha = alpha_max * (1.0 - u) * (u > 0)
    kappa = torch.ones_like(u)
    b = torch.exp(-(sigma / kappa + alpha) * dt)
    denom = kappa * (sigma + kappa * alpha)
    a = torch.where(sigma > 0, sigma * (b - 1.0) / denom.clamp(min=1e-300), torch.zeros_like(b))
    return b, a, kappa
```

- [ ] **Step 4: Run tests** — Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat: CFS-PML (CPML) coefficient profiles"`

---

### Task 4: TM coeffs builder + stepper (vacuum physics)

**Files:**
- Create: `meep_torch/core/coeffs.py`, `tests/analytic/test_energy.py`, `tests/analytic/test_propagation.py`
- Modify: `meep_torch/core/stepper.py` (add `step_tm`)

**Interfaces:**
- Consumes: `Grid`, `TMState`, `init_tm_state`, `cpml_profiles`.
- Produces:
  - `TMCoeffs(NamedTuple)`: `dt, dx: float`; `eps_inf: Tensor (nx,ny)`; `chi3: Tensor (nx,ny)`; `has_kerr: bool`; `c1, c2, c3: Tensor (n_poles,nx,ny)`; `y_pmc: bool`; PML tensors `b_hy,a_hy,k_hy (1,ny-1)`, `b_hx,a_hx,k_hx (nx-1,1)`, `b_ex,a_ex,k_ex (nx-2,1)`, `b_ey,a_ey,k_ey (1,ny_e)` where `ny_e = ny if y_pmc else ny-2`.
  - `build_tm_coeffs(grid, eps_inf=None, poles=(), pole_masks=None, chi3=None, npml=0, y_pmc=False) -> TMCoeffs`. `poles` = sequence of `LorentzPole` (Task 8 defines the dataclass; until then a 3-tuple `(fn, gamma, sigma)` is accepted — builder uses index access so both work). `eps_inf`/`chi3` accept float or (nx,ny) tensor; None → 1.0 / 0.0.
  - `step_tm(state: TMState, m: TMCoeffs, src_amp: float | Tensor = 0.0, src_mask: Tensor | None = None) -> TMState` — one full leapfrog step (H then D then P then E). Source: `Dz += dt * src_amp * src_mask`.

- [ ] **Step 1: Write failing tests**

`tests/analytic/test_energy.py`:
```python
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper import init_tm_state, step_tm


def gaussian_blob(g, x0, y0, w):
    x = torch.arange(g.nx, dtype=g.dtype)[:, None]
    y = torch.arange(g.ny, dtype=g.dtype)[None, :]
    return torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * w**2))


def total_energy(s, m, dx):
    e = 0.5 * (m.eps_inf * s.Ez**2).sum()
    h = 0.5 * (s.Hx**2).sum() + 0.5 * (s.Hy**2).sum()
    return ((e + h) * dx * dx).item()


def test_energy_conserved_in_pec_box():
    g = Grid(nx=120, ny=120, dx=0.02)
    m = build_tm_coeffs(g, npml=0)
    s = init_tm_state(g)
    s = s._replace(Ez=gaussian_blob(g, 60.0, 60.0, 6.0), Dz=gaussian_blob(g, 60.0, 60.0, 6.0))
    energies = []
    for i in range(2000):
        s = step_tm(s, m)
        if i % 50 == 0:
            energies.append(total_energy(s, m, g.dx))
    e = torch.tensor(energies)
    assert (e.max() - e.min()) / e.mean() < 1e-3          # bounded oscillation only
    assert abs(e[-5:].mean() - e[:5].mean()) / e.mean() < 1e-4  # no drift
    assert torch.isfinite(e).all()
```

`tests/analytic/test_propagation.py` (quasi-1D via y_pmc):
```python
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper import init_tm_state, step_tm


def plane_pulse_state(g, x0, w, y_pmc=True):
    s = init_tm_state(g, y_pmc=y_pmc)
    x = torch.arange(g.nx, dtype=g.dtype)
    ez = torch.exp(-((x - x0) ** 2) / (2 * w**2))
    Ez = ez[:, None].repeat(1, g.ny)
    # +x propagating: Hy = -Ez interpolated to half-grid
    Hy = -0.5 * (Ez[1:, :] + Ez[:-1, :])
    return s._replace(Ez=Ez, Dz=Ez.clone(), Hy=Hy)


def test_pulse_speed_is_c():
    g = Grid(nx=1000, ny=8, dx=0.02)
    m = build_tm_coeffs(g, npml=0, y_pmc=True)
    s = plane_pulse_state(g, x0=150.0, w=15.0)
    n_steps = 1200  # travels 1200*0.5 = 600 cells
    for _ in range(n_steps):
        s = step_tm(s, m)
    peak = s.Ez[:, 4].abs().argmax().item()
    travelled_cells = peak - 150
    expected_cells = n_steps * g.courant  # c=1: distance = n*dt = n*courant*dx
    assert abs(travelled_cells - expected_cells) / expected_cells < 0.01


def test_pulse_is_unidirectional():
    g = Grid(nx=1000, ny=8, dx=0.02)
    m = build_tm_coeffs(g, npml=0, y_pmc=True)
    s = plane_pulse_state(g, x0=500.0, w=15.0)
    for _ in range(400):
        s = step_tm(s, m)
    left = s.Ez[:400, 4].abs().max().item()
    right = s.Ez[600:, 4].abs().max().item()
    assert left < 0.05 * right
```

- [ ] **Step 2: Run to verify failure** — Expected: ImportError (no `build_tm_coeffs` / `step_tm`).

- [ ] **Step 3: Implement coeffs builder**

`meep_torch/core/coeffs.py`:
```python
import math
from typing import NamedTuple, Sequence

import torch

from .grid import Grid
from .pml import cpml_profiles


class TMCoeffs(NamedTuple):
    dt: float
    dx: float
    eps_inf: torch.Tensor
    chi3: torch.Tensor
    has_kerr: bool
    c1: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor
    y_pmc: bool
    b_hy: torch.Tensor
    a_hy: torch.Tensor
    k_hy: torch.Tensor
    b_hx: torch.Tensor
    a_hx: torch.Tensor
    k_hx: torch.Tensor
    b_ex: torch.Tensor
    a_ex: torch.Tensor
    k_ex: torch.Tensor
    b_ey: torch.Tensor
    a_ey: torch.Tensor
    k_ey: torch.Tensor


def _as_field(val, grid: Grid, default: float) -> torch.Tensor:
    if val is None:
        val = default
    if isinstance(val, torch.Tensor):
        if val.shape != (grid.nx, grid.ny):
            raise ValueError(f"expected shape {(grid.nx, grid.ny)}, got {tuple(val.shape)}")
        return val.to(dtype=grid.dtype, device=grid.device)
    return torch.full((grid.nx, grid.ny), float(val), dtype=grid.dtype, device=grid.device)


def build_tm_coeffs(
    grid: Grid,
    eps_inf=None,
    poles: Sequence = (),
    pole_masks: Sequence[torch.Tensor] | None = None,
    chi3=None,
    npml: int = 0,
    y_pmc: bool = False,
) -> TMCoeffs:
    dt, dx = grid.dt, grid.dx
    eps = _as_field(eps_inf, grid, 1.0)
    x3 = _as_field(chi3, grid, 0.0)
    has_kerr = bool((x3 != 0).any())

    n_poles = len(poles)
    c1 = grid.zeros(n_poles, grid.nx, grid.ny)
    c2 = grid.zeros(n_poles, grid.nx, grid.ny)
    c3 = grid.zeros(n_poles, grid.nx, grid.ny)
    for k, pole in enumerate(poles):
        fn, gamma, sigma = pole[0], pole[1], pole[2]
        w0 = 2 * math.pi * fn
        G = 2 * math.pi * gamma
        den = 1.0 + G * dt / 2.0
        mask = pole_masks[k] if pole_masks is not None else torch.ones_like(eps)
        c1[k] = (2.0 - w0 * w0 * dt * dt) / den * torch.ones_like(eps)
        c2[k] = -(1.0 - G * dt / 2.0) / den * torch.ones_like(eps)
        c3[k] = (sigma * w0 * w0 * dt * dt / den) * mask.to(eps)

    if npml * 2 >= grid.nx or (npml * 2 >= grid.ny and not y_pmc):
        raise ValueError(f"npml={npml} too large for grid {grid.nx}x{grid.ny}")

    def prof(n, offset, use_pml=True):
        return cpml_profiles(
            n, npml if use_pml else 0, offset, dt, dx, dtype=grid.dtype, device=grid.device
        )

    # H updates: dEz/dy at half-y (ny-1); dEz/dx at half-x (nx-1)
    b_hy, a_hy, k_hy = prof(grid.ny - 1, 0.5, use_pml=not y_pmc)
    b_hx, a_hx, k_hx = prof(grid.nx - 1, 0.5)
    # D update: dHy/dx at interior integer x (positions 1..nx-2)
    b_ex_f, a_ex_f, k_ex_f = prof(grid.nx, 0.0)
    b_ex, a_ex, k_ex = b_ex_f[1:-1], a_ex_f[1:-1], k_ex_f[1:-1]
    # dHx/dy at integer y: interior (PEC) or full (PMC)
    b_ey_f, a_ey_f, k_ey_f = prof(grid.ny, 0.0, use_pml=not y_pmc)
    if y_pmc:
        b_ey, a_ey, k_ey = b_ey_f, a_ey_f, k_ey_f
    else:
        b_ey, a_ey, k_ey = b_ey_f[1:-1], a_ey_f[1:-1], k_ey_f[1:-1]

    return TMCoeffs(
        dt=dt, dx=dx, eps_inf=eps, chi3=x3, has_kerr=has_kerr,
        c1=c1, c2=c2, c3=c3, y_pmc=y_pmc,
        b_hy=b_hy[None, :], a_hy=a_hy[None, :], k_hy=k_hy[None, :],
        b_hx=b_hx[:, None], a_hx=a_hx[:, None], k_hx=k_hx[:, None],
        b_ex=b_ex[:, None], a_ex=a_ex[:, None], k_ex=k_ex[:, None],
        b_ey=b_ey[None, :], a_ey=a_ey[None, :], k_ey=k_ey[None, :],
    )
```

- [ ] **Step 4: Implement step_tm** (append to `meep_torch/core/stepper.py`)

```python
import torch.nn.functional as F

from .coeffs import TMCoeffs


def step_tm(
    state: TMState,
    m: TMCoeffs,
    src_amp=0.0,
    src_mask: torch.Tensor | None = None,
) -> TMState:
    dt, dx = m.dt, m.dx

    # --- H update (from Ez at time n) ---
    dEz_dy = (state.Ez[:, 1:] - state.Ez[:, :-1]) / dx            # (nx, ny-1)
    psi_Ezy = m.b_hy * state.psi_Ezy + m.a_hy * dEz_dy
    Hx = state.Hx - dt * (dEz_dy / m.k_hy + psi_Ezy)

    dEz_dx = (state.Ez[1:, :] - state.Ez[:-1, :]) / dx            # (nx-1, ny)
    psi_Ezx = m.b_hx * state.psi_Ezx + m.a_hx * dEz_dx
    Hy = state.Hy + dt * (dEz_dx / m.k_hx + psi_Ezx)

    # --- D update (from H at n+1/2) ---
    if m.y_pmc:
        Hx_pad = F.pad(Hx, (1, 1))                                # PMC: H=0 beyond wall
        dHx_dy = (Hx_pad[1:-1, 1:] - Hx_pad[1:-1, :-1]) / dx      # (nx-2, ny)
        dHy_dx = (Hy[1:, :] - Hy[:-1, :]) / dx                    # (nx-2, ny)
        pad_spec = (0, 0, 1, 1)                                   # pad x only
    else:
        dHx_dy = (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dx              # (nx-2, ny-2)
        dHy_dx = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx              # (nx-2, ny-2)
        pad_spec = (1, 1, 1, 1)

    psi_Hyx = m.b_ex * state.psi_Hyx + m.a_ex * dHy_dx
    psi_Hxy = m.b_ey * state.psi_Hxy + m.a_ey * dHx_dy
    curl = dHy_dx / m.k_ex + psi_Hyx - dHx_dy / m.k_ey - psi_Hxy
    Dz = state.Dz + dt * F.pad(curl, pad_spec)
    if src_mask is not None:
        Dz = Dz + dt * src_amp * src_mask

    # --- ADE Lorentz poles: P(n+1) = c1 P(n) + c2 P(n-1) + c3 Ez(n) ---
    P_new = m.c1 * state.P_cur + m.c2 * state.P_prev + m.c3 * state.Ez

    # --- E from D (instantaneous Kerr via fixed point) ---
    rhs = Dz - P_new.sum(0)
    Ez = rhs / m.eps_inf
    if m.has_kerr:
        for _ in range(3):
            Ez = rhs / (m.eps_inf + m.chi3 * Ez * Ez)

    return TMState(Ez, Hx, Hy, Dz, P_new, state.P_cur, psi_Ezy, psi_Ezx, psi_Hyx, psi_Hxy)
```

- [ ] **Step 5: Run tests** — `uv run pytest tests/analytic/test_energy.py tests/analytic/test_propagation.py -v` — Expected: 3 PASS. If speed test fails marginally, check peak indexing off-by-one before touching physics.

- [ ] **Step 6: Commit** — `git commit -am "feat: TM Yee stepper with CPML hooks, ADE poles, Kerr fixed-point"`

---

### Task 5: Sources

**Files:**
- Create: `meep_torch/core/sources.py`, `tests/analytic/test_sources.py`

**Interfaces:**
- Consumes: `Grid`.
- Produces:
  - `GaussianPulse(fcen: float, fwidth: float, cutoff: float = 5.0)` with `.amplitude(t: float) -> float` = `cos(2π·fcen·(t−t0))·exp(−(t−t0)²/(2w²))`, `w = 1/fwidth`, `t0 = cutoff·w`; `.t_end = 2·t0`.
  - `CWRamp(fcen: float, t_ramp: float)` with `.amplitude(t) = sin(2π·fcen·t)·min(1, t/t_ramp)`.
  - `LineSource(grid, ix: int, j0: int, j1: int, profile) -> .mask (nx,ny) Tensor` (1.0 on the segment), `.amplitude(t_idx) = profile.amplitude((t_idx + 0.5) * grid.dt)`.

- [ ] **Step 1: Write failing tests**

`tests/analytic/test_sources.py`:
```python
import math

import torch

from meep_torch.core.grid import Grid
from meep_torch.core.sources import CWRamp, GaussianPulse, LineSource


def test_gaussian_peak_and_decay():
    p = GaussianPulse(fcen=0.65, fwidth=0.1)
    t0 = 5.0 / 0.1
    assert p.amplitude(t0) == 1.0                       # cos(0)*exp(0)
    assert abs(p.amplitude(0.0)) < 1e-5                 # cut off at start
    assert abs(p.amplitude(2 * t0)) < 1e-5


def test_gaussian_spectrum_peaks_at_fcen():
    p = GaussianPulse(fcen=0.65, fwidth=0.1)
    dt = 0.01
    n = int(p.t_end / dt) + 1
    ts = torch.arange(n, dtype=torch.float64) * dt
    sig = torch.tensor([p.amplitude(float(t)) for t in ts], dtype=torch.float64)
    spec = torch.fft.rfft(sig).abs()
    freqs = torch.fft.rfftfreq(n, dt)
    assert abs(freqs[spec.argmax()].item() - 0.65) < 0.02


def test_line_source_mask():
    g = Grid(nx=20, ny=10, dx=0.02)
    src = LineSource(g, ix=5, j0=2, j1=8, profile=GaussianPulse(0.65, 0.1))
    assert src.mask.shape == (20, 10)
    assert src.mask[5, 2:8].sum().item() == 6.0
    assert src.mask.sum().item() == 6.0


def test_cw_ramp():
    p = CWRamp(fcen=0.5, t_ramp=10.0)
    assert abs(p.amplitude(5.0)) <= 0.5 + 1e-12          # still ramping
    assert p.amplitude(20.25) == math.sin(2 * math.pi * 0.5 * 20.25) * 1.0
```

- [ ] **Step 2: Run** — Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

`meep_torch/core/sources.py`:
```python
import math

import torch

from .grid import Grid


class GaussianPulse:
    def __init__(self, fcen: float, fwidth: float, cutoff: float = 5.0):
        self.fcen = fcen
        self.w = 1.0 / fwidth
        self.t0 = cutoff * self.w
        self.t_end = 2.0 * self.t0

    def amplitude(self, t: float) -> float:
        return math.cos(2 * math.pi * self.fcen * (t - self.t0)) * math.exp(
            -((t - self.t0) ** 2) / (2 * self.w**2)
        )


class CWRamp:
    def __init__(self, fcen: float, t_ramp: float):
        self.fcen = fcen
        self.t_ramp = t_ramp

    def amplitude(self, t: float) -> float:
        ramp = min(1.0, t / self.t_ramp)
        return math.sin(2 * math.pi * self.fcen * t) * ramp


class LineSource:
    """Soft current source on the vertical segment x=ix, y in [j0, j1)."""

    def __init__(self, grid: Grid, ix: int, j0: int, j1: int, profile):
        if not (0 <= ix < grid.nx and 0 <= j0 < j1 <= grid.ny):
            raise ValueError(f"source segment out of bounds: ix={ix}, j0={j0}, j1={j1}")
        self.profile = profile
        self.dt = grid.dt
        mask = grid.zeros(grid.nx, grid.ny)
        mask[ix, j0:j1] = 1.0
        self.mask = mask

    def amplitude(self, t_idx: int) -> float:
        return self.profile.amplitude((t_idx + 0.5) * self.dt)
```

- [ ] **Step 4: Run tests** — Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat: Gaussian pulse, CW ramp, soft line source"`

---

### Task 6: PML absorption test

**Files:**
- Create: `tests/analytic/test_pml.py`

**Interfaces:**
- Consumes: everything from Tasks 2–5. No new production code expected; this task validates the CPML already wired into `step_tm`. If it fails, the bug is in `coeffs.py` profile placement or `step_tm` psi indexing.

- [ ] **Step 1: Write the test**

`tests/analytic/test_pml.py`:
```python
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper import init_tm_state, step_tm
from tests.analytic.test_energy import total_energy
from tests.analytic.test_propagation import plane_pulse_state


def test_pml_absorbs_below_minus_40db():
    g = Grid(nx=800, ny=8, dx=0.02)
    npml = 40
    m = build_tm_coeffs(g, npml=npml, y_pmc=True)
    s = plane_pulse_state(g, x0=400.0, w=15.0)
    e0 = total_energy(s, m, g.dx)
    # pulse reaches +x PML after ~360 cells -> 720 steps; run 3x for full decay
    for _ in range(2400):
        s = step_tm(s, m)
    e1 = total_energy(s, m, g.dx)
    assert e1 / e0 < 1e-4          # -40 dB energy


def test_pec_reflects_without_pml():
    g = Grid(nx=800, ny=8, dx=0.02)
    m = build_tm_coeffs(g, npml=0, y_pmc=True)
    s = plane_pulse_state(g, x0=400.0, w=15.0)
    e0 = total_energy(s, m, g.dx)
    for _ in range(2400):
        s = step_tm(s, m)
    e1 = total_energy(s, m, g.dx)
    assert e1 / e0 > 0.99          # sanity contrast: PEC box keeps energy
```

- [ ] **Step 2: Run** — `uv run pytest tests/analytic/test_pml.py -v` — Expected: PASS (implementation exists from Task 4). If absorption test fails: verify `a` sign in pml.py, psi shapes vs profile shapes in coeffs.py, and that `use_pml=not y_pmc` disabled y-PML.

- [ ] **Step 3: Commit** — `git commit -am "test: CPML broadband absorption below -40dB energy"`

---

### Task 7: Checkpointed run loop + monitor protocol

**Files:**
- Create: `meep_torch/core/run.py`, `meep_torch/core/monitors.py` (PointProbe only), `tests/autodiff/test_checkpoint_equivalence.py`

**Interfaces:**
- Consumes: `TMState`, `step_tm`, `TMCoeffs`, sources.
- Produces:
  - Monitor protocol: `.sample(state, t_idx) -> Tensor`, attribute `.reduce` in `{"sum", "stack"}`.
  - `PointProbe(ix: int, iy: int)` — `reduce="stack"`, sample returns scalar tensor `state.Ez[ix, iy]`.
  - `run_tm(state, coeffs, n_steps, *, source=None, monitors=(), checkpoint_every=0) -> tuple[TMState, list[Tensor]]`. Results list ordered like monitors: "sum" → accumulated tensor; "stack" → tensor of per-step samples stacked on dim 0 (length n_steps).
  - `checkpoint_every=0` disables checkpointing; `k>0` wraps every k-step chunk in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.

- [ ] **Step 1: Write failing test**

`tests/autodiff/test_checkpoint_equivalence.py`:
```python
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state


def make_sim():
    g = Grid(nx=64, ny=48, dx=0.02)
    m = build_tm_coeffs(
        g, eps_inf=2.25, poles=[(2.0, 0.1, 1.5)], chi3=0.01, npml=8
    )
    src = LineSource(g, ix=20, j0=16, j1=32, profile=GaussianPulse(0.65, 0.2))
    return g, m, src


def test_checkpointed_equals_uncheckpointed():
    g, m, src = make_sim()
    probes = [PointProbe(40, 24)]
    s0 = init_tm_state(g, n_poles=1)
    sA, rA = run_tm(s0, m, 60, source=src, monitors=probes, checkpoint_every=0)
    s0 = init_tm_state(g, n_poles=1)
    sB, rB = run_tm(s0, m, 60, source=src, monitors=probes, checkpoint_every=7)
    for a, b in zip(sA, sB):
        assert torch.allclose(a, b, atol=1e-12, rtol=0)
    assert torch.allclose(rA[0], rB[0], atol=1e-12, rtol=0)
    assert rA[0].shape == (60,)


def test_gradient_flows_through_checkpointed_run():
    g, m, src = make_sim()
    eps = torch.full((g.nx, g.ny), 2.25, dtype=torch.float64, requires_grad=True)
    m2 = build_tm_coeffs(g, eps_inf=eps, poles=[(2.0, 0.1, 1.5)], chi3=0.01, npml=8)
    s0 = init_tm_state(g, n_poles=1)
    s, _ = run_tm(s0, m2, 40, source=src, checkpoint_every=8)
    loss = (s.Ez**2).sum()
    loss.backward()
    assert eps.grad is not None
    assert torch.isfinite(eps.grad).all()
    assert eps.grad.abs().sum() > 0
```

- [ ] **Step 2: Run** — Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

`meep_torch/core/monitors.py` (start):
```python
import torch


class PointProbe:
    reduce = "stack"

    def __init__(self, ix: int, iy: int):
        self.ix, self.iy = ix, iy

    def sample(self, state, t_idx: int) -> torch.Tensor:
        return state.Ez[self.ix, self.iy]
```

`meep_torch/core/run.py`:
```python
import torch
from torch.utils.checkpoint import checkpoint

from .coeffs import TMCoeffs
from .stepper import TMState, step_tm


def _run_chunk(state: TMState, coeffs: TMCoeffs, step0: int, k: int, source, monitors):
    partials = [[] for _ in monitors]
    for s in range(k):
        t_idx = step0 + s
        if source is not None:
            state = step_tm(state, coeffs, source.amplitude(t_idx), source.mask)
        else:
            state = step_tm(state, coeffs)
        for i, mon in enumerate(monitors):
            partials[i].append(mon.sample(state, t_idx))
    outs = []
    for mon, ps in zip(monitors, partials):
        stacked = torch.stack(ps)
        outs.append(stacked.sum(0) if mon.reduce == "sum" else stacked)
    return state, outs


def run_tm(
    state: TMState,
    coeffs: TMCoeffs,
    n_steps: int,
    *,
    source=None,
    monitors=(),
    checkpoint_every: int = 0,
):
    monitors = list(monitors)
    acc = [None] * len(monitors)

    def absorb(outs):
        for i, (mon, out) in enumerate(zip(monitors, outs)):
            if acc[i] is None:
                acc[i] = out
            elif mon.reduce == "sum":
                acc[i] = acc[i] + out
            else:
                acc[i] = torch.cat([acc[i], out])

    if checkpoint_every <= 0:
        state, outs = _run_chunk(state, coeffs, 0, n_steps, source, monitors)
        absorb(outs)
        return state, acc

    n_tensors = len(state)

    def chunk_fn(step0, k, *tensors):
        st, outs = _run_chunk(TMState(*tensors), coeffs, step0, k, source, monitors)
        return (*st, *outs)

    step0 = 0
    while step0 < n_steps:
        k = min(checkpoint_every, n_steps - step0)
        flat = checkpoint(chunk_fn, step0, k, *state, use_reentrant=False)
        state = TMState(*flat[:n_tensors])
        absorb(list(flat[n_tensors:]))
        step0 += k
    return state, acc
```

- [ ] **Step 4: Run tests** — Expected: both PASS. Known trap: `checkpoint` requires at least one input tensor with `requires_grad` when grads are wanted — the eps test covers this via closure-captured `eps` (supported by `use_reentrant=False`).

- [ ] **Step 5: Commit** — `git commit -am "feat: checkpointed run loop with monitor partial accumulation"`

---

### Task 8: Flux monitors

**Files:**
- Modify: `meep_torch/core/monitors.py` (add FluxMonitorTM)
- Create: `tests/analytic/test_flux.py`

**Interfaces:**
- Consumes: run loop + monitor protocol from Task 7.
- Produces:
  - `FluxMonitorTM(grid, freqs: Sequence[float], *, x_normal: bool, idx: int, lo: int, hi: int)` — line monitor. `x_normal=True`: vertical line at `ix=idx`, y in `[lo,hi)`, measures +x flux. `x_normal=False`: horizontal line at `iy=idx`, x in `[lo,hi)`, measures +y flux. `reduce="sum"`.
  - `.sample(state, t_idx)` returns `(n_freqs, 4, n_pts)` tensor: dt-weighted `[Re Ê, Im Ê, Re Ĥ, Im Ĥ]` DFT contributions. E sampled at time `(t_idx+1)·dt`, H at `(t_idx+0.5)·dt` (Yee time stagger). H spatially averaged to the E line (two-point mean).
  - `.flux(acc, dx) -> Tensor (n_freqs,)`: x-normal: `Sx = −½·Σ Re(Êz·Ĥy*)·dx`; y-normal: `Sy = +½·Σ Re(Êz·Ĥx*)·dx`.

- [ ] **Step 1: Write failing tests**

`tests/analytic/test_flux.py`:
```python
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state

FREQS = [0.55, 0.65, 0.75]


def quasi_1d_sim():
    g = Grid(nx=1200, ny=8, dx=0.02)
    m = build_tm_coeffs(g, npml=40, y_pmc=True)
    src = LineSource(g, ix=100, j0=0, j1=8, profile=GaussianPulse(0.65, 0.2))
    return g, m, src


def test_flux_conserved_between_two_lines():
    g, m, src = quasi_1d_sim()
    monA = FluxMonitorTM(g, FREQS, x_normal=True, idx=400, lo=0, hi=8)
    monB = FluxMonitorTM(g, FREQS, x_normal=True, idx=900, lo=0, hi=8)
    s = init_tm_state(g, y_pmc=True)
    _, (accA, accB) = run_tm(s, m, 5000, source=src, monitors=[monA, monB])
    fA, fB = monA.flux(accA, g.dx), monB.flux(accB, g.dx)
    assert torch.all(fA > 0)                      # +x flux positive to the right of source
    assert torch.allclose(fA, fB, rtol=0.01)      # vacuum: no loss between lines


def test_flux_sign_flips_left_of_source():
    g, m, src = quasi_1d_sim()
    mon = FluxMonitorTM(g, FREQS, x_normal=True, idx=60, lo=0, hi=8)
    s = init_tm_state(g, y_pmc=True)
    _, (acc,) = run_tm(s, m, 5000, source=src, monitors=[mon])
    assert torch.all(mon.flux(acc, g.dx) < 0)     # energy flows -x left of source
```

- [ ] **Step 2: Run** — Expected: ImportError (FluxMonitorTM missing).

- [ ] **Step 3: Implement** (append to `meep_torch/core/monitors.py`)

```python
import math
from typing import Sequence

from .grid import Grid


class FluxMonitorTM:
    reduce = "sum"

    def __init__(
        self,
        grid: Grid,
        freqs: Sequence[float],
        *,
        x_normal: bool,
        idx: int,
        lo: int,
        hi: int,
    ):
        self.x_normal = x_normal
        self.idx, self.lo, self.hi = idx, lo, hi
        self.dt = grid.dt
        self.freqs = torch.tensor(freqs, dtype=grid.dtype, device=grid.device)
        if x_normal and not (1 <= idx < grid.nx - 1):
            raise ValueError(f"x-normal monitor ix={idx} must be interior")
        if not x_normal and not (1 <= idx < grid.ny - 1):
            raise ValueError(f"y-normal monitor iy={idx} must be interior")

    def sample(self, state, t_idx: int) -> torch.Tensor:
        if self.x_normal:
            E = state.Ez[self.idx, self.lo : self.hi]
            H = 0.5 * (
                state.Hy[self.idx - 1, self.lo : self.hi]
                + state.Hy[self.idx, self.lo : self.hi]
            )
        else:
            E = state.Ez[self.lo : self.hi, self.idx]
            H = 0.5 * (
                state.Hx[self.lo : self.hi, self.idx - 1]
                + state.Hx[self.lo : self.hi, self.idx]
            )
        tE = (t_idx + 1.0) * self.dt
        tH = (t_idx + 0.5) * self.dt
        w = 2 * math.pi * self.freqs[:, None]
        return self.dt * torch.stack(
            [
                torch.cos(w * tE) * E,
                -torch.sin(w * tE) * E,
                torch.cos(w * tH) * H,
                -torch.sin(w * tH) * H,
            ],
            dim=1,
        )

    def flux(self, acc: torch.Tensor, dx: float) -> torch.Tensor:
        ReE, ImE, ReH, ImH = acc.unbind(1)
        s = (ReE * ReH + ImE * ImH).sum(-1) * dx      # Re(Ê Ĥ*) integrated
        return -0.5 * s if self.x_normal else 0.5 * s
```

- [ ] **Step 4: Run tests** — Expected: PASS. If sign test fails, flip only the final sign convention in `flux` (physics: TM Sx = −Ez·Hy, Sy = +Ez·Hx) — never adjust `sample`.

- [ ] **Step 5: Commit** — `git commit -am "feat: DFT flux monitors with Yee time-stagger correction"`

---

### Task 9: Lorentz dispersion vs transfer matrix

**Files:**
- Create: `meep_torch/core/materials.py` (LorentzPole + analytic_eps only), `tests/analytic/test_dispersion.py`

**Interfaces:**
- Produces:
  - `LorentzPole(NamedTuple)`: `fn: float; gamma: float; sigma: float` (meep convention — indexable, so Task 4's builder accepts it unchanged).
  - `analytic_eps(eps_inf: float, poles: Sequence[LorentzPole], f: float) -> complex` = `eps_inf + Σ σ fn²/(fn² − f² − 1j f γ)`.
  - `slab_transmission(n2: complex, d: float, f: float) -> float` (test helper in test file): power transmission of slab (index n2, thickness d) in vacuum: `t = t12·t21·e^{iδ} / (1 − r12²·e^{2iδ})`, `δ = 2π f n2 d`, `r12 = (1−n2)/(1+n2)`, `t12 = 2/(1+n2)`, `t21 = 2n2/(1+n2)`; returns `|t|²`.

- [ ] **Step 1: Write failing test**

`tests/analytic/test_dispersion.py`:
```python
import cmath

import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.materials import LorentzPole, analytic_eps
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state

POLE = LorentzPole(fn=2.0, gamma=0.1, sigma=1.5)
EPS_INF = 2.25
FREQS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def slab_transmission(n2: complex, d: float, f: float) -> float:
    delta = 2 * cmath.pi * f * n2 * d
    r12 = (1 - n2) / (1 + n2)
    t = (2 / (1 + n2)) * (2 * n2 / (1 + n2)) * cmath.exp(1j * delta)
    t /= 1 - r12**2 * cmath.exp(2j * delta)
    return abs(t) ** 2


def run_transmission(with_slab: bool) -> torch.Tensor:
    g = Grid(nx=1400, ny=8, dx=0.02)
    slab = torch.zeros(g.nx, g.ny, dtype=torch.float64)
    slab[600:700, :] = 1.0                     # 2 um slab
    eps = 1.0 + (EPS_INF - 1.0) * slab if with_slab else None
    poles = [POLE] if with_slab else []
    masks = [slab] if with_slab else None
    m = build_tm_coeffs(g, eps_inf=eps, poles=poles, pole_masks=masks, npml=40, y_pmc=True)
    src = LineSource(g, ix=100, j0=0, j1=8, profile=GaussianPulse(0.65, 0.35))
    mon = FluxMonitorTM(g, FREQS, x_normal=True, idx=1200, lo=0, hi=8)
    s = init_tm_state(g, n_poles=len(poles), y_pmc=True)
    _, (acc,) = run_tm(s, m, 9000, source=src, monitors=[mon])
    return mon.flux(acc, g.dx)


def test_dispersive_slab_matches_transfer_matrix():
    ref = run_transmission(with_slab=False)
    tr = run_transmission(with_slab=True)
    T_sim = tr / ref
    for i, f in enumerate(FREQS):
        n2 = cmath.sqrt(analytic_eps(EPS_INF, [POLE], f))
        T_th = slab_transmission(n2, d=2.0, f=f)
        assert abs(T_sim[i].item() - T_th) / T_th < 0.03, (f, T_sim[i].item(), T_th)
```

- [ ] **Step 2: Run** — Expected: ImportError (materials module missing).

- [ ] **Step 3: Implement**

`meep_torch/core/materials.py`:
```python
from typing import NamedTuple, Sequence


class LorentzPole(NamedTuple):
    """Meep-convention Lorentzian: chi(f) = sigma * fn^2 / (fn^2 - f^2 - i*f*gamma)."""

    fn: float
    gamma: float
    sigma: float


def analytic_eps(eps_inf: float, poles: Sequence[LorentzPole], f: float) -> complex:
    eps = complex(eps_inf)
    for p in poles:
        eps += p.sigma * p.fn**2 / (p.fn**2 - f**2 - 1j * f * p.gamma)
    return eps
```

- [ ] **Step 4: Run test** — Expected: PASS (takes ~1–2 min CPU float64; that's fine). Failure >3%: check pole-mask alignment vs eps mask (both must cover cells 600:700) and that the reference run uses identical source/monitor placement.

- [ ] **Step 5: Commit** — `git commit -am "feat: Lorentz pole materials with transfer-matrix-validated ADE dispersion"`

---

### Task 10: Material library (cSi, SiO2, Si3N4)

**Files:**
- Modify: `meep_torch/core/materials.py`
- Create: `tests/analytic/test_materials.py`

**Interfaces:**
- Produces: `Material(NamedTuple)`: `eps_inf: float; poles: tuple[LorentzPole, ...]; valid_fmin: float; valid_fmax: float`, method `.n(f: float) -> complex` = `sqrt(analytic_eps(...))`. Constants `cSi`, `SiO2`, `Si3N4` (module-level). Values ported verbatim from `D:\Github\meep\python\materials.py` (um_scale = 1):
  - `cSi` (Deinega 2011, valid 0.4–1.0 µm): eps_inf=1, poles `(3.64, 0, 8)`, `(2.76, 0.126, 2.85)`, `(1.73, 5.0, -0.107)`.
  - `SiO2` (Horiba, valid 0.25–1.77 µm): eps_inf=1, pole `(1/0.103320160833333, 1/12.3984193, 1.12)`.
  - `Si3N4` (Luke 2015, valid 0.31–5.504 µm): eps_inf=1, poles `(1/0.1353406, 0, 3.0249)`, `(1/1239.842, 0, 40314)`.

- [ ] **Step 1: Write failing test**

`tests/analytic/test_materials.py`:
```python
import pytest

from meep_torch.core.materials import Si3N4, SiO2, cSi

F_1550 = 1 / 1.55


def test_si3n4_at_1550():
    # Luke fit is excellent here: published n(1.55) = 1.9963
    assert cSi is not Si3N4
    assert Si3N4.n(F_1550).real == pytest.approx(1.9963, abs=0.002)


def test_sio2_at_1550():
    # Horiba broadband fit: expect ~1.458 from the model (textbook 1.444; fit bias known)
    assert SiO2.n(F_1550).real == pytest.approx(1.4577, abs=0.003)
    assert SiO2.n(F_1550).real == pytest.approx(1.444, abs=0.02)   # sanity vs textbook


def test_csi_at_1550_extrapolated():
    # Deinega fit valid 0.4-1.0 um; extrapolates to ~3.497 at 1.55 (textbook 3.4757)
    n = cSi.n(F_1550).real
    assert n == pytest.approx(3.497, abs=0.01)
    assert n == pytest.approx(3.4757, abs=0.05)                    # sanity vs textbook
    assert F_1550 < cSi.valid_fmin  # document that 1.55um is outside fit range


def test_all_lossless_in_telecom():
    for mat in (SiO2, Si3N4):
        assert abs(mat.n(F_1550).imag) < 1e-3
```

- [ ] **Step 2: Run** — Expected: ImportError.

- [ ] **Step 3: Implement** (append to `meep_torch/core/materials.py`)

```python
import cmath


class Material(NamedTuple):
    eps_inf: float
    poles: tuple
    valid_fmin: float
    valid_fmax: float

    def n(self, f: float) -> complex:
        return cmath.sqrt(analytic_eps(self.eps_inf, self.poles, f))


# Ported from meep python/materials.py (um_scale = 1, meep natural units).
# cSi: A. Deinega et al., JOSA A 28, 770 (2011). Valid 0.4-1.0 um.
cSi = Material(
    eps_inf=1.0,
    poles=(
        LorentzPole(3.64, 0.0, 8.0),
        LorentzPole(2.76, 2 * 0.063, 2.85),
        LorentzPole(1.73, 2 * 2.5, -0.107),
    ),
    valid_fmin=1.0,
    valid_fmax=1 / 0.4,
)

# SiO2: Horiba Technical Note 08. Valid 0.25-1.77 um.
SiO2 = Material(
    eps_inf=1.0,
    poles=(LorentzPole(1 / 0.103320160833333, 1 / 12.3984193, 1.12),),
    valid_fmin=1 / 1.77,
    valid_fmax=1 / 0.25,
)

# Si3N4: K. Luke et al., Opt. Lett. 40, 4823 (2015). Valid 0.31-5.504 um.
Si3N4 = Material(
    eps_inf=1.0,
    poles=(
        LorentzPole(1 / 0.1353406, 0.0, 3.0249),
        LorentzPole(1 / 1239.842, 0.0, 40314.0),
    ),
    valid_fmin=1 / 5.504,
    valid_fmax=1 / 0.31,
)
```

- [ ] **Step 4: Run tests** — Expected: PASS. If the pinned model values differ in the 3rd decimal, re-derive by evaluating `analytic_eps` by hand before loosening tolerances — a mismatch means a transcription error in the constants.

- [ ] **Step 5: Commit** — `git commit -am "feat: port cSi/SiO2/Si3N4 Lorentz fits from meep materials library"`

---

### Task 11: Kerr χ³ validation (THG cube scaling)

**Files:**
- Create: `tests/analytic/test_kerr.py`

**Interfaces:**
- Consumes: Kerr path already implemented in `step_tm` (Task 4). This task validates it; expect no production changes.

- [ ] **Step 1: Write the test**

`tests/analytic/test_kerr.py`:
```python
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import CWRamp, LineSource
from meep_torch.core.stepper import init_tm_state

F0 = 0.25          # fundamental; THG at 0.75 still resolved at dx=0.02


def probe_spectrum_at(amp: float) -> tuple[float, float]:
    g = Grid(nx=1000, ny=8, dx=0.02)
    m = build_tm_coeffs(g, eps_inf=2.25, chi3=0.02, npml=40, y_pmc=True)

    class Scaled:
        def __init__(self, base, a):
            self.base, self.a = base, a

        def amplitude(self, t):
            return self.a * self.base.amplitude(t)

    src = LineSource(g, ix=100, j0=0, j1=8, profile=Scaled(CWRamp(F0, 40.0), amp))
    probe = PointProbe(700, 4)
    s = init_tm_state(g, y_pmc=True)
    _, (series,) = run_tm(s, m, 8000, source=src, monitors=[probe])
    steady = series[4000:]                      # discard ramp transient
    spec = torch.fft.rfft(steady).abs()
    freqs = torch.fft.rfftfreq(len(steady), g.dt)
    def peak(f):
        i = (freqs - f).abs().argmin()
        return spec[i - 2 : i + 3].max().item()
    return peak(F0), peak(3 * F0)


def test_thg_scales_as_cube():
    _, thg1 = probe_spectrum_at(0.02)
    _, thg2 = probe_spectrum_at(0.04)
    ratio = thg2 / thg1
    assert 8 * 0.7 < ratio < 8 * 1.3            # 2x input -> 8x THG (+-30%)


def test_thg_absent_without_chi3():
    g = Grid(nx=1000, ny=8, dx=0.02)
    m = build_tm_coeffs(g, eps_inf=2.25, chi3=0.0, npml=40, y_pmc=True)
    src = LineSource(g, ix=100, j0=0, j1=8, profile=CWRamp(F0, 40.0))
    probe = PointProbe(700, 4)
    s = init_tm_state(g, y_pmc=True)
    _, (series,) = run_tm(s, m, 8000, source=src, monitors=[probe])
    steady = series[4000:]
    spec = torch.fft.rfft(steady).abs()
    freqs = torch.fft.rfftfreq(len(steady), g.dt)
    i0 = (freqs - F0).abs().argmin()
    i3 = (freqs - 3 * F0).abs().argmin()
    assert spec[i3 - 2 : i3 + 3].max() < 1e-6 * spec[i0 - 2 : i0 + 3].max()
```

- [ ] **Step 2: Run** — `uv run pytest tests/analytic/test_kerr.py -v` — Expected: PASS. If cube-scaling drifts: reduce source amplitudes (avoid pump depletion regime), don't loosen the ratio bounds first.

- [ ] **Step 3: Commit** — `git commit -am "test: Kerr chi3 third-harmonic cube scaling"`

---

### Task 12: TE polarization

**Files:**
- Create: `meep_torch/core/stepper_te.py`, `tests/analytic/test_te.py`
- Modify: `meep_torch/core/coeffs.py` (add `TECoeffs`, `build_te_coeffs`), `meep_torch/core/run.py` (add `run_te`), `meep_torch/core/monitors.py` (add `FluxMonitorTE`)

**Interfaces:**
- Produces:
  - `TEState(NamedTuple)`: `Hz (nx-1,ny-1); Ex (nx-1,ny); Ey (nx,ny-1); Dx (nx-1,ny); Dy (nx,ny-1); Px_cur, Px_prev (np,nx-1,ny); Py_cur, Py_prev (np,nx,ny-1); psi_Eyx, psi_Exy (nx-1,ny-1); psi_Hzy (nx-1,ny-2); psi_Hzx (nx-2,ny-1)`.
  - `init_te_state(grid, n_poles=0) -> TEState`. TE is PEC-only in v1 (no y_pmc — quasi-1D tests use TM).
  - `build_te_coeffs(grid, eps_inf=None, poles=(), pole_masks=None, chi3=None, npml=0) -> TECoeffs` — eps/chi3 given at cell centers (nx,ny), builder averages to Ex positions (`0.5*(v[:-1,:]+v[1:,:])` → (nx-1,ny)) and Ey positions (`0.5*(v[:,:-1]+v[:,1:])` → (nx,ny-1)).
  - `step_te(state, m, src_amp=0.0, src_mask_ey: Tensor | None = None) -> TEState` (source injects into Dy).
  - `run_te(...)` mirroring `run_tm` exactly (share `_run_chunk` by parameterizing the step function and state class).
  - `FluxMonitorTE(grid, freqs, *, ix, lo, hi)` — x-normal only in v1: `Sx(ω) = +½ Σ Re(Êy Ĥz*) dx`, Ey sampled on line ix (`state.Ey[ix, lo:hi]`), Hz averaged: `0.5*(Hz[ix-1, lo:hi] + Hz[ix, lo:hi])`.
- Kerr in TE uses colocation approximation: at Ex, `|E|² ≈ Ex² + avg4(Ey)²`; at Ey symmetric. `avg4` = mean of 4 nearest other-component values with zero-padding at boundaries.

- [ ] **Step 1: Write failing tests**

`tests/analytic/test_te.py`:
```python
import torch

from meep_torch.core.coeffs import build_te_coeffs, build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper import init_tm_state, step_tm
from meep_torch.core.stepper_te import init_te_state, step_te


def test_te_state_shapes():
    g = Grid(nx=8, ny=6, dx=0.02)
    s = init_te_state(g, n_poles=1)
    assert s.Hz.shape == (7, 5)
    assert s.Ex.shape == (7, 6)
    assert s.Ey.shape == (8, 5)
    assert s.Px_cur.shape == (1, 7, 6)
    assert s.Py_cur.shape == (1, 8, 5)


def test_te_tm_vacuum_duality():
    # In vacuum the discrete TE Hz update graph is isomorphic to the TM Ez graph.
    g = Grid(nx=160, ny=160, dx=0.02)
    x = torch.arange(g.nx, dtype=g.dtype)[:, None]
    y = torch.arange(g.ny, dtype=g.dtype)[None, :]
    G = torch.exp(-((x - 80.0) ** 2 + (y - 80.0) ** 2) / (2 * 5.0**2))

    mtm = build_tm_coeffs(g, npml=0)
    stm = init_tm_state(g)._replace(Ez=G.clone(), Dz=G.clone())

    mte = build_te_coeffs(g, npml=0)
    ste = init_te_state(g)._replace(Hz=G[:-1, :-1].clone())

    for _ in range(40):   # 40 steps * courant 0.5 = 20 cells < 60-cell margin to walls
        stm = step_tm(stm, mtm)
        ste = step_te(ste, mte)

    win = slice(40, 120)
    assert torch.allclose(ste.Hz[win, win], stm.Ez[win, win], atol=1e-12, rtol=0)


def test_te_energy_bounded():
    g = Grid(nx=100, ny=100, dx=0.02)
    m = build_te_coeffs(g, npml=0)
    x = torch.arange(g.nx - 1, dtype=g.dtype)[:, None]
    y = torch.arange(g.ny - 1, dtype=g.dtype)[None, :]
    G = torch.exp(-((x - 50.0) ** 2 + (y - 50.0) ** 2) / (2 * 5.0**2))
    s = init_te_state(g)._replace(Hz=G)
    def energy(s):
        return (0.5 * (s.Hz**2).sum() + 0.5 * (m.eps_x * s.Ex**2).sum()
                + 0.5 * (m.eps_y * s.Ey**2).sum()).item()
    e0 = None
    for i in range(1500):
        s = step_te(s, m)
        if i == 100:
            e0 = energy(s)
    assert abs(energy(s) - e0) / e0 < 1e-3
```

- [ ] **Step 2: Run** — Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

`meep_torch/core/stepper_te.py`:
```python
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .grid import Grid


class TEState(NamedTuple):
    Hz: torch.Tensor       # (nx-1, ny-1) at n-1/2
    Ex: torch.Tensor       # (nx-1, ny)   at n
    Ey: torch.Tensor       # (nx, ny-1)   at n
    Dx: torch.Tensor       # (nx-1, ny)
    Dy: torch.Tensor       # (nx, ny-1)
    Px_cur: torch.Tensor   # (np, nx-1, ny)
    Px_prev: torch.Tensor
    Py_cur: torch.Tensor   # (np, nx, ny-1)
    Py_prev: torch.Tensor
    psi_Eyx: torch.Tensor  # (nx-1, ny-1)
    psi_Exy: torch.Tensor  # (nx-1, ny-1)
    psi_Hzy: torch.Tensor  # (nx-1, ny-2)
    psi_Hzx: torch.Tensor  # (nx-2, ny-1)


def init_te_state(grid: Grid, n_poles: int = 0) -> TEState:
    nx, ny = grid.nx, grid.ny
    return TEState(
        Hz=grid.zeros(nx - 1, ny - 1),
        Ex=grid.zeros(nx - 1, ny),
        Ey=grid.zeros(nx, ny - 1),
        Dx=grid.zeros(nx - 1, ny),
        Dy=grid.zeros(nx, ny - 1),
        Px_cur=grid.zeros(n_poles, nx - 1, ny),
        Px_prev=grid.zeros(n_poles, nx - 1, ny),
        Py_cur=grid.zeros(n_poles, nx, ny - 1),
        Py_prev=grid.zeros(n_poles, nx, ny - 1),
        psi_Eyx=grid.zeros(nx - 1, ny - 1),
        psi_Exy=grid.zeros(nx - 1, ny - 1),
        psi_Hzy=grid.zeros(nx - 1, ny - 2),
        psi_Hzx=grid.zeros(nx - 2, ny - 1),
    )


def _avg4_to_ex(Ey: torch.Tensor) -> torch.Tensor:
    # Ey (nx, ny-1) -> value at Ex positions (nx-1, ny); zero-pad y boundaries
    Eyp = F.pad(Ey, (1, 1))                                    # (nx, ny+1)
    return 0.25 * (Eyp[:-1, :-1] + Eyp[:-1, 1:] + Eyp[1:, :-1] + Eyp[1:, 1:])


def _avg4_to_ey(Ex: torch.Tensor) -> torch.Tensor:
    # Ex (nx-1, ny) -> value at Ey positions (nx, ny-1); zero-pad x boundaries
    Exp = F.pad(Ex, (0, 0, 1, 1))                              # (nx+1, ny)
    return 0.25 * (Exp[:-1, :-1] + Exp[:-1, 1:] + Exp[1:, :-1] + Exp[1:, 1:])


def step_te(state: TEState, m, src_amp=0.0, src_mask_ey: torch.Tensor | None = None) -> TEState:
    dt, dx = m.dt, m.dx

    dEy_dx = (state.Ey[1:, :] - state.Ey[:-1, :]) / dx          # (nx-1, ny-1)
    dEx_dy = (state.Ex[:, 1:] - state.Ex[:, :-1]) / dx          # (nx-1, ny-1)
    psi_Eyx = m.b_hx * state.psi_Eyx + m.a_hx * dEy_dx
    psi_Exy = m.b_hy * state.psi_Exy + m.a_hy * dEx_dy
    Hz = state.Hz - dt * (
        dEy_dx / m.k_hx + psi_Eyx - dEx_dy / m.k_hy - psi_Exy
    )

    dHz_dy = (Hz[:, 1:] - Hz[:, :-1]) / dx                      # (nx-1, ny-2)
    psi_Hzy = m.b_ey * state.psi_Hzy + m.a_ey * dHz_dy
    Dx = state.Dx + dt * F.pad(dHz_dy / m.k_ey + psi_Hzy, (1, 1))

    dHz_dx = (Hz[1:, :] - Hz[:-1, :]) / dx                      # (nx-2, ny-1)
    psi_Hzx = m.b_ex * state.psi_Hzx + m.a_ex * dHz_dx
    Dy = state.Dy - dt * F.pad(dHz_dx / m.k_ex + psi_Hzx, (0, 0, 1, 1))
    if src_mask_ey is not None:
        Dy = Dy + dt * src_amp * src_mask_ey

    Px_new = m.c1x * state.Px_cur + m.c2x * state.Px_prev + m.c3x * state.Ex
    Py_new = m.c1y * state.Py_cur + m.c2y * state.Py_prev + m.c3y * state.Ey

    rhs_x = Dx - Px_new.sum(0)
    rhs_y = Dy - Py_new.sum(0)
    Ex = rhs_x / m.eps_x
    Ey = rhs_y / m.eps_y
    if m.has_kerr:
        for _ in range(3):
            E2x = Ex * Ex + _avg4_to_ex(Ey) ** 2
            E2y = Ey * Ey + _avg4_to_ey(Ex) ** 2
            Ex = rhs_x / (m.eps_x + m.chi3_x * E2x)
            Ey = rhs_y / (m.eps_y + m.chi3_y * E2y)

    return TEState(
        Hz, Ex, Ey, Dx, Dy,
        Px_new, state.Px_cur, Py_new, state.Py_cur,
        psi_Eyx, psi_Exy, psi_Hzy, psi_Hzx,
    )
```

`build_te_coeffs` (append to coeffs.py) — `TECoeffs(NamedTuple)` fields: `dt, dx, eps_x (nx-1,ny), eps_y (nx,ny-1), chi3_x, chi3_y, has_kerr, c1x,c2x,c3x (np,nx-1,ny), c1y,c2y,c3y (np,nx,ny-1)`, PML: `b_hx,a_hx,k_hx (nx-1,1)` at half-x; `b_hy,a_hy,k_hy (1,ny-1)` half-y; `b_ey,a_ey,k_ey (1,ny-2)` integer-y interior; `b_ex,a_ex,k_ex (nx-2,1)` integer-x interior. Build with the same `cpml_profiles` calls as TM (PEC both axes); average cell-center eps/chi3/pole-mask fields to Ex and Ey positions with the two-point means given in Interfaces.

`run_te`: refactor `run.py` — extract `_make_runner(step_fn, state_cls)` returning a `run` function; `run_tm = _make_runner(step_tm, TMState)`, `run_te = _make_runner(step_te, TEState)`. Source arg for TE passes `src_mask_ey`. Keep the public signatures unchanged.

`FluxMonitorTE` (append to monitors.py):
```python
class FluxMonitorTE:
    reduce = "sum"

    def __init__(self, grid: Grid, freqs, *, ix: int, lo: int, hi: int):
        if not (1 <= ix < grid.nx - 1):
            raise ValueError(f"ix={ix} must be interior")
        self.ix, self.lo, self.hi = ix, lo, hi
        self.dt = grid.dt
        self.freqs = torch.tensor(freqs, dtype=grid.dtype, device=grid.device)

    def sample(self, state, t_idx: int) -> torch.Tensor:
        E = state.Ey[self.ix, self.lo : self.hi]
        H = 0.5 * (state.Hz[self.ix - 1, self.lo : self.hi] + state.Hz[self.ix, self.lo : self.hi])
        tE = (t_idx + 1.0) * self.dt
        tH = (t_idx + 0.5) * self.dt
        w = 2 * torch.pi * self.freqs[:, None]
        return self.dt * torch.stack(
            [torch.cos(w * tE) * E, -torch.sin(w * tE) * E,
             torch.cos(w * tH) * H, -torch.sin(w * tH) * H], dim=1)

    def flux(self, acc: torch.Tensor, dx: float) -> torch.Tensor:
        ReE, ImE, ReH, ImH = acc.unbind(1)
        return 0.5 * (ReE * ReH + ImE * ImH).sum(-1) * dx   # TE: Sx = +Ey*Hz
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/analytic/test_te.py tests/autodiff/test_checkpoint_equivalence.py -v` — Expected: PASS (including the untouched TM checkpoint tests after the run.py refactor).

- [ ] **Step 5: Commit** — `git commit -am "feat: TE polarization stepper with duality-validated vacuum dynamics"`

---

### Task 13: Autodiff gradcheck suite

**Files:**
- Create: `tests/autodiff/test_gradcheck.py`

**Interfaces:**
- Consumes: everything. No production changes expected; failures here are bugs in earlier tasks (in-place ops, dtype leaks, non-differentiable branches).

- [ ] **Step 1: Write the tests**

`tests/autodiff/test_gradcheck.py`:
```python
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.run import run_tm
from meep_torch.core.stepper import init_tm_state

G = Grid(nx=16, ny=16, dx=0.05)


def _blob():
    x = torch.arange(16, dtype=torch.float64)[:, None]
    y = torch.arange(16, dtype=torch.float64)[None, :]
    return torch.exp(-((x - 8.0) ** 2 + (y - 8.0) ** 2) / 4.0)


def _loss_for(eps=None, chi3=None, pole_mask=None, n_steps=10, checkpoint_every=4):
    poles = [(1.5, 0.2, 0.8)] if pole_mask is not None else []
    masks = [pole_mask] if pole_mask is not None else None
    m = build_tm_coeffs(G, eps_inf=eps, poles=poles, pole_masks=masks, chi3=chi3, npml=3)
    s = init_tm_state(G, n_poles=len(poles))
    s = s._replace(Ez=_blob(), Dz=_blob())
    s, _ = run_tm(s, m, n_steps, checkpoint_every=checkpoint_every)
    return (s.Ez**2).sum()


def test_gradcheck_eps():
    def f(eps):
        return _loss_for(eps=eps)
    eps0 = torch.full((16, 16), 2.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (eps0,), eps=1e-6, atol=1e-4)


def test_gradcheck_chi3():
    def f(chi3):
        return _loss_for(eps=2.0, chi3=chi3)
    x0 = torch.full((16, 16), 0.05, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (x0,), eps=1e-6, atol=1e-4)


def test_gradcheck_pole_mask():
    def f(mask):
        return _loss_for(eps=2.0, pole_mask=mask)
    m0 = torch.full((16, 16), 0.5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (m0,), eps=1e-6, atol=1e-4)


def test_gradcheck_uncheckpointed_matches():
    eps0 = torch.full((16, 16), 2.0, dtype=torch.float64, requires_grad=True)
    l1 = _loss_for(eps=eps0, checkpoint_every=0)
    (g1,) = torch.autograd.grad(l1, eps0)
    l2 = _loss_for(eps=eps0, checkpoint_every=3)
    (g2,) = torch.autograd.grad(l2, eps0)
    assert torch.allclose(g1, g2, atol=1e-12, rtol=1e-10)
```

- [ ] **Step 2: Run** — `uv run pytest tests/autodiff/test_gradcheck.py -v` — Expected: PASS (~1 min; gradcheck is O(n_inputs) forward passes on a 16×16 grid). If eps gradcheck fails but chi3 passes, suspect `_as_field` dtype/`.to()` detachment — `.to()` preserves grad; `torch.full` from a tensor value does not.

- [ ] **Step 3: Commit** — `git commit -am "test: gradcheck through eps, chi3, and pole-mask paths incl. PML"`

---

### Task 14: Memory flatness test

**Files:**
- Create: `tests/autodiff/test_memory.py`

**Interfaces:**
- Consumes: run_tm. No production changes.

- [ ] **Step 1: Write the test**

`tests/autodiff/test_memory.py`:
```python
import gc

import psutil
import pytest
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state


def forward_rss_delta(checkpoint_every: int) -> tuple[float, torch.Tensor]:
    gc.collect()
    g = Grid(nx=384, ny=384, dx=0.02, dtype=torch.float32)
    eps = torch.full((384, 384), 2.0, requires_grad=True)
    m = build_tm_coeffs(g, eps_inf=eps, npml=16)
    src = LineSource(g, ix=60, j0=100, j1=284, profile=GaussianPulse(0.65, 0.2))
    s = init_tm_state(g)
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    s, _ = run_tm(s, m, 300, source=src, checkpoint_every=checkpoint_every)
    loss = (s.Ez**2).sum()
    rss1 = proc.memory_info().rss
    loss.backward()
    return (rss1 - rss0) / 1e6, eps.grad


@pytest.mark.slow
def test_checkpointing_bounds_memory():
    delta_ckpt, grad_ckpt = forward_rss_delta(checkpoint_every=20)
    delta_full, grad_full = forward_rss_delta(checkpoint_every=0)
    # 300 steps, ~10 saved tensors/step at 384*384*4B ~ 0.6MB each:
    # uncheckpointed retains O(GB); checkpointed retains O(state * n_chunks).
    assert delta_ckpt < delta_full / 3, (delta_ckpt, delta_full)
    assert torch.allclose(grad_ckpt, grad_full, rtol=1e-4, atol=1e-6)
```

- [ ] **Step 2: Run** — `uv run pytest tests/autodiff/test_memory.py -m slow -v` — Expected: PASS (~2–4 min). RSS is coarse; the 3x margin against a physically ~15x difference absorbs allocator noise. If flaky on the machine, raise grid to 512×512 rather than shrinking the margin.

- [ ] **Step 3: Commit** — `git commit -am "test: checkpointed run keeps autodiff memory bounded"`

---

### Task 15: float32 stability at production scale

**Files:**
- Create: `tests/autodiff/test_float32.py`

- [ ] **Step 1: Write the test**

`tests/autodiff/test_float32.py`:
```python
import pytest
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state


@pytest.mark.slow
def test_float32_stable_at_2000x2000():
    g = Grid(nx=2000, ny=2000, dx=0.02, dtype=torch.float32)
    m = build_tm_coeffs(g, eps_inf=2.25, npml=20)
    src = LineSource(g, ix=200, j0=400, j1=1600, profile=GaussianPulse(0.65, 0.2))
    s = init_tm_state(g)
    s, _ = run_tm(s, m, 2000, source=src, checkpoint_every=0)
    with torch.no_grad():
        assert torch.isfinite(s.Ez).all()
        assert torch.isfinite(s.Hx).all() and torch.isfinite(s.Hy).all()
        assert s.Ez.abs().max() < 1e3       # bounded, no runaway growth
```

Wrap the whole run in `torch.no_grad()` (add `with torch.no_grad():` around the run too — this test is about numerics, not gradients; without it the uncheckpointed 2000-step graph would exhaust RAM, which is Task 14's demonstration, not this one's).

- [ ] **Step 2: Run** — `uv run pytest tests/autodiff/test_float32.py -m slow -v` — Expected: PASS (~5–10 min CPU; fine).

- [ ] **Step 3: Commit** — `git commit -am "test: float32 stability at 2000x2000 production grid"`

---

### Task 16: Meep parity skeleton

**Files:**
- Create: `tests/parity/test_meep_parity.py`

**Interfaces:**
- Consumes: full engine. Runs only where real meep is importable (future Linux box) — must skip cleanly on Windows.

- [ ] **Step 1: Write the tests**

`tests/parity/test_meep_parity.py`:
```python
import pytest
import torch

mp = pytest.importorskip("meep")

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state

# Both engines: quasi-1D vacuum, Gaussian line source, two flux planes.
# Compare the RATIO flux_far/flux_near (normalization-independent).
FREQS = [0.55, 0.65, 0.75]


def torch_flux_ratio() -> torch.Tensor:
    g = Grid(nx=1200, ny=8, dx=0.02)
    m = build_tm_coeffs(g, npml=40, y_pmc=True)
    src = LineSource(g, ix=100, j0=0, j1=8, profile=GaussianPulse(0.65, 0.2))
    monA = FluxMonitorTM(g, FREQS, x_normal=True, idx=400, lo=0, hi=8)
    monB = FluxMonitorTM(g, FREQS, x_normal=True, idx=900, lo=0, hi=8)
    s = init_tm_state(g, y_pmc=True)
    _, (a, b) = run_tm(s, m, 5000, source=src, monitors=[monA, monB])
    return monB.flux(b, g.dx) / monA.flux(a, g.dx)


def meep_flux_ratio():
    cell = mp.Vector3(24, 0, 0)
    sim = mp.Simulation(
        cell_size=cell,
        resolution=50,
        boundary_layers=[mp.PML(0.8, direction=mp.X)],
        sources=[mp.Source(mp.GaussianSource(0.65, fwidth=0.2), component=mp.Ez,
                           center=mp.Vector3(-10, 0))],
        dimensions=1,
    )
    fA = sim.add_flux(0.65, 0.2, len(FREQS),
                      mp.FluxRegion(center=mp.Vector3(-4, 0)))
    fB = sim.add_flux(0.65, 0.2, len(FREQS),
                      mp.FluxRegion(center=mp.Vector3(6, 0)))
    sim.run(until_after_sources=100)
    import numpy as np
    return np.array(mp.get_fluxes(fB)) / np.array(mp.get_fluxes(fA))


def test_vacuum_flux_ratio_parity():
    ours = torch_flux_ratio()
    theirs = torch.tensor(meep_flux_ratio(), dtype=torch.float64)
    assert torch.allclose(ours, theirs, rtol=0.01)
```

- [ ] **Step 2: Run on Windows** — `uv run pytest tests/parity -v` — Expected: `1 skipped` (meep not importable). That skip IS the pass criterion today.

- [ ] **Step 3: Commit** — `git commit -am "test: meep parity skeleton, activates where meep is installed"`

---

### Task 17: Final gate

- [ ] **Step 1: Full suite** — `uv run pytest -v` then `uv run pytest -m slow -v` — Expected: all pass, parity skipped.
- [ ] **Step 2: Verify success criteria from spec** — tier-1 float64 ✓ (Tasks 4,6,8,9,10,11,12), gradcheck all paths ✓ (13), memory-flat ✓ (14), float32 at scale ✓ (15), parity-skips-clean ✓ (16). Fix anything unmet before declaring done.
- [ ] **Step 3: Commit any stragglers; tag** — `git tag v0.1.0-phase1`

## Self-review notes

- Spec coverage: every spec bullet maps to a task; spec's "10k-step memory test" realized as 300-step RSS-ratio test (Task 14) — same property, tractable runtime; spec's scipy dep dropped (constants ported instead, flagged as deviation).
- Type consistency: `TMState` field order fixed in Task 2 and used positionally in Task 7's `chunk_fn`; `LorentzPole` is index-compatible with the 3-tuples used in Tasks 4–7 tests.
- Known risks called out inline: flux sign conventions (Task 8 step 4), gradcheck dtype traps (Task 13 step 2), RSS flakiness mitigation (Task 14 step 2).
