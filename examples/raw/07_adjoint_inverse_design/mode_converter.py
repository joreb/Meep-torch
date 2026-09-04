# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Mode-coefficient optimization across a waveguide gap
#
# A slab waveguide has a gap in the middle; one scalar design variable alpha
# fills the gap footprint with eps = 1 + alpha. Adam raises alpha to restore
# the guide, maximizing the forward mode coefficient |a_p|^2 measured by mode
# decomposition on the output port — the adjoint is `loss.backward()` through
# the FDTD run plus the mode overlap.

# %%
"""meep-torch port of meep/python/examples/adjoint_optimization/mode_converter.py.

Topology optimization of a waveguide mode coupler. The meep original uses
meep.adjoint + NLopt CCSA worst-case optimization (minimize max{R, 1-T}
over six wavelengths) of a silicon design region between two waveguides,
maximizing |S21|^2 into the target mode. meep-torch has no meep.adjoint --
run_tm is itself differentiable via torch autograd (W8), and
ports/mode_monitor.decompose gives the mode coefficients, so the adjoint is
`loss.backward()` through the FDTD run + the mode-overlap.

Reduction (per port instructions):
  * a single-mode slab waveguide (TM, x-propagation) with a GAP in the
    middle; a tunable eps blob (one scalar alpha) fills the gap. The
    optimizer raises alpha (eps = 1 + alpha over the gap waveguide
    footprint) to restore the waveguide and recover forward-mode
    transmission -- maximizing the forward mode coefficient |a_p|^2;
  * ModeSourceTM launches the fundamental on the input port;
    FluxMonitorTM + ports.mode_monitor.decompose reads the forward mode
    coefficient on the output port;
  * six-wavelength CCSA worst-case -> single-frequency Adam, 5 steps,
    objective per step printed.

Run: uv run python examples/07_adjoint_inverse_design/mode_converter.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_tm import init_tm_state
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Geometry, ports, and differentiable objective
#
# `ModeSourceTM` launches the fundamental mode at the input port; a flux
# monitor at the output port feeds `ports.mode_monitor.decompose`, whose
# forward coefficient |a_p|^2 is the (differentiable) objective.

# %%
FCEN, DF = 0.2, 0.2
N_STEPS, CKPT = 3000, 200
NX, NY, PML = 64, 20, 6
DX = 0.05
GRID = Grid(nx=NX, ny=NY, dx=DX)
EPS_WG = 3.5 ** 2                       # waveguide core
WG_LO, WG_HI = 7, 13                   # waveguide row band (6 cells = 0.3 wide)
GAP0, GAP1 = 28, 36                    # gap x-range where the core is removed
# lo/hi span the full transverse extent (incl. air cladding) so the
# mode solver sees core + cladding and finds a guided mode; the core
# itself is only rows WG_LO..WG_HI.
IN_PORT = Port(axis="x", index=10, lo=0, hi=NY, direction=+1, n_modes=1)
OUT_PORT = Port(axis="x", index=52, lo=0, hi=NY, direction=+1, n_modes=1)


def _eps(alpha):
    """Waveguide core everywhere except the gap, where eps = 1 + alpha."""
    eps = torch.ones(NX, NY, dtype=GRID.dtype, device=GRID.device)
    eps[:, WG_LO:WG_HI] = EPS_WG
    gap = torch.full((GAP1 - GAP0, WG_HI - WG_LO), 1.0, dtype=GRID.dtype, device=GRID.device)
    eps[GAP0:GAP1, WG_LO:WG_HI] = 1.0 + alpha
    return eps


def _objective(alpha):
    eps = _eps(alpha)
    coeffs = build_tm_coeffs(GRID, eps_inf=eps, npml=PML)
    src = ModeSourceTM(GRID, coeffs, IN_PORT, GaussianPulse(FCEN, DF), FCEN, mode=0)
    mon = FluxMonitorTM(GRID, [FCEN], x_normal=True,
                        idx=OUT_PORT.index, lo=OUT_PORT.lo, hi=OUT_PORT.hi)
    _, (acc,) = run_tm(init_tm_state(GRID), coeffs, N_STEPS,
                       source=src, monitors=[mon], checkpoint_every=CKPT)
    eps_line = eps[OUT_PORT.index, OUT_PORT.lo:OUT_PORT.hi].detach()
    a_p, _ = decompose(acc, eps_line, GRID.dx, [FCEN], "tm", OUT_PORT.n_modes,
                       axis_sign=-1.0)
    return (a_p[0, 0].real ** 2 + a_p[0, 0].imag ** 2)


# %% [markdown]
# ## Optimization loop
#
# Five Adam steps on -|a_p|^2 with alpha clamped to [0, 30]; the objective
# must improve or the assert fires.

# %%
def main():
    alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=GRID.dtype, device=GRID.device))
    opt = torch.optim.Adam([alpha], lr=0.8)

    objs = []
    for step in range(5):
        opt.zero_grad()
        loss = -_objective(alpha)
        loss.backward()
        opt.step()
        with torch.no_grad():
            alpha.clamp_(min=0.0, max=30.0)
            f = float(-loss)
        objs.append(f)
        print(f"mode_converter  step {step + 1}: |a_p|^2 = {f:.6g}  "
              f"alpha={float(alpha.detach()):.3f}")
    assert objs[-1] > objs[0], f"objective did not improve: {objs}"

    with torch.no_grad():
        eps = _eps(alpha.detach()).cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.imshow(eps.T, origin="lower", cmap="viridis", aspect="auto")
    ax1.set_title(f"optimized eps (alpha={float(alpha):.2f})")
    ax2.plot([1, 2, 3, 4, 5], objs, "bo-")
    ax2.set_title("forward mode coefficient |a_p|^2")
    ax2.set_xlabel("Adam step")
    fig.tight_layout()
    fig.savefig(OUT / "mode_converter.png", dpi=110)
    plt.close(fig)
    print(f"saved {OUT / 'mode_converter.png'}  "
          f"(|a_p|^2 {objs[0]:.6g} -> {objs[-1]:.6g})")


if __name__ == "__main__":
    main()