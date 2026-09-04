# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Differentiable multilayer-stack shape optimization
#
# The design parameters are the five layer thicknesses of a lossy 1D stack,
# parameterized as a softmax over logits (positive thicknesses, fixed total
# length) and painted into eps with tanh-smoothed edges so the layer
# boundaries are autograd-differentiable. Adam thins the absorbing layers to
# recover transmitted flux.

# %%
"""meep-torch port of meep/python/examples/adjoint_optimization/multilayer_opt.py.

Shape optimization of a 1D multilayer stack. The meep original uses
meep.adjoint + autograd + NLopt to minimize the worst-case (over two
wavelengths) integral of the DFT fields in the stack, with the N layer
thicknesses as design parameters. meep-torch has no meep.adjoint -- but
run_tm is itself differentiable via torch autograd (checkpointed backward,
W8), so the adjoint is just `loss.backward()` through the FDTD run.

Reduction (per port instructions):
  * 1D stack via thin 2D TM + y_pmc; layer thicknesses are the design
    parameters, painted into eps with differentiable soft (tanh) edges so
    the boundary positions are autograd-differentiable;
  * the worst-case two-wavelength DFT-field objective -> maximize the
    single-frequency transmitted flux through the stack; the high-index
    layers are made lossy (sigma_d) so a thick stack absorbs the beam and
    the optimizer thins them to recover transmission -- a smooth monotone
    objective the 5 Adam steps descend (the differentiable-stack machinery
    is the point of the demo);
  * NLopt CCSA -> 5 Adam steps, objective printed per step.

The layer thicknesses are parameterized as a softmax over raw logits ->
positive thicknesses with a fixed total stack length (always fits the
cell), and the eps at cell centre x is

    eps(x) = eps_lo + (eps_hi - eps_lo) * sum_k b_k * rect_soft(x; b_{k-1}, b_k)

with b_k the cumulative thickness, b_k in {0,1} the alternating band
pattern, and rect_soft a tanh-smoothed rectangle of width ~1 cell. All of
that is differentiable in the logits, hence in the FDTD objective.

Run: uv run python examples/07_adjoint_inverse_design/multilayer_opt.py
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
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state

OUT = pathlib.Path(__file__).parent / "output"

# %% [markdown]
# ## Soft-edged stack painter and flux objective
#
# `_stack_eps` turns logits into cumulative layer boundaries and paints the
# alternating high/low band with ~1-cell tanh rectangles; high-index layers
# carry conductivity, so the transmitted-flux objective is smooth and
# monotone in the thicknesses. 1D is modeled as thin 2D TM with PMC on y.

# %%
FCEN, DF = 1.0, 0.4
N_STEPS, CKPT = 1200, 200
NX, NY, PML = 120, 3, 12
DX = 0.05
GRID = Grid(nx=NX, ny=NY, dx=DX)
N_LAYERS = 5
EPS_HI, EPS_LO = 4.0, 1.0          # alternating high/low index layers
BAND = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])  # high on even layers
EDGE = 1.0 * DX                    # soft-edge width (~1 cell)
LOSS = 0.3                         # conductivity on high-index layers (absorbing)


def _stack_eps(logits):
    """Differentiable multilayer (eps, sigma_d) from layer-thickness logits.

    softmax -> positive thicknesses summing to the inner cell span; tanh
    rectangles paint the alternating band pattern with smooth edges. The
    high-index layers are lossy (sigma_d = LOSS * band) so a thick stack
    absorbs most of the incident beam -- the optimizer thins the absorbing
    layers to recover transmission (the adjoint_gradient "clear the lossy
    obstacle" pattern), giving a smooth, monotone objective the 5 Adam
    steps actually descend.
    """
    span = (NX - 2 * PML) * DX
    thick = torch.softmax(logits, dim=0) * span
    bounds = torch.cumsum(thick, dim=0) - thick        # b_0=0, ..., b_{N-1}
    x = (torch.arange(NX, dtype=GRID.dtype) + 0.5) * DX
    x0 = PML * DX
    band = torch.zeros(NX, dtype=GRID.dtype)
    b_prev = x0
    for k in range(N_LAYERS):
        b_k = x0 + bounds[k] + thick[k]
        rect = 0.25 * (torch.tanh((x - b_prev) / EDGE) + 1.0) * \
            (1.0 - torch.tanh((x - b_k) / EDGE))      # 1 inside (b_prev, b_k)
        band = band + BAND[k] * rect
        b_prev = b_k
    eps = (EPS_LO + (EPS_HI - EPS_LO) * band)[:, None].expand(NX, NY)
    sigma = (LOSS * band)[:, None].expand(NX, NY)
    return eps, sigma


def _objective(logits):
    eps, sigma = _stack_eps(logits)
    coeffs = build_tm_coeffs(GRID, eps_inf=eps, sigma_d=sigma, npml=PML, y_pmc=True)
    src = LineSource(GRID, ix=PML + 3, j0=0, j1=NY,
                     profile=GaussianPulse(FCEN, DF))
    mon = FluxMonitorTM(GRID, [FCEN], x_normal=True,
                        idx=NX - PML - 3, lo=0, hi=NY)
    _, (acc,) = run_tm(init_tm_state(GRID, y_pmc=True), coeffs, N_STEPS,
                       source=src, monitors=[mon], checkpoint_every=CKPT)
    return mon.flux(acc, GRID.dx).sum()


# %% [markdown]
# ## Optimization loop

# %%
def main():
    logits = torch.nn.Parameter(
        torch.full((N_LAYERS,), 0.0, dtype=GRID.dtype))
    opt = torch.optim.Adam([logits], lr=0.15)

    flux = []
    for step in range(5):
        opt.zero_grad()
        loss = -_objective(logits)
        loss.backward()
        opt.step()
        with torch.no_grad():
            f = float(-loss)
        flux.append(f)
        print(f"multilayer_opt  step {step + 1}: transmitted flux = {f:.6g}")
    assert flux[-1] > flux[0], f"objective did not improve: {flux}"

    with torch.no_grad():
        eps = _stack_eps(logits.detach())[0].numpy()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.imshow(eps.T, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title("optimized multilayer eps")
    ax.set_xlabel("x (cells)")
    fig.tight_layout()
    fig.savefig(OUT / "multilayer_opt.png", dpi=110)
    plt.close(fig)
    print(f"saved {OUT / 'multilayer_opt.png'}  "
          f"(flux {flux[0]:.6g} -> {flux[-1]:.6g})")


if __name__ == "__main__":
    main()