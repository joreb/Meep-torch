# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Phase of a mode coefficient
#
# meep-torch port of meep/python/examples/mode_coeff_phase.py. The meep
# tutorial verifies the *phase* of a TIR reflection coefficient at an oblique
# planewave interface against the Fresnel equations, using an oblique
# planewave source (k_point + amp_func) and direction=NO_DIRECTION mode
# decomposition. meep-torch has no oblique-planewave TFSF source and its
# decompose/solve_1d is a 1D slab solver at normal incidence, so the oblique
# Fresnel-TIR setup is not reproducible here.
#
# Instead, this example demonstrates the same feature the tutorial is really
# about -- the *phase* of a mode coefficient returned by decompose -- in a
# setup meep-torch can do: a straight waveguide. A TM eigenmode source
# launches the fundamental mode; monitors at several downstream x planes are
# decomposed into the guided mode, and the phase of the forward coefficient
# a_p vs propagation distance is the modal propagation phase
# exp(i*2*pi*neff*fcen*x). The slope of the unwrapped phase vs x gives neff,
# which we check against the 1D solver.
#
# Key numeric: neff extracted from the phase slope vs solve_1d neff.
# Output: `output/mode_coeff_phase.png`.
#
# Run: `uv run python examples/05_monitors_analysis/mode_coeff_phase.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_tm import init_tm_state
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.port import Port

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
FCEN, FWIDTH = 1.0 / 1.55, 0.2 / 1.55  # telecom, fwidth = 0.2*fcen

# %% [markdown]
# ## Setup, run, and phase-slope fit
# A straight eps=12 guide driven by the TFSF eigenmode source; 33 closely
# spaced monitors keep the phase advance per step below pi so np.unwrap is
# unambiguous, and the fitted |slope|/(2 pi f) is neff.

# %%
def main():
    # straight waveguide: w=1.0 um, eps=12 (Si), in air
    dx = 0.05
    nx, ny = 320, 80          # 16 x 4 um
    grid = Grid(nx=nx, ny=ny, dx=dx)
    j0, j1 = 30, 50           # waveguide y in [-1.0, 1.0] -> rows 30:50 (centered at ny/2=40)
    eps = torch.ones(nx, ny, dtype=torch.float64, device=grid.device)
    eps[:, j0:j1] = 12.0
    npml = 10
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=npml)
    # port/monitor span must include cladding so solve_1d finds the guided mode
    # (a span that is entirely inside the waveguide sees uniform eps=12, neff==clad).
    mlo, mhi = j0 - 10, j1 + 10     # waveguide + 0.5 um cladding each side

    # TFSF eigenmode source at the left; monitors at several x downstream.
    src_port = Port(axis="x", index=40, lo=mlo, hi=mhi, direction=+1)
    src = ModeSourceTM(grid, coeffs, src_port, GaussianPulse(FCEN, FWIDTH), FCEN)
    # closely-spaced monitors so the phase advance (<pi per step) is unambiguous
    # to np.unwrap: beta=2*pi*neff*fcen ~ 13.8 rad/um, so spacing 0.2 um -> 2.76 rad < pi.
    mon_idx = list(range(80, 210, 4))   # x planes every 4 cells (0.2 um)
    monitors = [FluxMonitorTM(grid, [FCEN], x_normal=True, idx=ix, lo=mlo, hi=mhi)
               for ix in mon_idx]

    n_steps = 7000
    state, accs = run_tm(init_tm_state(grid), coeffs, n_steps,
                        source=src, monitors=monitors)

    # reference neff from the 1D slab solver on the port eps line
    eps_line = eps[src_port.index, mlo:mhi]
    neff_ref = float(solve_1d(eps_line, dx, FCEN, "tm", k=1).neff[0])

    xs, phases = [], []
    for ix, acc in zip(mon_idx, accs):
        a_p, _ = decompose(acc, eps_line, dx, [FCEN], "tm", n_modes=1, axis_sign=-1.0)
        ph = float(np.angle(complex(a_p[0, 0])))
        xs.append(ix * dx)
        phases.append(ph)
    xs = np.array(xs)
    phases = np.array(phases)
    # unwrap vs the first monitor, take the principal branch by differencing
    phases = np.unwrap(phases)

    # neff from phase slope: phase(x) = phase0 (+/-) 2*pi*neff*fcen*x (forward wave);
    # the decompose convention yields a_p ~ exp(-i*beta*x), so the slope is
    # negative. Use |slope| so the sign convention does not matter.
    slope = np.polyfit(xs, phases, 1)[0]
    neff_meas = abs(slope) / (2 * np.pi * FCEN)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    ax1.plot(xs, phases, "o-", color="#1f77b4", label="measured phase(decompose)")
    # decompose convention: a_p ~ exp(-i*beta*x) -> phase decreases with x
    slope_ref = -2 * np.pi * neff_ref * FCEN
    ax1.plot(xs, slope_ref * (xs - xs[0]) + phases[0],
             "k--", label=f"analytic -2*pi*neff*f*x (neff={neff_ref:.4f})")
    ax1.set_xlabel("monitor x (um)")
    ax1.set_ylabel("phase of forward mode coeff a_p (rad)")
    ax1.set_title("Mode-coefficient phase vs propagation (mode_coeff_phase)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    # field snapshot (transient Ez at the final step, from the same run)
    ax2.imshow(eps.cpu().T, origin="lower", cmap="binary", alpha=0.9)
    v = float(state.Ez.abs().max()) or 1.0
    ax2.imshow(state.Ez.cpu().T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
    ax2.set_title("Ez field snapshot (straight waveguide)")
    ax2.set_xlabel("x (cells)")
    ax2.set_ylabel("y (cells)")
    fig.tight_layout()
    fig.savefig(OUT / "mode_coeff_phase.png", dpi=150)
    plt.close(fig)
    print(f"mode_coeff_phase.png written; neff_ref={neff_ref:.4f}, "
          f"neff_from_phase={neff_meas:.4f} (err {abs(neff_meas-neff_ref)/neff_ref:.3e})")


if __name__ == "__main__":
    main()
