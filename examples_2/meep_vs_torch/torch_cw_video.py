# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # meep-torch CW videos
#
# meep-torch CW beam-propagation videos -- the counterpart of the gmeep
# CW mp4s: same ported+extended geometry (torch_sparams.build_sim), same
# 674 nm CW mode source on o1, Ez snapshot every 0.25 time units, rendered
# to <device>_torch_cw_fdtd.mp4 next to the gmeep videos.
#
# Run: `uv run python examples_2/meep_vs_torch/torch_cw_video.py`

# %%
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ports_util import CASES, WL
from torch_sparams import FCEN, N_SIN, N_SIO2, build_sim

from meep_torch.core.sources import ContinuousSource
from meep_torch.core.stepper_tm import init_tm_state, step_tm
from meep_torch.ports.mode_source import ModeSourceTM

HERE = pathlib.Path(__file__).parent
CW_TIME = 80.0       # matches the gmeep videos
SNAP_EVERY = 0.25
FPS = 24

# %% [markdown]
# ## CW snapshots
# Reuse torch_sparams.build_sim for the identical ported geometry, drive
# the CW TE0 mode on o1, and collect Ez every 0.25 time units.

# %%
def cw_snapshots(name):
    s = build_sim(name)
    grid, coeffs = s["grid"], s["coeffs"]
    src = ModeSourceTM(grid, coeffs, s["p_src"],
                       ContinuousSource(frequency=FCEN), FCEN)
    state = init_tm_state(grid)
    h_masks = getattr(src, "h_masks", None) or {}
    every = int(round(SNAP_EVERY / grid.dt))
    snaps = []
    for t_idx in range(int(round(CW_TIME / grid.dt))):
        state = step_tm(state, coeffs, src.amplitude(t_idx), src.mask,
                        src.h_amplitude(t_idx), **h_masks)
        if (t_idx + 1) % every == 0:
            snaps.append(state.Ez.detach().cpu().numpy().astype(np.float32))
    return np.stack(snaps), s["eps"].cpu().numpy(), s["extent"], grid

# %% [markdown]
# ## Render
# Steady-state color scale from the last quarter of the run; each device
# gets its own <device>_torch_cw_fdtd.mp4.

# %%
def render(name):
    snaps, eps, extent, grid = cw_snapshots(name)
    # steady-state color scale from the last quarter of the run
    vmax = np.percentile(np.abs(snaps[3 * len(snaps) // 4:]), 99.5)
    fig, ax = plt.subplots(figsize=(9, 9 * grid.ny / grid.nx + 0.6))
    im = ax.imshow(snaps[0].T, origin="lower", cmap="RdBu",
                   vmin=-vmax, vmax=vmax, extent=extent, aspect="equal")
    ax.contour(eps.T, levels=[(N_SIO2 ** 2 + N_SIN ** 2) / 2], extent=extent,
               colors="k", linewidths=0.4)
    title = ax.set_title("")
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    mp4 = HERE / f"{name}_torch_cw_fdtd.mp4"
    writer = FFMpegWriter(fps=FPS)
    with writer.saving(fig, str(mp4), dpi=100):
        for i, fr in enumerate(snaps):
            im.set_data(fr.T)
            title.set_text(f"{name}  meep-torch CW Ez  "
                           f"{WL * 1e3:.0f} nm   t={(i + 1) * SNAP_EVERY:.2f}")
            writer.grab_frame()
    plt.close(fig)
    print(f"{name}: torch CW video -> {mp4.name} ({len(snaps)} frames)")

# %% [markdown]
# ## Run all cases

# %%
if __name__ == "__main__":
    for case in CASES:
        render(case)
