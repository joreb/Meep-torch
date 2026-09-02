# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Power and mode-profile comparison
#
# Power-output + mode-profile comparison plots, meep vs meep-torch,
# for all devices.
#
# In-plane devices (c_y_splitter, c_edge_coupler): |S|^2 per output port
# (torch_sparams vs gmeep npz) as bars, and the DFT |Ez(y)| profile at each
# monitor plane overlaid (both normalized to the o1 input-plane peak so
# relative power between ports survives).
#
# Grating coupler (aafgc_ion_729): focusing-efficiency proxy (5 um ion-plane
# window over input-line power, steady-state CW |Ez|^2) as bars, plus the
# ion-plane |Ez| focal-spot profile overlaid (needs aafgc_final_fields.npz
# from aafgc_cw_video.py).
#
# Run: `uv run python examples_2/meep_vs_torch/compare_power_mode.py`

# %%
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ports_util import CASES
from torch_sparams import gmeep_sparams, torch_sparams
import aafgc_cw_video as ag

HERE = pathlib.Path(__file__).parent

# %% [markdown]
# ## In-plane devices
# Bars of |S|^2 per output port from both engines, plus overlaid |Ez(y)|
# profiles at every monitor plane.

# %%
def plot_inplane(name):
    sp_t, prof_t = torch_sparams(name)
    sp_m = gmeep_sparams(name)
    prof_m = np.load(HERE / f"{name}_profiles_meep.npz")
    port_names = ["o1"] + sorted(sp_m)

    fig, axs = plt.subplots(1, 1 + len(port_names),
                            figsize=(4 + 3.2 * len(port_names), 3.6))
    ax = axs[0]
    xs = np.arange(len(sp_m))
    ax.bar(xs - 0.17, [sp_t[p] for p in sorted(sp_m)], 0.34,
           label="meep-torch", color="#4878d0")
    ax.bar(xs + 0.17, [sp_m[p] for p in sorted(sp_m)], 0.34,
           label="pymeep (gmeep)", color="#ee854a")
    ax.set_xticks(xs, sorted(sp_m))
    ax.set_ylabel("|S|$^2$ @ 674 nm")
    ax.set_title("output power")
    ax.legend(fontsize=8)

    # normalize both engines to their own o1 input-plane peak, so the
    # output profiles keep relative amplitude
    n_t = prof_t["o1"]["ez"].max()
    n_m = prof_m["o1_ez"].max()
    for ax, pn in zip(axs[1:], port_names):
        ax.plot(prof_t[pn]["ys"], prof_t[pn]["ez"] / n_t, "-",
                color="#4878d0", label="meep-torch")
        ax.plot(prof_m[f"{pn}_ys"], prof_m[f"{pn}_ez"] / n_m, "--",
                color="#ee854a", label="pymeep")
        ax.set_title(f"|Ez| at {pn} plane")
        ax.set_xlabel("y (um)")
        ax.legend(fontsize=8)
    fig.suptitle(f"{name}: power + mode profiles, meep-torch vs pymeep")
    fig.tight_layout()
    png = HERE / f"{name}_power_mode.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"{name} -> {png.name}")

# %% [markdown]
# ## Grating coupler
# Focused-fraction bars at the ion plane and the focal-spot profile,
# from the steady-state CW fields of both engines.

# %%
def plot_grating():
    d = np.load(HERE / "aafgc_final_fields.npz")
    ds = int(d["ds"])
    fields = {"meep-torch": (np.abs(d["torch_ez"].astype(np.float64)), ds),
              "pymeep": (np.abs(d["meep_ez"]), 1)}
    res = ag.RES
    ia_x_um = ag.X0_DEV + ag.L_DEV / 2 + ag.D_LAT
    ia_z_um = ag.OX_BOT + ag.H_FILM + ag.OX_TOP + ag.H_ION
    icol_um, k0f_um = ag.X0_DEV - 2.0, ag.OX_BOT
    fe, prof = {}, {}
    for eng, (f, dsf) in fields.items():
        r = res / dsf
        iz = int(round(ia_z_um * r))
        ix = int(round(ia_x_um * r))
        win = int(round(2.5 * r))
        band = int(round(1.0 * r))     # +-1 um z-band: averages CW fringes
        iz = min(iz, f.shape[1] - band - 1)
        I = (f[:, iz - band:iz + band] ** 2).mean(axis=1)
        # focused fraction: 5 um window over the whole ion-plane line
        fe[eng] = float(I[ix - win:ix + win].sum() / I.sum())
        xs = np.arange(len(I)) / r
        m = (xs > ia_x_um - 10) & (xs < ia_x_um + 10)
        prof[eng] = (xs[m], np.sqrt(I[m]) / np.sqrt(I[m]).max())

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].bar([0, 1], [fe["meep-torch"], fe["pymeep"]], 0.5,
               color=["#4878d0", "#ee854a"])
    axs[0].set_xticks([0, 1], ["meep-torch", "pymeep"])
    axs[0].set_ylabel("focused fraction (5 um window / line)")
    axs[0].set_title("grating: focused power @ ion plane (CW)")
    for eng, col, lsty in [("meep-torch", "#4878d0", "-"),
                           ("pymeep", "#ee854a", "--")]:
        axs[1].plot(*prof[eng], lsty, color=col, label=eng)
    axs[1].axvline(ia_x_um, color="k", lw=0.5)
    axs[1].set_xlabel("x (um)"); axs[1].set_ylabel("|Ez| (norm)")
    axs[1].set_title("ion-plane focal spot (H=68 um)")
    axs[1].legend(fontsize=8)
    fig.suptitle("aafgc_ion_729: power + spot profile, meep-torch vs pymeep")
    fig.tight_layout()
    png = HERE / "aafgc_ion_729_power_mode.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"aafgc_ion_729 -> {png.name}  focused fraction: "
          f"torch={fe['meep-torch']:.4f} pymeep={fe['pymeep']:.4f}")

# %% [markdown]
# ## Run all cases

# %%
if __name__ == "__main__":
    for case in CASES:
        plot_inplane(case)
    plot_grating()
