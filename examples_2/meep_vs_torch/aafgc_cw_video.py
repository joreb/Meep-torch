# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # AAFGC grating coupler: CW videos, both engines
#
# AAFGC grating-coupler CW beam-propagation videos, both engines.
#
# Rebuilds the aafgc_ion_729 final design (saved rho -> conic filter ->
# tanh projection @64 -> painted into the SiN220/SiO2 stack, identical
# constants to examples_2/grating_coupler/aafgc_ion_729.py), then:
# * meep-torch: CW TE0 mode source in the feed waveguide, GPU FDTD,
#   Ez snapshots -> aafgc_ion_729_torch_cw_fdtd.mp4
# * pymeep: the SAME eps array via epsilon_input_file (HDF5), CW
#   EigenModeSource at the same plane -> aafgc_ion_729_meep_cw_fdtd.mp4
#   (Animate2D), final Ez saved for comparison
#
# Parity metric: Pearson correlation of the normalized steady-state |Ez|.
#
# Run: `uv run python examples_2/meep_vs_torch/aafgc_cw_video.py`

# %%
import json
import pathlib
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from meep_torch.adjoint import DesignRegion, MaterialGrid
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import ContinuousSource
from meep_torch.core.stepper_tm import init_tm_state, step_tm
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port

HERE = pathlib.Path(__file__).parent
GDS_DIR = HERE.parent
ORACLE = pathlib.Path.home() / "miniforge3/envs/meep-oracle/bin/python"

# %% [markdown]
# ## Design constants
# Copied verbatim from examples_2/grating_coupler/aafgc_ion_729.py so the
# rebuilt eps is identical to the optimized design.

# %%
# --- constants copied from examples_2/grating_coupler/aafgc_ion_729.py ---
WL = 0.729
FCEN = 1 / WL
N_SIN, N_SIO = 2.0, 1.44
RES = 20
DX = 1.0 / RES
NPML = RES
H_FILM = 0.22
OX_BOT, OX_TOP = 3.0, 5.0
H_ION = 68.0
D_LAT = 30.0
L_DEV = 30.0
X0_DEV = 6.0
LX = X0_DEV + L_DEV + D_LAT + 14.0
LZ = OX_BOT + H_FILM + OX_TOP + H_ION + 3.0

T_CW = 150.0
SNAP_EVERY = 1.0
FPS = 24
DS = 2                 # display downsample

# %% [markdown]
# ## Rebuild the design eps
# The saved rho (examples_2/grating_coupler/output/aafgc_ion_729_rho.npy)
# is filtered, projected at BETA=64, and painted into the film layer.

# %%
def build_eps():
    nx, nz = int(LX * RES), int(LZ * RES)
    k0f = int(OX_BOT * RES)
    k1f = k0f + max(int(H_FILM * RES), 4)
    k_surf = int((OX_BOT + H_FILM + OX_TOP) * RES)
    base = torch.full((nx, nz), 1.0, dtype=torch.float64)
    base[:, :k_surf] = N_SIO ** 2
    base[:, k0f:k1f] = N_SIN ** 2
    rho = torch.from_numpy(np.load(
        GDS_DIR / "grating_coupler" / "output" / "aafgc_ion_729_rho.npy"))
    NRX, NRZ = rho.shape
    mg = MaterialGrid((NRX, NRZ), eps_min=N_SIO ** 2, eps_max=N_SIN ** 2)
    region = DesignRegion(mg, ix0=int(X0_DEV * RES),
                          ix1=int((X0_DEV + L_DEV) * RES), iy0=k0f, iy1=k1f)
    r = tanh_projection(conic_filter(rho, 0.12, DX), 64.0)
    eps = region.paint(base, r).detach()
    zc = (k0f + k1f) // 2
    span = int(1.5 * RES)
    p_in = Port(axis="x", index=NPML + 8, lo=zc - span, hi=zc + span,
                direction=+1)
    return eps, p_in, zc

# %% [markdown]
# ## meep-torch CW video
# CW TE0 mode source in the feed waveguide, GPU FDTD, snapshots every
# time unit rendered to mp4; the last frame is kept for the parity check.

# %%
def torch_video(eps, p_in):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = Grid(nx=eps.shape[0], ny=eps.shape[1], dx=DX, device=dev)
    coeffs = build_tm_coeffs(grid, eps_inf=eps.to(dev), npml=NPML)
    src = ModeSourceTM(grid, coeffs, p_in, ContinuousSource(frequency=FCEN),
                       FCEN)
    h_masks = getattr(src, "h_masks", None) or {}
    state = init_tm_state(grid)
    every = int(round(SNAP_EVERY / grid.dt))
    snaps = []
    with torch.no_grad():
        for t_idx in range(int(round(T_CW / grid.dt))):
            state = step_tm(state, coeffs, src.amplitude(t_idx), src.mask,
                            src.h_amplitude(t_idx), **h_masks)
            if (t_idx + 1) % every == 0:
                snaps.append(state.Ez[::DS, ::DS].cpu().numpy()
                             .astype(np.float16))
    snaps = np.stack(snaps).astype(np.float32)
    vmax = np.percentile(np.abs(snaps[3 * len(snaps) // 4:]), 99.9)
    fig, ax = plt.subplots(figsize=(8, 8 * LZ / LX + 0.8))
    ext = [0, LX, 0, LZ]
    im = ax.imshow(snaps[0].T, origin="lower", cmap="RdBu",
                   vmin=-vmax, vmax=vmax, extent=ext, aspect="equal")
    ax.axhline(OX_BOT + H_FILM, color="k", lw=0.3)
    ax.axhline(OX_BOT + H_FILM + OX_TOP, color="k", lw=0.3)
    ax.plot([X0_DEV + L_DEV / 2 + D_LAT], [OX_BOT + H_FILM + OX_TOP + H_ION],
            "k+", ms=10)  # ion
    title = ax.set_title("")
    ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
    mp4 = HERE / "aafgc_ion_729_torch_cw_fdtd.mp4"
    writer = FFMpegWriter(fps=FPS)
    with writer.saving(fig, str(mp4), dpi=100):
        for i, fr in enumerate(snaps):
            im.set_data(fr.T)
            title.set_text(f"AAFGC 729nm meep-torch CW Ez  "
                           f"t={(i + 1) * SNAP_EVERY:.0f}")
            writer.grab_frame()
    plt.close(fig)
    print(f"torch CW video -> {mp4.name} ({len(snaps)} frames)")
    return snaps[-1]

# %% [markdown]
# ## pymeep oracle
# The same eps array goes to real pymeep (meep-oracle conda env) via
# epsilon_input_file; Animate2D renders the CW run and the final Ez comes
# back for comparison.

# %%
ORACLE_SCRIPT = r"""
import json, sys
import numpy as np
import meep as mp
import matplotlib
matplotlib.use("Agg")

p = json.load(open(sys.argv[1]))
import h5py
eps = np.load(p["eps_npy"])
with h5py.File(p["eps_h5"], "w") as f:
    f.create_dataset("eps", data=eps)
sim = mp.Simulation(
    cell_size=mp.Vector3(p["LX"], p["LZ"]),
    resolution=p["RES"],
    epsilon_input_file=p["eps_h5"],
    boundary_layers=[mp.PML(p["NPML"] / p["RES"])],
    sources=[mp.EigenModeSource(
        mp.ContinuousSource(p["FCEN"]),
        center=mp.Vector3(p["x_src"] - p["LX"] / 2, p["z_src"] - p["LZ"] / 2),
        size=mp.Vector3(0, p["span"]),
        eig_parity=mp.ODD_Z)],
)
animate = mp.Animate2D(fields=mp.Ez, realtime=False, normalize=True,
                       eps_parameters={"contour": False, "alpha": 0.25},
                       field_parameters={"alpha": 0.9, "cmap": "RdBu",
                                         "interpolation": "none"})
sim.run(mp.at_every(p["snap"], animate), until=p["T"])
animate.to_mp4(p["fps"], p["mp4"])
ez = sim.get_array(center=mp.Vector3(), size=mp.Vector3(p["LX"], p["LZ"]),
                   component=mp.Ez)
np.savez(p["out_npz"], ez=ez)
print("MEEP_DONE")
"""


def meep_video(eps, p_in, zc, tmp):
    npy = tmp / "aafgc_eps.npy"
    np.save(npy, eps.numpy())
    params = dict(LX=LX, LZ=LZ, RES=RES, NPML=NPML, FCEN=FCEN,
                  eps_npy=str(npy), eps_h5=str(tmp / "aafgc_eps.h5"),
                  x_src=(p_in.index) * DX, z_src=zc * DX,
                  span=3.0, snap=SNAP_EVERY, T=T_CW, fps=FPS,
                  mp4=str(HERE / "aafgc_ion_729_meep_cw_fdtd.mp4"),
                  out_npz=str(tmp / "aafgc_meep_final.npz"))
    pf = tmp / "aafgc_params.json"
    pf.write_text(json.dumps(params))
    sf = tmp / "aafgc_oracle.py"
    sf.write_text(ORACLE_SCRIPT)
    r = subprocess.run([str(ORACLE), str(sf), str(pf)], capture_output=True,
                       text=True, timeout=3600)
    assert "MEEP_DONE" in r.stdout, f"meep failed:\n{r.stderr[-2000:]}"
    print(f"meep CW video -> aafgc_ion_729_meep_cw_fdtd.mp4")
    return np.load(params["out_npz"])["ez"]

# %% [markdown]
# ## Parity check
# Both videos are rendered, then the steady-state intensity envelopes
# (blurred over ~half a wavelength to defeat fringe aliasing) are
# correlated and saved to aafgc_final_fields.npz.

# %%
def main():
    tmp = HERE / "_tmp"
    tmp.mkdir(exist_ok=True)
    eps, p_in, zc = build_eps()
    last_torch = torch_video(eps, p_in)
    last_meep = meep_video(eps, p_in, zc, tmp)
    # compare steady-state intensity ENVELOPES (raw |Ez| fringes are ~3 px
    # at DS=2 and alias under strided sampling; blur ~half a wavelength)
    from scipy.ndimage import uniform_filter
    a = uniform_filter(np.abs(last_torch.astype(np.float64)) ** 2, 5)
    b = uniform_filter(np.abs(last_meep)[::DS, ::DS] ** 2, 5)
    n0, n1 = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:n0, :n1] / a.max(), b[:n0, :n1] / b.max()
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    print(f"steady-state intensity-envelope correlation: {corr:.4f}")
    np.savez(HERE / "aafgc_final_fields.npz",
             torch_ez=last_torch, meep_ez=last_meep, ds=DS, corr=corr)


if __name__ == "__main__":
    main()
