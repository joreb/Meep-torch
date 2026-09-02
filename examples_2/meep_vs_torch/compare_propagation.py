# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Side-by-side propagation comparison
#
# Side-by-side FDTD propagation: meep-torch vs real pymeep on the SAME
# exported GDS geometry (setup shared with
# examples_2/test_gds_meep_equivalence.py; the device GDS files come from
# each device folder's output/ directory).
#
# For each device the Ez field is snapshotted at several times in both
# engines, normalized per engine (global max), and rendered as a panel
# grid [meep-torch | pymeep | difference] per time, with the Pearson
# correlation of the two field patterns annotated per row. Raw stacks
# land in <device>.npz.
#
# Run: `uv run python examples_2/meep_vs_torch/compare_propagation.py`
# Requires the meep-oracle conda env (real pymeep).

# %%
import json
import pathlib
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import test_gds_meep_equivalence as base

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state, step_tm
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene

HERE = pathlib.Path(__file__).parent
SNAP_TIMES = [20.0, 30.0, 40.0, 50.0, 60.0]

# %% [markdown]
# ## pymeep oracle script
# Runs in the meep-oracle env on the cleaned GDS: same cell, source, and
# snapshot times as the meep-torch run below.

# %%
ORACLE_SNAP_SCRIPT = r"""
import json, sys
import numpy as np
import meep as mp

p = json.load(open(sys.argv[1]))
prisms = mp.get_GDSII_prisms(mp.Medium(index=p["n_core"]), p["gds"],
                             p["layer"], -100, 100)
# geometry_center frame: every coordinate below is a raw GDS coordinate
sim = mp.Simulation(
    cell_size=mp.Vector3(p["sx"], p["sy"]),
    resolution=p["res"],
    geometry=prisms,
    geometry_center=mp.Vector3(p["cx"], p["cy"]),
    default_material=mp.Medium(index=p["n_clad"]),
    boundary_layers=[mp.PML(p["pml"])],
    sources=[mp.Source(
        mp.GaussianSource(p["fcen"], fwidth=p["fwidth"]),
        component=mp.Ez,
        center=mp.Vector3(p["x_src"], p["y_c"]),
        size=mp.Vector3(0, p["y_span"]))],
)
snaps, prev = [], 0.0
for t in p["times"]:
    sim.run(until=t - prev)
    prev = t
    snaps.append(sim.get_array(center=mp.Vector3(p["cx"], p["cy"]),
                               size=mp.Vector3(p["sx"], p["sy"]),
                               component=mp.Ez))
np.savez(p["out"], ez=np.stack(snaps), times=np.array(p["times"]))
print("SNAPSHOTS_SAVED")
"""

# %% [markdown]
# ## Snapshot runners
# torch_snapshots rasterizes the polygons and steps to each snapshot
# time; meep_snapshots ships the same parameters to the oracle env.

# %%
def torch_snapshots(polys, ox, oy, sx, sy, x_src, y_lo, y_hi):
    grid = Grid(nx=int(round(sx * base.RES)), ny=int(round(sy * base.RES)),
                dx=base.DX)
    items = [(Polygon(v - [ox, oy]), base.N_CORE ** 2) for v in polys]
    eps = rasterize_scene(items, grid, background=base.N_CLAD ** 2).eps_tm
    coeffs = build_tm_coeffs(grid, eps_inf=eps,
                             npml=int(round(base.PMLT * base.RES)))
    src = LineSource(grid, ix=int(round((x_src - ox) * base.RES)),
                     j0=int(round((y_lo - oy) * base.RES)),
                     j1=int(round((y_hi - oy) * base.RES)),
                     profile=GaussianPulse(base.FCEN, base.FWIDTH))
    state = init_tm_state(grid)
    snap_steps = [int(round(t / grid.dt)) for t in SNAP_TIMES]
    snaps, t_idx = [], 0
    for target in snap_steps:
        while t_idx < target:
            state = step_tm(state, coeffs, src.amplitude(t_idx), src.mask)
            t_idx += 1
        snaps.append(state.Ez.detach().cpu().numpy().copy())
    return np.stack(snaps), eps.numpy()


def meep_snapshots(clean_gds, ox, oy, sx, sy, x_src, y_lo, y_hi, out_npz):
    params = dict(
        gds=str(clean_gds), layer=base.LAYER, n_core=base.N_CORE,
        n_clad=base.N_CLAD, sx=sx, sy=sy, cx=ox + sx / 2, cy=oy + sy / 2,
        pml=base.PMLT, res=base.RES, fcen=base.FCEN, fwidth=base.FWIDTH,
        x_src=x_src, y_c=(y_lo + y_hi) / 2, y_span=y_hi - y_lo,
        times=SNAP_TIMES, out=str(out_npz),
    )
    pfile = out_npz.with_suffix(".params.json")
    pfile.write_text(json.dumps(params))
    sfile = out_npz.with_suffix(".oracle.py")
    sfile.write_text(ORACLE_SNAP_SCRIPT)
    out = subprocess.run([str(base.ORACLE), str(sfile), str(pfile)],
                         capture_output=True, text=True, timeout=1800)
    assert "SNAPSHOTS_SAVED" in out.stdout, f"oracle failed:\n{out.stderr[-2000:]}"
    pfile.unlink(); sfile.unlink()
    return np.load(out_npz)["ez"]

# %% [markdown]
# ## Per-device comparison
# Crop to a common shape, align source polarity, compute per-snapshot
# Pearson correlations, and render the panel grid.

# %%
def compare(ddir, cell):
    polys = base.load_clean_polys(base.HERE / ddir / "output" / f"{cell}.gds", cell)
    clean_gds = HERE / f"{cell}_clean.gds"
    base.write_clean_gds(polys, cell, clean_gds)
    x0, x1, ox, oy, sx, sy = base._domain(polys)
    x_src = x0 - 0.6
    y_lo, y_hi = oy + base.PMLT + 0.2, oy + sy - base.PMLT - 0.2

    ours, eps = torch_snapshots(polys, ox, oy, sx, sy, x_src, y_lo, y_hi)
    theirs = meep_snapshots(clean_gds, ox, oy, sx, sy, x_src, y_lo, y_hi,
                            HERE / f"{cell}_meep.npz")

    # crop to common shape (meep get_array is one cell larger), one
    # engine-global scale each so relative amplitude across time survives
    n0, n1 = min(ours.shape[1], theirs.shape[1]), min(ours.shape[2], theirs.shape[2])
    a = ours[:, :n0, :n1] / np.abs(ours).max()
    b = theirs[:, :n0, :n1] / np.abs(theirs).max()
    corr_last = np.corrcoef(a[-1].ravel(), b[-1].ravel())[0, 1]
    if corr_last < 0:          # opposite source polarity convention
        b = -b

    corrs = [np.corrcoef(a[i].ravel(), b[i].ravel())[0, 1] for i in range(len(a))]
    np.savez(HERE / f"{cell}_compare.npz", torch_ez=a, meep_ez=b,
             times=np.array(SNAP_TIMES), corr=np.array(corrs), eps=eps)

    nrow = len(SNAP_TIMES)
    fig, axes = plt.subplots(nrow, 3, figsize=(12, 2.2 * nrow),
                             constrained_layout=True)
    vmax = max(np.abs(a).max(), np.abs(b).max())
    ext = [ox, ox + sx, oy, oy + sy]
    for i, t in enumerate(SNAP_TIMES):
        for j, (fld, name) in enumerate([(a[i], "meep-torch"), (b[i], "pymeep"),
                                         (a[i] - b[i], "difference")]):
            ax = axes[i, j]
            ax.imshow(fld.T, origin="lower", cmap="RdBu", vmin=-vmax, vmax=vmax,
                      extent=ext, aspect="equal")
            ax.contour(eps[:n0, :n1].T, levels=[np.mean([base.N_CLAD ** 2,
                       base.N_CORE ** 2])], extent=ext, colors="k",
                       linewidths=0.3)
            if i == 0:
                ax.set_title(name)
            if j == 0:
                ax.set_ylabel(f"t={t:g}\ncorr={corrs[i]:.4f}")
    fig.suptitle(f"{cell}: Ez propagation, meep-torch vs pymeep "
                 f"(res={base.RES}, {base.WL * 1e3:.0f} nm)")
    png = HERE / f"{cell}_propagation.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"{cell}: corr per snapshot {[round(c, 4) for c in corrs]} -> {png.name}")
    return corrs

# %% [markdown]
# ## Run all cases

# %%
if __name__ == "__main__":
    for ddir, cell in base.CASES:
        compare(ddir, cell)
