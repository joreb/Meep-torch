# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Device-suite GDSII exports
#
# device-suite: GDSII exports -- ring (a), SiN + LiNbO3 rings (b), and the
# inverse designs (c, binarized density -> contour polygons).
#
# Run: `uv run python examples_2/export_gds.py`
# (each GDS lands in the corresponding device folder's `output/` subdirectory)

# %%
import pathlib

import gdsfactory as gf
import numpy as np
import torch

EX2 = pathlib.Path(__file__).parent


def _out(device_dir):
    d = EX2 / device_dir / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


# %% [markdown]
# ## Ring exports (a, b)
# gdsfactory ring_single/ring_double flattened and remapped onto the SiN
# layer (4, 0) -- the fab-deck convention; the LiNbO3 copy keeps the same
# layer, its material assignment lives in the deck.

# %%
gf.gpdk.PDK.activate()

# All waveguide geometry goes on the SiN layer (4, 0) in every GDS
# (user-specified fab-deck convention; the LiNbO3 copy keeps the same
# layer, its material assignment lives in the deck).
SIN_LAYER = (4, 0)

def _ring_gds(radius, gap, width, length_x=0.01, double=False):
    xs = gf.get_cross_section("strip").copy(width=width, radius_min=min(radius, 3.5))
    fac = gf.components.ring_double if double else gf.components.ring_single
    r = fac(radius=radius, gap=gap, length_x=length_x,
            cross_section=xs).copy()
    r.flatten()                      # remap_layers skips referenced subcells
    r.remap_layers({(1, 0): SIN_LAYER})
    return r


# (a): 674 nm devices -- W=1 um, r=50, racetrack coupler LC=4, gap 0.2
_ring_gds(50.0, 0.2, 1.0, length_x=4.0).write_gds(_out("ring") / "a_ring_SiN.gds")
_ring_gds(50.0, 0.2, 1.0, length_x=4.0, double=True).write_gds(
    _out("ring") / "a_ring_adddrop_SiN.gds")
# (b): poled TFLN SHG ring -- r=50, W=1.2 ridge, single bus gap 0.25
# (QPM poling is a fab layer, not drawn here)
_ring_gds(50.0, 0.25, 1.2, length_x=4.0).write_gds(
    _out("ring_nonlinear") / "b_ring_Lithium_Niobate.gds")
print("export_gds  rings written (a_ring_SiN, a_ring_adddrop_SiN, b_ring_Lithium_Niobate)")


# %% [markdown]
# ## Density -> polygon conversion
# contourf (not contour) closes regions along the domain boundary and turns
# cladding islands into hole rings; gdstk.boolean cuts the holes into
# GDS-legal keyholes.

# %%
def design_to_gds(npz_or_recompute, name, out_dir):
    """Binarized design density -> filled-region polygons -> GDS.

    contourf (NOT contour) so regions touching the domain boundary come back
    closed along it, and cladding islands come back as hole rings instead of
    solid polygons; gdstk.boolean cuts the holes into GDS-legal keyholes.
    """
    import gdstk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho, extent = npz_or_recompute
    fig, ax = plt.subplots()
    cs = ax.contourf(rho.T, levels=[0.5, rho.max() + 1.0],
                     extent=[extent[0], extent[1], extent[2], extent[3]])
    exteriors, holes = [], []
    for path in cs.get_paths():
        for ring in path.to_polygons():
            r = np.asarray(ring)
            if len(r) < 3:
                continue
            rn = np.roll(r, -1, axis=0)   # shoelace (np.cross 2D gone in numpy 2)
            signed2 = float((r[:, 0] * rn[:, 1] - rn[:, 0] * r[:, 1]).sum())
            (exteriors if signed2 > 0 else holes).append(gdstk.Polygon(r))
    plt.close(fig)
    comp = gf.Component(name)
    npoly = 0
    for p in gdstk.boolean(exteriors, holes, "not"):
        if len(p.points) >= 3:
            comp.add_polygon(np.asarray(p.points), layer=SIN_LAYER)
            npoly += 1
    comp.write_gds(out_dir / f"{name}.gds")
    print(f"export_gds  {name}.gds written ({npoly} polygons)")


# %% [markdown]
# ## Inverse-design exports (c)
# Each c script saves its final rho as npy in its own output/ folder; here the
# PROJECTED density (conic filter + tanh projection at final beta -- the
# fabricated shape) is contoured with that script's pixel and filter settings.

# %%
# --- (c): recompute the two final designs deterministically ---
def final_design(script):
    """Re-run the optimization script's deterministic loop is expensive;
    instead re-derive the design from the saved rho if present, else run
    the script (it prints and we import its artifacts)."""
    raise NotImplementedError


# store rho at optimization end: the c scripts are rerun here quickly in
# import mode? ponytail: re-run both scripts IS the deterministic source,
# but they take minutes. Instead the scripts now save rho (npy) -- see
# the small patches applied alongside this file.
# all current designs are FULL-domain rhos; contour the PROJECTED density
# (conic filter + tanh_projection at final beta -- the fabricated shape),
# with each script's own design-pixel size and filter parameters.
from meep_torch.adjoint.filters import conic_filter, tanh_projection

DESIGNS = {                # name -> (device_dir, pixel_um, filter_radius, grid_dx)
    "c_edge_coupler": ("edge_coupler", 0.10, 0.15, 1 / 20),
    "c_y_splitter": ("y_splitter", 0.08, 0.12, 1 / 30),
    "c_y_coupler_mmi_tapered": ("mmi_coupler", 0.08, 0.12, 1 / 30),
}
for name, (ddir, px, fr, dx) in DESIGNS.items():
    npy = _out(ddir) / f"{name}_rho.npy"
    if not npy.exists():
        print(f"export_gds  {name}_rho.npy missing -- "
              f"run examples_2/{ddir}/{name}.py first")
        continue
    rho = torch.from_numpy(np.load(npy)).to(torch.float64)
    r = tanh_projection(conic_filter(rho, fr, dx), 64.0).cpu().numpy()
    design_to_gds((r, (0, r.shape[0] * px, 0, r.shape[1] * px)), name, _out(ddir))
