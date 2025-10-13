# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite: GDSII exports -- ring (a), SiN + LiNbO3 rings (b), and the
two inverse designs (c, binarized density -> contour polygons).

Run: uv run python examples_2/export_gds.py
"""
import pathlib
import sys

import gdsfactory as gf
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()

# All waveguide geometry goes on the SiN layer (4, 0) in every GDS
# (user-specified fab-deck convention; the LiNbO3 copy keeps the same
# layer, its material assignment lives in the deck).
SIN_LAYER = (4, 0)

def _ring_gds(radius, gap):
    # gpdk strip DRC has radius_min=3.5; the r=3 nonlinear device is a
    # simulation geometry, so relax the floor (same override as b_ring sim)
    xs = gf.get_cross_section("strip").copy(radius_min=radius)
    r = gf.components.ring_single(radius=radius, gap=gap,
                                  cross_section=xs).copy()
    r.flatten()                      # remap_layers skips referenced subcells
    r.remap_layers({(1, 0): SIN_LAYER})
    return r


# (a): r=5 gap=0.2 (S-matrix device); (b): r=3 gap=0.30 (nonlinear device,
# single FSR-spaced mode in band -- see b_ring_nonlinear redo)
_ring_gds(5.0, 0.2).write_gds(OUT / "a_ring_SiN.gds")
ring_b = _ring_gds(3.0, 0.30)
ring_b.write_gds(OUT / "b_ring_SiN.gds")
ring_b.write_gds(OUT / "b_ring_Lithium_Niobate.gds")
print("export_gds  rings written (a_ring_SiN, b_ring_SiN, b_ring_Lithium_Niobate)")


def design_to_gds(npz_or_recompute, name):
    """Binarized design density -> contour polygons -> GDS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho, extent = npz_or_recompute
    fig, ax = plt.subplots()
    cs = ax.contour(rho.T, levels=[0.5],
                    extent=[extent[0], extent[1], extent[2], extent[3]])
    comp = gf.Component(name)
    npoly = 0
    for path in cs.allsegs[0]:
        if len(path) >= 3:
            comp.add_polygon(np.asarray(path), layer=SIN_LAYER)
            npoly += 1
    plt.close(fig)
    comp.write_gds(OUT / f"{name}.gds")
    print(f"export_gds  {name}.gds written ({npoly} polygons)")


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
for name in ("c_edge_coupler", "c_y_splitter"):
    npy = OUT / f"{name}_rho.npy"
    if not npy.exists():
        print(f"export_gds  {name}_rho.npy missing -- run examples_2/{name}.py first")
        continue
    rho = np.load(npy)
    design_to_gds((rho, (0, rho.shape[0] * 0.05, 0, rho.shape[1] * 0.05)), name)
