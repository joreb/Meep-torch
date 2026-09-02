# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # gmeep FDTD on the exported designs
#
# gdsfactory + gplugins.gmeep FDTD on the exported design GDS files,
# following https://gdsfactory.github.io/gplugins/notebooks/meep_01_sparameters/
#
# Per device: gf.import_gds -> auto-detect waveguide stubs at the left/right
# bbox edges -> add ports -> (a) gm.write_sparameters_meep (pulse source in
# the input waveguide, mode monitors on every output waveguide) ->
# S-parameters saved to npz for comparison against meep-torch; (b) a second
# run with continuous_source=True recorded with mp.Animate2D -> CW FDTD
# video (mp4). The design GDS files are read from each device folder's
# output/ directory.
#
# Run:
# `~/miniforge3/envs/meep-oracle/bin/python examples_2/meep_vs_torch/gmeep_fdtd.py`
# (real pymeep env; gdsfactory + gplugins installed there)

# %%
import pathlib

import gdsfactory as gf
import matplotlib

matplotlib.use("Agg")
import meep as mp
import numpy as np
import gplugins.gmeep as gm
from gdsfactory.technology import LayerLevel, LayerStack
from gdsfactory.technology.layer_stack import LogicalLayer
from gplugins.gmeep.get_simulation import get_simulation

gf.gpdk.PDK.activate()

# %% [markdown]
# ## Compatibility patch
# gplugins 2.1.5 compares component layer tuples against LogicalLayer
# stack keys; gdsfactory 9.x LogicalLayer.__eq__ raises on that, so it is
# patched to compare by (layer, datatype).

# %%
# gplugins 2.1.5 compares component layer TUPLES against LogicalLayer stack
# keys; gdsfactory 9.x LogicalLayer.__eq__ raises on that instead of
# comparing -- patch it to compare by (layer, datatype)
def _logical_eq(self, other):
    if isinstance(other, LogicalLayer):
        return tuple(self.layer) == tuple(other.layer)
    if isinstance(other, (tuple, list)):
        return tuple(self.layer) == tuple(other)
    return NotImplemented

LogicalLayer.__eq__ = _logical_eq
LogicalLayer.__hash__ = lambda self: hash(tuple(self.layer))

import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ports_util import (CASES, EXTEND, LAYER, RES, TPML, WL, clamped_margin, design_gds,
                        with_ports)

HERE = pathlib.Path(__file__).parent
GDS_DIR = HERE.parent
CW_TIME = 80.0

# %% [markdown]
# ## Layer stack and materials
# One SiN 150 nm core layer; the exact meep-torch indices at 674 nm so
# both engines simulate identical materials.

# %%
STACK = LayerStack(layers=dict(
    core=LayerLevel(layer=LogicalLayer(layer=LAYER), thickness=0.15,
                    zmin=-0.075, material="sin", mesh_order=2),
))
# exact meep-torch indices @674nm (Si3N4.n, SiO2.n) so both engines
# simulate identical materials; floats -> mp.Medium(index=...)
MAT = {"sin": 2.0347185067116333, "sio2": 1.4652476212523187}

COMMON = dict(
    resolution=RES, layer_stack=STACK, tpml=TPML,
    extend_ports_length=EXTEND, wavelength_start=WL - 0.02,
    wavelength_stop=WL + 0.02, wavelength_points=3,
    material_name_to_meep=MAT,
)

# %% [markdown]
# ## Per-device run
# (a) write_sparameters_meep on port o1, (b) DFT |Ez| mode profiles at
# the S-param monitor planes, (c) an optional CW Animate2D video.

# %%
def run_case(name, video=True):
    c = with_ports(design_gds(GDS_DIR, name), name)
    print(f"{name}: ports {[(p.name, tuple(np.round(p.center, 2)), p.width) for p in c.ports]}")

    # (a) S-parameters, pulse source on o1 (notebook flow)
    # default plane offsets (source 0.1 um inside o1, monitors 0.1 um inside
    # the outputs) -- torch_sparams.py mirrors these exact planes
    sp = gm.write_sparameters_meep(
        c, port_source_names=["o1"], filepath=HERE / f"{name}_sp.npz",
        overwrite=True, dirpath=HERE, verbosity=0,
        port_margin=clamped_margin(c), **COMMON)
    mid = len(sp["wavelengths"]) // 2
    for k in sorted(sp):
        if k != "wavelengths" and k.endswith("o1@0"):
            print(f"  |{k}|^2 = {abs(sp[k][mid]) ** 2:.4f} @ {sp['wavelengths'][mid]:.3f} um")

    # (c) DFT |Ez| mode profiles at the S-param monitor planes
    margin = clamped_margin(c)
    sd = get_simulation(c, is_3d=False, port_source_name="o1",
                        port_margin=margin, **COMMON)
    sim = sd["sim"]
    fcen = 1.0 / WL
    vols, spans = {}, {}
    for p in c.ports:
        inward = 0.3 if p.name == "o1" else 0.1
        x = p.center[0] + (-inward if p.orientation == 0 else inward)
        span = p.width + 2 * margin
        spans[p.name] = (p.center[1] - span / 2, p.center[1] + span / 2)
        vols[p.name] = sim.add_dft_fields(
            [mp.Ez], fcen, fcen, 1,
            where=mp.Volume(center=mp.Vector3(x, p.center[1]),
                            size=mp.Vector3(0, span)))
    decay_pt = mp.Vector3(c.ports["o2"].center[0] - 0.1, c.ports["o2"].center[1])
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        25, mp.Ez, decay_pt, 1e-6))
    prof = {}
    for pn, d in vols.items():
        ez = np.abs(sim.get_dft_array(d, mp.Ez, 0))
        prof[f"{pn}_ez"] = ez
        prof[f"{pn}_ys"] = np.linspace(*spans[pn], len(ez))
    np.savez(HERE / f"{name}_profiles_meep.npz", **prof)
    print(f"{name}: mode profiles -> {name}_profiles_meep.npz")

    # (b) CW-source FDTD video
    if not video:
        return
    sd = get_simulation(c, is_3d=False, port_source_name="o1",
                        continuous_source=True,
                        port_margin=clamped_margin(c), **COMMON)
    sim = sd["sim"]
    animate = mp.Animate2D(fields=mp.Ez, realtime=False, normalize=True,
                           eps_parameters={"contour": True},
                           field_parameters={"alpha": 0.8, "cmap": "RdBu",
                                             "interpolation": "none"})
    sim.run(mp.at_every(0.25, animate), until=CW_TIME)
    mp4 = HERE / f"{name}_cw_fdtd.mp4"
    animate.to_mp4(24, str(mp4))
    print(f"{name}: CW video -> {mp4.name}")

# %% [markdown]
# ## Run all cases

# %%
if __name__ == "__main__":
    sparams_only = "--sparams-only" in sys.argv
    for case in CASES:
        run_case(case, video=not sparams_only)
