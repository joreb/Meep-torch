# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port: meep/python/examples/edge_emitter_3D.py -- extraction
efficiency of a cleaved multilayer stack via a 3D sim.

Runs through the W9 wave-3 ``mp.Simulation(dimensions=3, ...)`` shim (3D
stepper_3d + monitors_3d). A point dipole sits in the active region of a
substrate / cladding / active / cladding / air stack (all axis-aligned
Blocks, invariant in z); a 3-region flux box measures the power into air
(top +y face + cleaved-facet side +x face, minus the substrate-bound -y face).

Substitutions from the meep original (noted per the W9 wave-3 plan):

* ``mp.Mirror(direction=mp.Z)`` symmetries -> no-op. The shim has no Mirror
  (W1 limitation); the full cell is run instead of the half-cell, ~2x cost.
* ``stop_when_fields_decayed`` -> fixed ``n_steps`` (the 3D runner is
  forward-only, no adaptive stop).
* ``dft_ldos`` + ``ldos_Fdata`` / ``ldos_Jdata`` -> deferred (3D LDOS is not
  this wave). The dipole_flux denominator is not available, so the
  extraction_efficiency ratio is not computed; this port reports the three
  flux-box readings and the air flux (top + side - bottom) only.
* resolution and ``SIDE_Z`` / ``SIDE`` reduced for a fast smoke run (the
  physics is unchanged; the meep original uses resolution 25 and SIDE_Z=10).
"""
import argparse

import meep_torch.mp as mp
from meep_torch.mp import (Block, FluxRegion, GaussianSource, Medium, PML,
                           Simulation, Source, Vector3)

RESOLUTION_UM = 10
WAVELENGTH_UM = 1.0
CLADDING_UM = 0.5
ACTIVE_UM = 0.3
SUBSTRATE_UM = 1.0
PML_UM = 0.5
AIR_UM = 1.0
SIDE_UM = 1.0
SIDE_Z_UM = 2.0            # reduced from 10 for the smoke run
N_SUBSTRATE = 3.2
N_CLADDING = 1.5
N_ACTIVE = 2.3
T_END = 30.0               # fixed run length (no stop_when_fields_decayed)


def _flux_box(dipole_pol: str, dipole_pos_um: float) -> tuple[float, float, float]:
    """Run the cleaved-stack 3D sim; return (top, side, bottom) flux through
    the three flux-box faces (all at the source frequency)."""
    src_cmpt = {"x": mp.Ex, "y": mp.Ey, "z": mp.Ez}[dipole_pol]

    cell_x_um = PML_UM + SIDE_UM + AIR_UM + PML_UM
    cell_y_um = (PML_UM + SUBSTRATE_UM + CLADDING_UM + ACTIVE_UM
                 + CLADDING_UM + AIR_UM + PML_UM)
    cell_z_um = PML_UM + SIDE_Z_UM + PML_UM
    cell_size = Vector3(cell_x_um, cell_y_um, cell_z_um)

    src_pos = Vector3(
        -0.5 * cell_x_um + PML_UM + SIDE_UM - dipole_pos_um,
        -0.5 * cell_y_um + PML_UM + SUBSTRATE_UM + CLADDING_UM + 0.5 * ACTIVE_UM,
        0,
    )
    sources = [Source(src=GaussianSource(1 / WAVELENGTH_UM, fwidth=0.1 / WAVELENGTH_UM),
                      center=src_pos, component=src_cmpt)]

    x0 = -0.5 * cell_x_um + 0.5 * (PML_UM + SIDE_UM)
    geometry = [
        Block(material=Medium(index=N_SUBSTRATE),
              center=Vector3(x0, -0.5 * cell_y_um + 0.5 * (PML_UM + SUBSTRATE_UM), 0),
              size=Vector3(PML_UM + SIDE_UM, PML_UM + SUBSTRATE_UM, mp.inf)),
        Block(material=Medium(index=N_CLADDING),
              center=Vector3(x0,
                             -0.5 * cell_y_um + PML_UM + SUBSTRATE_UM + 0.5 * CLADDING_UM, 0),
              size=Vector3(PML_UM + SIDE_UM, CLADDING_UM, mp.inf)),
        Block(material=Medium(index=N_ACTIVE),
              center=Vector3(x0, -0.5 * cell_y_um + PML_UM + SUBSTRATE_UM
                             + CLADDING_UM + 0.5 * ACTIVE_UM, 0),
              size=Vector3(PML_UM + SIDE_UM, ACTIVE_UM, mp.inf)),
        Block(material=Medium(index=N_CLADDING),
              center=Vector3(x0, -0.5 * cell_y_um + PML_UM + SUBSTRATE_UM
                             + CLADDING_UM + ACTIVE_UM + 0.5 * CLADDING_UM, 0),
              size=Vector3(PML_UM + SIDE_UM, CLADDING_UM, mp.inf)),
    ]

    sim = Simulation(resolution=RESOLUTION_UM, cell_size=cell_size,
                     sources=sources, geometry=geometry,
                     boundary_layers=[PML(thickness=PML_UM)], dimensions=3)

    freq = 1 / WAVELENGTH_UM
    # Flux box: top (+y, into air), side (+x, cleaved facet), bottom (-y,
    # into substrate; subtracted). The meep original hands all three to one
    # add_flux call with a -1 weight on the bottom; the shim takes one region
    # per call, so they are summed here by hand.
    sim.add_flux(freq, 0, 1,
                 FluxRegion(center=Vector3(-0.5 * cell_x_um + PML_UM
                                           + 0.5 * (cell_x_um - 2 * PML_UM),
                                           0.5 * cell_y_um - PML_UM, 0),
                            size=Vector3(cell_x_um - 2 * PML_UM, 0,
                                         cell_z_um - 2 * PML_UM)), name="top")
    sim.add_flux(freq, 0, 1,
                 FluxRegion(center=Vector3(0.5 * cell_x_um - PML_UM,
                                           0.5 * cell_y_um - PML_UM
                                           - 0.5 * (cell_y_um - 2 * PML_UM), 0),
                            size=Vector3(0, cell_y_um - 2 * PML_UM,
                                         cell_z_um - 2 * PML_UM)), name="side")
    sim.add_flux(freq, 0, 1,
                 FluxRegion(center=Vector3(0.5 * cell_x_um - PML_UM - 0.5 * AIR_UM,
                                           -0.5 * cell_y_um + PML_UM, 0),
                            size=Vector3(AIR_UM, 0, cell_z_um - 2 * PML_UM)),
                 name="bottom")

    sim.run(until=T_END)
    top = float(sim.get_fluxes("top")[0])
    side = float(sim.get_fluxes("side")[0])
    bottom = float(sim.get_fluxes("bottom")[0])
    return top, side, bottom


def dipole_in_cleaved_stack(dipole_pol: str, dipole_pos_um: float):
    """Return (air_flux, dipole_flux). ``dipole_flux`` (the LDOS denominator
    of the extraction efficiency) is deferred to a 3D LDOS wave and returned
    as None here; ``air_flux`` = top + side - bottom is the power into air."""
    top, side, bottom = _flux_box(dipole_pol, dipole_pos_um)
    air_flux = top + side - bottom
    return air_flux, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dipole_pol", type=str, choices=["x", "y", "z"],
                        help="polarization of the electric dipole (x, y, z)")
    parser.add_argument("dipole_pos_um", type=float,
                        help="position of the dipole relative to the cleaved facet")
    args = parser.parse_args()
    air_flux, dipole_flux = dipole_in_cleaved_stack(args.dipole_pol, args.dipole_pos_um)
    print(f"air_flux:, {args.dipole_pol}, {args.dipole_pos_um}, {air_flux:.6g}")
    print("# extraction_efficiency deferred: 3D dft_ldos denominator not implemented (W9 wave 3)")