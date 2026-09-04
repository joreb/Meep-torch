# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Per-direction/per-side PML (meep mp.PML(direction=..., side=...)):
layers apply only on their edges; edges with no layer are plain PEC walls."""
import numpy as np

import meep_torch.mp as mp


def _late_energy(boundary_layers, until=24.0):
    """2D vacuum pulse from the cell center; total sum(Ez^2) after the pulse
    has crossed the cell several times (t=24 in an 8x8 cell)."""
    sim = mp.Simulation(
        cell_size=mp.Vector3(8, 8), resolution=10,
        boundary_layers=boundary_layers,
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5),
                           component=mp.Ez, center=mp.Vector3())])
    sim.run(until=until)
    ez = np.asarray(sim.get_array(component=mp.Ez))
    return float((ez ** 2).sum())


def test_pml_direction_x_absorbs_x_but_y_pec_reflects():
    e_all = _late_energy([mp.PML(1.0)])
    e_x = _late_energy([mp.PML(1.0, direction=mp.X)])
    # full PML drains the pulse; x-only PML leaves the y-bouncing energy
    # trapped between the PEC y walls
    assert e_x > 50 * e_all
    assert e_x > 1e-6


def _late_field_1d(boundary_layers, until=30.0):
    """1D vacuum pulse from the cell center (cell along z -> engine x axis);
    |Ex| profile after the pulse has reached both ends and come back."""
    sim = mp.Simulation(
        cell_size=mp.Vector3(0, 0, 12), resolution=20, dimensions=1,
        boundary_layers=boundary_layers,
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5),
                           component=mp.Ex, center=mp.Vector3())])
    sim.run(until=until)
    return np.abs(np.asarray(sim.get_array(component=mp.Ex)))


def test_pml_side_high_leaves_low_edge_pec():
    f_both = _late_field_1d([mp.PML(1.0, direction=mp.Z)])
    f_high = _late_field_1d([mp.PML(1.0, direction=mp.Z, side=mp.High)])
    # both-sides PML: nothing comes back; High-only: the -z half reflects
    # off the PEC low wall and survives
    assert f_high.max() > 50 * f_both.max()
    assert f_high.max() > 1e-6


def test_pml_all_skips_axis_with_no_extent():
    """meep's grid_volume has no boundary on a direction of zero extent, so
    direction=ALL must leave a degenerate axis PEC rather than give it a PML
    thicker than the axis itself (upstream refl-angular-kz2d: a 12-unit x-only
    cell with dpml=1 at resolution 100 vs the 3-cell y axis)."""
    sim = mp.Simulation(cell_size=mp.Vector3(12), resolution=100,
                        boundary_layers=[mp.PML(1.0)])
    assert sim._npml_sides[0] == [100, 100]
    assert sim._npml_sides[1] == [0, 0]
