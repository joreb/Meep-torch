# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""A centred source must produce a symmetric field.

WHY SYMMETRY AND NOT A COMPARISON AGAINST MEEP. The first attempt to check
get_array against pymeep compared an instantaneous ContinuousWave snapshot and
read a 73% difference that meant nothing: at t = 2 both engines are in their
turn-on transient, and the correlation was -0.17 with 33% sign agreement -- not
even a phase offset, just two ramps sampled at an arbitrary instant.

Symmetry avoids all of that. It is a property of a SINGLE snapshot, so it needs
no oracle, no phase alignment and no second engine, and it is sensitive to
exactly the error being hunted: a source displaced by half a cell breaks the
mirror symmetry of an otherwise symmetric problem, while leaving every
integrated quantity looking fine (a full-cell source move changes the flux
through a wide plane by 2.5e-16 -- see wiki/bugs/point-source-snapped-not-
interpolated.md).

pymeep's numbers for the same setup are the target: 3.1e-16 on Ez, 1.4e-16 on
Hx, 2.6e-16 on Hy, 8.6e-16 on Hz.
"""
import numpy as np
import pytest

import meep_torch.mp as mp

RES, CELL = 20, 3.0


def _run(component):
    sim = mp.Simulation(resolution=RES, cell_size=mp.Vector3(CELL, CELL),
                        boundary_layers=[mp.PML(1.0)],
                        sources=[mp.Source(mp.ContinuousSource(1.0),
                                           component=component,
                                           center=mp.Vector3(0, 0))])
    sim.run(until=2)
    return sim


def _mirror_residual(a, axis):
    """max |a - flip(a)| / max|a| -- 0 for a field even about the cell centre."""
    return float(np.abs(a - np.flip(a, axis=axis)).max() / max(np.abs(a).max(), 1e-30))


def test_centred_geometry_is_symmetric():
    """The GRID is not the suspect: a centred block must mirror exactly.

    This runs first so that a failure of the source test below cannot be
    blamed on the grid. Measured: residual 0 here, 8.9e-16 in pymeep.
    """
    sim = mp.Simulation(resolution=RES, cell_size=mp.Vector3(CELL, CELL),
                        geometry=[mp.Block(size=mp.Vector3(1, 1),
                                           center=mp.Vector3(0, 0),
                                           material=mp.Medium(epsilon=12))],
                        sources=[])
    eps = np.asarray(sim.get_epsilon())
    for axis in (0, 1):
        assert _mirror_residual(eps, axis) < 1e-12


def test_centred_point_source_gives_a_symmetric_field():
    """Ez from a centred Ez source is even in both x and y.

    A point at x = 0 on a 60-cell grid sits at grid coordinate 29.5 in cell
    centres -- exactly between two of them. meep splits it, giving a 2x2
    footprint symmetric about that half-integer axis; we round to a single
    cell, so the field mirrors about an axis one cell away.
    """
    ez = np.asarray(_run(mp.Ez).get_array(component="Ez"))
    for axis in (0, 1):
        assert _mirror_residual(ez, axis) < 1e-9
