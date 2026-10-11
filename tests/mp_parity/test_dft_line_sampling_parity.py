# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""DFT and zero-thickness line sampling, against pymeep's conventions.

Measured with pymeep on a 10x6 cell, resolution 20, Block eps=13 on |y|<0.6:

* get_array(Hz, center=(0, 2.8), size=(9, 0)) is a 182-point line whose
  coordinate is exactly y = 2.8 (the requested point, not a cell centre). Its
  values are the mean of the centre rows at y = 2.775 and 2.825 (difference
  0.0 to double precision).

The shim used to return the nearest row (y = 2.825), and a DFT Hz plane
came back raw from the Yee corner lattice, one half cell off the centres
get_array reports. Both defects sat in the shared sampling path, so the
checks below go through get_dft_array and get_array_metadata as well.
"""
import numpy as np

import meep_torch.mp as mp

# pymeep numbers (see the docstring): point count, x extent, y coordinate
MEEP_LINE_NPTS = 182
MEEP_LINE_X = (-4.525, 4.525)
MEEP_LINE_Y = 2.8


def _sim():
    return mp.Simulation(
        cell_size=mp.Vector3(10.0, 6.0, 0),
        geometry=[mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, 1.2, mp.inf),
                           material=mp.Medium(epsilon=13))],
        sources=[mp.Source(mp.GaussianSource(0.25, fwidth=0.2), component=mp.Hz,
                           center=mp.Vector3(0.3, 0.2))],
        boundary_layers=[mp.PML(1.0)],
        resolution=20,
    )


def test_line_metadata_matches_pymeep_point_and_coordinate():
    sim = _sim()
    x, y, _, _ = sim.get_array_metadata(center=mp.Vector3(0, 2.8), size=mp.Vector3(9.0, 0))
    assert len(x) == MEEP_LINE_NPTS
    np.testing.assert_allclose([x[0], x[-1]], MEEP_LINE_X, atol=1e-9)
    np.testing.assert_allclose(y, [MEEP_LINE_Y], atol=1e-9)


def test_dft_plane_is_on_centres_and_line_is_the_interpolated_point():
    sim = _sim()
    full_mon = sim.add_dft_fields([mp.Hz], 0.25, 0, 1,
                                  center=mp.Vector3(), size=mp.Vector3(10.0, 6.0))
    line_mon = sim.add_dft_fields([mp.Hz], 0.25, 0, 1,
                                  center=mp.Vector3(0, 2.8), size=mp.Vector3(9.0, 0))
    sim.run(until=12)

    full = np.asarray(sim.get_dft_array(full_mon, mp.Hz, 0))
    line = np.asarray(sim.get_dft_array(line_mon, mp.Hz, 0))
    # the DFT plane has the shape get_array reports for the same component
    assert full.shape == sim.get_array(component=mp.Hz).shape == (200, 120)
    assert line.shape == (MEEP_LINE_NPTS,)

    fx, fy, _, _ = sim.get_array_metadata()
    lx, ly, _, _ = sim.get_array_metadata(dft_cell=line_mon)
    np.testing.assert_allclose(ly, [MEEP_LINE_Y], atol=1e-9)
    # y = 2.8 is a cell edge between the centres 2.775 and 2.825: the line is
    # their mean, column by column (meep: get_array at 2.8 == mean of the rows)
    j1 = int(np.argmin(np.abs(np.asarray(fy) - 2.775)))
    j2 = int(np.argmin(np.abs(np.asarray(fy) - 2.825)))
    ix = [int(np.argmin(np.abs(np.asarray(fx) - v))) for v in lx]
    np.testing.assert_allclose(line, 0.5 * (full[ix, j1] + full[ix, j2]),
                               rtol=1e-12, atol=1e-30)
