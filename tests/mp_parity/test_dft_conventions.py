# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""The DFT conventions, pinned against answers that are exact.

Both checks here avoid an oracle on purpose: they ask questions whose answer
is known in closed form, so they stay meaningful with no pymeep installed.

Vacuum has eps = 1, so a DFT of D and a DFT of E taken at the same instant
must have ratio exactly 1 + 0j. `core/monitors.py` chose the DFT time offset
with `component.startswith("E")`, and "Dz" does not, so D carried the
MAGNETIC half-step: the measured ratio was 0.99692 + 0.07846j, a phase of
omega*dt/2 to six digits. Nothing raised on it. It is ruinous for absorbed
power `Im(conj(E) D)`, whose signal for SiO2 is `Im(eps)/|eps| = 4.6e-4` --
some 68x SMALLER than the error -- and it made
`examples/11_upstream/absorbed_power_density` report 29.63 against meep's
0.131.

The same line had been written four times (`core/monitors.py`,
`core/monitors_3d.py`, `core/near2far_cyl.py`, `mp/cyl_shim.py`), so the
offset now has one home and the second test pins THAT rather than one caller.
"""
import numpy as np
import pytest

import meep_torch.mp as mp
from meep_torch.core.monitors import dft_time_offset


@pytest.mark.parametrize("component", ["Ez", "Dz", "Ex", "Dy", "Er"])
def test_electric_family_shares_one_offset(component):
    """E and D are the same family and must be stamped at the same time."""
    assert dft_time_offset(component) == 1.0


@pytest.mark.parametrize("component", ["Hy", "By", "Hx", "Bz", "Hp"])
def test_magnetic_family_is_the_half_step(component):
    assert dft_time_offset(component) == 0.5


def test_vacuum_dft_of_D_equals_dft_of_E():
    """eps = 1, so D/E is exactly 1 + 0j at every point the field reaches."""
    fcen = 1.0
    sim = mp.Simulation(
        resolution=20, cell_size=mp.Vector3(8, 8), k_point=mp.Vector3(),
        boundary_layers=[mp.PML(thickness=1.0)],
        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=0.2),
                           component=mp.Ez, center=mp.Vector3(-3.0))])
    df = sim.add_dft_fields([mp.Dz, mp.Ez], fcen, 0, 1, center=mp.Vector3(),
                            size=mp.Vector3(2, 2), yee_grid=True)
    sim.run(until_after_sources=40)
    D = np.asarray(sim.get_dft_array(df, mp.Dz, 0))
    E = np.asarray(sim.get_dft_array(df, mp.Ez, 0))
    live = np.abs(E) > 0.1 * np.abs(E).max()
    assert live.sum() > 10, "probe is not seeing the field"
    ratio = D[live] / E[live]
    # A half-step error would put omega*dt/2 = 0.0785 rad of phase here, so
    # the tolerance is two orders below the bug it exists to catch.
    assert np.abs(ratio - 1.0).max() < 1e-9
