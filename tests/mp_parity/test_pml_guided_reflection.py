# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Regression guard: how much of a guided mode the CPML reflects.

A straight waveguide terminated by PML should show a FLAT complex-amplitude
envelope along its axis. Ours does not: the CPML reflects the guided mode at
roughly 0.22 in amplitude with ``dpml=1``, so the guide carries a standing
wave. Measured 2026-09-07 at the full tutorial cell size (16 x 32, until=300):

    meep-torch   depth 0.227   (uniform along the guide -> a reflector)
    real meep    depth 0.064   (decaying -> mode beating, no reflector)

This is the cause of the bend-flux reflectance disagreement: bend-flux gets R
by subtracting two large nearly equal fluxes, and its reflection plane sits in
a cavity between this PML and the bend corner, so the reflector becomes a
frequency fringe on R (torch mean 0.313 oscillating, meep 0.221 smooth) while
T, measured past the bend, stays fine.

The gate below is a REGRESSION GUARD at the value we currently have, not a
parity gate -- it fails if the reflection gets worse, and it is deliberately
loose. When the CPML is fixed, tighten ``MAX_DEPTH`` toward meep's 0.064 and
turn this into a parity gate.

Do not "fix" this by raising sigma: the reflection scales UP with sigma_max
(R0 1e-3/1e-6/1e-15 -> depth 0.184/0.227/0.302) and DOWN with layer
thickness (dpml 1/2/4 um -> 0.227/0.090/0.059). Full evidence, seven refuted
hypotheses and one invalid test not to repeat:
wiki/tasks/bend-flux_reflection-ripple.md
"""
import numpy as np

import meep_torch.mp as mp

RES = 10
DPML = 1.0
W = 1.0          # guide width
FCEN = 0.15
# A 16 x 12 cell reproduces the 16 x 32 tutorial number (0.230 vs 0.229) in
# about a tenth of the time; the transverse padding does not matter, the
# reflection is off the x layers.
SX, SY = 16, 12
UNTIL = 200      # long enough for the CW field to settle (checked: identical at 300)

MAX_DEPTH = 0.25     # currently 0.230; meep is 0.064


def _guide_sim():
    return mp.Simulation(
        cell_size=mp.Vector3(SX, SY, 0), resolution=RES,
        boundary_layers=[mp.PML(DPML)],
        geometry=[mp.Block(size=mp.Vector3(mp.inf, W, mp.inf),
                           center=mp.Vector3(),
                           material=mp.Medium(epsilon=12))],
        sources=[mp.Source(mp.ContinuousSource(FCEN), component=mp.Ez,
                           center=mp.Vector3(-0.5 * SX + DPML, 0, 0),
                           size=mp.Vector3(0, W, 0))])


def _axis_amplitude(sim):
    """|Ez| complex amplitude along the guide axis, between the source and the
    far PML. Two snapshots a quarter period apart give the amplitude without
    the phase scrambling a single CW snapshot suffers."""
    a = np.asarray(sim.get_array(component=mp.Ez))
    sim.run(until=1 / FCEN / 4)
    c = np.asarray(sim.get_array(component=mp.Ez))
    F = np.abs(a + 1j * c)
    lo = int(round((-0.5 * SX + DPML + 0.5 + SX / 2) * RES))
    hi = int(round((0.5 * SX - DPML - 0.5 + SX / 2) * RES))
    return F[lo:hi, F.shape[1] // 2]


def test_pml_guided_reflection_does_not_regress():
    sim = _guide_sim()
    sim.run(until=UNTIL)
    p = _axis_amplitude(sim)
    depth = (p.max() - p.min()) / (p.max() + p.min())
    far = p[len(p) // 2:]
    depth_far = (far.max() - far.min()) / (far.max() + far.min())
    print(f"CPML guided-mode standing wave: depth {depth:.4f} "
          f"(far half {depth_far:.4f}); meep 0.064")
    assert depth < MAX_DEPTH, (
        f"CPML reflection got worse: standing-wave depth {depth:.4f} exceeds "
        f"{MAX_DEPTH} (was 0.230; real meep 0.064). See "
        f"wiki/tasks/bend-flux_reflection-ripple.md -- and note that raising "
        f"sigma makes this worse, not better.")
    # The envelope is currently UNIFORM, which is what marks it as a reflector
    # rather than mode beating; meep's decays (0.064 near, 0.019 far). If this
    # ever starts decaying strongly the character has changed and the numbers
    # above need re-measuring rather than the threshold nudging.
    assert depth_far > 0.5 * depth, (
        f"envelope no longer uniform (far {depth_far:.4f} vs all {depth:.4f}) "
        f"-- re-measure against meep before touching MAX_DEPTH")
