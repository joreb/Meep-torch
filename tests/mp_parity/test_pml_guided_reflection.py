# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Regression guard: how much of a guided mode the CPML reflects.

A straight waveguide terminated by PML should show a FLAT complex-amplitude
envelope along its axis, give or take the mode beating any line source
launches. It now does.

Root cause of the original defect: ``cpml_profiles`` anchored the HIGH-edge
ramp at ``n - 1 - n_hi`` in array-index space, but a half-offset H array is
one node shorter than its E partner, so its ramp started a full cell inside
the E ramp. The resulting E/H stretch mismatch reflected a returning guided
mode at ~0.22 amplitude. The low edge was always correct (an H node genuinely
sits half a cell inside that wall), which is why the two edges reflected
0.0007 and 0.18. Fixed by anchoring both at the same physical wall.

    before the fix   depth 0.230   (uniform -> a reflector)
    after the fix    depth 0.057   (decaying -> mode beating)
    real meep        depth 0.064   (decaying)

That reflection was the cause of the bend-flux reflectance disagreement:
bend-flux gets R by subtracting two large nearly equal fluxes, and its
reflection plane sat in a cavity between this PML and the bend corner, so the
reflector became a frequency fringe on R while T, measured past the bend,
stayed fine.

If this ever regresses, do NOT "fix" it by raising sigma. The old defect
scaled UP with sigma_max (R0 1e-3/1e-6/1e-15 -> depth 0.184/0.227/0.302) and
DOWN with layer thickness (dpml 1/2/4 um -> 0.227/0.090/0.059), because both
knobs move the sigma GRADIENT that the E/H mismatch multiplies. Every
profile-shape knob was inert (cell-integrated ramp, m, alpha, node anchoring
shifted equally). Full evidence and the refuted hypotheses:
wiki/tasks/bend-flux_reflection-ripple.md and
wiki/tasks/fix_cpml-entrance-reflection.md
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

MAX_DEPTH = 0.08     # currently 0.057; meep 0.064; was 0.230 before the fix


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
        f"CPML reflection regressed: standing-wave depth {depth:.4f} exceeds "
        f"{MAX_DEPTH} (0.057 after the high-edge anchor fix; 0.230 before it; "
        f"real meep 0.064). See wiki/tasks/fix_cpml-entrance-reflection.md -- "
        f"and note that raising sigma makes this worse, not better.")
    # Character check, not a magnitude one. A DECAYING envelope is mode
    # beating from the line source, which is what meep shows (0.064 near,
    # 0.019 far). A uniform one means a reflector is back, and that is a
    # different bug from the depth merely drifting -- so it gets its own
    # assert rather than being folded into the threshold.
    assert depth_far < 0.5 * depth, (
        f"envelope is uniform again (far {depth_far:.4f} vs all {depth:.4f}): "
        f"a reflector has returned, not just a magnitude drift. Re-measure "
        f"against meep before touching MAX_DEPTH")
