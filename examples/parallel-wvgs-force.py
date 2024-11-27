# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port stub: meep/python/examples/parallel-wvgs-force.py

BLOCKED: EigenModeSource in Simulation.run.

W6 UPDATE (2026-08-08): the Maxwell-stress-tensor force itself landed
(core/forces.py mst_force_tm, radiation-pressure gate vs analytic + meep
oracle). This example remains blocked on driving an EigenModeSource through
Simulation.run (the shim routes eigenmode sources to the S-param path
only); revisit when eigenmode sources become first-class run() sources.

meep-torch does not implement optical force (Maxwell stress tensor). To port, meep-torch would need
ForceRegion + the Maxwell stress-tensor force accumulation between waveguides. See the meep original for the physics.

meep original: meep/python/examples/parallel-wvgs-force.py
"""
