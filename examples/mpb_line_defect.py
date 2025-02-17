# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port stub: meep/python/examples/mpb_line_defect.py

MPB-TRACK UPDATE (2026-08-09): band machinery landed (Bloch+harminv:
stepper_bloch2 doubly-periodic, mpb-oracle gates <=2%; see
holey-wvg-bands.py and mpb_bragg_sine.py). This stub still needs: supercell defect bands (large-cell run; schedulable).

BLOCKED: MPB eigenmode band solver.

meep-torch does not implement MPB eigenmode band solver. To port, meep-torch would need
meep.mpb.ModeSolver to compute the photonic band structure (frequency vs k) of a periodic dielectric lattice. See the meep original for the physics.

meep original: meep/python/examples/mpb_line_defect.py
"""
