# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port stub: meep/python/examples/diffracted_planewave.py

BLOCKED: mp.DiffractedPlanewave single-order eigenmode launch.

W2-DEFERRAL UPDATE (2026-08-08): the TE Bloch stepper itself landed
(stepper_bloch step_bloch_te, oblique Fresnel p-pol gate <=0.02). This
example still needs launching a SINGLE diffraction order as the source
(mp.DiffractedPlanewave -- an eigenmode-decomposed injection), which is
Bloch-order TFSF machinery beyond the plain stepper.

The meep original computes a binary grating's transmitted diffraction orders at
oblique incidence in TE polarization (mp.EVEN_Z, Hz). meep-torch's Bloch stepper
(core/stepper_bloch.py) is TM-only (Ez, s-pol); the TM twin of this calculation
is examples/binary_grating_oblique.py (+ its meep-oracle gate). To port this TE
variant, meep-torch would need a two-quadrature envelope Bloch stepper for TE
(Hz/Ex/Ey, the TE Yee update). See the meep original for the physics.

meep original: meep/python/examples/diffracted_planewave.py
"""