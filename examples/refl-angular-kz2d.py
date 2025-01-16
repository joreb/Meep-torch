# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port stub: meep/python/examples/refl-angular-kz2d.py

UNBLOCKED CORE (2026-08-08): core/stepper_kz.py (2.5D k_z, 1D-in-x
{Ex, Ez, Hy} set, Fresnel p-pol gate <=0.02) and step_bloch_te both
landed. Port pending: the example's full angular sweep + mp shim wiring
(kz_2d + special_kz handling in Simulation) -- scheduled with the next
example-port pass.

The meep original computes angular reflectance of a planar interface using
meep's kz_2d 2.5D effective-index mode (a quasi-2D solve that folds a 3D
out-of-plane waveguide k_z into the 2D cell) in TE (Ey). meep-torch has no 2.5D
kz_2d engine and its Bloch stepper is TM-only. The full 2D angular-reflectance
twin (no kz_2d) is examples/refl-angular.py (+ Fresnel gate). To port this 2.5D
variant, meep-torch would need a kz_2d effective-index FDTD + TE Bloch. See the
meep original for the physics.

meep original: meep/python/examples/refl-angular-kz2d.py
"""