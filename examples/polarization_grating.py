# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port stub: meep/python/examples/polarization_grating.py

BLOCKED: full-vector 2D stepper (ez-coupled 3x3 permittivity tensor).

W5 UPDATE (2026-08-08): meep-torch now has an IN-PLANE anisotropic tensor
(exx, exy, eyy) in the TE stepper (build_te_coeffs(eps_tensor=...), gate:
rotated-uniaxial n_eff <1% vs the index ellipse). That does NOT unblock this
example: the twisted-nematic director rotates out of the xy-plane, coupling
Ez with (Ex, Ey) -- the TM/TE split itself breaks, and a full-vector 2D
stepper (all six field components) is required.

The meep original models a polarization grating from a uniaxial twisted-nematic
liquid crystal whose permittivity is a full 3x3 tensor with off-diagonal entries
that vary spatially (the director twists along the propagation direction and
across the grating period). meep's mp.Medium(epsilon_diag=..., epsilon_offdiag=...)
accepts a tensor material, and the FDTD stepper propagates all coupled field
components.

meep-torch's TM/TE steppers (step_tm / step_te) only support an isotropic scalar
permittivity (eps_inf is a scalar field; build_tm_coeffs / build_te_coeffs have no
off-diagonal epsilon path). The geometry/polygons + scene rasterizer also only
paints a scalar eps. There is no anisotropic FDTD in meep-torch, and the fd1d
mode solver is scalar-only (the fd2d_tensor solver handles off-diagonal eps for
mode solves but is not wired into the FDTD stepper).

To port, meep-torch would need: (1) an anisotropic FDTD stepper that advances
coupled Ex/Ey/Ez (or Er/Ep in cylindrical) from a tensor eps with off-diagonal
entries, and (2) a scene rasterizer that paints a per-cell tensor, not a scalar.
Both are substantial additions. See the meep original for the physics.

meep original: meep/python/examples/polarization_grating.py
"""