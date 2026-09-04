# Out-of-plane angular reflectance (kz stepper), from the Meep tutorial.
#
# meep-torch port: STUB -- not ported. The upstream example uses kz_2d="complex"
# to simulate fields F(x) e^{i k_z z} -- a 1D-in-x reduction with out-of-plane
# wavevector. This requires the core-API stepper_kz (build_kz_coeffs, step_kz).
#
# UNPORTED FEATURE: kz_2d reduction (out-of-plane wavevector in 2D simulation).
# The mp.Simulation shim does not expose kz_2d. Core-API has stepper_kz but
# this is not an mp.Simulation twin pattern.
#
# Run: uv run python examples/04_boundaries/refl-angular-kz2d_mp.py
# (raises NotImplementedError)

import meep_torch.mp as mp

raise NotImplementedError(
    "refl-angular-kz2d.py requires kz_2d='complex' for out-of-plane wavevector "
    "(fields F(x) e^{i k_z z}). Unported feature: the mp.Simulation shim does not "
    "expose kz_2d. Core-API equivalent: build_kz_coeffs + step_kz (not a twin)."
)
