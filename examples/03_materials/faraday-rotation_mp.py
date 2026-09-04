# Faraday rotation in a gyrotropic medium, from the Meep tutorial.
#
# meep-torch port: STUB -- not ported. The upstream example uses a gyrotropic
# Lorentzian susceptibility with bias vector, which requires the full-vector
# 1D stepper with gyrotropic ADE (build_gyro_1d_coeffs, step_1d). This is
# core-API, not mp.Simulation.
#
# UNPORTED FEATURE: GyrotropicLorentzianSusceptibility with bias vector.
# The mp.Simulation shim does not expose gyrotropic media.
#
# Run: uv run python examples/03_materials/faraday-rotation_mp.py
# (raises NotImplementedError)

import meep_torch.mp as mp

raise NotImplementedError(
    "faraday-rotation.py requires GyrotropicLorentzianSusceptibility(bias=Vector3(0,0,b0)) "
    "-- unported feature. The mp.Simulation shim does not expose gyrotropic media. "
    "Core-API equivalent: build_gyro_1d_coeffs + step_1d (not a twin)."
)
