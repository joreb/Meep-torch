# Angular reflectance of a dielectric interface, from the Meep tutorial.
#
# meep-torch port: STUB -- not ported. The upstream example uses Bloch-periodic
# boundaries with k_y = 2*pi*f*sin(theta) to simulate oblique incidence.
# Bloch boundaries are an unported feature in meep-torch.
#
# UNPORTED FEATURE: Bloch-periodic boundaries (k_point with non-zero transverse component).
# The core-API has init_bloch_tm_state + step_bloch_tm, but the mp.Simulation
# shim does not expose Bloch boundaries.
#
# Run: uv run python examples/04_boundaries/refl-angular_mp.py
# (raises NotImplementedError)

import meep_torch.mp as mp

raise NotImplementedError(
    "refl-angular.py requires Bloch-periodic boundaries with k_y = 2*pi*f*sin(theta) "
    "for oblique incidence. Unported feature: the mp.Simulation shim does not expose "
    "Bloch boundaries. Core-API equivalent: init_bloch_tm_state + step_bloch_tm "
    "(not a twin)."
)
