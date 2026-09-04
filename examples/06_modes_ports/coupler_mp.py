# Directional coupler: coupling vs length, from the Meep tutorial.
#
# meep-torch port: STUB -- not ported. The upstream example reads a directional
# coupler from coupler.gds and computes per-branch transmittance via
# eigenmode coefficients (get_eigenmode_coefficients) for a swept coupling
# length.
#
# UNPORTED FEATURES:
#   - get_eigenmode_coefficients for modal decomposition at port planes
#   - S-matrix computation (compute_smatrix) as an mp.Simulation method
# The core-API has compute_smatrix + Port machinery, but this rebuilds the
# geometry parametrically rather than being the upstream file with only the
# import swapped. A faithful twin would need the mp.Simulation shim to expose
# eigenmode expansion and S-matrix methods.
#
# Run: uv run python examples/06_modes_ports/coupler_mp.py
# (raises NotImplementedError)

import meep_torch.mp as mp

raise NotImplementedError(
    "coupler.py requires get_eigenmode_coefficients and S-matrix computation "
    "for modal transmittance. Unported features: the mp.Simulation shim does not "
    "expose eigenmode expansion or smatrix methods. Core-API equivalent: "
    "compute_smatrix + Port machinery (not a twin -- also rewrites geometry "
    "parametrically rather than loading coupler.gds)."
)
