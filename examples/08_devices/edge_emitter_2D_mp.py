# 2D edge emitter at a cleaved facet, from the Meep tutorial.
#
# meep-torch port: STUB -- not ported. The upstream example computes extraction
# efficiency via Brillouin-zone integration over kz (k_point Bloch boundaries),
# dft_ldos, and DFT flux monitors.
#
# UNPORTED FEATURES:
#   - k_point Bloch boundaries for Brillouin-zone integration
#   - dft_ldos (LDOS computation)
#   - DFT flux monitors for extraction efficiency
# This port keeps only the near-field physics (Ez field snapshot), but the
# upstream example's main point is the extraction-efficiency calculation.
#
# Run: uv run python examples/08_devices/edge_emitter_2D_mp.py
# (raises NotImplementedError)

import meep_torch.mp as mp

raise NotImplementedError(
    "edge_emitter_2D.py requires k_point Bloch boundaries for Brillouin-zone "
    "integration, dft_ldos, and DFT flux monitors to compute extraction efficiency. "
    "Unported features: the mp.Simulation shim does not expose Bloch boundaries, "
    "LDOS, or DFT flux. Core-API can produce a near-field snapshot only "
    "(not a faithful twin -- the extraction-efficiency calculation is omitted)."
)
