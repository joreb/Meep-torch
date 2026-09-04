# Far fields of the holey-waveguide cavity, from the Meep tutorial.
#
# meep-torch port: STUB -- not ported. The upstream example uses add_near2far
# to compute far fields from a near-field contour. The meep-torch near2far
# machinery (faces_from_accs + farfield_tm) is core-API only.
#
# UNPORTED FEATURE: mp.Simulation.add_near2far() shim. The core-API has
# FluxMonitorTM + faces_from_accs + farfield_tm, but the mp.Simulation
# shim does not expose near2far as a simulation method.
#
# Run: uv run python examples/05_monitors_analysis/cavity-farfield_mp.py
# (raises NotImplementedError)

import meep_torch.mp as mp

raise NotImplementedError(
    "cavity-farfield.py requires add_near2far + get_farfield for near-to-far "
    "transform. Unported feature: the mp.Simulation shim does not expose "
    "near2far. Core-API equivalent: FluxMonitorTM + faces_from_accs + farfield_tm "
    "(not a twin)."
)
