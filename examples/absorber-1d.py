"""STUB -- meep-torch port of meep/python/examples/absorber-1d.py.

Blockers (unsupported by meep-torch at this time):

  1. mp.Absorber boundary layer. meep-torch only provides PML
     (cpml_profiles in meep_torch/core/pml.py); there is no absorber /
     slowly-ramped conductivity boundary. The whole point of the original
     example is to contrast mp.PML vs mp.Absorber for a dipole in aluminum,
     so without an Absorber the comparison cannot be made.

  2. meep.materials.Al (aluminum, a dispersive Drude metal). meep-torch's
     meep_torch/core/materials.py only ships cSi, SiO2, Si3N4; there is no Al
     material and no Drude pole convention wired into build_tm_coeffs
     (LorentzPole uses the Lorentzian form sigma*fn^2/(fn^2-f^2-i*f*gamma),
     which can approximate a Drude term only at fn=0, and build_tm_coeffs
     does not special-case the fn=0 Drude limit).

When meep-torch gains an Absorber boundary layer and a Drude/Al material,
revisit this port. Run: uv run python examples/absorber-1d.py  (no-op stub).
"""
print("absorber-1d.py: STUB -- see docstring for blockers (no mp.Absorber, no Al material).")