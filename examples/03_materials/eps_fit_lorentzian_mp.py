# Fitting eps(f) with Lorentzian poles, from the Meep tutorial.
#
# meep-torch port: STUB -- not ported. The upstream example is a standalone
# nonlinear least-squares fit of complex permittivity data to a sum of
# Lorentzian/Drude susceptibility terms using scipy.optimize.least_squares.
# There is no FDTD simulation involved -- it's pure curve fitting.
#
# This does not belong in the meep-torch examples: it's a materials/fitting
# script, not a photonics simulation. The meep-torch porting directive is to
# port upstream meep python/examples that drive mp.Simulation; this file
# imports scipy and NLopt, not meep's FDTD engine.
#
# Run: uv run python examples/03_materials/eps_fit_lorentzian_mp.py
# (prints this notice and exits)

"""STUB -- eps_fit_lorentzian.py is NOT ported.

The upstream file is a standalone curve-fitting script:
  - Reads a CSV of wavelength, real(n), imag(n)
  - Fits complex eps(f) to a sum of Lorentzian/Drude poles via scipy.optimize.least_squares
  - Plots actual vs fit permittivity

No FDTD simulation is involved. This is a materials characterization utility,
not a meep simulation example. The meep-torch examples port directive covers
upstream meep python/examples that drive mp.Simulation -- this file drives
scipy.optimize.

If you need to fit material data for use in meep-torch simulations, use
meep_torch.mp.LorentzPole or meep_torch.core.coeffs.build_tm_coeffs directly
with parameters obtained from your own fitting script.
"""
import sys

print(__doc__)
print("eps_fit_lorentzian_mp.py: STUB -- standalone scipy curve fit, no FDTD simulation")
sys.exit(0)
