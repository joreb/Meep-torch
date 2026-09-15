# SPDX-License-Identifier: MIT
# Investigating whether the whole-curl derived form (discrete_curl, see
# meep_torch/core/symbolic.py) describes different physics from meep's
# per-difference composition, or only conditions worse.
#
# Background: wiring stepper_tm's D phase to the whole-curl form failed both
# solve_cw parity gates -- on "BiCGSTAB(2) did not converge" at tol=1e-10, NOT
# on a field value. So the question is open. See
# wiki/concepts/finite-difference-deviation.md.
#
# Run each against whichever meep_torch is on sys.path, e.g. a worktree at the
# whole-curl commit vs the current checkout:
#   PYTHONPATH=<checkout> uv run python symbolic_test/<script>.py <tag>
"""Vacuum point-source CW solve vs the analytic 2D Green's function.

2D scalar Helmholtz with a point source has the closed form
    Ez(r) = (i/4) H_0^(1)(k r)
so this has real resolving power, unlike comparing against a two-snapshot
time-domain run. Saves the solved field so the two curl forms can also be
differenced against EACH OTHER -- which bounds whether they can differ
physically at all.
"""
import sys, numpy as np
from scipy.special import hankel1
import meep_torch.mp as mp

TAG = sys.argv[1]
RES, FCEN, SXY, DPML = 20, 0.3, 12.0, 2.0
k = 2 * np.pi * FCEN                      # vacuum

sim = mp.Simulation(
    resolution=RES, cell_size=mp.Vector3(SXY, SXY),
    sources=[mp.Source(mp.ContinuousSource(FCEN), component=mp.Ez,
                       center=mp.Vector3(0, 0))],
    boundary_layers=[mp.PML(DPML)], force_complex_fields=True)
sim.init_sim()
ok = bool(sim.solve_cw(1e-9, 8000, 2))
ez = np.asarray(sim.get_array(component=mp.Ez))
np.save(f"symbolic_test/output/cw_ez_{TAG}.npy", ez)
print(f"{TAG}: converged={ok}  shape={ez.shape}")

ny, nx = ez.shape
x = (np.arange(nx) - nx / 2 + 0.5) / RES
y = (np.arange(ny) - ny / 2 + 0.5) / RES
X, Y = np.meshgrid(x, y, indexing="ij")
R = np.sqrt(X**2 + Y**2)

# annulus: clear of the source singularity and of the PML
inner, outer = 1.0, SXY / 2 - DPML - 0.5
m = (R > inner) & (R < outer)
ana = 0.25j * hankel1(0, k * R[m])
num = ez[m]
s = (ana @ num.conj()) / (num @ num.conj())        # best complex scale
rel = np.linalg.norm(num * s - ana) / np.linalg.norm(ana)
print(f"{TAG}: rel-L2 vs (i/4)H_0^(1)(kr) on the annulus = {rel:.6e}  "
      f"(N={m.sum()} points)")
