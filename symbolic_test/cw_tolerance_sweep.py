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
"""solve_cw convergence + field accuracy vs tolerance, for whichever
meep_torch is on sys.path. Mirrors tests/mp_parity/test_solve_cw_parity.py."""
import sys, numpy as np
import meep_torch.mp as mp

RES, FCEN, SXY, DPML = 10, 0.15, 8.0, 1.0

def _sim(force_complex=True, symmetries=None, src_x=-2.0):
    return mp.Simulation(
        resolution=RES, cell_size=mp.Vector3(SXY, SXY),
        geometry=[mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf),
                           material=mp.Medium(epsilon=12))],
        sources=[mp.Source(mp.ContinuousSource(FCEN), component=mp.Ez,
                           center=mp.Vector3(src_x, 0))],
        symmetries=symmetries, boundary_layers=[mp.PML(DPML)],
        force_complex_fields=force_complex)

# time-domain steady state (independent of tolerance) -- the reference
td = _sim(force_complex=False)
td.run(until=400)
a = np.asarray(td.get_array(component=mp.Ez))
td.run(until=1 / FCEN / 4)
ez_td = a + 1j * np.asarray(td.get_array(component=mp.Ez))

print(f"{'tol':>8}  {'converged':>9}  {'rel-L2 vs time-domain':>22}")
for tol in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6):
    sim = _sim(); sim.init_sim()
    ok = bool(sim.solve_cw(tol, 5000, 2))
    if not ok:
        print(f"{tol:8.0e}  {'NO':>9}  {'-':>22}")
        continue
    ez_cw = np.asarray(sim.get_array(component=mp.Ez))
    i = np.unravel_index(np.abs(ez_td).argmax(), ez_td.shape)
    k = ez_td[i] / ez_cw[i]
    rel = np.linalg.norm(ez_cw * k - ez_td) / np.linalg.norm(ez_td)
    print(f"{tol:8.0e}  {'yes':>9}  {rel:22.3e}   (gate: < 5e-2)")
