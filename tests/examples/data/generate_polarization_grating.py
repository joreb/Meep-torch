# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep oracle: uniaxial polarization-grating diffraction efficiencies.

The twisted-nematic grating of Oh & Escuti (Opt. Lett. 33, 2287, 2008), as
meep ships it in `python/examples/polarization_grating.py`. This is the
chiral-twist-free case (`ph = 0`), three thicknesses instead of the example's
34, at a lower resolution so it finishes in minutes.

WHY AN ORACLE AND NOT THE ANALYTIC FORMULA. The example carries its own
closed form, `eff_m0 = cos^2(pi dn d / wvl)` and `eff_m1 = sin^2(...)`, and
BOTH engines depart from it substantially -- at phase 0.75 the formula says
0.50 and meep says 0.78. The formula ignores the interface reflections and
the finite-`delta_n` correction that a real simulation has. Gating against it
would therefore either fail a correct engine or need a tolerance so loose it
gates nothing. meep's own numbers are the thing worth matching.

WHAT IT GATES. `meep_torch.core.stepper_fv2d`, the full-vector 2D stepper
with a 3x3 symmetric eps -- it exists because a z-coupled permittivity breaks
the TM/TE split, and this grating is the problem it was written for. It is
not reachable from the `mp` shim (whose scalar eps_tm/eps_te grids cannot
hold a tensor), so nothing else in the suite touches it.

Run: `mamba run -n meep-oracle python tests/examples/data/generate_polarization_grating.py`
     (or `/home/papac/miniforge3/envs/meep-oracle/bin/python <this file>`)
"""
import json
import math
import os

import meep as mp

RES = 20
DPML = DSUB = DPAD = 1.0
N0, DELTA_N, WVL, GP, NMODE = 1.55, 0.159, 0.54, 6.5, 5
PHASES = (0.25, 0.5, 0.75)
FCEN = 1 / WVL
EPS_DIAG = mp.Matrix(mp.Vector3(N0**2, 0, 0), mp.Vector3(0, N0**2, 0),
                     mp.Vector3(0, 0, (N0 + DELTA_N) ** 2))


def pol_grating(d, ph, gp, nmode):
    """meep's own pol_grating, verbatim apart from the reduced resolution."""
    sx = DPML + DSUB + d + d + DPAD + DPML

    def phi(p):
        xx = p.x - (-0.5 * sx + DPML + DSUB)
        if 0 <= xx <= d:
            return math.pi * p.y / gp + ph * xx / d
        return math.pi * p.y / gp - ph * xx / d + 2 * ph

    def lc_mat(p):
        Rx = mp.Matrix(mp.Vector3(1, 0, 0),
                       mp.Vector3(0, math.cos(phi(p)), math.sin(phi(p))),
                       mp.Vector3(0, -math.sin(phi(p)), math.cos(phi(p))))
        e = Rx * EPS_DIAG * Rx.transpose()
        return mp.Medium(epsilon_diag=mp.Vector3(e[0].x, e[1].y, e[2].z),
                         epsilon_offdiag=mp.Vector3(e[1].x, e[2].x, e[2].y))

    geometry = [
        mp.Block(center=mp.Vector3(-0.5 * sx + 0.5 * (DPML + DSUB)),
                 size=mp.Vector3(DPML + DSUB, mp.inf, mp.inf),
                 material=mp.Medium(index=N0)),
        mp.Block(center=mp.Vector3(-0.5 * sx + DPML + DSUB + d),
                 size=mp.Vector3(2 * d, mp.inf, mp.inf), material=lc_mat)]
    src_pt = mp.Vector3(-0.5 * sx + DPML + 0.3 * DSUB, 0, 0)
    sources = [mp.Source(mp.GaussianSource(FCEN, fwidth=0.05 * FCEN),
                         component=c, center=src_pt, size=mp.Vector3(0, gp, 0))
               for c in (mp.Ez, mp.Ey)]
    common = dict(resolution=RES, cell_size=mp.Vector3(sx, gp, 0),
                  boundary_layers=[mp.PML(thickness=DPML, direction=mp.X)],
                  k_point=mp.Vector3(0, 0, 0), sources=sources)
    tran_pt = mp.Vector3(0.5 * sx - DPML - 0.5 * DPAD, 0, 0)
    region = mp.FluxRegion(center=tran_pt, size=mp.Vector3(0, gp, 0))

    sim = mp.Simulation(default_material=mp.Medium(index=N0), **common)
    tf = sim.add_flux(FCEN, 0, 1, region)
    sim.run(until_after_sources=100)
    input_flux = mp.get_fluxes(tf)
    sim.reset_meep()

    sim = mp.Simulation(geometry=geometry, **common)
    tf = sim.add_flux(FCEN, 0, 1, region)
    sim.run(until_after_sources=300)
    r1 = sim.get_eigenmode_coefficients(tf, range(1, nmode + 1),
                                        eig_parity=mp.ODD_Z + mp.EVEN_Y)
    r2 = sim.get_eigenmode_coefficients(tf, range(1, nmode + 1),
                                        eig_parity=mp.EVEN_Z + mp.ODD_Y)
    angles = [math.degrees(math.acos(k.x / FCEN)) for k in r1.kdom]
    return input_flux[0], angles, r1.alpha[:, 0, 0], r2.alpha[:, 0, 0]


rows = []
for phase in PHASES:
    dd = phase * WVL / DELTA_N
    inp, angles, c1, c2 = pol_grating(0.5 * dd, 0.0, GP, NMODE)
    tran = (abs(c1) ** 2 + abs(c2) ** 2) / inp
    tot = tran[0] + 2 * tran[1]
    cosang = math.cos(math.radians(angles[1]))
    rows.append({
        "phase": phase,
        "d": dd,
        "eff_m0": float(tran[0] / tot),
        "eff_m1": float((2 * tran[1] / tot) / cosang),
        "angle_m1_deg": float(angles[1]),
        "eff_m0_analytic": math.cos(math.pi * phase) ** 2,
        "eff_m1_analytic": math.sin(math.pi * phase) ** 2,
    })

out = {"source": "meep python/examples/polarization_grating.py, ph=0 (uniaxial)",
       "resolution": RES, "n_0": N0, "delta_n": DELTA_N, "wvl": WVL,
       "grating_period": GP, "dpml": DPML, "dsub": DSUB, "dpad": DPAD,
       "rows": rows}
path = os.path.join(os.path.dirname(__file__), "polarization_grating.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", path)
for r in rows:
    print(f"  phase {r['phase']:.2f}  m0 {r['eff_m0']:.4f}  m1 {r['eff_m1']:.4f}"
          f"  (analytic {r['eff_m0_analytic']:.2f}/{r['eff_m1_analytic']:.2f})")
