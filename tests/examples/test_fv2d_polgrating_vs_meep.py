# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""`stepper_fv2d` against meep's own polarization-grating numbers.

The companion file `test_fv2d_polarization_grating.py` gates the stepper on
physics it can check by itself -- conversion, deflection, and the nulls. This
one gates it against an ORACLE: `data/polarization_grating.json`, produced by
running meep's `python/examples/polarization_grating.py` under a real pymeep.

The distinction matters because the example's own analytic formula is NOT the
thing to match. `eff_m0 = cos^2(pi dn d / wvl)` ignores interface reflections
and the finite-`delta_n` correction, and real meep departs from it by 0.28 at
phase 0.75 (formula 0.50, meep 0.78). An engine that matched the formula there
would be wrong. `test_the_gate_follows_meep_and_not_the_formula` is the
assertion that keeps this honest: where the two references disagree, fv2d has
to side with meep.
"""
import json
import math
import pathlib

import numpy as np
import pytest
import torch

from meep_torch.core.stepper_fv2d import (build_fv2d_coeffs, init_fv2d_state,
                                          step_fv2d)

DATA = pathlib.Path(__file__).parent / "data"
REF = json.load(open(DATA / "polarization_grating.json"))

#: absolute tolerance on a diffraction efficiency. The measured gaps are 0.001
#: and 0.006 at phases 0.25/0.75 and 0.026 at half-wave, where all the power is
#: meant to leave in the first orders and the residual m0 leakage is most
#: sensitive to the absorber. 0.04 keeps margin over that without admitting the
#: 0.28 gap to the analytic formula, which is the error this must still catch.
TOL = 0.04


def _efficiencies(d, res, t_end=120.0):
    """fv2d at meep's geometry; d is the HALF thickness, as meep passes it."""
    dpml = dsub = dpad = 1.0
    n0, dn, gp = REF["n_0"], REF["delta_n"], REF["grating_period"]
    fcen = 1.0 / REF["wvl"]
    sx = dpml + dsub + d + d + dpad + dpml
    dx = 1.0 / res
    nx, ny = int(round(sx * res)), int(round(gp * res))
    xs = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * dx
    a, b = n0 ** 2, (n0 + dn) ** 2

    one = torch.ones(nx, ny, dtype=torch.float64)
    exx, eyy, ezz = one.clone(), one.clone(), one.clone()
    eyz = torch.zeros(nx, ny, dtype=torch.float64)
    sub = xs <= (dpml + dsub)                       # substrate, index n_0
    exx[sub, :] = eyy[sub, :] = ezz[sub, :] = a
    lc = (xs > (dpml + dsub)) & (xs <= (dpml + dsub + 2 * d))
    phi = math.pi * ys / gp                          # ph = 0: no chiral twist
    c2, s2 = torch.cos(phi) ** 2, torch.sin(phi) ** 2
    cs = torch.cos(phi) * torch.sin(phi)
    exx[lc, :] = a
    eyy[lc, :] = (a * c2 + b * s2)[None, :]
    ezz[lc, :] = (a * s2 + b * c2)[None, :]
    eyz[lc, :] = ((b - a) * cs)[None, :]
    eps6 = (exx, torch.zeros_like(one), torch.zeros_like(one), eyy, eyz, ezz)

    c = build_fv2d_coeffs(eps6, dx, npml=int(dpml * res))
    st = init_fv2d_state(nx, ny)
    mask = torch.zeros(nx, ny, dtype=torch.float64)
    mask[int(round((dpml + 0.3 * dsub) * res)), :] = 1.0
    omega = 2 * math.pi * fcen
    n_steps = int(round(t_end / c.dt))
    ix_out = int(round((sx - dpml - 0.5 * dpad) * res))
    accY = np.zeros(ny, dtype=complex)
    accZ = np.zeros(ny, dtype=complex)
    for t in range(n_steps):
        tt = (t + 0.5) * c.dt
        amp = math.cos(omega * tt) * min(1.0, tt / 10.0)   # linear drive
        st = step_fv2d(st, c, amp, mask, "Ey", src2=("Ez", mask, amp))
        if t >= n_steps // 2:                        # steady state only
            ph = np.exp(-2j * math.pi * fcen * (t + 1.0) * c.dt) * c.dt
            accY += st.Ey[ix_out, :].cpu().numpy() * ph
            accZ += st.Ez[ix_out, :].cpu().numpy() * ph

    p = np.abs(np.fft.fft(accY)) ** 2 + np.abs(np.fft.fft(accZ)) ** 2
    m0, m1 = p[0], 0.5 * (p[1] + p[-1])
    sin_t = (1.0 / gp) / fcen
    cos_t = math.sqrt(max(1.0 - sin_t ** 2, 1e-12))
    tot = m0 + 2 * m1
    return m0 / tot, (2 * m1 / tot) / cos_t


@pytest.mark.parametrize("row", REF["rows"], ids=[f"phase{r['phase']}" for r in REF["rows"]])
def test_fv2d_matches_meep_diffraction_efficiencies(row):
    e0, e1 = _efficiencies(0.5 * row["d"], REF["resolution"])
    assert abs(e0 - row["eff_m0"]) < TOL, (
        f"phase {row['phase']}: m0 {e0:.4f} vs meep {row['eff_m0']:.4f}")
    assert abs(e1 - row["eff_m1"]) < TOL, (
        f"phase {row['phase']}: m1 {e1:.4f} vs meep {row['eff_m1']:.4f}")


def test_the_gate_follows_meep_and_not_the_formula():
    """Where the oracle and the textbook disagree, fv2d must side with meep.

    Without this the parametrised test above is just a band, and a tolerance
    wide enough to pass would also pass an engine that reproduced the
    idealised formula instead of the simulation. At phase 0.75 the two
    references are 0.28 apart, which is far outside TOL -- so exactly one of
    them can be matched, and it has to be the one with the reflections in it.
    """
    row = max(REF["rows"],
              key=lambda r: abs(r["eff_m0"] - r["eff_m0_analytic"]))
    gap = abs(row["eff_m0"] - row["eff_m0_analytic"])
    assert gap > 3 * TOL, (
        f"no row separates meep from the formula by more than {3*TOL}: "
        f"largest gap {gap:.3f} at phase {row['phase']}. This test cannot "
        "discriminate and the oracle rows need a phase where they diverge.")

    e0, _ = _efficiencies(0.5 * row["d"], REF["resolution"])
    assert abs(e0 - row["eff_m0"]) < abs(e0 - row["eff_m0_analytic"]), (
        f"phase {row['phase']}: fv2d m0 {e0:.4f} is closer to the analytic "
        f"{row['eff_m0_analytic']:.4f} than to meep {row['eff_m0']:.4f}")
