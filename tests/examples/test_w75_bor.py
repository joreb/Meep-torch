# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W7.5: (r, z) 2D BOR stepper — 3D vacuum LDOS + disc resonator gates."""
import json
import math
import pathlib

import numpy as np
import torch

from meep_torch.core.ldos import ldos_spectrum
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bor import (build_bor_coeffs, init_bor_state,
                                         run_bor)
from meep_torch.modes.harminv import harminv

DATA = pathlib.Path(__file__).parent / "data" / "disc_bor.json"


def test_m0_vacuum_ldos_f_squared():
    """m = 0 on-axis Ez dipole = a physical 3D point dipole: LDOS must go
    as f^2 (3D), not f (2D)."""
    res = 20
    dr = 1.0 / res
    nr, nz = int(3.0 * res), int(6.0 * res)
    c = build_bor_coeffs(nr, nz, dr, m=0, n_abs_r=res, n_abs_z=res)
    pulse = GaussianPulse(1.0, 0.4)
    src = ("Ez", 0, nz // 2)
    n_steps = int(round(60.0 / c.dt))
    _, series = run_bor(init_bor_state(nr, nz), c, n_steps,
                        src=src, src_profile=pulse, probe=src)
    j = np.array([pulse.amplitude((t + 0.5) * c.dt) for t in range(n_steps)])
    band = np.linspace(0.85, 1.15, 7)
    ld = ldos_spectrum(series.real, j, c.dt, band)
    assert (ld > 0).all()
    r = ld / band ** 2
    spread = (r.max() - r.min()) / r.mean()
    assert spread <= 0.04, f"ldos/f^2 spread {spread:.4f} ({r})"


def test_disc_resonance_vs_oracle():
    with open(DATA) as f:
        ref = json.load(f)
    res = ref["resolution"]
    dr = 1.0 / res
    sr = ref["r_disc"] + ref["pad"] + ref["dpml"]
    sz = ref["h_disc"] + 2 * (ref["pad"] + ref["dpml"])
    nr, nz = int(round(sr * res)), int(round(sz * res))
    # coverage-fraction eps (cell around each node vs the disc extent) --
    # the same edge averaging that reproduced meep subpixel in ring-cyl
    i_ax = torch.arange(nr, dtype=torch.float64)
    j_ax = torch.arange(nz, dtype=torch.float64)
    frac_r = torch.clamp(ref["r_disc"] * res - (i_ax - 0.5), 0.0, 1.0)
    z0 = (sz / 2 - ref["h_disc"] / 2) * res
    z1 = (sz / 2 + ref["h_disc"] / 2) * res
    frac_z = torch.clamp(torch.minimum(j_ax + 0.5, torch.tensor(z1))
                         - torch.maximum(j_ax - 0.5, torch.tensor(z0)), 0.0, 1.0)
    eps = 1.0 + (ref["n"] ** 2 - 1.0) * frac_r[:, None] * frac_z[None, :]
    c = build_bor_coeffs(nr, nz, dr, m=ref["m"], eps=eps,
                         n_abs_r=int(ref["dpml"] * res),
                         n_abs_z=int(ref["dpml"] * res))
    src = ("Er", int(0.8 * ref["r_disc"] * res), nz // 2)
    pulse = GaussianPulse(ref["fcen"], ref["df"])
    n_steps = int(round(340.0 / c.dt))
    _, series = run_bor(init_bor_state(nr, nz), c, n_steps,
                        src=src, src_profile=pulse, probe=src)
    tail = series[int(round(40.0 / c.dt)):]
    modes = [x for x in harminv(tail, c.dt, ref["fcen"] - ref["df"],
                                ref["fcen"] + ref["df"]) if x.Q > 10]
    want = ref["modes"][0]
    near = [x for x in modes if abs(x.freq - want["freq"]) / want["freq"] < 0.03]
    assert near, f"no mode near {want['freq']:.4f}; got {[round(x.freq, 4) for x in modes[:6]]}"
    got = max(near, key=lambda x: x.amp)
    # 1.5%: our isotropic coverage-fraction averaging vs meep's anisotropic
    # subpixel tensor scheme -- they bracket the resonance (+1.6% binary,
    # -1.0% averaged at res 25, 7.25-cell disc); converges with resolution
    assert abs(got.freq - want["freq"]) / want["freq"] <= 0.015, \
        f"f={got.freq:.5f} vs oracle {want['freq']:.5f}"
    assert abs(got.Q - want["Q"]) / want["Q"] <= 0.25, \
        f"Q={got.Q:.1f} vs oracle {want['Q']:.1f}"
