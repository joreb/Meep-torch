# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W7.75: cylindrical near2far (m = 0) — on-axis dipole pattern gates."""
import json
import math
import pathlib

import numpy as np

from meep_torch.core.near2far_cyl import farfield_bor_m0, run_bor_dft
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bor import build_bor_coeffs, init_bor_state

DATA = pathlib.Path(__file__).parent / "data" / "dipole_cyl_n2f.json"
FCEN = 1.0


def _pattern(thetas):
    res = 25
    dr = 1.0 / res
    nr, nz = int(3.0 * res), int(6.0 * res)
    c = build_bor_coeffs(nr, nz, dr, m=0, n_abs_r=res, n_abs_z=res)
    pulse = GaussianPulse(FCEN, 0.2)
    src = ("Ez", 0, nz // 2)
    n_steps = int(round(60.0 / c.dt))
    _, planes = run_bor_dft(init_bor_state(nr, nz), c, n_steps,
                            src=src, src_profile=pulse, freqs=[FCEN])
    return farfield_bor_m0(planes, c, ir=int(1.5 * res),
                           jz0=int(1.5 * res), jz1=int(4.5 * res),
                           thetas=thetas, k=2 * math.pi * FCEN)[0]


def test_dipole_pattern_sin2_and_oracle():
    with open(DATA) as f:
        ref = json.load(f)
    thetas = np.radians(ref["theta_deg"])
    got = _pattern(thetas)
    sin2 = np.sin(thetas) ** 2
    rms_an = float(np.sqrt(np.mean((got - sin2) ** 2)))
    assert rms_an <= 0.03, f"rms vs sin^2 {rms_an:.4f}"
    rms_or = float(np.sqrt(np.mean((got - np.array(ref["pattern"])) ** 2)))
    assert rms_or <= 0.05, f"rms vs oracle {rms_or:.4f}"
