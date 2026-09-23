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


def _pattern(thetas, eps=1.0, k_scale=None, Z=None):
    """On-axis Ez dipole in a homogeneous ``eps``; the far pattern from the
    medium kernel (k = n omega, Z = 1/n) unless overridden."""
    import torch
    res = 25
    dr = 1.0 / res
    nr, nz = int(3.0 * res), int(6.0 * res)
    n = math.sqrt(eps)
    e = None if eps == 1.0 else torch.full((nr, nz), float(eps), dtype=torch.float64)
    c = build_bor_coeffs(nr, nz, dr, m=0, eps=e, n_abs_r=res, n_abs_z=res)
    pulse = GaussianPulse(FCEN, 0.2)
    src = ("Ez", 0, nz // 2)
    n_steps = int(round(60.0 / c.dt))
    _, planes = run_bor_dft(init_bor_state(nr, nz), c, n_steps,
                            src=src, src_profile=pulse, freqs=[FCEN])
    k = 2 * math.pi * FCEN * (n if k_scale is None else k_scale)
    return farfield_bor_m0(planes, c, ir=int(1.5 * res),
                           jz0=int(1.5 * res), jz1=int(4.5 * res),
                           thetas=thetas, k=k, Z=(1.0 / n if Z is None else Z))[0]


def test_dipole_pattern_sin2_in_medium():
    """n = 1.5 background: sin^2 only when the kernel gets the medium k and
    Z. The vacuum k on the same planes is a shape error, not a scale one."""
    thetas = np.radians(np.linspace(5, 175, 35))
    sin2 = np.sin(thetas) ** 2
    got = _pattern(thetas, eps=2.25)
    rms = float(np.sqrt(np.mean((got - sin2) ** 2)))
    assert rms <= 0.03, f"rms vs sin^2 in n=1.5: {rms:.4f}"
    wrong = _pattern(thetas, eps=2.25, k_scale=1.0, Z=1.0)
    assert float(np.sqrt(np.mean((wrong - sin2) ** 2))) > 0.1


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


def test_m1_dipole_pattern_vs_oracle():
    """W7.9: |m| = 1 far fields -- off-axis Er dipole vs meep oracle."""
    from meep_torch.core.near2far_cyl import farfield_bor_m

    with open(pathlib.Path(__file__).parent / "data" / "dipole_m1_n2f.json") as f:
        ref = json.load(f)
    res = 25
    dr = 1.0 / res
    nr, nz = int(3.0 * res), int(6.0 * res)
    c = build_bor_coeffs(nr, nz, dr, m=1, n_abs_r=res, n_abs_z=res)
    pulse = GaussianPulse(FCEN, 0.2)
    src = ("Er", int(ref["r_src"] * res), nz // 2)
    n_steps = int(round(60.0 / c.dt))
    _, planes = run_bor_dft(init_bor_state(nr, nz), c, n_steps,
                            src=src, src_profile=pulse, freqs=[FCEN])
    thetas = np.radians(ref["theta_deg"])
    got = farfield_bor_m(planes, c, 1, ir=int(1.5 * res),
                         jz0=int(1.5 * res), jz1=int(4.5 * res),
                         thetas=thetas, k=2 * math.pi * FCEN)[0]
    rms = float(np.sqrt(np.mean((got - np.array(ref["pattern"])) ** 2)))
    assert rms <= 0.05, f"m=1 pattern rms vs oracle {rms:.4f}"
