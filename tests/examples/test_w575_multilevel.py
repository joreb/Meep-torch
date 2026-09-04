# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W5.75: two-level gain medium — 1D Fabry-Perot laser vs meep oracle."""
import json
import math
import pathlib

import numpy as np
import torch

from meep_torch.core.grid import Grid
from meep_torch.core.stepper_1d import build_1d_coeffs, init_1d_state, step_1d

DATA = pathlib.Path(__file__).parent / "data" / "multilevel_lasing.json"


def _laser_run(Rp, endt, res=100):
    with open(DATA) as f:
        ref = json.load(f)
    p = ref["params"]
    Lcav, dpad, dpml = p["Lcav"], 1.0, 1.0
    sz = Lcav + dpad + dpml
    grid = Grid(nx=int(sz * res), ny=4, dx=1.0 / res)
    nx = grid.nx
    ncav_cells = int(Lcav * res)
    mask = torch.zeros(nx, dtype=grid.dtype, device=grid.device)
    mask[:ncav_cells] = 1.0
    eps = torch.ones(nx, dtype=grid.dtype, device=grid.device)
    eps[:ncav_cells] = p["ncav"] ** 2
    coeffs = build_1d_coeffs(
        grid, eps=eps, npml=(0, int(dpml * res)),      # left wall = PEC mirror
        mla=(p["omega_a"], p["gamma_21"], p["sigma_21"], Rp, p["rate_21"], mask))
    state = init_1d_state(grid, n0=p["N0"], n0_mask=mask)
    state.Ey[ncav_cells // 2] = 1.0                    # seed field (meep delta)
    probe_ix = int((Lcav + 0.5 * dpad) * res)
    n_steps = int(round(endt / grid.dt))
    tail_n = int(round(ref["tail_t"] / grid.dt))
    series = np.empty(min(tail_n, n_steps))
    k = 0
    for t in range(n_steps):
        state = step_1d(state, coeffs)
        if t >= n_steps - len(series):
            series[k] = state.Ey[probe_ix].item()
            k += 1
    return series, grid.dt, ref


def test_below_threshold_decays():
    s, dt, _ = _laser_run(Rp=1e-5, endt=400)
    early = np.abs(s[: len(s) // 4]).max()
    late = np.abs(s[-len(s) // 4:]).max()
    assert late < 0.5 * early or late < 1e-8, f"early {early:.2e} late {late:.2e}"


def test_lasing_peak_vs_oracle():
    s, dt, ref = _laser_run(Rp=ref_rp(), endt=ref_endt())
    assert np.abs(s).max() > 1e-3, "did not lase"
    spec = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(len(s), dt)
    band = (freqs > 5.0) & (freqs < 8.0)
    f_peak = float(freqs[band][spec[band].argmax()])
    assert abs(f_peak - ref["f_peak"]) / ref["f_peak"] <= 0.05, \
        f"f_peak={f_peak:.3f} vs oracle {ref['f_peak']:.3f}"


def ref_rp():
    with open(DATA) as f:
        return json.load(f)["params"]["Rp"]


def ref_endt():
    with open(DATA) as f:
        return json.load(f)["endt"]
