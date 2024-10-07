# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Photonics parity group 1: slab-collapsed effective index of the SiN guide.

Mirrors the first step of Photonics_Simulation's `EffectiveIndex_Waveguide`
(`src/photonic/waveguide/__init__.py`): the vertical SiN/SiO2 slab of height h
is collapsed to a 1D effective index. The notebook fits the resulting strip
neff(lambda) to a Cauchy model n = A + B/lambda^2 + C/lambda^4 (the lmfit
coefficients are runtime-only, not persisted), so the gate here is a ballpark
sanity check against our own 2D fd2d strip solver (the meep-torch equivalent of
the notebook's tidy3d 2D solve):

  - the slab mode is guided: n_SiO2 < neff_slab < n_Si3N4
  - normal dispersion (Cauchy-like): neff decreases with lambda
  - the slab-collapsed neff is ballpark-close to the 2D strip neff (the DBR is
    TM, so the TM slab collapse is the relevant one)

DBR geometry from `Testing_DBR_documented.ipynb`: SiN core, SiO2 clad,
height = 0.150 um, design lambda = 0.674 um, sweep [.574, .874] um.
"""
import numpy as np
import pytest
import torch

from meep_torch.core.grid import Grid
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.modes.effindex import neff_table
from meep_torch.modes.fd2d import solve_2d

H = 0.150          # waveguide height (um) -- the slab we collapse
LAMS = [0.574, 0.674, 0.774, 0.874]   # notebook wavelength_range [.574, .874]
FREQS = [1.0 / lam for lam in LAMS]
DESIGN_LAM = 0.674
DX_SLAB = 0.005     # effindex slab discretization


def _bulk_n(mat, lams):
    return [mat.n(1.0 / lam).real for lam in lams]


def _slab_neff(pol):
    # vertical slab: SiO2 clad | SiN core (h) | SiO2 clad; effindex pads 2um.
    return neff_table([(2.0, SiO2), (H, Si3N4), (2.0, SiO2)], FREQS, pol, dx=DX_SLAB)


@pytest.fixture(scope="module")
def strip_2d_neff():
    # 2D fd2d strip fundamental at the design lambda (notebook tidy3d proxy).
    w, lam = 1.14, DESIGN_LAM
    f = 1.0 / lam
    n_sin = Si3N4.n(f).real
    n_sio2 = SiO2.n(f).real
    dx = 0.01
    pad = 1.0
    nx, ny = int((w + 2 * pad) / dx), int((H + 2 * pad) / dx)
    g = Grid(nx=nx, ny=ny, dx=dx)
    core = Polygon(np.array([[pad, pad], [pad + w, pad], [pad + w, pad + H], [pad, pad + H]]))
    rm = rasterize_scene([(core, n_sin ** 2)], g, background=n_sio2 ** 2)
    return float(solve_2d(rm.eps_tm, dx, f, k=1).neff[0].real)


def test_slab_mode_is_guided():
    n_clad = _bulk_n(SiO2, LAMS)
    n_core = _bulk_n(Si3N4, LAMS)
    for pol in ("te", "tm"):
        neff = _slab_neff(pol)
        for i, lam in enumerate(LAMS):
            assert n_clad[i] < float(neff[i]) < n_core[i], (
                f"{pol} slab neff {float(neff[i])} not between clad {n_clad[i]} "
                f"and core {n_core[i]} at lambda={lam}")


def test_slab_normal_dispersion():
    # Cauchy-like: neff decreases as lambda increases (shorter lambda -> higher n).
    for pol in ("te", "tm"):
        neff = _slab_neff(pol)
        assert float(neff[0]) > float(neff[-1]), (
            f"{pol} slab neff not decreasing with lambda: "
            f"{float(neff[0])} -> {float(neff[-1])}")


def test_slab_collapsed_ballpark_vs_2d(strip_2d_neff):
    # The DBR is TM (DBR_TM_SM); the TM slab collapse should be ballpark-close to
    # the 2D strip fundamental (the slab collapse overestimates confinement
    # slightly because it ignores the lateral leakage, so it sits a touch high).
    neff_tm = _slab_neff("tm")
    n_slab = float(neff_tm[LAMS.index(DESIGN_LAM)])
    assert abs(n_slab - strip_2d_neff) < 0.05, (
        f"TM slab-collapsed neff {n_slab} vs 2D strip neff {strip_2d_neff} "
        f"diverge by {abs(n_slab - strip_2d_neff):.4f} (>0.05)")