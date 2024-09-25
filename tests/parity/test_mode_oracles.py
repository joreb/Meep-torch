# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Mode-solver oracle parity tests.

femwell is the live oracle (gmsh + scikit-fem FEM waveguide mode solver). It
requires Python < 3.10 (its setup.py hard-bounds ``>=3.6,<3.10``) and the
shapely / gmsh / scikit-fem deps; on this Windows + Python 3.14 machine it
cannot be installed, so the femwell test skips here and runs live only where
femwell is properly installed (Linux / Python < 3.10 box).

The gate targets the real femwell entry point ``femwell.maxwell.waveguide``
(not the bare ``femwell`` name): a reference clone checked out under
``./femwell/`` would otherwise shadow as a namespace package at the clone
root and let ``import femwell`` succeed without the inner package or its
deps, making the test fail instead of skip. Importing the real submodule
either resolves to the installed package (live) or raises ImportError
(skip).

tidy3d / tidy3d-extras / MPB are guarded stubs that mark the integration
points and skip by construction today (wiring them is future-machine work,
per the Phase 1 parity-skeleton pattern).
"""
import numpy as np
import pytest
import torch

from meep_torch.modes.fd2d import solve_2d


def test_si_strip_parity_vs_femwell():
    """Si strip 500x220nm, lam=1.55, n_Si=3.4777, n_SiO2=1.444: |dneff| < 5e-4."""
    # Gate on the real femwell waveguide entry point (not the bare ``femwell``
    # name): a reference clone under ``./femwell/`` shadows as a namespace
    # package at the clone root and lets ``import femwell`` succeed without
    # the inner package or its deps. Importing the real submodule either
    # resolves to the installed package (live) or raises ImportError (skip).
    # Skips here -- femwell needs Python < 3.10 + shapely/gmsh/scikit-fem, none
    # available on this Windows + Python 3.14 machine.
    pytest.importorskip("femwell.maxwell.waveguide", reason="femwell oracle not installed")
    # femwell side: gmsh mesh of the same cross-section, order=2 elements
    from collections import OrderedDict

    import shapely
    from femwell.maxwell.waveguide import compute_modes
    from skfem import Basis, ElementTriP0
    from femwell.mesh import mesh_from_OrderedDict

    core = shapely.geometry.box(-0.25, -0.11, 0.25, 0.11)
    clad = shapely.geometry.box(-1.5, -1.0, 1.5, 1.0)
    shapes = OrderedDict(core=core, clad=clad)
    resolutions = dict(core={"resolution": 0.02, "distance": 0.3})
    mesh = mesh_from_OrderedDict(shapes, resolutions, default_resolution_max=0.1)
    basis0 = Basis(mesh, ElementTriP0())
    eps = basis0.zeros()
    eps[basis0.get_dofs(elements="core")] = 3.4777**2
    eps[basis0.get_dofs(elements="clad")] = 1.444**2
    fem_modes = compute_modes(basis0, eps, wavelength=1.55, num_modes=1, order=2)
    n_fem = float(np.real(fem_modes[0].n_eff))

    dx = 0.01
    eps_t = torch.full((300, 200), 1.444**2, dtype=torch.float64)
    eps_t[125:175, 89:111] = 3.4777**2
    n_fd = solve_2d(eps_t, dx=dx, f=1 / 1.55, k=1).neff[0].real.item()
    assert n_fd == pytest.approx(n_fem, abs=5e-4)


def test_tidy3d_diagonal_parity():
    try:
        from tidy3d.components.mode.solver import compute_modes as t3d_compute  # noqa: F401
    except ImportError:
        pytest.skip("wire up when tidy3d is added as a dev dep on a machine with it installed")
    pytest.skip("wire up when tidy3d is added as a dev dep on a machine with it installed")


def test_extras_tensorial_oracle():
    extras = pytest.importorskip("tidy3d_extras")
    if getattr(extras, "extension", None) is None:
        pytest.skip("tidy3d-extras extension not licensed on this machine")
    pytest.skip("wire up on a machine with a valid extras key")


def test_mpb_parity():
    mpb = pytest.importorskip("meep.mpb", reason="MPB not available (Linux box)")
    pytest.skip("wire up on the Linux machine")