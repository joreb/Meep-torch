# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Photonics parity group 4: mode-Gaussian overlap vs waveguide width.

Mirrors Photonics_Simulation's `mode_overlap_example.py` (SiN edge coupler
mode vs an astigmatic Gaussian beam, coupling swept vs waveguide width). The
notebook uses a 2D tidy3d mode and a 2D astigmatic Gaussian beam
(`td.AstigmaticGaussianBeamProfile`, waist 0.48um) with the full bi-orthogonal
E/H overlap; its lens curve peaks at width 0.181um (a 2D height-confinement
effect: the strip's lateral extent at w=0.181 matches the 0.48 beam).

Per the task spec, this is the 1D meep-torch equivalent: the fd1d lateral slab
mode (slab-collapsed EIM core, width w) vs a 1D Gaussian beam of waist w0.
The 1D model cannot reproduce the notebook's 0.181 peak (that needs the 2D
height confinement), but it has its own clean optimum: the fundamental slab
mode is a half-cosine in the core, so its effective width ~ w/2, and the
overlap with a Gaussian of waist w0 peaks when w/2 ~ w0, i.e. w_peak ~ 2*w0.
The gate is the existence and location of that 1D coupling peak.
"""
import numpy as np
import pytest

from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.modes.effindex import neff_table
from meep_torch.modes.fd1d import solve_1d

H = 0.150          # waveguide height (um) -- notebook value
LAM = 0.690        # notebook mode_overlap_example lda0 (um)
W0 = 0.48          # Gaussian beam waist (um) -- notebook lens beam
DX = 0.005
PAD = 3.0           # clad padding each side (um)
# Sweep the well-confined regime (w >= 0.3): below that the cutoff mode's wide
# evanescent tail gives a second local maximum that obscures the half-cosine
# peak the gate targets.
WIDTHS = np.array([0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 1.8, 2.2])


def _n_v():
    return float(neff_table([(2.0, SiO2), (H, Si3N4), (2.0, SiO2)], [1.0 / LAM], "tm", dx=DX)[0])


def _coupling(w):
    f = 1.0 / LAM
    n_v = _n_v()
    n_sio2 = SiO2.n(f).real
    eps_line = np.concatenate([
        np.full(int(PAD / DX), n_sio2 ** 2),
        np.full(int(w / DX), n_v ** 2),
        np.full(int(PAD / DX), n_sio2 ** 2),
    ])
    E = solve_1d(eps_line, DX, f, "tm", k=1).profiles[0].numpy()
    y = (np.arange(len(E)) + 0.5) * DX - len(E) * DX / 2
    y = y - y[int(np.argmax(np.abs(E)))]  # center on the mode peak
    Eg = np.exp(-y ** 2 / W0 ** 2)
    num = abs(np.sum(E * Eg) * DX) ** 2
    den = (np.sum(E ** 2) * DX) * (np.sum(Eg ** 2) * DX)
    return num / den


@pytest.fixture(scope="module")
def coupling_curve():
    return np.array([_coupling(w) for w in WIDTHS])


def test_coupling_has_interior_peak(coupling_curve):
    i = int(coupling_curve.argmax())
    assert 0 < i < len(WIDTHS) - 1, (
        f"coupling peak at sweep edge (index {i}/{len(WIDTHS)-1}); no interior optimum")


def test_peak_coupling_is_high(coupling_curve):
    assert coupling_curve.max() > 0.9, (
        f"peak coupling {coupling_curve.max():.4f} not > 0.9 (mode cannot match the beam)")


def test_peak_width_matches_half_cosine(coupling_curve):
    # The fundamental slab mode is a half-cosine in the core (effective width
    # ~ w/2); it best matches a Gaussian of waist w0 when w/2 ~ w0, i.e.
    # w_peak ~ 2*w0. Gate the peak width to that band.
    i = int(coupling_curve.argmax())
    w_peak = WIDTHS[i]
    assert 1.5 * W0 < w_peak < 2.5 * W0, (
        f"peak width {w_peak:.3f} not in half-cosine band [1.5*w0, 2.5*w0] = "
        f"[{1.5*W0:.3f}, {2.5*W0:.3f}]")


def test_coupling_decays_away_from_peak(coupling_curve):
    # Unimodal: the curve falls off on both sides of the peak.
    i = int(coupling_curve.argmax())
    peak = coupling_curve[i]
    assert coupling_curve[0] < peak - 0.03 and coupling_curve[-1] < peak - 0.03, (
        f"coupling does not decay from the peak: "
        f"edges {coupling_curve[0]:.3f}, {coupling_curve[-1]:.3f} vs peak {peak:.3f}")