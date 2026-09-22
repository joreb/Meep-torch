# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""3D modal decomposition gates (wiki/tasks/design_3d-modal-decompose.md).

``decompose_3d`` overlaps the DFT'd tangential (E, H) of a FluxMonitor3D
x-normal plane against the full-vector ``solve_2d`` mode of that
cross-section (bi-orthogonal, power-normalized like the 2D ``decompose``).
One module-scoped pair of forward runs (straight guide + crude Y splitter,
extruded SiN-like stack at res 20) feeds three gates:

(a) straight guide |S21| = |a+_out / a+_in| = 1 to < 1% -- mode-pure
    transmission, insensitive to the windowed-flux path's PML tail loss;
(b) in-mode launched fraction of ModePlaneSourceX (Ey-only injection that
    relaxes into the mode): modal power / windowed Poynting flux at the
    input plane, ~1 within the two estimators' discretization skew;
(c) 50/50 splitter per-arm modal |S|^2 agrees with the windowed-flux
    numbers (the examples_2 3D validations' estimator, reduced resolution)
    within the radiation share. The splitter is two coupled guides driven
    in their even supermode (ModePlaneSourceX solves the full two-guide
    cross-section, so k=1 IS the even supermode): an eigenmode 50/50 split
    with near-zero radiation. Both estimators normalize by the same run's
    input plane (full-aperture even-supermode decompose vs full-aperture
    flux) -- a straight-guide reference would mis-normalize, since the
    max-1 source profile launches ~2x the power into two guides. Each arm
    window is decomposed against the mode of its own PEC-truncated
    sub-cross-section: the estimator comparison the gate is about. (A
    crude abrupt-Y was tried first: its forward junk inside the arm
    windows made windowed flux read 0.44/arm vs modal 0.17/arm -- exactly
    the windowed-flux pathology modal decomposition removes, but useless
    for an agreement gate.)
"""
import math

import pytest
import torch

from meep_torch.core.monitors_3d import FluxMonitor3D
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d_lean import build_lean, run_3d_lean
from meep_torch.ports.mode_monitor import decompose_3d
from meep_torch.ports.validate3d import ModePlaneSourceX, extrude

RES = 20
DX = 1.0 / RES
N_CORE, N_CLAD = 2.0, 1.444
W, H_CORE = 0.8, 0.4
FCEN, FWIDTH = 1.0 / 1.55, 0.4
NPML = 8
NX, NY = 88, 96
HW = int(round(W / 2 * RES))    # guide half-width (cells)
SEP = 14                        # coupled-guide center offset (0.7 um)
JC = NY // 2
IX_SRC = NPML + 6
IX_IN = NPML + 16
IX_OUT = NX - NPML - 8
ARM_HW = 12                     # arm decompose/flux window half-height
T_RUN = 2 * (5 / FWIDTH) + NX * DX * 1.7 + 15.0


def _extrude(eps2):
    return extrude(eps2, n_clad=N_CLAD, n_core=N_CORE, h_core=H_CORE,
                   pad_z=1.0, res=RES)


def _straight2d():
    e = torch.full((NX, NY), N_CLAD ** 2, dtype=torch.float64)
    e[:, JC - HW:JC + HW] = N_CORE ** 2
    return e


def _splitter2d():
    # Two full-length coupled guides at +/-SEP; the mode source launches
    # their even supermode, i.e. an eigenmode 50/50 splitter.
    e = torch.full((NX, NY), N_CLAD ** 2, dtype=torch.float64)
    e[:, JC + SEP - HW:JC + SEP + HW] = N_CORE ** 2
    e[:, JC - SEP - HW:JC - SEP + HW] = N_CORE ** 2
    return e


def _run(eps3, mons):
    nx, ny, nz = eps3.shape
    # eps3 lands on the default device (conftest sets it), so the
    # coefficients have to follow it rather than being pinned to the CPU.
    c = build_lean(nx, ny, nz, DX, eps3, NPML, dtype=torch.float64,
                   device=eps3.device)
    src = ModePlaneSourceX(eps3, DX, IX_SRC, FCEN,
                           GaussianPulse(FCEN, FWIDTH), c.dt)
    _, accs = run_3d_lean(c, int(round(T_RUN / c.dt)), source=src,
                          monitors=mons)
    return accs


def _mode_power(mon, acc, eps_cs):
    """(P_mode+, P_mode-, flux) at the plane; lossless cross-section, so
    0.5*|a|^2 is the modal power in mon.flux units."""
    a_p, a_m = decompose_3d(mon, acc, eps_cs, DX)
    return (0.5 * float(abs(a_p[0, 0]) ** 2), 0.5 * float(abs(a_m[0, 0]) ** 2),
            float(mon.flux(acc, DX)[0]))


@pytest.fixture(scope="module")
def runs():
    e3_str = _extrude(_straight2d())
    e3_spl = _extrude(_splitter2d())
    nz = e3_str.shape[2]
    dt = 0.5 * DX / math.sqrt(3.0)
    win_z = (NPML, nz - NPML)
    win = lambda jlo, jhi: (jlo, jhi, *win_z)
    mk = lambda idx, **kw: FluxMonitor3D(dt, [FCEN], normal="x", idx=idx, **kw)
    m_str = [mk(IX_IN, ap=NPML), mk(IX_OUT, ap=NPML)]
    a_str = _run(e3_str, m_str)
    # splitter: full-aperture input plane (normalization) + arm windows
    m_spl = [mk(IX_IN, ap=NPML),
             mk(IX_OUT, win=win(JC + SEP - ARM_HW, JC + SEP + ARM_HW)),
             mk(IX_OUT, win=win(JC - SEP - ARM_HW, JC - SEP + ARM_HW))]
    a_spl = _run(e3_spl, m_spl)
    return dict(e3_str=e3_str, e3_spl=e3_spl, m_str=m_str, a_str=a_str,
                m_spl=m_spl, a_spl=a_spl)


def test_straight_guide_modal_s21(runs):
    e3 = runs["e3_str"]
    a_in, _ = decompose_3d(runs["m_str"][0], runs["a_str"][0], e3[IX_IN], DX)
    a_out, _ = decompose_3d(runs["m_str"][1], runs["a_str"][1], e3[IX_OUT], DX)
    s21 = abs(complex(a_out[0, 0] / a_in[0, 0]))
    print(f"straight |S21| = {s21:.5f}")
    assert abs(s21 - 1.0) < 0.01, f"straight-guide modal |S21|={s21:.5f}"


def test_in_mode_launched_fraction(runs):
    p_p, p_m, flx = _mode_power(runs["m_str"][0], runs["a_str"][0],
                                runs["e3_str"][IX_IN])
    frac = p_p / flx
    print(f"in-mode launched fraction = {frac:.4f} "
          f"(P+={p_p:.4e}, P-={p_m:.4e}, flux={flx:.4e})")
    # ModePlaneSourceX relaxes into the mode: essentially all guided power
    # is in-mode; the +/-few-% band is the two estimators' staggering skew.
    assert 0.90 < frac < 1.10
    assert p_m < 0.1 * p_p          # negligible backward wave


def test_splitter_modal_vs_windowed_flux(runs):
    # input-plane even-supermode decompose (full cross-section) + flux
    p_in_mode, _, p_in_flx = _mode_power(runs["m_spl"][0], runs["a_spl"][0],
                                         runs["e3_spl"][IX_IN])
    arms = [_mode_power(m, a, runs["e3_spl"][IX_OUT])
            for m, a in zip(runs["m_spl"][1:], runs["a_spl"][1:])]
    t_mod = [pm / p_in_mode for pm, _, _ in arms]
    t_flx = [fl / p_in_flx for _, _, fl in arms]
    rad = 1.0 - sum(t_flx)          # radiation/window-loss share
    print(f"splitter modal T={t_mod[0]:.4f}/{t_mod[1]:.4f} "
          f"flux T={t_flx[0]:.4f}/{t_flx[1]:.4f} radiation={rad:.4f}")
    assert sum(t_flx) > 0.85        # eigenmode split: low radiation
    # half-cell window mirror offset on the Ey sublattice costs a few %
    assert abs(t_mod[0] - t_mod[1]) < 0.05          # symmetric split
    # Per arm, modal vs windowed flux agree within the radiation share plus
    # a 5% estimator-skew allowance (sub-window PEC-truncated mode vs the
    # supermode's gap/tail power that only the flux window counts).
    for tm, tf in zip(t_mod, t_flx):
        assert abs(tm - tf) <= abs(rad) + 0.05, (
            f"modal {tm:.4f} vs flux {tf:.4f} beyond radiation share {rad:.4f}")
    assert sum(t_mod) <= sum(t_flx) + 0.02          # modal excludes radiation
