# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math

import pytest

import torch

from meep_torch.core.grid import Grid
from meep_torch.core.sources import CWRamp, GaussianPulse, LineSource


def test_gaussian_peak_and_decay():
    p = GaussianPulse(fcen=0.65, fwidth=0.1)
    t0 = 5.0 / 0.1
    assert p.amplitude(t0) == 1.0                       # cos(0)*exp(0)
    assert abs(p.amplitude(0.0)) < 1e-5                 # cut off at start
    assert abs(p.amplitude(2 * t0)) < 1e-5


def test_gaussian_spectrum_peaks_at_fcen():
    p = GaussianPulse(fcen=0.65, fwidth=0.1)
    dt = 0.01
    n = int(p.t_end / dt) + 1
    ts = torch.arange(n, dtype=torch.float64) * dt
    sig = torch.tensor([p.amplitude(float(t)) for t in ts], dtype=torch.float64)
    spec = torch.fft.rfft(sig).abs()
    freqs = torch.fft.rfftfreq(n, dt)
    assert abs(freqs[spec.argmax()].item() - 0.65) < 0.02


def test_line_source_mask():
    g = Grid(nx=20, ny=10, dx=0.02)
    src = LineSource(g, ix=5, j0=2, j1=8, profile=GaussianPulse(0.65, 0.1))
    assert src.mask.shape == (20, 10)
    # The mask is a current DENSITY: each of the 6 cells carries dx of length
    # over the dx^2 cell area, i.e. 1/dx. Injecting a bare 1.0 per cell made
    # the field scale as dx instead of being resolution-independent as in
    # meep, so the weight is 1/dx = 50 here (dx = 0.02).
    assert src.mask[5, 2:8].sum().item() == pytest.approx(6.0 / g.dx)
    assert src.mask.sum().item() == pytest.approx(6.0 / g.dx)
    # a single-cell (point) source is a delta in both axes: 1/dx^2
    pt = LineSource(g, ix=5, j0=2, j1=3, profile=GaussianPulse(0.65, 0.1))
    assert pt.mask.sum().item() == pytest.approx(1.0 / g.dx ** 2)


def test_cw_ramp():
    p = CWRamp(fcen=0.5, t_ramp=10.0)
    assert abs(p.amplitude(5.0)) <= 0.5 + 1e-12          # still ramping
    assert p.amplitude(20.25) == math.sin(2 * math.pi * 0.5 * 20.25) * 1.0


# ---------------------------------------------------------------------------
# The injection sign is a consequence of Ampere's law, not a per-file choice.
# ---------------------------------------------------------------------------

def test_the_source_sign_is_derived_from_maxwell():
    """`dD/dt = curl H - J`, so a current ENTERS NEGATIVE. meep's `f -= A`.

    It used to live nowhere: `maxwell_rhs` had no current term, so twelve
    steppers each typed the sign by hand and nothing could contradict a wrong
    one. `stepper_3d` wrote it the other way round in two of its three
    branches, and its own lean variant disagreed with it.
    """
    from meep_torch.core.symbolic import (COMPONENTS, EPS, source_coefficient,
                                          source_sign)

    for c in COMPONENTS:
        assert source_sign(c) == -1, c
    assert source_coefficient("Ez") == -1 / EPS      # the constitutive half
    assert source_coefficient("Hz") == -1


def _one_step_response(step, state, coeffs, comp, **kw):
    """One step from rest with a POSITIVE amplitude; return the driven field."""
    out = step(state, coeffs, 1.0, **kw)
    return float(getattr(out, comp).flatten()[getattr(out, comp).abs().argmax()])


def test_every_stepper_injects_with_the_derived_sign():
    """Drive each stepper from rest and check which way the field moves.

    A stepper that adds where it should subtract still runs and still
    converges; it radiates with the opposite phase, and the only gate that sees
    it is the LDOS. This is the gate that would have caught `stepper_3d`.
    """
    import torch

    from meep_torch.core.coeffs import build_tm_coeffs
    from meep_torch.core.grid import Grid
    from meep_torch.core.stepper_3d import (build_3d_coeffs, init_3d_state,
                                            step_3d)
    from meep_torch.core.stepper_tm import init_tm_state, step_tm
    from meep_torch.core.symbolic import source_sign

    want = source_sign("Ez")                          # -1, derived above

    # 2D TM, mask form
    g = Grid(nx=24, ny=20, dx=0.05)
    m = build_tm_coeffs(g, npml=4)
    mask = torch.zeros(24, 20, dtype=torch.float64)
    mask[12, 10] = 1.0
    got = _one_step_response(step_tm, init_tm_state(g), m, "Ez", src_mask=mask)
    assert got != 0.0 and (1 if got > 0 else -1) == want, ("TM", got)

    # 3D, all three source forms
    n = 12
    c = build_3d_coeffs(n, n, n, 0.1, n_abs=2)
    st = init_3d_state(n, n, n)
    vol = torch.zeros_like(st.Ez)                  # Ez is (nx, ny, nz-1)
    vol[n // 2, n // 2, n // 2] = 1.0
    forms = {
        "point": ("Ez", n // 2, n // 2, n // 2),
        "plane": ("Ez", n // 2, torch.ones_like(st.Ez[n // 2])),
        "mask": ("Ez", vol),
    }
    for name, src in forms.items():
        got = _one_step_response(step_3d, st, c, "Ez", src=src)
        assert got != 0.0, (name, "no response at all")
        assert (1 if got > 0 else -1) == want, (name, got)


def test_a_3d_step_does_not_mutate_its_input():
    """`_inject_src` writes its indexed forms in place, and that has to stay safe.

    It is safe today because both call sites build a fresh `State3D` from the
    update before handing it over, so the tensor written is never the one the
    caller holds. That is an invariant of the CALL SITES, not of the writing
    code, which is exactly the kind of thing that stops being true quietly --
    hand `_inject_src` the incoming state and every caller's array changes
    underneath. Pinned here rather than defended with a per-step clone of a
    full field.
    """
    from meep_torch.core.stepper_3d import (build_3d_coeffs, init_3d_state,
                                            step_3d)

    n, mid = 10, 5
    c = build_3d_coeffs(n, n, n, 0.1, n_abs=2)
    st = init_3d_state(n, n, n)
    before = {f: getattr(st, f).clone() for f in st._fields
              if hasattr(getattr(st, f), "clone")}
    step_3d(st, c, 1.0, ("Ez", mid, mid, mid))
    for f, v in before.items():
        assert torch.equal(getattr(st, f), v), f


def test_a_3d_point_dipole_is_resolution_independent():
    """The `1/dx^3` measure, gated by the property it exists for.

    A source is a current DENSITY, so a point dipole of a given amplitude has
    to produce the same field at the same PHYSICAL distance whichever grid it
    is on. Without the measure the amplitude carries a factor of `dx^3` and the
    field moves with resolution -- the same defect
    `source-normalization-extent` fixed in 2D, never applied to the 3D point
    path.
    """
    import numpy as np

    from meep_torch.core.sources import GaussianPulse
    from meep_torch.core.stepper_3d import (build_3d_coeffs, init_3d_state,
                                            step_3d)

    def peak_at(res, size=2.4, offset=0.6):
        n = int(round(size * res))
        dx = 1.0 / res
        c = build_3d_coeffs(n, n, n, dx, n_abs=max(2, res // 4))
        st = init_3d_state(n, n, n)
        pulse = GaussianPulse(1.0, 0.5)
        mid = n // 2
        probe = mid + int(round(offset * res))
        n_steps = int(round(3.0 / c.dt))
        peak = 0.0
        for t in range(n_steps):
            st = step_3d(st, c, pulse.amplitude((t + 0.5) * c.dt),
                         ("Ez", mid, mid, mid))
            peak = max(peak, abs(float(st.Ez[probe, mid, mid])))
        return peak

    lo, hi = peak_at(8), peak_at(16)
    assert lo > 0 and hi > 0, (lo, hi)
    ratio = hi / lo
    # doubling the resolution must not move the field by anything like the
    # factor of 8 a missing dx^3 would produce
    assert 0.5 < ratio < 2.0, f"peak moved {ratio:.3f}x with resolution"


def test_delta_weights_matches_meeps_measured_footprint():
    """A point between cell centres spreads over both, as meep does.

    The reference is meep's MEASURED footprint, not a derivation of ours: a
    centred source on a 60-cell, 3.0 um grid gives cells 29 and 30 at 0.5 each
    (wiki/bugs/point-source-snapped-not-interpolated). Snapping to round()
    puts the whole delta on one cell, displaced up to half a cell.

    Both edge cases here bit a real attempt and are not hypothetical: at an
    exact cell centre floating point gives g = -1.8e-15, so floor() lands on
    cell -1 and the weight leaks off the grid; and mirrored positions give
    weights equal only to ~4e-15, so a symmetry check on top needs a
    tolerance rather than ==.
    """
    from meep_torch.core.sources import delta_weights

    assert delta_weights(0.0, 3.0, 0.05, 60) == [(29, 0.5), (30, 0.5)]
    # on a cell centre: one weight, identical to the old placement
    assert delta_weights(-1.475, 3.0, 0.05, 60) == [(0, 1.0)]
    # at either edge the off-grid neighbour is dropped, never index -1 or n
    assert delta_weights(-1.5, 3.0, 0.05, 60) == [(0, 0.5)]
    assert delta_weights(1.5, 3.0, 0.05, 60) == [(59, 0.5)]
    # weights always sum to 1 inside the grid: the delta is distributed,
    # not rescaled
    for c in (0.0, 0.013, -0.37, 1.0):
        assert sum(w for _, w in delta_weights(c, 3.0, 0.05, 60)) == pytest.approx(1.0)
    # mirrored positions mirror, to floating point
    a = [w for _, w in delta_weights(0.3, 3.0, 0.05, 60)]
    b = [w for _, w in delta_weights(-0.3, 3.0, 0.05, 60)][::-1]
    assert a == pytest.approx(b, abs=1e-12)
