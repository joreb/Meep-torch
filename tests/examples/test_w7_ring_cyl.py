# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W7 wave 1: radial BOR stepper — ring-cyl resonances vs meep oracle."""
import json
import pathlib

import numpy as np
import torch

from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_cyl import (build_cyl_coeffs, init_cyl_state,
                                         run_cyl)
from meep_torch.modes.harminv import harminv

DATA = pathlib.Path(__file__).parent / "data" / "ring_cyl.json"


def _modes(m):
    with open(DATA) as f:
        ref = json.load(f)
    res = ref["resolution"]
    dr = 1.0 / res
    sr = ref["r"] + ref["w"] + ref["pad"] + ref["dpml"]
    nr = int(sr * res)
    eps = torch.ones(nr, dtype=torch.float64)
    i0, i1 = int(ref["r"] * res), int((ref["r"] + ref["w"]) * res)
    # node-centered eps: interior nodes full, edge nodes half-covered ->
    # effective interfaces at exactly r and r+w (matches meep's staircase)
    eps[i0 + 1: i1] = ref["n"] ** 2
    eps[i0] = eps[i1] = (1.0 + ref["n"] ** 2) / 2.0
    c = build_cyl_coeffs(nr, dr, m, eps=eps, n_absorber=int(ref["dpml"] * res))
    src_ix = int((ref["r"] + 0.1) * res)
    pulse = GaussianPulse(ref["fcen"], ref["df"])
    n_steps = int(round(260.0 / c.dt))
    _, series = run_cyl(init_cyl_state(nr), c, n_steps,
                        src_ix=src_ix, src_profile=pulse, probe_ix=src_ix)
    tail = series[int(round(60.0 / c.dt)):]
    return harminv(tail, c.dt, ref["fcen"] - ref["df"], ref["fcen"] + ref["df"]), ref


def _gate(m, key):
    modes, ref = _modes(m)
    want = ref[key][0]
    near = [x for x in modes if abs(x.freq - want["freq"]) / want["freq"] < 0.03]
    assert near, f"m={m}: no mode near f={want['freq']:.4f}; got {[round(x.freq, 4) for x in modes[:5]]}"
    got = max(near, key=lambda x: x.amp)
    assert abs(got.freq - want["freq"]) / want["freq"] <= 0.01, \
        f"m={m}: f={got.freq:.5f} vs oracle {want['freq']:.5f}"
    assert abs(got.Q - want["Q"]) / want["Q"] <= 0.20, \
        f"m={m}: Q={got.Q:.1f} vs oracle {want['Q']:.1f}"


def test_ring_cyl_m3():
    _gate(3, "m3")


def test_ring_cyl_m4():
    _gate(4, "m4")


def test_the_azimuthal_coupling_carries_the_sign_of_m():
    """`dBr/dt = -(i m / r) Ez`, including the sign, which the oracle cannot see.

    WHY THIS EXISTS. `test_ring_cyl_m3`/`m4` gate resonance frequency and Q
    against a meep oracle, and both are INVARIANT under `m -> -m`: those are
    the two counter-propagating members of a degenerate pair. Flipping the sign
    of the `i m / r` coupling leaves them passing -- verified by injecting
    exactly that defect. meep's own examples share the blind spot, because they
    report the same invariant observables, so no oracle closes it.

    Energy has to circulate the way `m` says. A travelling wave is awkward to
    set up here (a REAL drive gives `Re(F e^{i m phi}) = F cos(m phi)`, a
    standing wave in phi that carries no net azimuthal flux -- measured as
    identically zero), so this probes the coupling itself: seed a known Ez,
    take one step, and read the Hr it induces. The value is predicted by the
    equation in the module docstring, not measured from a run.
    """
    import torch
    from meep_torch.core.stepper_cyl import (build_cyl_coeffs, init_cyl_state,
                                             step_cyl)

    nr, dr, ir = 60, 1.0 / 20, 30
    r = ir * dr

    def induced(m):
        c = build_cyl_coeffs(nr, dr, m, eps=1.0, n_absorber=0)
        st = init_cyl_state(nr)
        ez = torch.zeros(nr, dtype=torch.complex128)
        ez[ir] = 1.0
        st = st._replace(Ez=ez, Dz=ez.clone())
        return complex(step_cyl(st, c, 0.0, None).Hr[ir]), c.dt

    for m in (1, 3):
        got, dt = induced(m)
        want = -1j * m * dt / r                      # -(i m / r) Ez * dt
        assert abs(got - want) < 1e-12, f"m={m}: {got} vs {want}"
        neg, _ = induced(-m)
        assert abs(neg + want) < 1e-12, f"m={-m} did not flip the sign"

    # linear in m, so a coupling that dropped or squared it is caught too
    one, _ = induced(1)
    three, _ = induced(3)
    assert abs(three - 3 * one) < 1e-12
