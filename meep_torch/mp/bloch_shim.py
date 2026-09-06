# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Bloch-periodic Simulation runs (``k_point=...``): the adapter between the
meep Simulation surface and core.stepper_bloch.

Scope: a 2D cell whose y axis carries no boundary layers (Bloch-periodic
wrap, k_y = 2*pi*k_point.y, PML on x) and the effectively-1D z-only cell
with an in-plane transverse k (refl-angular). When EVERY cell axis carries
boundary layers there is no periodic wrap and ``needs_bloch`` returns False:
Simulation.run keeps the ordinary real-field path (meep only switches to
complex fields there; the fluxes are identical).

The engine steps the periodic Bloch envelope in two real quadratures
(E = (E_c + i E_s) exp(i k_y y)). Flux monitors sample BOTH quadratures with
the ordinary core FluxMonitor samplers; the pair is combined into the
standard (ReE, ImE, ReH, ImH) accumulator at the end of the run, so
get_fluxes / get_flux_data / load_(minus_)flux_data work unchanged.
"""
import math
from types import SimpleNamespace

import numpy as np
import torch

from .run_control import (_eval_step_func, fire_per_step,
                          fire_scheduled, plan_hooks)
import torch.nn.functional as F

from ..core.stepper_bloch import (init_bloch_te_state, init_bloch_tm_state,
                                  step_bloch_te, step_bloch_tm)
from .sources import EigenModeSource, Source, build_line_source
from .vec import Vector3


def needs_bloch(sim) -> bool:
    """True when a k_point run has a periodic wrap the Bloch stepper must
    handle: a boundary-layer-free y axis (2D), or a nonzero transverse k on
    the z-only 1D cell."""
    if sim.dimensions == 1:
        k = sim.k_point
        return bool(k.x or k.y)
    return not (any(sim._npml_sides[1]) or any(sim._nabs_sides[1]))


def _bloch_ky(sim) -> float:
    """Engine k_y (rad/length) = 2*pi * the k_point component along the
    periodic grid axis. 1D cells lie along z (grid x); the transverse
    components are the Bloch precession."""
    k = sim.k_point
    if sim.dimensions == 1:
        return 2 * math.pi * math.hypot(k.x, k.y)
    return 2 * math.pi * k.y


def _quadratures(prof, t):
    """(Re, Im) of meep's complex source J(t) = envelope * exp(-i w t).
    Exact for the engine GaussianPulse (cos carrier); other profiles use a
    quarter-period phase ADVANCE for the Im quadrature (the narrowband
    analytic signal)."""
    f = getattr(prof, "fcen", 0.0)
    t0, w = getattr(prof, "t0", None), getattr(prof, "w", None)
    if t0 is not None and w is not None and not hasattr(prof, "_inner"):
        env = math.exp(-((t - t0) ** 2) / (2 * w * w))
        ph = 2 * math.pi * f * (t - t0)
        return env * math.cos(ph), -env * math.sin(ph)
    c = prof.amplitude(t)
    # ponytail: quarter-period advance approximates the Hilbert quadrature;
    # add exact per-profile pairs if a broadband non-Gaussian source needs them
    s = prof.amplitude(t + 0.25 / f) if f else 0.0
    return c, s


def _scope_check(sim, step_funcs):
    nie = NotImplementedError
    if sim.dimensions == 3:
        raise nie("k_point: 3D Bloch runs are not supported")
    if sim.dimensions != 1:
        if sim.k_point.z:
            raise nie("k_point.z (oblique out-of-plane, kz_2d) is not wired "
                      "to the Simulation Bloch path (core/stepper_kz)")
        if not any(sim._npml_sides[0]):
            raise nie("k_point: the x axis needs a PML (fully periodic cells "
                      "are not supported)")
    if any(sim._nabs_sides[0]) or any(sim._nabs_sides[1]):
        raise nie("k_point with Absorber is not supported (the Bloch stepper "
                  "is conductivity-free); use PML")
    if sim._sym_cx or sim._sym_cy:
        raise nie("k_point with Mirror symmetry is not supported")
    if sim._cond is not None:
        raise nie("k_point with conductivity/mu materials is not supported "
                  "(the Bloch stepper has no D/B damping)")


def _source_terms(sim):
    """Per source: (profile, mask_c, mask_s|None) -- the complex envelope
    line mask amp * amp_func(r) * exp(-i k_y y) on the Ez/Hz grid."""
    ky = _bloch_ky(sim)
    g, gs = sim.grid, sim.gs
    terms = []
    for spec in sim._sources:
        if isinstance(spec, EigenModeSource) or type(spec).__name__.startswith(
                "GaussianBeam"):
            raise NotImplementedError(
                f"{type(spec).__name__} under a k_point run is not supported")
        if sim.dimensions == 1 and spec.center is not None:
            # 1D: source coords live in z -> map to x (same as _make_sources)
            sz = spec.size if spec.size is not None else Vector3(0, math.inf, 0)
            spec = Source(spec.src, spec.component,
                          center=Vector3(spec.center.z, 0, 0), size=sz,
                          amplitude=spec.amplitude, amp_func=spec.amp_func)
        _, _, ix, j0, j1, prof, amp, amp_func, center, _ = build_line_source(
            spec, gs, sim._pol)
        c = center or Vector3()
        ys = np.array([c.y - gs.sy / 2 + (j + 0.5) * gs.dx
                       for j in range(j0, j1)])
        vals = complex(amp) * np.exp(-1j * ky * ys)      # physical -> envelope
        if amp_func is not None:
            vals = vals * np.array([complex(amp_func(Vector3(c.x, y, 0.0)))
                                    for y in ys])
        # Same current-density integral as core.sources.LineSource: each
        # occupied cell carries the source measure it holds (dx per extended
        # axis) over the cell area. This path builds its own masks, so it
        # needs the normalization too -- without it a k_point run produced a
        # field a factor of 1/dx below the identical non-Bloch run.
        norm = (g.dx if (j1 - j0) > 1 else 1.0) / g.dx ** 2
        mask_c = g.zeros(g.nx, g.ny)
        mask_c[ix, j0:j1] = torch.tensor(vals.real, dtype=g.dtype,
                                         device=g.device) * norm
        mask_s = None
        if np.abs(vals.imag).max() > 0.0:
            mask_s = g.zeros(g.nx, g.ny)
            mask_s[ix, j0:j1] = torch.tensor(vals.imag, dtype=g.dtype,
                                             device=g.device) * norm
        terms.append((prof, mask_c, mask_s))
    return terms


def _flux_mons(sim):
    mons = []
    for (name, kind, mon, meta) in sim._monitors:
        if kind != "flux":
            raise NotImplementedError(
                f"k_point run supports flux monitors only ({kind!r} monitor "
                f"{name!r}: near2far/energy/force/dft-fields are not wired)")
        if not getattr(mon, "x_normal", True):
            raise NotImplementedError(
                "k_point run supports x-normal flux monitors only (the y axis "
                "is the periodic wrap)")
        mons.append((name, mon, meta))
    return mons


def _combine(acc_c, acc_s):
    """(c, s) quadrature accumulators -> the complex-field DFT accumulator.

    The samplers store (R, I) = (sum cos(wt) F, -sum sin(wt) F) per
    quadrature. The complex field runs as e^{-i w t}, so its DFT must use
    meep's e^{+i w t} kernel (the e^{-i w t} kernel would demodulate to
    ~zero): F^ = sum e^{+iwt}(F_c + i F_s) -> Re = R_c + I_s,
    Im = R_s - I_c."""
    return torch.stack([acc_c[:, 0] + acc_s[:, 1], acc_s[:, 0] - acc_c[:, 1],
                        acc_c[:, 2] + acc_s[:, 3], acc_s[:, 2] - acc_c[:, 3]],
                       dim=1)


def run_bloch(sim, step_funcs, until, until_after_sources, n_steps):
    _scope_check(sim, step_funcs)
    sim._init_structure()
    m = sim._coeffs                       # TM: TMCoeffs; TE: same-named x profiles
    if getattr(m, "has_chi2", False):
        raise NotImplementedError("k_point with chi2 media is not supported")
    if getattr(m, "mc1x", None) is not None:
        raise NotImplementedError("k_point with magnetic poles is not supported")
    te = sim._pol == "te"
    n_poles = len(sim.structure._poles)
    if te and (n_poles or m.has_kerr):
        raise NotImplementedError(
            "k_point TE runs support nondispersive linear media only "
            "(step_bloch_te has no poles/Kerr)")
    g = sim.grid
    ky = _bloch_ky(sim)
    if te:
        # ponytail: cell-center eps Yee-averaged to the Bloch TE staggering
        # (Ex at j+1/2 periodic, Ey at i+1/2); no TE tensor rasterization
        eps = sim._rm.eps_tm
        eps_x = 0.5 * (eps + torch.roll(eps, -1, dims=1))
        eps_y = 0.5 * (eps[:-1, :] + eps[1:, :])
        st = init_bloch_te_state(g)
    else:
        st = init_bloch_tm_state(g, n_poles=n_poles)
    # meep's run() continues the simulation it is called on; a second call
    # must not wipe the fields. This built a fresh state every time, so
    # sim.run(until=10) twice re-started from vacuum -- previously masked
    # because the clock also reset, which re-fired the source and produced a
    # plausible-looking (but 10 time units late) pulse. reset_meep() is what
    # clears the state, exactly as in meep.
    if sim._bloch_state is not None:
        st = sim._bloch_state
    terms = _source_terms(sim)
    mons = _flux_mons(sim)
    accs = {name: [None, None] for name, _, _ in mons}

    dt = g.dt
    cutoff = sim._source_cutoff()
    conds, gate, n_total = None, 0.0, n_steps
    stop = until if until is not None else until_after_sources
    if callable(stop) or isinstance(stop, (list, tuple)):
        conds = list(stop) if isinstance(stop, (list, tuple)) else [stop]
        if until is None:
            gate = cutoff
        n_total = None
    elif n_total is None:
        if until is not None:
            n_total = max(1, int(round(float(until) / dt)))
        else:
            n_total = max(1, int(round((cutoff + (until_after_sources or 0)) / dt)))

    # meep step functions: same plan/fire helpers the cartesian path uses, so
    # at_beginning / at_every / at_end / to_appended behave identically under
    # a Bloch run. get_array reconstructs the physical field from the
    # envelope, which is what output_* ultimately reads.
    begin, per_step, scheduled = plan_hooks(step_funcs, cutoff)

    # meep's clock accumulates across run() calls, and the source waveform is
    # a function of ABSOLUTE time -- restarting t_idx re-fires the pulse from
    # zero on every subsequent run(). _run_until was fixed for this; run_bloch
    # had the same defect, which showed up as the pulse being 10 time units
    # behind meep after a second run(until=...).
    start = sim._t_step
    sim._bloch_state = st
    for gfn in begin:
        _eval_step_func(sim, gfn, "step")
    fire_scheduled(sim, scheduled, start * dt)
    t_idx = start
    if n_total is not None:
        n_total = start + n_total
    while n_total is None or t_idx < n_total:
        tt = (t_idx + 0.5) * dt
        inj_c = inj_s = None
        # cutoff == 0 means "no known turn-off" (CustomSource/ContinuousSource
        # carry no t_end -- see Simulation._source_cutoff), NOT "already off".
        # Gating on `tt < cutoff` alone therefore never injected for those, and
        # the run produced a uniformly zero field with no error.
        if cutoff <= 0.0 or tt < cutoff:
            for prof, mask_c, mask_s in terms:
                qc, qs = _quadratures(prof, tt)
                ic = qc * mask_c - (qs * mask_s if mask_s is not None else 0.0)
                s_ = qs * mask_c + (qc * mask_s if mask_s is not None else 0.0)
                inj_c = ic if inj_c is None else inj_c + ic
                inj_s = s_ if inj_s is None else inj_s + s_
        if te:
            st = step_bloch_te(st, m, eps_x, eps_y, ky, src_c=1.0, src_s=0.0,
                               src_mask=inj_c, src_mask_s=inj_s)
        else:
            st = step_bloch_tm(st, m, ky, src_c=1.0, src_s=0.0,
                               src_mask=inj_c, src_mask_s=inj_s)
        sim._bloch_state = st
        if te:
            # FluxMonitorTE reads Ey[ix] collocated with avg(Hz[ix-1], Hz[ix]);
            # the Bloch Ey is x-staggered, so shift it one node with a pad
            view_c = SimpleNamespace(Hz=st.Hz_c, Ey=F.pad(st.Ey_c, (0, 0, 1, 0)))
            view_s = SimpleNamespace(Hz=st.Hz_s, Ey=F.pad(st.Ey_s, (0, 0, 1, 0)))
        else:
            view_c = SimpleNamespace(Ez=st.Ez_c, Hy=st.Hy_c, Hx=st.Hx_c)
            view_s = SimpleNamespace(Ez=st.Ez_s, Hy=st.Hy_s, Hx=st.Hx_s)
        for name, mon, _ in mons:
            sc = mon.sample(view_c, t_idx)
            ss = mon.sample(view_s, t_idx)
            a = accs[name]
            a[0] = sc if a[0] is None else a[0] + sc
            a[1] = ss if a[1] is None else a[1] + ss
        t_idx += 1
        sim._t_step = t_idx
        t_now = t_idx * dt
        fire_per_step(sim, per_step, t_now, dt)
        fire_scheduled(sim, scheduled, t_now)
        if conds is not None:
            t = t_idx * dt
            flags = [bool(c(sim)) for c in conds]
            if t >= gate - 1e-9 and any(flags):
                break

    for name, mon, meta in mons:
        acc_c, acc_s = accs[name]
        if acc_c is None:                 # zero-step run
            continue
        comb = _combine(acc_c, acc_s)
        seed = getattr(sim, "_acc_seed", {}).get(name)
        if seed is not None:
            comb = comb + seed
        sim._last_acc[name] = (comb, mon, meta)
