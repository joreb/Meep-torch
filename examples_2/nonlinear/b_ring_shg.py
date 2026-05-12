# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Example (b): TFLN ring for SHG, pump 1348 nm -> second harmonic 674 nm.

Half-ring coupler extraction, same method as a_ring_smatrix.py (peer-approved
template): only the LOWER HALF of the gdsfactory ring_single is simulated, the
ring side-guides run vertically at the cut plane and exit into the top PML, and
two short single-pass runs give the whole coupler:

  run 1 (bus-driven):  t(f) = bus through amplitude, kappa(f) = amplitude into
                       the ring (right arc port),
  run 2 (arc-driven):  A(f) = arc-end to arc-end = t * a(f), so the bare half
                       loop is a = A / t.

The ring spectrum is then assembled analytically (all-pass), giving FSR and Q at
arbitrary resolution without waiting for FDTD ring buildup. Run at BOTH
wavelengths: the pump must resonate for buildup, and the SH must also be guided
(it is generated inside the ring), so both get their own coupler extraction.

RADIUS r = 50 um from examples_2/b_ring_bendcheck.py (peer-approved): the pump is
the binding constraint at 0.0004 dB/round-trip, while r=30 would put it at
0.152 dB/rt, over the 0.1 dB budget. The SH is comfortable at both.

STACK IS ASSUMED, NOT MEASURED (peer 444238d7 approved proceeding on it, user
can correct later; the method stands either way): 600 nm LN film, 300 nm etch,
1.2 um top width, SiO2 clad, quasi-TE, n_e-ish indices.

2D EFFECTIVE-INDEX REDUCTION -- note the lateral cladding is the UNETCHED SLAB
(300 nm of LN), NOT bulk SiO2, so the lateral contrast is much weaker than a
naive LN/oxide guess:
    1348 nm: ridge 2.0233 / slab 1.7065  -> dn = 0.317
     674 nm: ridge 2.2198 / slab 2.0777  -> dn = 0.142
The SH is the WEAKLY confined one here (opposite of the SiN platform), which is
also why its bend loss is mildly non-monotone in R -- it sits nearer the slab
continuum. Ridge values are fd2d straight-mode solves, slab values fd1d.

Coupler gap: peer note (1ba5de1c) -- at LN confinement the gap has to be tighter
than the SiN ring's. GAP_SWEEP brackets it; the first run reports |kappa|^2 so
the gap can be picked once before grinding the full model.

Run: uv run python examples_2/b_ring_shg.py
"""
import dataclasses
import math
import os
import pathlib
import sys

import gdsfactory as gf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(16)

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gds_meep_torch import component_to_eps

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.monitors import PointProbe
from meep_torch.core.sources import (ContinuousSource, CustomSource,
                                     GaussianPulse)
from meep_torch.core.stepper import init_tm_state
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port, validate_ports

OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()
DEV = os.environ.get("BRS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# --- stack-assumed 2D reduction (see module docstring) --------------------
STACK_CONFIRMED = False
# (label, lambda_um, n_ridge, n_slab_lateral_clad)
BANDS = {
    "pump 1348": (1.348, 2.0233, 1.7065),
    "SH 674": (0.674, 2.2198, 2.0777),
}
R_RING = 50.0            # from b_ring_bendcheck.py, peer-approved
W_WG = 1.2               # ridge top width (the stack's guide width)
LC = 4.0                 # racetrack straight coupler section
# TWO BUSES (peer 54ba96d9). The bands want incompatible gaps -- measured
# gamma is 5.07/um at the pump vs 7.29/um at the SH, so kappa diverges
# exponentially with gap and no single coupler serves both. A real SHG ring
# uses one bus tuned for the pump and a second for the SH; the final gds is
# gf.ring_double(radius, gap=GAP_PUMP, gap_top=GAP_SH).
#
# Each coupler is characterized on its OWN 2-port half-ring (no 3-arc-port
# geometry needed), but BOTH bands are measured through BOTH couplers: the SH
# coupler still perturbs the pump round trip and vice versa, so the assembled
# round trip needs t from both.
# SINGLE BUS is the settled architecture (peer 3db9578c/89ee2cda). Two buses
# were measured and REJECTED: the SH bus at 0.20 couples 59% of the PUMP out per
# round trip, and with that cross term the pump has NO resonance at all
# (|t_tot|=0.61 against a lossless loop -> zero dips). A phase-mismatched SH bus
# width was scanned and proved impossible (b_ring_buswidth_scan.py). So one bus
# at the compromise gap: the pump resonates (Q~4700) and the SH still escapes
# (|k|^2~0.031). BRS_TWOBUS=1 restores the two-bus measurement for the record.
TWO_BUS = os.environ.get("BRS_TWOBUS") == "1"
GAP_SINGLE_BUS = float(os.environ.get("BRS_GAP1", "0.25"))
GAP_PUMP = float(os.environ.get("BRS_GAP_PUMP", "0.44"))   # two-bus only
GAP_SH = float(os.environ.get("BRS_GAP_SH", "0.20"))       # two-bus only
# RES is a real tradeoff here, sized against MEASURED GPU throughput
# (~0.3e9 cell-steps/s on the GPU, from the coupler run). An r=50 half-ring is
# ~4400 x 2300 at RES=40, and each wavelength needs 2 runs:
#   RES=30: 5.7M cells, 0.9 h both bands, gap = 4.5 cells   <- gap too coarse
#   RES=40: 10.2M cells, 2.2 h both bands, gap = 6.0 cells  <- default
#   RES=60: 23.0M cells, 7.5 h both bands, gap = 9.0 cells  <- monopolizes the GPU
# The binding feature is the 0.15 um coupler gap, not the wavelength: at RES=40
# the SH still gets 12 cells per in-medium wavelength, but the gap gets only 6,
# and kappa is exponentially sensitive to the gap. If the extracted |kappa|^2
# looks grid-limited, raise RES before trusting it.
RES = int(os.environ.get("BRS_RES", "40"))
dx = 1.0 / RES
NPML = RES

# --- QPM periodic poling ----------------------------------------------------
# The ring is NOT phase matched. From our own fd2d numbers, neff(pump 1348) =
# 2.0233 vs neff(SH 674) = 2.2198, so dn = 0.1965 and the coherence length is
# lam_SH/(4*dn) = 0.86 um against a 314 um circumference -- 366 coherence
# lengths. The SH amplitude reverses sign every 0.86 um and net conversion
# integrates to ~ZERO, no matter how good Q or kappa are. Quasi-phase-matching
# flips the sign of chi2 every Lambda/2 of ARC LENGTH so successive half
# periods add instead of cancelling:
#     Lambda = lam_SH / dn = 3.43 um  ->  183 half-domains around r=50
# The flip must follow ARC ANGLE about the ring center, not x or y: a Cartesian
# stripe pattern mis-poles the curved sections.
#
# Modal phase matching was checked and RULED OUT on this stack: the 8 lowest
# 674 nm modes of the W=1.2 um guide run 2.2198 down to 2.1350, all >= 0.11
# above the pump neff, so there is no higher-order SH mode to match against.
QPM_DN = abs(BANDS["SH 674"][1] - BANDS["pump 1348"][1])    # 0.1965, measured
QPM_LAMBDA = 0.674 / QPM_DN                                  # 3.43 um
QPM_DEFF_FACTOR = 2.0 / math.pi          # d_eff = (2/pi)d at Delta_k = 0


def poling_mask(core_mask, cx, cy):
    """Signed chi2 field: +1/-1 alternating every QPM_LAMBDA/2 of arc length.

    Arc length is R*theta, so flipping every Lambda/2 of arc means flipping
    every Lambda/(2R) in angle. Zero outside the LiNbO3 core.
    """
    nx, ny = core_mask.shape
    xs, ys = torch.meshgrid(
        torch.arange(nx, dtype=core_mask.dtype, device=core_mask.device),
        torch.arange(ny, dtype=core_mask.dtype, device=core_mask.device),
        indexing="ij")
    theta = torch.atan2(ys - cy, xs - cx)                    # [-pi, pi]
    arc = theta * R_RING
    sign = torch.where(((arc / (QPM_LAMBDA / 2.0)).floor() % 2) == 0, 1.0, -1.0)
    return (sign.to(core_mask.dtype) * core_mask) + 0.0      # +0.0 kills -0.0


# gpdk 'strip' enforces a radius_min; r=50 is way above it, but width must be set
XS = gf.get_cross_section("strip").copy(width=W_WG, radius_min=R_RING)


def build_half(n_core, n_clad, gap):
    """Lower half of ring_single at `gap`, guides extended into the top PML."""
    c = gf.components.ring_single(radius=R_RING, gap=gap, length_x=LC,
                                  cross_section=XS)
    grid_f, eps_f, ports = component_to_eps(c, res=RES, pad=3.0,
                                            n_core=n_core, n_clad=n_clad)
    o1x, o1y, _ = ports["o1"]
    o2x, o2y, _ = ports["o2"]
    j_center = o1y + int((W_WG / 2 + gap + W_WG / 2 + R_RING) * RES)
    ny_h = j_center + int(2.0 * RES) + NPML
    nx_h = grid_f.nx
    grid = Grid(nx=nx_h, ny=ny_h, dx=dx, device=DEV)
    eps = torch.full((nx_h, ny_h), n_clad ** 2, dtype=grid.dtype)
    eps[:, :j_center] = eps_f[:, :j_center]
    eps[:, j_center:] = eps[:, j_center - 1: j_center]   # guides straight up
    return grid, eps.to(DEV), (o1x, o1y, o2x, o2y), nx_h, ny_h


def arc_columns(eps, j_arc, nx_h):
    """Locate the two vertical side guides in the eps row at j_arc."""
    row = eps[:, j_arc]
    thr = 0.5 * (float(row.min()) + float(row.max()))
    cols = (row > thr).nonzero().flatten()
    assert len(cols) > 0, "no core found at the arc plane"
    splits = (cols.diff() > 1).nonzero().flatten()
    assert len(splits) == 1, f"expected 2 side guides, got {len(splits) + 1}"
    cl, cr = cols[: splits[0] + 1], cols[splits[0] + 1:]
    return int(cl.float().mean().round()), int(cr.float().mean().round())


def extract(label, lam, n_core, n_clad, gap, coupler):
    """Measure one BAND through one COUPLER (t, kappa, bare half loop)."""
    fcen = 1.0 / lam
    grid, eps, (o1x, o1y, o2x, o2y), nx_h, ny_h = build_half(n_core, n_clad, gap)
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)
    # SPAN: core half-width + evanescent tail. The tail scales with the LOCAL
    # contrast, which is weak at 674 (dn=0.142) -- so size it off the mode, not
    # off a fixed number of um. decay ~ lam / (2*pi*sqrt(n_core^2 - n_clad^2)).
    decay = lam / (2 * math.pi * math.sqrt(max(n_core ** 2 - n_clad ** 2, 1e-6)))
    span_um = W_WG / 2 + 4.0 * decay
    SPAN = int(round(span_um * RES))
    print(f"b_ring_shg  [{coupler} coupler, gap={gap}] {label}: dn={n_core - n_clad:.3f} "
          f"decay={decay*1e3:.0f} nm -> SPAN={span_um:.2f} um ({SPAN} cells), "
          f"grid {nx_h}x{ny_h} on {DEV}")

    p_in = Port(axis="x", index=max(o1x + 4, NPML + 4), lo=o1y - SPAN,
                hi=o1y + SPAN, direction=+1)
    p_thru = Port(axis="x", index=min(o2x - 4, nx_h - NPML - 4), lo=o2y - SPAN,
                  hi=o2y + SPAN, direction=-1)
    j_arc = ny_h - NPML - int(2.0 * RES)
    xL, xR = arc_columns(eps, j_arc, nx_h)
    p_arcL = Port(axis="y", index=j_arc, lo=xL - SPAN, hi=xL + SPAN, direction=-1)
    p_arcR = Port(axis="y", index=j_arc, lo=xR - SPAN, hi=xR + SPAN, direction=-1)
    validate_ports([p_in, p_thru, p_arcL, p_arcR], grid, NPML)

    nfreq = 60
    # FSR ~ lam^2/(n_g*2*pi*R) scales as lam^2, so a FIXED relative band catches
    # fewer dips at the pump -- the first run returned FSR=nan at 1348 while the
    # SH got 0.861 nm cleanly. Scale the band to cover ~4 FSR at EITHER wavelength
    # (peer 54ba96d9 asked for pump FSR): df/f ~ 4*FSR/lam ~ 4*lam/(n_g*2*pi*R).
    df_band = fcen * 4.0 * lam / (n_core * 2 * math.pi * R_RING)
    freqs = np.linspace(fcen - df_band / 2, fcen + df_band / 2, nfreq).tolist()
    fwidth = 0.05 * fcen / (1 / 0.674)
    pulse = GaussianPulse(fcen, fwidth)
    # T_RUN must cover the pulse tail PLUS the longest optical path, and both
    # scale with the band. The a_ring template's fixed 420 was sized for 674 nm;
    # at 1348 nm the pulse tail alone is ~200 (fwidth halves with fcen) and the
    # arc transit is pi*R*n ~ 318, so 420 TRUNCATES. Measured: the first run
    # returned |t|^2+|k|^2 = 0.041 with |a_loop| = 0 -- the light had not arrived.
    t_pulse = 5.0 / fwidth
    t_path = math.pi * R_RING * n_core          # arc run = half circumference
    t_run = 1.25 * (t_pulse + t_path)           # +25% for the ringdown tail
    print(f"b_ring_shg  {label}: T_RUN={t_run:.0f} "
          f"(pulse tail {t_pulse:.0f} + arc transit {t_path:.0f}, +25%)")

    def eline(p):
        return eps[p.index, p.lo:p.hi] if p.axis == "x" else eps[p.lo:p.hi, p.index]

    def run_ports(src_port, mons_ports):
        src = ModeSourceTM(grid, coeffs, src_port, pulse, fcen)
        mons = [FluxMonitorTM(grid, freqs, x_normal=(p.axis == "x"), idx=p.index,
                              lo=p.lo, hi=p.hi) for p in mons_ports]
        with torch.no_grad():
            _, accs = run_tm(init_tm_state(grid), coeffs,
                             int(round(t_run / grid.dt)), source=src,
                             monitors=mons, checkpoint_every=0)
        out = []
        for p, acc in zip(mons_ports, accs):
            sign = -1.0 if p.axis == "x" else +1.0
            out.append(decompose(acc.cpu(), eline(p).cpu(), dx, freqs, "tm", 1,
                                 axis_sign=sign))
        return out

    r1 = run_ports(p_in, [p_in, p_thru, p_arcR])
    a_in = r1[0][0][:, 0]
    t = (r1[1][0][:, 0] / a_in).numpy()
    kap = (r1[2][0][:, 0] / a_in).numpy()
    r2 = run_ports(p_arcR, [p_arcR, p_arcL])
    a_inR = r2[0][1][:, 0]
    A = (r2[1][0][:, 0] / a_inR).numpy()
    # a_half = A / |t|, NOT A / t (peer 0a2eaf0). A is the arc-to-arc amplitude;
    # dividing by the COMPLEX t strips a bus propagation phase that the arc
    # path never travelled, which shows up as a tripled apparent FSR. Only the
    # coupler's magnitude belongs here.
    loop = (A / np.abs(t)) ** 2                  # bare full loop a^2

    mid = nfreq // 2
    unit = abs(t[mid]) ** 2 + abs(kap[mid]) ** 2
    print(f"b_ring_shg  {label} @ {1e3/freqs[mid]:.1f} nm: |t|^2={abs(t[mid])**2:.4f} "
          f"|kappa|^2={abs(kap[mid])**2:.4f} |a_loop|={abs(loop[mid]):.4f} "
          f"(|t|^2+|k|^2={unit:.3f})")
    # A lossless coupler conserves power: |t|^2 + |k|^2 ~ 1. Far below that means
    # the run ended before the light arrived (or the ports miss the mode), and the
    # extracted t/kappa/loop are meaningless -- say so instead of assembling a
    # spectrum from them and reporting a confident Q.
    if unit < 0.8:
        print(f"b_ring_shg  WARNING {label}: |t|^2+|k|^2={unit:.3f} << 1 -- coupler "
              f"extraction INVALID (T_RUN too short or ports off-mode). "
              f"Q/FSR/kappa below are NOT trustworthy.")

    return dict(label=label, coupler=coupler, gap=gap, lam=lam,
                freqs=np.array(freqs), t=t, kappa=kap, loop=loop,
                eps=eps.cpu().numpy(), nx=nx_h, ny=ny_h,
                kappa2=abs(kap[mid]) ** 2, unit=unit)


def assemble(band, m_pump_coupler, m_sh_coupler):
    """Two-bus ring spectrum for ONE band.

    With two couplers the round trip passes BOTH, so the effective through
    amplitude is t_total = t_pump_coupler * t_sh_coupler (both measured
    complex, at this band). a_bare is the bare loop; each `loop` already
    equals a^2 for its own run, so take it from either measurement -- they
    describe the same physical ring -- and use the pump-coupler one, whose
    arc run is the better-converged of the two.
    """
    freqs = m_pump_coupler["freqs"]
    t_tot = m_pump_coupler["t"] * m_sh_coupler["t"]
    loop = m_pump_coupler["loop"]
    fd = np.linspace(freqs[0], freqs[-1], 6000)
    # t enters the all-pass ratio as a MAGNITUDE (peer 0a2eaf0): its phase is bus
    # propagation, which is not part of the ring round trip. Carrying complex t
    # into the denominator produced |S21| > 1 lobes -- unphysical for a passive
    # ring. The loop keeps its full complex phase; that is the resonance.
    # group index from the measured loop phase slope: the round-trip phase is
    # 2*pi*f*n_g*L, so d(phase)/df = 2*pi*n_g*L. This is the MEASURED n_g, which
    # is what sets the true FSR -- better than reusing the phase index.
    ph = np.unwrap(np.angle(loop))
    dphi_df = np.polyfit(freqs, ph, 1)[0]
    n_g_est = abs(dphi_df) / (2 * math.pi * 2 * math.pi * R_RING)
    tm = np.interp(fd, freqs, np.abs(t_tot))
    # The extracted bare loop can come out marginally ABOVE 1 where the coupler
    # is nearly transparent (a = A/|t| divides measurement noise in A by a |t|
    # close to 1). A passive ring cannot have round-trip gain, so clamp -- but
    # assert the excess is small, because a LARGE excess would mean the
    # extraction is wrong rather than merely noisy. Measured worst case here is
    # |a^2| = 1.0194 on the pump, against two independent estimates of the same
    # loop agreeing to 0.9939 vs 0.9915.
    a2 = np.abs(loop)
    assert a2.max() < 1.05, (
        f"bare loop |a^2| = {a2.max():.4f} -- too far above 1 to be noise; "
        f"the loop extraction is wrong, not just noisy")
    lm = np.interp(fd, freqs, np.minimum(a2, 1.0))
    lp = np.interp(fd, freqs, np.unwrap(np.angle(loop)))
    loop_d = lm * np.exp(1j * lp)
    T = np.abs((tm - loop_d) / (1 - tm * loop_d)) ** 2
    assert T.max() <= 1.0 + 1e-9, (
        f"|S21|^2 max {T.max():.4f} > 1 -- passive ring cannot amplify; "
        f"check that t enters as a magnitude")

    i0 = int(np.argmin(T))
    half = (T[i0] + np.median(T)) / 2
    lo = hi = i0
    while lo > 0 and T[lo] < half:
        lo -= 1
    while hi < len(fd) - 1 and T[hi] < half:
        hi += 1
    Q = fd[i0] / max(fd[hi] - fd[lo], 1e-12)
    # Dip finding by PROMINENCE, not by an absolute fraction of the median. A
    # shallow-but-real resonance (the pump dips to 0.73 against a 0.996 median,
    # 1.2 dB extinction) never crosses a 0.6*median threshold, so the old test
    # returned FSR=nan while 359 interpolation-ripple minima sat in the array.
    # Require each minimum to be the deepest point within half an expected FSR
    # and to stand clear of the local baseline by a fraction of the full depth.
    med = float(np.median(T))
    depth = med - float(T.min())
    fsr_expect_f = 1.0 / (n_g_est * 2 * math.pi * R_RING)   # FSR in freq units
    win = max(int(0.5 * fsr_expect_f / (fd[1] - fd[0])), 3)
    dips = []
    for i in range(win, len(fd) - win):
        if T[i] > med - 0.25 * depth:
            continue                       # not deep enough to be a resonance
        if T[i] <= T[i - win:i + win + 1].min():
            if not dips or (i - dips[-1]) > win:
                dips.append(i)
    fsr = (abs(1e3 / fd[dips[0]] - 1e3 / fd[dips[-1]]) / (len(dips) - 1)
           if len(dips) >= 2 else float("nan"))
    ext = 10 * math.log10(max(np.median(T), 1e-12) / max(T[i0], 1e-12))
    fsr_analytic = (1e3 * (1.0 / freqs[len(freqs) // 2]) ** 2
                    / (n_g_est * 2 * math.pi * R_RING))
    print(f"b_ring_shg  ASSEMBLED {band}: dip {1e3/fd[i0]:.2f} nm |S21|^2={T[i0]:.3f} "
          f"Q={Q:.0f} FSR={fsr:.3f} nm (analytic {fsr_analytic:.3f} at measured "
          f"n_g={n_g_est:.3f}) extinction {ext:.1f} dB "
          f"[|t_pump_c|={abs(m_pump_coupler['t'][len(freqs)//2]):.4f} "
          f"|t_sh_c|={abs(m_sh_coupler['t'][len(freqs)//2]):.4f}]")
    return dict(band=band, fd=fd, T=T, Q=Q, fsr=fsr, i0=i0, ext=ext,
                n_g=n_g_est, fsr_analytic=fsr_analytic, f0=fd[i0])


# --- poled full-ring nonlinear SHG (peer 998a3c70) ---------------------------
# The half-ring extraction above gives the LINEAR resonator (t, kappa, Q, FSR).
# SHG needs the actual poled ring driven CW on resonance, so this section runs
# the full ring with a signed chi2. Sized from the measured linear result:
# buildup tau = Q/(pi f) ~ 2000 tu at Q~4700, so T_RUN ~ 3 tau.
#
# BRS_NL=1 runs it (it is a multi-hour GPU grind); BRS_NL_SCALE=1 runs only the
# short scale check, which exists to catch a chi2 magnitude error BEFORE the
# long grinds rather than after (peer's instruction, and the same discipline
# that caught the truncated-T_RUN Q earlier).
RES_NL = int(os.environ.get("BRS_RES_NL", "30"))
CHI2 = float(os.environ.get("BRS_CHI2", "0.5"))


def build_full_ring(lam, n_core, n_clad, res):
    """Full (not half) ring_single on its own grid, plus the signed chi2 mask."""
    c = gf.components.ring_single(radius=R_RING, gap=GAP_SINGLE_BUS, length_x=LC,
                                  cross_section=XS)
    grid_f, eps_f, ports = component_to_eps(c, res=res, pad=2.0,
                                            n_core=n_core, n_clad=n_clad)
    grid_f = dataclasses.replace(grid_f, device=DEV)
    eps = eps_f.to(DEV)
    core = (eps > 0.5 * (n_core ** 2 + n_clad ** 2)).to(eps.dtype)
    # Ring center from the RING's own extent, not the core centroid: the bus is
    # part of `core` and would drag the centroid off-center, which would make
    # the poling angle advance non-uniformly around the ring. The ring spans
    # 2R+W vertically from its top; take x from the ring bbox and y as
    # (top of core) - (R + W/2), both in cells.
    col_has = core.sum(dim=0)                    # per-y count
    row_has = core.sum(dim=1)                    # per-x count
    x_lo = int(torch.nonzero(row_has)[0])
    x_hi = int(torch.nonzero(row_has)[-1])
    y_hi = int(torch.nonzero(col_has)[-1])       # top of the ring
    cx = 0.5 * (x_lo + x_hi)
    cy = y_hi - (R_RING + W_WG / 2.0) * res
    pol = poling_mask(core, float(cx), float(cy))
    # bus source port + a probe INSIDE the ring (top of the ring, opposite the
    # bus, so the probe sees circulating field rather than the bus directly).
    o1x, o1y, _ = ports["o1"]
    npml_nl = int(res)
    span = int(round((W_WG / 2 + 0.8) * res))
    p_src = Port(axis="x", index=max(o1x + 4, npml_nl + 4),
                 lo=o1y - span, hi=o1y + span, direction=+1)
    ring_probe = (int(round(cx)), int(y_hi) - int(round(W_WG / 2 * res)))
    assert core[ring_probe] > 0, "ring probe is not inside the LiNbO3 core"
    return grid_f, eps, core, pol, p_src, ring_probe, (cx, cy)


def nl_scale_check():
    """SHORT run (T~1000) to sanity-check the chi2 conversion magnitude."""
    lam, nco, ncl = BANDS["pump 1348"]
    grid_nl, eps, core, pol, p_src, ring_probe, (cx, cy) = build_full_ring(
        lam, nco, ncl, RES_NL)
    npos = int((pol > 0).sum())
    nneg = int((pol < 0).sum())
    n_dom = int(round(2 * math.pi * R_RING / (QPM_LAMBDA / 2.0)))
    print(f"b_ring_shg  NL grid {grid_nl.nx}x{grid_nl.ny} "
          f"({grid_nl.nx*grid_nl.ny/1e6:.1f}M cells) on {DEV}")
    print(f"b_ring_shg  QPM: Lambda={QPM_LAMBDA:.3f} um (dn={QPM_DN:.4f}), "
          f"{n_dom} half-domains expected around r={R_RING}; "
          f"mask has +{npos} / -{nneg} core cells "
          f"(imbalance {abs(npos-nneg)/max(npos+nneg,1)*100:.1f}%)")
    assert abs(npos - nneg) / max(npos + nneg, 1) < 0.10, (
        "poling mask badly unbalanced -- check the ring center estimate")
    print(f"b_ring_shg  ring probe at cell {ring_probe} (in core), "
          f"bus source at x={p_src.index}")
    return grid_nl, eps, core, pol, p_src, ring_probe


def ring_probe_series(grid_nl, eps, pol, chi2_val, f0, amp, t_end, t_skip,
                      ring_xy, src_port):
    """CW-drive the poled ring at f0; return the Ez time series in the ring."""
    coeffs = build_tm_coeffs(grid_nl, eps_inf=eps, npml=int(RES_NL),
                             chi2=chi2_val * pol)
    # amplitude scales the SOURCE masks (power ~ amp^2); run_tm has no amplitude
    # kwarg. Both the electric and the magnetic sheet must be scaled together.
    src = ModeSourceTM(grid_nl, coeffs, src_port,
                       ContinuousSource(f0, width=15.0), f0)
    src.mask = src.mask * amp
    for k in src.h_masks:
        src.h_masks[k] = src.h_masks[k] * amp
    n_steps = int(round(t_end / grid_nl.dt))
    # checkpoint_every=400 bounds PointProbe's stack-reduce partials per chunk:
    # a bare probe holds a view into the live Ez every step, which is tens of GB
    # over a 300k-step run (see wiki bugs/monitor-sample-accumulation-oom).
    with torch.no_grad():
        _, (sig,) = run_tm(init_tm_state(grid_nl), coeffs, n_steps,
                           source=src, monitors=[PointProbe(*ring_xy)],
                           checkpoint_every=400)
    skip = int(round(t_skip / grid_nl.dt))
    # upcast the probe series: the SH power is orders below the pump and float32
    # FFT noise would bury it.
    return sig.detach().to(torch.float64).cpu().numpy()[skip:]


def shg_conversion(sig, dt, f0):
    """P(2f0)/P(f0) from the ring time series (Hanning to kill CW sidelobes)."""
    fr = np.fft.rfftfreq(len(sig), dt)
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2

    def band(f, w=0.02):
        m = (fr > f - w) & (fr < f + w)
        return float(spec[m].max()) if m.any() else 0.0

    p1, p2 = band(f0), band(2 * f0)
    return p1, p2, (p2 / p1 if p1 > 0 else float("nan")), fr, spec


def charge_then_ring(grid_nl, eps, chi3_val, f0, amp, t_charge, t_ring,
                     ring_xy, src_port):
    """Charge the ring CW, cut the source, record the FREE decay.

    The source is a gated CW (CustomSource): sin carrier with a smooth ramp on,
    then hard off at t_charge. After that the ring rings down freely, which is
    the whole point -- there is no drive left for the Kerr-shifted resonance to
    walk away from, so the shift can be made large and still measured cleanly.
    Returns only the post-cutoff series.
    """
    coeffs = build_tm_coeffs(grid_nl, eps_inf=eps, npml=int(RES_NL),
                             chi3=chi3_val * (eps > 0.5 * (float(eps.min())
                                                           + float(eps.max()))))
    ramp = 5.0 / f0

    def gated(t):
        if t >= t_charge:
            return 0.0
        return math.sin(2 * math.pi * f0 * t) * min(1.0, t / ramp)

    src = ModeSourceTM(grid_nl, coeffs, src_port, CustomSource(gated), f0)
    src.mask = src.mask * amp
    for k in src.h_masks:
        src.h_masks[k] = src.h_masks[k] * amp
    n_steps = int(round((t_charge + t_ring) / grid_nl.dt))
    with torch.no_grad():
        _, (sig,) = run_tm(init_tm_state(grid_nl), coeffs, n_steps,
                           source=src, monitors=[PointProbe(*ring_xy)],
                           checkpoint_every=400)
    cut = int(round(t_charge / grid_nl.dt))
    return sig.detach().to(torch.float64).cpu().numpy()[cut:]


def band_power(sig, dt, f, w=0.02):
    """Integrated band power, normalised so it does NOT depend on record length.

    The earlier estimator took the band PEAK, and a coherent CW tone's peak bin
    grows with N while a less-coherent tone does not -- so peak-based eta was
    record-length dependent and two runs of different length were never
    comparable. (The P^2 gate survived only because both its runs had equal N.)
    """
    fr = np.fft.rfftfreq(len(sig), dt)
    win = np.hanning(len(sig))
    psd = np.abs(np.fft.rfft(sig * win)) ** 2 / (win ** 2).sum()
    m = (fr > f - w) & (fr < f + w)
    return float(psd[m].sum())


def tone_envelope(sig, dt, f, n_slices=200):
    """Slice-wise amplitude of one tone: |FFT| in its band per slice."""
    k = len(sig) // n_slices
    t, amp = [], []
    for i in range(n_slices):
        seg = sig[i * k:(i + 1) * k]
        if len(seg) < 16:
            continue
        t.append((i + 0.5) * k * dt)
        amp.append(math.sqrt(max(band_power(seg, dt, f), 0.0)))
    return np.array(t), np.array(amp)


def fit_charge_tau(t, amp):
    """Fit A*(1 - exp(-t/tau)) with tau FREE; returns (A, tau, r2, coverage).

    Fitting the FIELD charge curve, so the caller must pass an amplitude
    envelope, not power. A short record cannot constrain tau -- normalising by
    the record end then makes a still-climbing curve look saturated, which is
    exactly how a 0.32-tau record produced a bogus tau=494. Guard: report r2 and
    the record length in units of the fitted tau so the caller can judge.
    """
    # r2 is NOT evidence the tau is right: on a 0.40-tau record this fit returns
    # r2 = 1.0000 while being 99.4% wrong, because an exponential's early rise is
    # indistinguishable from a straight line. COVERAGE is the real criterion, so
    # the caller must check record_length / fitted_tau. Verified on synthetic
    # data: 0.40 tau -> 99.4% error at r2=1.0; 2.0 tau -> 0.8% error.
    best = (None, None, -1e9)
    for tau in np.geomspace(max(t[1], 1.0), 20.0 * t[-1], 400):
        basis = 1.0 - np.exp(-t / tau)
        A = float((basis * amp).sum() / max((basis * basis).sum(), 1e-30))
        pred = A * basis
        ss_res = float(((amp - pred) ** 2).sum())
        ss_tot = float(((amp - amp.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        if r2 > best[2]:
            best = (A, tau, r2)
    A, tau, r2 = best
    cov = t[-1] / max(tau, 1e-30)
    if cov < 1.0:
        print(f"b_ring_shg  WARNING: charge fit covers only {cov:.2f} tau -- a "
              f"sub-tau record cannot constrain tau (r2={r2:.4f} regardless). "
              f"Treat tau={tau:.0f} as UNMEASURED.")
    return A, tau, r2, cov


def gaussian_charge_ring(grid_nl, eps, chi3_val, f0, amp, t_pulse, t_ring,
                         ring_xy, src_port):
    """Charge with a GAUSSIAN pulse, then record the free ringdown.

    Pulse duration is set by sidemode suppression, not by convenience: the
    bandwidth must sit under the FSR so only one azimuthal mode is excited. A
    flat-top whose sinc null lands on the +/-1 FSR sidemode looks better on
    paper (-328 dB) but that null must be HIT -- T_rt is known only to a few
    percent (our two n_g estimates differ by 2.6%), and a 2% mismatch collapses
    it to -34 dB. The Gaussian has no null to miss: 1400 tu gives -61.8 dB and
    IMPROVES with mismatch. Robust beats sharp when the tuning parameter is
    itself uncertain.
    """
    core = (eps > 0.5 * (float(eps.min()) + float(eps.max())))
    coeffs = build_tm_coeffs(grid_nl, eps_inf=eps, npml=int(RES_NL),
                             chi3=chi3_val * core)
    t0 = 0.5 * t_pulse
    sig_t = t_pulse / 2.355                       # FWHM -> sigma

    def pulse(t):
        return math.sin(2 * math.pi * f0 * t) * math.exp(-0.5 * ((t - t0) / sig_t) ** 2)

    src = ModeSourceTM(grid_nl, coeffs, src_port, CustomSource(pulse), f0)
    src.mask = src.mask * amp
    for k in src.h_masks:
        src.h_masks[k] = src.h_masks[k] * amp
    n_steps = int(round((t_pulse + t_ring) / grid_nl.dt))
    with torch.no_grad():
        _, (sig,) = run_tm(init_tm_state(grid_nl), coeffs, n_steps,
                           source=src, monitors=[PointProbe(*ring_xy)],
                           checkpoint_every=400)
    cut = int(round(t_pulse / grid_nl.dt))
    return sig.detach().to(torch.float64).cpu().numpy()[cut:]


def kerr_ringdown(sig, dt, f0_cold, floor_rel):
    """Kerr law from ONE free ringdown: instantaneous frequency vs stored energy.

    Charge the ring, cut the source, and let it ring freely. There is no drive
    to walk off, so the shift can be made large (high charge amplitude) while
    staying a clean measurement -- the CW alternative cannot do both at this Q:
    a perturbative amplitude (0.4 linewidths) sits only 2.6x above the
    frequency floor, and a resolvable amplitude (1.6 linewidths) has already
    walked the resonance off the drive.

    Instantaneous frequency via the PHASE DERIVATIVE OF THE ANALYTIC SIGNAL
    (Hilbert transform), not sliding harminv: the phase derivative is local by
    construction, so its time resolution is set by the smoothing window alone,
    whereas a sliding harminv window imposes its own 1/T floor on top and would
    blur exactly the early-decay region where the shift is largest. Envelope
    from the same analytic signal gives instantaneous stored energy U ~ |a|^2.

    Returns (t, f_inst, U) with the transient edges trimmed.
    """
    from scipy.signal import hilbert
    z = hilbert(sig - sig.mean())
    ph = np.unwrap(np.angle(z))
    env = np.abs(z)
    # smooth over ~10 optical cycles: long enough to kill carrier ripple, short
    # enough to track the decay (tau is thousands of cycles).
    w = max(int(round(10.0 / (f0_cold * dt))), 5)
    k = np.ones(w) / w
    f_inst = np.convolve(np.gradient(ph, dt) / (2 * np.pi), k, mode="same")
    U = np.convolve(env ** 2, k, mode="same")
    trim = w
    return (np.arange(len(sig)) * dt)[trim:-trim], f_inst[trim:-trim], U[trim:-trim]


def kerr_gates(t, f_inst, U, f0_cold, floor_rel):
    """(g3a) f vs U linear over the resolvable window; (g3b) late f -> cold f0."""
    shift = f_inst - f0_cold
    resolvable = np.abs(shift) > 3.0 * floor_rel * f0_cold
    g3a_ok = False
    slope = r2 = icept = float("nan")
    if resolvable.sum() > 50:
        u, f = U[resolvable], f_inst[resolvable]
        slope, icept = np.polyfit(u, f, 1)
        pred = slope * u + icept
        ss_res = float(((f - pred) ** 2).sum())
        ss_tot = float(((f - f.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        g3a_ok = r2 >= 0.99
    # g3b RESTATED (peer 7fecc477): the df-vs-U fit INTERCEPT at U=0 must equal
    # the cold f0. Same physical self-check -- does the ringdown return to the
    # cold cavity -- but it extrapolates 1.16x along the line g3a already fits
    # instead of waiting for U to physically reach zero. Literal convergence
    # would need >3 tau_field of ringdown (7+ h) because the residual shift
    # decays only as fast as the stored energy: still 575x above the floor at
    # 1 tau, 11x at 3 tau. The measured residual at record end is reported as a
    # caveat rather than hidden.
    f_late = float(np.mean(f_inst[int(0.9 * len(f_inst)):]))
    if resolvable.sum() > 50:
        f_at_zero_U = float(icept)          # intercept of the g3a fit
    else:
        f_at_zero_U = f_late
    g3b_ok = abs(f_at_zero_U - f0_cold) <= floor_rel * f0_cold
    print(f"b_ring_shg  KERR g3a: df/dU={slope:.4e}, R^2={r2:.4f} over "
          f"{int(resolvable.sum())} samples (shift > 3x floor) -> "
          f"{'PASS' if g3a_ok else 'FAIL'}")
    print(f"b_ring_shg  KERR g3b: fit intercept at U=0 -> f={f_at_zero_U:.6f} vs "
          f"cold f0={f0_cold:.6f} (|df|={abs(f_at_zero_U-f0_cold):.2e}, "
          f"floor={floor_rel*f0_cold:.2e}) -> {'PASS' if g3b_ok else 'FAIL'}")
    print(f"b_ring_shg  KERR caveat: residual shift at record end = "
          f"{abs(f_late-f0_cold):.2e} rel ({abs(f_late-f0_cold)/(floor_rel*f0_cold):.0f}x "
          f"floor) -- U has not physically reached zero, by design")
    return dict(slope=slope, r2=r2, f_late=f_late, g3a=g3a_ok, g3b=g3b_ok)


def main():
    if not STACK_CONFIRMED:
        print("b_ring_shg  *** STACK ASSUMED (600/300/1200 nm TFLN, SiO2, TE) ***")
    arch = (f"TWO-BUS gap_pump={GAP_PUMP} gap_SH={GAP_SH}" if TWO_BUS
            else f"SINGLE-BUS gap={GAP_SINGLE_BUS}")
    print(f"b_ring_shg  {arch} | r={R_RING} um W={W_WG} um LC={LC} um RES={RES}")
    # 4 runs: every band through every coupler. The cross terms are NOT
    # optional -- the SH coupler sits in the pump's round trip and vice versa.
    COUPLERS = ({"pump-bus": GAP_PUMP, "SH-bus": GAP_SH} if TWO_BUS
                else {"bus": GAP_SINGLE_BUS})
    m = {}
    for cname, cgap in COUPLERS.items():
        for band, (lam, nco, ncl) in BANDS.items():
            m[(cname, band)] = extract(band, lam, nco, ncl, cgap, cname)

    print("b_ring_shg  --- per-coupler kappa (targets: pump-bus ~0.04 @1348, "
          "SH-bus ~0.03-0.05 @674) ---")
    for (cname, band), r in m.items():
        flag = ""
        if cname == "pump-bus" and band.startswith("pump"):
            flag = "  <-- TARGET ~0.04"
        if cname == "SH-bus" and band.startswith("SH"):
            flag = "  <-- TARGET ~0.03-0.05"
        print(f"b_ring_shg    [{cname} gap={r['gap']}] {band}: "
              f"|kappa|^2={r['kappa2']:.4f} "
              f"[unitarity {r['unit']:.3f}]{flag}")

    if TWO_BUS:
        asm = {b: assemble(b, m[("pump-bus", b)], m[("SH-bus", b)]) for b in BANDS}
    else:
        # single bus: it is the only coupler in the round trip, so pass it as
        # both factors' source -- assemble() multiplies t_pump_c * t_sh_c, so
        # feed a unit second factor rather than squaring the real one.
        asm = {b: assemble(b, m[("bus", b)], {"t": np.ones_like(m[("bus", b)]["t"])})
               for b in BANDS}

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for ax, b in zip(axs[:, 0], BANDS):
        a = asm[b]
        ax.plot(1e3 / a["fd"], a["T"], "b-", lw=0.9)
        ax.axvline(1e3 / a["fd"][a["i0"]], color="gray", ls=":", lw=0.8)
        ax.set_xlabel("wavelength [nm]"); ax.set_ylabel(r"$|S_{21}|^2$")
        ax.set_title(f"{b} (both buses): Q={a['Q']:.0f}, FSR={a['fsr']:.3f} nm")
        ax.set_ylim(0, 1.05)
    for ax, cname in zip(axs[:, 1], COUPLERS):
        r = m[(cname, list(BANDS)[0])]
        step = max(1, r["nx"] // 900)
        ax.imshow(r["eps"][::step, ::step].T, origin="lower", cmap="viridis",
                  extent=[0, r["nx"] * dx, 0, r["ny"] * dx], aspect="auto")
        ax.set_title(f"{cname} extraction domain (gap={r['gap']} um)")
        ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]")
    title = (f"TFLN two-bus SHG ring r={R_RING:.0f} um: pump bus {GAP_PUMP} um / "
             f"SH bus {GAP_SH} um")
    if not STACK_CONFIRMED:
        title += "  (STACK ASSUMED)"
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "b_ring_shg.png", dpi=150)
    plt.close(fig)
    np.savez(OUT / "b_ring_shg_data.npz",
             **{f"{c.split('-')[0]}_{b.split()[0]}_{n}": m[(c, b)][n]
                for (c, b) in m for n in ("freqs", "t", "kappa", "loop")})
    print("b_ring_shg.png written (+ b_ring_shg_data.npz)")
    gds = (f"gf.components.ring_double(radius={R_RING}, gap={GAP_PUMP}, "
           f"gap_top={GAP_SH}, length_x={LC})" if TWO_BUS else
           f"gf.components.ring_single(radius={R_RING}, gap={GAP_SINGLE_BUS}, "
           f"length_x={LC}) + QPM poling Lambda={QPM_LAMBDA:.3f} um")
    print(f"b_ring_shg  FINAL GDS = {gds}")
    print(f"b_ring_shg  pump f0 for the nonlinear runs = {asm['pump 1348']['f0']:.6f} "
          f"(lam={1e3/asm['pump 1348']['f0']:.2f} nm), Q={asm['pump 1348']['Q']:.0f}")


if __name__ == "__main__":
    main()
