# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (b): TFLN ring for SHG, pump 1348 nm -> second harmonic 674 nm.

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
from meep_torch.core.sources import GaussianPulse
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
GAP_PUMP = float(os.environ.get("BRS_GAP_PUMP", "0.44"))   # target |k|^2 ~ 0.04 @1348
GAP_SH = float(os.environ.get("BRS_GAP_SH", "0.20"))       # target |k|^2 ~ 0.03-0.05 @674
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
    loop = (A / t) ** 2                          # bare full loop a^2

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
    ti = np.interp(fd, freqs, t_tot.real) + 1j * np.interp(fd, freqs, t_tot.imag)
    lm = np.interp(fd, freqs, np.abs(loop))
    lp = np.interp(fd, freqs, np.unwrap(np.angle(loop)))
    loop_d = lm * np.exp(1j * lp)
    T = np.abs((ti - loop_d) / (1 - ti * loop_d)) ** 2

    i0 = int(np.argmin(T))
    half = (T[i0] + np.median(T)) / 2
    lo = hi = i0
    while lo > 0 and T[lo] < half:
        lo -= 1
    while hi < len(fd) - 1 and T[hi] < half:
        hi += 1
    Q = fd[i0] / max(fd[hi] - fd[lo], 1e-12)
    dips = [i for i in range(2, len(fd) - 2)
            if T[i] < 0.6 * np.median(T) and T[i] <= T[i - 1] and T[i] < T[i + 1]]
    fsr = (abs(1e3 / fd[dips[0]] - 1e3 / fd[dips[-1]]) / (len(dips) - 1)
           if len(dips) >= 2 else float("nan"))
    ext = 10 * math.log10(max(np.median(T), 1e-12) / max(T[i0], 1e-12))
    print(f"b_ring_shg  ASSEMBLED {band}: dip {1e3/fd[i0]:.2f} nm |S21|^2={T[i0]:.3f} "
          f"Q={Q:.0f} FSR={fsr:.3f} nm extinction {ext:.1f} dB "
          f"[|t_pump_c|={abs(m_pump_coupler['t'][len(freqs)//2]):.4f} "
          f"|t_sh_c|={abs(m_sh_coupler['t'][len(freqs)//2]):.4f}]")
    return dict(band=band, fd=fd, T=T, Q=Q, fsr=fsr, i0=i0, ext=ext)


def main():
    if not STACK_CONFIRMED:
        print("b_ring_shg  *** STACK ASSUMED (600/300/1200 nm TFLN, SiO2, TE) ***")
    print(f"b_ring_shg  TWO-BUS r={R_RING} um W={W_WG} um LC={LC} um RES={RES} | "
          f"gap_pump={GAP_PUMP} gap_SH={GAP_SH}")
    # 4 runs: every band through every coupler. The cross terms are NOT
    # optional -- the SH coupler sits in the pump's round trip and vice versa.
    COUPLERS = {"pump-bus": GAP_PUMP, "SH-bus": GAP_SH}
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

    asm = {b: assemble(b, m[("pump-bus", b)], m[("SH-bus", b)]) for b in BANDS}

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
    print(f"b_ring_shg  FINAL GDS = gf.components.ring_double(radius={R_RING}, "
          f"gap={GAP_PUMP}, gap_top={GAP_SH}, length_x={LC})")


if __name__ == "__main__":
    main()
