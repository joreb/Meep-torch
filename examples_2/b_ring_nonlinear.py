# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (b): LiNbO3 (n=2.2) racetrack ring with chi2 / chi3 -- redo.

Driven at the fdtdx quality bar: RES 30, bus excited by a ports
ModeSourceTM (TFSF eigenmode of the bus, not a soft line source), three numeric
gates (peer-rebased 2026-08-14 after the bus-driven loaded-Q + multi-mode
findings):

  (1) Bus-coupled loaded Q: widen the bus-ring gap to near critical coupling
      (loaded Q ~ 213 = Q_unloaded/2). harminv loaded Q >= 150 AND a resonance
      in 1500-1600 nm. Radius r=5 -> r=3: larger FSR isolates ONE mode in the
      band and kills the 1534/1538 doublet that blocked gates (2)/(3).
  (2) SHG power scaling: pump CW AT the harminv resonance, two amplitudes
      A=0.1 and 2A=0.2 (undepleted -- depletion at 0.6 is physics, not a bug);
      eta = P(2f0)/P(f0) ~ pump power, so eta(2A)/eta(A) = 4.0 within 15%.
  (3) Kerr red-shift linear in power via FREE RINGDOWN (no driven-state
      harminv): charge the ring with CW at f0 for ~3 decay times, switch the
      source OFF, harminv the free ringdown in an early window (~2.5 decay
      times) -- only the charged mode rings (no source leakage, doublet
      ambiguity gone). Kerr shift = f_ringdown(P) - f_ringdown(P->0 reference
      from a low-amp run); gate shift(4x)/shift(1x) = 4.0 within 20%.

chi2 / chi3 physics unchanged (core_mask = LiNbO3 regions); png layout and name
b_ring_Lithium_Niobate_nonlinear.png kept.

Run: uv run python examples_2/b_ring_nonlinear.py
"""
import dataclasses
import pathlib
import sys

import gdsfactory as gf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gds_meep_torch import component_to_eps

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import ContinuousSource, GaussianPulse
from meep_torch.core.stepper import init_tm_state
from meep_torch.modes.harminv import harminv
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port, validate_ports

OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()

RES = 30
N_LN = 2.2
NPML = RES
FCEN = 1.0 / 1.55                      # pump center (lambda=1.55 um)
DF_BROAD = 0.08                       # broadband ringdown covers the resonance
RADIUS = 3.0                          # r=5 -> r=3: larger FSR isolates one mode in band
Q_UNLOADED = 426                       # RES-20 soft-line intrinsic Q (r=5 ref; display only here)
Q_LOADED_LO, Q_LOADED_HI = 150, 280   # peer-rebased gate (1) band; target ~213 = Q_unloaded/2
LAM_LO, LAM_HI = 1.500, 1.600          # resonance must land in 1500-1600 nm
GAP_SWEEP = (0.20, 0.25, 0.30, 0.35)  # r=3: re-find critical (start 0.25 per peer; bracket)

torch.set_num_threads(16)             # 32 threads oversubscribe -> 1.5 steps/s; 16 -> ~900

# gpdk 'strip' enforces radius_min=3.5; r=3.0 < 3.5 so override the cross-section
# (fab DRC limit is irrelevant to an FDTD sim) to honor the peer's r=3 geometry.
XS_LN = gf.get_cross_section("strip").copy(radius_min=3.0)


@dataclasses.dataclass(frozen=True)
class Geom:
    grid: object
    eps: object
    core_mask: object
    coeffs: object
    p_in: object
    j_ring: tuple
    gap: float


def build_geom(gap):
    c = gf.components.ring_single(radius=RADIUS, gap=gap, cross_section=XS_LN)
    grid, eps, ports = component_to_eps(c, res=RES, pad=1.5,
                                        n_core=N_LN, n_clad=1.44)
    grid = dataclasses.replace(grid, dtype=torch.float32)
    eps = eps.to(torch.float32)
    core_mask = (eps > 3.0).to(eps.dtype)        # LiNbO3 regions (eps=4.84)
    o1x, o1y, _ = ports["o1"]
    span = min(int(0.9 * RES), o1y - NPML - 2, grid.ny - NPML - 2 - o1y)
    p_in = Port(axis="x", index=max(o1x, NPML + 4), lo=o1y - span,
                hi=o1y + span, direction=+1)
    validate_ports([p_in], grid, NPML)
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)
    col = grid.nx // 2
    ring_ys = torch.where(core_mask[col])[0].tolist()
    return Geom(grid, eps, core_mask, coeffs, p_in, (col, max(ring_ys)), gap)


def _mode_source(g, profile, amp):
    """Bus eigenmode source scaled to amplitude `amp` (power ~ amp^2)."""
    src = ModeSourceTM(g.grid, g.coeffs, g.p_in, profile, FCEN)
    src.mask = src.mask * amp
    for k in src.h_masks:                          # scale the magnetic sheet too
        src.h_masks[k] = src.h_masks[k] * amp
    return src


def _ring_series(g, coeffs, profile, amp, t_end, t_skip):
    """Launch a (scaled) bus mode source; return the ring-probe time series."""
    src = _mode_source(g, profile, amp)        # bus eigenmode is linear (g.coeffs);
    n = int(round(t_end / g.grid.dt))
    # checkpoint_every=400 bounds the PointProbe stack-reduce partials per chunk
    # (a bare PointProbe stores a view into the live Ez each step -> 10s of GB on
    # long runs); no_grad: forward-only, no autograd graph.
    with torch.no_grad():
        _, (s,) = run_tm(init_tm_state(g.grid), coeffs, n, source=src,
                         monitors=[PointProbe(*g.j_ring)], checkpoint_every=400)
    # sim runs float32 (speed); upcast the small probe series to float64 so the
    # Kerr shift (Delta ~ 1e-4) and FWHM are resolved by harminv / FFT.
    return s.detach().to(torch.float64).numpy()[int(round(t_skip / g.grid.dt)):]


def _fwhm_q(f0, fr, spec, half_win=0.02):
    """Q = f0 / FWHM from the |FFT|^2 spectrum, with sub-bin half-max interp."""
    win = (fr > f0 - half_win) & (fr < f0 + half_win)
    frw, spw = fr[win], spec[win]
    if len(frw) < 3:
        return float("nan")
    peak = float(spw.max())
    half = peak / 2.0
    idx = np.where(spw >= half)[0]
    if len(idx) < 2:
        return float("nan")
    f_lo, f_hi = frw[idx[0]], frw[idx[-1]]
    if idx[0] > 0 and spw[idx[0] - 1] < half < spw[idx[0]]:
        a, b = frw[idx[0] - 1], frw[idx[0]]
        sa, sb = spw[idx[0] - 1], spw[idx[0]]
        f_lo = a + (half - sa) * (b - a) / (sb - sa)
    if idx[-1] + 1 < len(frw) and spw[idx[-1]] > half > spw[idx[-1] + 1]:
        a, b = frw[idx[-1]], frw[idx[-1] + 1]
        sa, sb = spw[idx[-1]], spw[idx[-1] + 1]
        f_hi = a + (half - sa) * (b - a) / (sb - sa)
    fwhm = f_hi - f_lo
    return f0 / fwhm if fwhm > 0 else float("nan")


def make_png(out, fr_shg, spec_shg, shift_1x, shift_4x, kerr_ratio,
             shg_ratio, q_harm, q_ext, gap, radius, f0, gq, gshg, gkerr):
    """2-panel figure: SHG spectrum (top, short title + gate box) and Kerr
    free-ringdown red-shift bars with the x4 expectation (bottom)."""
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.2, 6.6), height_ratios=[3, 2])
    ax0.semilogy(fr_shg, np.maximum(spec_shg, 1e-12), lw=0.8)
    ax0.axvline(f0, color="g", ls=":", label="pump $f_0$")
    ax0.axvline(2 * f0, color="r", ls=":", label="SHG $2 f_0$")
    ax0.set_xlim(0, 3 * f0)
    ax0.set_xlabel("frequency (c/um)")
    ax0.set_ylabel("ring spectrum (norm.)")
    ax0.set_title(f"LiNbO3 ring chi2 pump at resonance (r={radius}, Q={q_harm:.0f})")
    ax0.legend(fontsize=8, loc="upper right")
    gate_txt = (f"gap={gap:.2f}  Q_ext={q_ext:.0f}\n"
                f"SHG 4x={shg_ratio:.2f}  Kerr 4x={kerr_ratio:.2f}\n"
                f"gates: Q={'Y' if gq else 'N'}  SHG={'Y' if gshg else 'N'}  "
                f"Kerr={'Y' if gkerr else 'N'}")
    ax0.text(0.015, 0.97, gate_txt, transform=ax0.transAxes, fontsize=8,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.85))
    xs = ["1x", "4x"]
    vals = [shift_1x * 1e4, shift_4x * 1e4]
    ax1.bar(xs, vals, width=0.5, color="#3a7", label="measured $\\Delta f$")
    expect = [vals[0], vals[0] * 4.0]            # linear-in-power: shift(4x) = 4*shift(1x)
    ax1.plot(xs, expect, "k--o", ms=4, lw=1.0, label="x4 expectation")
    for x, v in zip(xs, vals):
        ax1.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Kerr red-shift $f_{ref}-f$  ($10^{-4}$)")
    ax1.set_title(f"Kerr free-ringdown shift (ratio={kerr_ratio:.2f}, want 4.0)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_ylim(0, max(vals + expect) * 1.25)
    fig.tight_layout()
    fig.savefig(out / "b_ring_Lithium_Niobate_nonlinear.png", dpi=150)
    plt.close(fig)
    print("b_ring_Lithium_Niobate_nonlinear.png written")


# --replot: regenerate the png from the saved npz without an 80min rerun.
if "--replot" in sys.argv:
    _d = np.load(OUT / "b_ring_nonlinear_data.npz")
    make_png(OUT, _d["FR_SHG"], _d["SPEC_SHG"], float(_d["SHIFT_1X"]),
             float(_d["SHIFT_4X"]), float(_d["KERR_RATIO"]), float(_d["SHG_RATIO"]),
             float(_d["Q_HARM"]), float(_d["Q_EXT"]), float(_d["GAP"]),
             float(_d["RADIUS"]), float(_d["F0"]), bool(_d["gate_q"]),
             bool(_d["gate_shg"]), bool(_d["gate_kerr"]))
    sys.exit(0)


# ---------------------------------------------------------------- gate (1): Q
def measure_resonance():
    """Sweep bus-ring gaps; pick the one whose harminv loaded Q is closest to
    critical (~213) and inside [150, 280]. Returns (Geom, f0, Q_harm, Q_fwhm,
    Q_ext). Q_ext = (1/Q_loaded - 1/Q_unloaded)^-1 (display; Q_unloaded is the
    r=5 ref, so Q_ext is only approximate for r=3)."""
    fmin, fmax = 1.0 / LAM_HI, 1.0 / LAM_LO     # 1500-1600 nm -> [0.625, 0.667]
    tried = []
    for gap in GAP_SWEEP:
        g = build_geom(gap)
        sig = _ring_series(g, g.coeffs, GaussianPulse(FCEN, DF_BROAD), 1.0,
                           t_end=800.0, t_skip=120.0)
        modes = [m for m in harminv(sig, g.grid.dt, fmin, fmax) if m.Q > 50]
        if not modes:                            # fall back to a wider harminv window
            modes = [m for m in harminv(sig, g.grid.dt, FCEN - 0.05, FCEN + 0.05)
                     if m.Q > 50 and LAM_LO <= 1e3 / m.freq <= LAM_HI]
        if not modes:
            print(f"b_ring  gap={gap:.2f} no resonance in {LAM_LO}-{LAM_HI} nm")
            continue
        m0 = max(modes, key=lambda m: m.Q)       # highest-Q resonance in 1500-1600 nm
        f0 = m0.freq
        fr = np.fft.rfftfreq(len(sig), g.grid.dt)
        spec = np.abs(np.fft.rfft(sig)) ** 2
        q_harm = m0.Q
        q_fwhm = _fwhm_q(f0, fr, spec)
        q_ext = 1.0 / (1.0 / q_harm - 1.0 / Q_UNLOADED) if q_harm < Q_UNLOADED else float("inf")
        print(f"b_ring  gap={gap:.2f} f0={f0:.5f} lam={1e3/f0:.1f}nm "
              f"Q_harm={q_harm:.0f} Q_fwhm={q_fwhm:.0f} Q_ext={q_ext:.0f}")
        tried.append((g, f0, q_harm, q_fwhm, q_ext))
    if not tried:
        raise RuntimeError("no resonance found at any gap")
    target = Q_UNLOADED / 2.0                    # critical coupling: loaded Q = Q_in/2
    in_band = [t for t in tried if Q_LOADED_LO <= t[2] <= Q_LOADED_HI]
    pool = in_band if in_band else tried         # if none in band, take closest overall
    return min(pool, key=lambda t: abs(t[2] - target))


G, F0, Q_HARM, Q_FWHM, Q_EXT = measure_resonance()
GAP = G.gap
print(f"b_ring  CHOSEN r={RADIUS} gap={GAP:.2f} f0={F0:.5f} (lam={1e3/F0:.1f} nm) "
      f"Q_harminv={Q_HARM:.0f} Q_fwhm={Q_FWHM:.0f} Q_ext={Q_EXT:.0f}")


# ---------------------------------------------------------------- gate (2): SHG
def shg_eta(g, chi2_val, amp, t_end=400.0, t_skip=100.0):
    """P(2f0)/P(f0) in the ring under a CW chi2 pump at amplitude `amp`,
    driven at the measured resonance f0 (resonant buildup -> clean SHG)."""
    coeffs = build_tm_coeffs(g.grid, eps_inf=g.eps, npml=NPML,
                             chi2=chi2_val * g.core_mask)
    sig = _ring_series(g, coeffs, ContinuousSource(F0, width=15.0), amp,
                       t_end=t_end, t_skip=t_skip)
    fr = np.fft.rfftfreq(len(sig), g.grid.dt)
    win = np.hanning(len(sig))            # suppress f0 CW sidelobe leakage into 2f0
    spec = np.abs(np.fft.rfft(sig * win)) ** 2

    def band_peak(f, w=0.04):
        m = (fr > f - w) & (fr < f + w)
        return float(spec[m].max()) if m.any() else 0.0
    return band_peak(F0), band_peak(2 * F0), fr, spec


CHI2 = 0.5
AMP_A = 0.1                            # undepleted (peer): 2A=0.2; depletion at 0.6
                                       # is physics (P(f0) drains at over-coupled gap)
p_f0_A, p_2f0_A, _, _ = shg_eta(G, CHI2, AMP_A)
p_f0_2A, p_2f0_2A, FR_SHG, SPEC_SHG = shg_eta(G, CHI2, 2 * AMP_A)
ETA_A = p_2f0_A / p_f0_A
ETA_2A = p_2f0_2A / p_f0_2A
SHG_RATIO = ETA_2A / ETA_A
print(f"b_ring  SHG: eta(A)={ETA_A:.3e} eta(2A)={ETA_2A:.3e} "
      f"ratio={SHG_RATIO:.3f} (want 4.0 +/-15%)")


# ---------------------------------------------------------------- gate (3): Kerr (free ringdown)
CHI3 = 0.05
KERR_REF = 0.02                        # P->0 reference: shift ~ (0.02/KERR_A)^2 ~ 1% of 1x
KERR_A = 0.2                           # 1x pump; 2*KERR_A = 4x power. Linear-Kerr regime.
DECAY = Q_HARM / (np.pi * F0)          # field 1/e time (Lorentzian; energy 1/e = 2*DECAY)
T_CHARGE = 3.0 * DECAY                 # charge ~3 decay times -> ring ~95% steady state
T_RING = 2.5 * DECAY                   # early free-ringdown window (high SNR, resolvable)


def kerr_ringdown_freq(g, amp, t_charge, t_ring):
    """Free-ringdown resonance under chi3 at charge amplitude `amp`.

    Phase 1: charge the ring with CW at f0 (source ON) for t_charge (~3 decay
    times) -> steady-state intracavity field. Phase 2: source OFF (run_tm
    source=None -> no injection), record the FREE ringdown for t_ring (~2.5
    decay times) and harminv it. Only the charged mode rings (no source
    leakage, doublet ambiguity gone); the Kerr shift chirps toward f0 as the
    field decays, so harminv returns an amplitude-weighted early-time freq.
    Identical window for ref/1x/4x -> ratio stays 4 in the linear-Kerr regime.
    """
    coeffs = build_tm_coeffs(g.grid, eps_inf=g.eps, npml=NPML,
                             chi3=CHI3 * g.core_mask)
    n_c = int(round(t_charge / g.grid.dt))
    n_r = int(round(t_ring / g.grid.dt))
    with torch.no_grad():
        state, _ = run_tm(init_tm_state(g.grid), coeffs, n_c,
                         source=_mode_source(g, ContinuousSource(F0, 15.0), amp),
                         monitors=[], checkpoint_every=400)
        _, (s,) = run_tm(state, coeffs, n_r, source=None,
                        monitors=[PointProbe(*g.j_ring)], checkpoint_every=400)
    sig = s.detach().to(torch.float64).numpy()
    modes = [m for m in harminv(sig, g.grid.dt, F0 - 0.01, F0 + 0.01) if m.Q > 20]
    return max(modes, key=lambda m: m.amp).freq if modes else float("nan")


print(f"b_ring  Kerr ringdown: decay={DECAY:.1f} t_charge={T_CHARGE:.0f} "
      f"t_ring={T_RING:.0f}")
f_ref = kerr_ringdown_freq(G, KERR_REF, T_CHARGE, T_RING)
f_k1 = kerr_ringdown_freq(G, KERR_A, T_CHARGE, T_RING)
f_k4 = kerr_ringdown_freq(G, 2 * KERR_A, T_CHARGE, T_RING)
SHIFT_1X = f_ref - f_k1                  # Kerr red-shift at 1x power (>0)
SHIFT_4X = f_ref - f_k4                  # Kerr red-shift at 4x power
KERR_RATIO = SHIFT_4X / SHIFT_1X if SHIFT_1X > 1e-7 else float("nan")
print(f"b_ring  Kerr: f_ref={f_ref:.6f} f(1x)={f_k1:.6f} f(4x)={f_k4:.6f} "
      f"D(1x)={SHIFT_1X*1e4:.3f}e-4 D(4x)={SHIFT_4X*1e4:.3f}e-4 "
      f"ratio={KERR_RATIO:.3f} (want 4.0 +/-20%)")


# ---------------------------------------------------------------- gates
lam0_nm = 1e3 / F0
in_band = 1.0 / LAM_HI <= F0 <= 1.0 / LAM_LO      # freq comparison (LAM_* are um)
gate_q = (Q_HARM >= Q_LOADED_LO) and in_band
gate_shg = 0.85 * 4.0 <= SHG_RATIO <= 1.15 * 4.0
gate_kerr = 0.80 * 4.0 <= KERR_RATIO <= 1.20 * 4.0
print(f"b_ring  GATES: r={RADIUS} gap={GAP:.2f} Q_harm={Q_HARM:.0f}>={Q_LOADED_LO} "
      f"& lam={lam0_nm:.1f}nm in [{LAM_LO*1e3:.0f},{LAM_HI*1e3:.0f}]={gate_q} | "
      f"SHG_ratio={SHG_RATIO:.3f} in [3.40,4.60]={gate_shg} | "
      f"Kerr_ratio={KERR_RATIO:.3f} in [3.20,4.80]={gate_kerr}")
assert gate_q, f"Q_harm {Q_HARM:.0f} < {Q_LOADED_LO} or lam {lam0_nm:.1f}nm outside band"
assert gate_shg, f"SHG ratio {SHG_RATIO:.3f} outside [3.40,4.60]"
assert gate_kerr, f"Kerr ratio {KERR_RATIO:.3f} outside [3.20,4.80]"


# ---------------------------------------------------------------- persist + png
# npz of the plot inputs so future presentation tweaks replot via --replot
# without an 80min rerun (spectra + ringdown shifts + gate scalars/flags).
np.savez(OUT / "b_ring_nonlinear_data.npz",
         FR_SHG=FR_SHG, SPEC_SHG=SPEC_SHG, SHIFT_1X=SHIFT_1X, SHIFT_4X=SHIFT_4X,
         KERR_RATIO=KERR_RATIO, SHG_RATIO=SHG_RATIO, Q_HARM=Q_HARM, Q_EXT=Q_EXT,
         Q_FWHM=Q_FWHM, GAP=GAP, RADIUS=RADIUS, F0=F0,
         gate_q=gate_q, gate_shg=gate_shg, gate_kerr=gate_kerr,
         ETA_A=ETA_A, ETA_2A=ETA_2A, f_ref=f_ref, f_k1=f_k1, f_k4=f_k4)
print("b_ring_nonlinear_data.npz written")
make_png(OUT, FR_SHG, SPEC_SHG, SHIFT_1X, SHIFT_4X, KERR_RATIO, SHG_RATIO,
         Q_HARM, Q_EXT, GAP, RADIUS, F0, gate_q, gate_shg, gate_kerr)