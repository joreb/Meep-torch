# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Dispersive pulse propagation vs non-dispersive reference
#
# The same Gaussian pulse runs twice: through a constant-eps reference medium
# (no broadening, phase delay only) and through a Lorentz medium with normal
# dispersion (group delay n_g > n and GVD broadening). Three numeric gates
# verify constant reference width, >= 20% dispersive broadening, and a
# group delay matching the analytic n_g via spectral phase.

# %%
"""Dispersive Gaussian pulse vs non-dispersive reference (W9 big-fdtdx item 2).

A Gaussian-envelope pulse (cos-modulated Gaussian, zero DC) propagates +x
through a uniform medium. Two runs share the SAME source:

  * reference -- a non-dispersive medium with eps = Re(eps(fc)) (a real
    constant), so the group velocity equals the phase velocity: the pulse
    keeps the width it was launched with and accrues only phase-velocity delay;
  * dispersive -- a Lorentz medium eps(f) = 1 + sigma*fn^2/(fn^2 - f^2
    - i*gamma*f) (Si-like, detuned above fc for normal dispersion), where
    n_g = n + f dn/df > n so the pulse travels slower (group delay) and the
    group-velocity dispersion beta2 broadens the envelope.

Three numeric gates (asserts; the PNG is for eyeballing):

  (1) reference envelope width is constant: the reference pulse width at the
      far probe matches the launched Gaussian width w = 1/FWIDTH within 5%
      (no GVD broadening -- sanity). The *near* probe is not used for width:
      a soft source leaves a turn-on transient in the near field that inflates
      the envelope RMS there, so width is only meaningful once the pulse has
      cleared the source (the far probe).
  (2) dispersive envelope broadens: the dispersive far width is >= 1.20x the
      reference far width (>=20% GVD broadening).
  (3) group delay matches the analytic n_g: the cross-spectrum
      FFT(far)*conj(FFT(near)) has phase -k(f)*SEG (propagation from near to
      far). The dispersive-vs-reference phase-difference slope is
      (n_g - n_ref)*SEG (c=1); the measured slope is compared to the analytic
      value within 3%. This is done in the spectral phase, NOT the envelope
      centroid: absorption (Im n) shapes the magnitude (which biases a
      centroid) but leaves the phase -- the propagation k -- clean.

The dispersive pole is masked to the interior x (NPML..NX-NPML): the TM PML
damps Ez but not the ADE polarization recurrence, so a pole active in the x-PML
(where Ez is PML-damped) feeds an un-damped Pz back -> an Ez-Pz growth mode
(the per-axis dispersion test documents the same mechanism). The pulse is
launched and measured inside the interior, so the vacuum-in-PML step is never
illuminated.

Run: uv run python examples/11_3d/fdtdx_ports/dispersive_gaussian_pulse.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, SoftSource
from meep_torch.core.stepper import init_tm_state

OUT = pathlib.Path(__file__).parent / "output"

# --- pulse + medium ---
FC = 0.3                       # center frequency (normalized, c=1)
FWIDTH = 0.12                   # Gaussian envelope sigma w = 1/FWIDTH
W_ENV = 1.0 / FWIDTH            # launched envelope RMS width
FN, GAM, SIG = 0.44, 0.008, 0.6   # Lorentz pole (detuned above fc, low loss)
EPS_INF = 1.0

# --- grid (1D TM, ny=3 + y_pmc) ---
NX, NY, DX, NPML = 2200, 3, 0.025, 60
GRID = Grid(nx=NX, ny=NY, dx=DX)
SRC_IX = 100
NEAR_IX = 500                   # near probe: source-to-here phase reference
FAR_IX = 2080                    # far probe: width + far phase
SEG = (FAR_IX - NEAR_IX) * DX    # near->far segment length (group-delay path)
T_END = 320.0
PULSE = GaussianPulse(FC, FWIDTH)


# %% [markdown]
# ## Analytic dispersion helpers and the two runs
#
# eps(f), n(f), and n_g(f) come from the Lorentz pole; the near/far point
# probes provide the envelope widths and the cross-spectrum phase used by
# the gates.

# %%
def _eps_complex(f):
    return EPS_INF + SIG * FN ** 2 / (FN ** 2 - f ** 2 - 1j * GAM * f)


def _n_and_ng(f):
    """n(f)=sqrt(eps(f)); n_g = n + f dn/df (= n + omega dn/domega)."""
    eps = _eps_complex(f)
    n = np.sqrt(eps)
    den = FN ** 2 - f ** 2 - 1j * GAM * f
    deps = SIG * FN ** 2 * (2 * f + 1j * GAM) / den ** 2
    return n, n + f * (deps / (2 * n))


def _run(dispersive: bool):
    """Propagate the pulse; return (near_series, far_series) as numpy."""
    eps = torch.ones(NX, NY, dtype=GRID.dtype)
    poles, pole_masks, n_poles = [], None, 0
    if dispersive:
        mask = torch.zeros(NX, NY, dtype=GRID.dtype)
        mask[NPML:NX - NPML, :] = 1.0          # interior x only (PML is vacuum)
        poles, pole_masks, n_poles = [(FN, GAM, SIG)], [mask], 1
    else:
        n_ref = math.sqrt(_eps_complex(FC).real)   # eps_ref = Re(eps(fc))
        eps = torch.full((NX, NY), n_ref ** 2, dtype=GRID.dtype)
    coeffs = build_tm_coeffs(GRID, eps_inf=eps, poles=poles,
                             pole_masks=pole_masks, npml=NPML, y_pmc=True)
    smask = GRID.zeros(NX, NY)
    smask[SRC_IX, 1] = 1.0
    src = SoftSource(GRID, smask, PULSE)
    n_steps = int(round(T_END / GRID.dt))
    _, (near, far) = run_tm(init_tm_state(GRID, n_poles=n_poles, y_pmc=True),
                            coeffs, n_steps, source=src,
                            monitors=[PointProbe(NEAR_IX, 1),
                                      PointProbe(FAR_IX, 1)])
    return near.detach().numpy(), far.detach().numpy()


def _env_rms_width(series, dt):
    """RMS width of the |Hilbert| envelope, weighted by the envelope."""
    env = np.abs(__import__("scipy.signal", fromlist=["hilbert"]).hilbert(series))
    t = np.arange(len(series)) * dt
    tot = env.sum()
    tc = float((t * env).sum() / tot)
    return math.sqrt(float(((t - tc) ** 2 * env).sum() / tot))


def _group_delay(near_r, far_r, near_d, far_d, dt, ng_re, n_ref):
    """Spectral-phase group delay over the near->far segment.

    cross-spec(far, near) phase = -k(f)*SEG. The dispersive-vs-reference
    phase-difference slope dphi/d(f) = -(n_g - n_ref)*SEG (c=1); the magnitude
    of the slope is the group-delay excess, compared to the analytic value.
    """
    n = min(len(far_r), len(far_d))
    freq = np.fft.fftfreq(n, dt)
    omega = 2 * np.pi * freq

    def cross_spec(s_far, s_near):
        return np.fft.fft(s_far[:n]) * np.conj(np.fft.fft(s_near[:n]))

    # phase of (dispersive cross-spec)/(reference cross-spec) = (k_disp-k_ref)*SEG
    ratio = cross_spec(far_d, near_d) / cross_spec(far_r, near_r)
    phi = np.unwrap(np.angle(ratio))
    band = (freq > FC * 0.9) & (freq < FC * 1.1)
    slope = float(np.polyfit(omega[band], phi[band], 1)[0])
    gd_sim = abs(slope)
    gd_an = abs(ng_re - n_ref) * SEG
    return gd_sim, gd_an


def main():
    dt = GRID.dt
    near_r, far_r = _run(dispersive=False)
    near_d, far_d = _run(dispersive=True)

    sig_far_r = _env_rms_width(far_r, dt)
    sig_far_d = _env_rms_width(far_d, dt)
    n_ref = math.sqrt(_eps_complex(FC).real)
    _, ng = _n_and_ng(FC)
    ng_re = float(ng.real)

    ref_const = abs(sig_far_r / W_ENV - 1.0)          # gate (1)
    disp_ratio = sig_far_d / max(sig_far_r, 1e-30)    # gate (2)
    gd_sim, gd_an = _group_delay(near_r, far_r, near_d, far_d, dt, ng_re, n_ref)
    gd_rel = abs(gd_sim - gd_an) / abs(gd_an)         # gate (3)

    print(f"dispersive_gaussian_pulse  fc={FC} fwidth={FWIDTH} "
          f"pole=(fn={FN}, gamma={GAM}, sigma={SIG})")
    print(f"  n_ref={n_ref:.4f}  n_g={ng_re:.4f}  (n_g - n_ref)={ng_re - n_ref:.4f}"
          f"  Im(n)={float(_n_and_ng(FC)[0].imag):.4f}")
    print(f"  launched width w={W_ENV:.4f}  ref far width={sig_far_r:.4f}  "
          f"|ref/w - 1|={ref_const:.4f} (want <= 0.05)")
    print(f"  dispersive far width={sig_far_d:.4f}  "
          f"disp/ref={disp_ratio:.4f} (want >= 1.20)")
    print(f"  group delay sim={gd_sim:.4f} analytic={gd_an:.4f} "
          f"rel={gd_rel:.4f} (want <= 0.03)")

    assert ref_const <= 0.05, f"ref width drifted: |{sig_far_r}/{W_ENV}-1|={ref_const:.4f}"
    assert disp_ratio >= 1.20, f"dispersive not broadened: {disp_ratio:.4f} < 1.20"
    assert gd_rel <= 0.03, f"group delay rel {gd_rel:.4f} > 0.03"

    # PNG: |envelope| at the far probe, ref vs dispersive (eyeball).
    from scipy.signal import hilbert
    env_r = np.abs(hilbert(far_r))
    env_d = np.abs(hilbert(far_d))
    t = np.arange(len(far_r)) * dt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, env_r, label=f"reference eps=Re(eps(fc)) (width {sig_far_r:.2f})")
    ax.plot(t, env_d, label=f"Lorentz dispersive (width {sig_far_d:.2f})")
    ax.set_xlabel("time")
    ax.set_ylabel("|envelope| at far probe")
    ax.set_title(f"Dispersive pulse broadening (n_g={ng_re:.2f}, seg={SEG:.1f}); "
                 f"group delay rel {gd_rel:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "dispersive_gaussian_pulse.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT / 'dispersive_gaussian_pulse.png'}")


if __name__ == "__main__":
    main()