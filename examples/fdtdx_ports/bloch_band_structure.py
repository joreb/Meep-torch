# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""1D binary Bragg crystal band structure (W9 big-fdtdx item 4).

A 1D binary photonic crystal -- alternating layers eps1 (thickness d1) and eps2
(thickness d2), period a = d1+d2 -- is simulated one period at a time with the
Bloch stepper (stepper_bloch2, TM Ez) at a sweep of Bloch wavevectors K in
[0, pi/a]. harminv pulls the eigenfrequencies at each K.

The exact 1D layered-medium dispersion (transfer matrix, mu=1 -- the same
formula for both polarizations since eta + 1/eta is symmetric) is

    cos(K a) = cos(d1p) cos(d2p) - 0.5 (n1/n2 + n2/n1) sin(d1p) sin(d2p)

with djp = 2*pi*f*n_j*d_j. Given K, the band frequencies are the f where the
RHS crosses cos(K a); each pass band (|RHS| <= 1) is one band, scanned and
root-found by interpolation.

Gate: the first two bands (lowest two frequencies) at 5 K points agree with the
exact transfer-matrix bands within 1%.

Two practical bits that make the sim resolve the low acoustic band 1
(f ~ 0.02-0.10, far below band 2 ~ 0.32):

  * a low-frequency source (GaussianPulse centered at 0.18, fwidth 0.4) so the
    spectrum actually illuminates band 1 -- a source centered in band 2
    starves band 1 and harminv never sees it;
  * harminv fmin below band 1 (0.005) and a long run (t_end 600) with the
    turn-on transient dropped (t0 80), so the slow band-1 oscillation has
    enough clean cycles to resolve.

Run: uv run python examples/fdtdx_ports/bloch_band_structure.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bloch2 import init_bloch2_state, step_bloch2_tm
from meep_torch.modes.harminv import harminv

OUT = pathlib.Path(__file__).parent

# --- 1D binary Bragg crystal ---
EPS1, EPS2 = 12.0, 1.0          # layer permittivities (Si-like / air)
D1, D2, A = 0.5, 0.5, 1.0        # layer thicknesses and period (a = d1+d2)
N1, N2 = math.sqrt(EPS1), math.sqrt(EPS2)
ETA = 0.5 * (N1 / N2 + N2 / N1)  # transfer-matrix impedance factor

# --- grid + source ---
RES = 48                         # cells per period
DX = 1.0 / RES
NX, NY = int(A * RES), 4
DT = 0.4 * DX
T_END = 600.0
T0 = 80.0                         # drop the source turn-on transient
FCEN, FWIDTH = 0.18, 0.4          # low-f source illuminates band 1 + band 2
FMIN, FMAX = 0.005, 0.5           # harminv window (band 1..band 2 only)
K_FRACS = (0.1, 0.2, 0.3, 0.4, 0.5)   # K in units of pi/a (5 points)


def _rhs(f):
    """Transfer-matrix RHS: cos(K a) as a function of frequency f."""
    d1p = 2 * math.pi * f * N1 * D1
    d2p = 2 * math.pi * f * N2 * D2
    return math.cos(d1p) * math.cos(d2p) - ETA * math.sin(d1p) * math.sin(d2p)


def exact_bands(K, fmax=0.6, df=2e-5):
    """Sorted band frequencies at Bloch wavevector K (reduced-zone scheme).

    Scan f; each contiguous |RHS|<=1 segment is one pass band. Within a band
    K(f) = arccos(RHS)/a is monotonic, so for target cos(K a) the frequency is
    found by interpolating (RHS -> f) on the sorted segment.
    """
    fs = np.arange(df, fmax, df)
    segs, cur = [], []
    for f in fs:
        if abs(_rhs(f)) <= 1.0:
            cur.append((f, _rhs(f)))
        else:
            if cur:
                segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    target = math.cos(K * A)
    out = []
    for seg in segs:
        fa = np.array([p[0] for p in seg])
        ra = np.array([p[1] for p in seg])
        if min(ra) <= target <= max(ra):
            o = np.argsort(ra)
            out.append(float(np.interp(target, ra[o], fa[o])))
    return sorted(out)


def sim_bands(K):
    """harminv frequencies at Bloch wavevector K via the one-period Bloch TM stepper."""
    eps = np.full((NX, NY), EPS2, dtype=np.float64)
    eps[: int(round(D1 / DX)), :] = EPS1            # eps1 then eps2 along x
    eps = torch.from_numpy(eps)
    g = type("G", (), {})()
    g.nx, g.ny = NX, NY
    g.zeros = lambda *s: torch.zeros(*s, dtype=torch.float64)
    st = init_bloch2_state(g)
    mask = torch.zeros(NX, NY, dtype=torch.float64)
    mask[NX // 3 + 1, NY // 5 + 2] = 1.0
    pulse = GaussianPulse(FCEN, FWIDTH)
    n_steps = int(round(T_END / DT))
    probes = [(NX // 3 + 1, NY // 5 + 2), (2 * NX // 3, NY // 2 + 1),
              (NX // 2 - 2, 0), (NX - 3, NY // 3), (NX // 4, NY // 3)]
    series = np.empty((len(probes), n_steps), dtype=complex)
    for t in range(n_steps):
        a = pulse.amplitude((t + 0.5) * DT)
        st = step_bloch2_tm(st, eps, DT, DX, K, 0.0, src_c=a, src_s=0.0,
                            src_mask=mask)
        for p, (pi, pj) in enumerate(probes):
            series[p, t] = complex(st.Ez_c[pi, pj].item(), st.Ez_s[pi, pj].item())
    t0 = int(round(T0 / DT))
    freqs = []
    for p in range(len(probes)):
        freqs += [m.freq for m in harminv(series[p, t0:], DT, FMIN, FMAX)
                  if m.Q > 50 and m.amp > 1e-9]
    return sorted(freqs)


def main():
    kpts = [frac * math.pi / A for frac in K_FRACS]
    sim1, sim2, ex1, ex2 = [], [], [], []
    max_rel = 0.0
    for frac, K in zip(K_FRACS, kpts):
        ex = exact_bands(K)
        sm = sim_bands(K)
        e_b1, e_b2 = ex[0], ex[1]
        s_b1 = min(sm, key=lambda s: abs(s - e_b1)) if sm else 9.99
        s_b2 = min(sm, key=lambda s: abs(s - e_b2)) if sm else 9.99
        r1 = abs(s_b1 - e_b1) / e_b1
        r2 = abs(s_b2 - e_b2) / e_b2
        max_rel = max(max_rel, r1, r2)
        print(f"K={frac:.2f}*pi/a  band1 exact={e_b1:.4f} sim={s_b1:.4f} "
              f"rel={r1:.4f} | band2 exact={e_b2:.4f} sim={s_b2:.4f} rel={r2:.4f}")
        sim1.append(s_b1); sim2.append(s_b2)
        ex1.append(e_b1); ex2.append(e_b2)
        assert r1 <= 0.01, f"K={frac}: band1 rel {r1:.4f} > 0.01"
        assert r2 <= 0.01, f"K={frac}: band2 rel {r2:.4f} > 0.01"
    print(f"max rel over 5 K points, 2 bands = {max_rel:.4f} (want <= 0.01)")
    assert max_rel <= 0.01

    # PNG: band diagram, exact lines (fine K grid) + sim markers (5 K points).
    kf = np.linspace(0.01, 0.5, 80)   # avoid K=0 (band-1 f=0 is degenerate)
    ex1_line = [exact_bands(k * math.pi / A)[0] for k in kf]
    ex2_line = [exact_bands(k * math.pi / A)[1] for k in kf]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(kf, ex1_line, "C0-", label="exact band 1")
    ax.plot(kf, ex2_line, "C1-", label="exact band 2")
    ax.plot(K_FRACS, sim1, "C0o", ms=6, mfc="none", label="sim band 1")
    ax.plot(K_FRACS, sim2, "C1s", ms=6, mfc="none", label="sim band 2")
    ax.set_xlabel("K (units of pi/a)")
    ax.set_ylabel("frequency")
    ax.set_title(f"1D Bragg bands (eps1={EPS1:g}, eps2={EPS2:g}, d1=d2={D1}); "
                 f"max rel {max_rel:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "bloch_band_structure.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT / 'bloch_band_structure.png'}")


if __name__ == "__main__":
    main()