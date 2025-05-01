# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/disc_extraction_efficiency.py.

Extraction efficiency of a dipole in a high-index disc (n=2.4, r=1.2,
h=0.29): fraction of the emitted power escaping through the TOP plane,
normalized by the CLOSED-surface flux (top + bottom + side) -- fully
normalized, no cross-convention constants (the earlier point_dipole_cyl
port's LDOS-vs-flux ratio was relative only).

SUBSTITUTIONS: on-axis vertical Ez dipole (m=0; the meep example sweeps
horizontal-dipole positions with per-m sums -- the machinery for that
landed with the off-axis port); res 25.

Run: uv run python examples/disc_extraction_efficiency.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.near2far_cyl import run_bor_dft
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bor import build_bor_coeffs, init_bor_state

OUT = pathlib.Path(__file__).parent
FCEN, RES = 1.0, 25
N_DISC, R_DISC, H_DISC = 2.4, 1.2, 0.29


def extraction(zfrac):
    dr = 1.0 / RES
    nr, nz = int(3.0 * RES), int(6.0 * RES)
    eps = torch.ones(nr, nz, dtype=torch.float64)
    jz0 = int((3.0 - H_DISC / 2) * RES)
    jz1 = int((3.0 + H_DISC / 2) * RES)
    eps[: int(R_DISC * RES), jz0:jz1] = N_DISC ** 2
    c = build_bor_coeffs(nr, nz, dr, m=0, eps=eps, n_abs_r=RES, n_abs_z=RES)
    j_src = jz0 + max(1, int(zfrac * (jz1 - jz0)))
    _, pl = run_bor_dft(init_bor_state(nr, nz), c, int(round(80.0 / c.dt)),
                        src=("Ez", 0, j_src),
                        src_profile=GaussianPulse(FCEN, 0.2), freqs=[FCEN])
    r = (np.arange(nr - 1) + 0.5) * dr

    def flux_z(jp, sign):
        er = pl["Er"][0][:, jp]
        hp = 0.5 * (pl["Hphi"][0][:, jp - 1] + pl["Hphi"][0][:, jp])
        return sign * 2 * math.pi * dr * float(np.sum(np.real(er * np.conj(hp)) * r))

    def flux_r(ip):
        ez = pl["Ez"][0][ip, :]
        hp = 0.5 * (pl["Hphi"][0][ip - 1, :] + pl["Hphi"][0][ip, :])
        R = ip * dr
        return -2 * math.pi * R * dr * float(np.sum(np.real(ez * np.conj(hp))))

    jt, jb, ir = jz1 + RES // 2, jz0 - RES // 2, int(2.0 * RES)
    top = flux_z(jt, +1)
    bottom = flux_z(jb, -1)
    side = flux_r(ir)
    total = abs(top) + abs(bottom) + abs(side)
    return abs(top) / total


fracs = [0.25, 0.5, 0.75]
effs = [extraction(f) for f in fracs]
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.plot(fracs, effs, "bo-")
ax.set_xlabel("dipole depth (fraction of disc)")
ax.set_ylabel("top-plane extraction efficiency")
ax.set_title(f"vertical dipole in n={N_DISC} disc (m=0, closed-surface norm)")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUT / "disc_extraction_efficiency.png", dpi=150)
plt.close(fig)
for f, e in zip(fracs, effs):
    print(f"disc_extraction_efficiency  zfrac={f}: eta_top = {e:.3f}")
