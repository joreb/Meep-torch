# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Extraction efficiency and the LDOS proportionality
#
# meep-torch port of meep/python/examples/extraction_eff_ldos.py. Extraction
# efficiency of a point dipole in a dielectric layer, and the proportionality
# between the total radiated power (closed-surface Poynting flux) and the LDOS
# (source-field overlap). The meep original places the layer above a lossless
# metallic ground plane and verifies cyl vs 3D agreement. meep-torch has no
# PEC ground-plane primitive and no 3D LDOS monitor, so per the port
# instructions this is re-cast as a single BOR m=0 calculation:
#
# * the dielectric layer is an axisymmetric slab (r < r_disc, thickness dmat,
#   n = 2.4) in air, driven by an on-axis vertical Ez dipole (m = 0) at
#   height h * dmat inside the slab;
# * extraction efficiency = top-plane Poynting flux / CLOSED-surface flux
#   (top + bottom + outer-r), the fully normalized disc_extraction convention;
# * total radiated power (closed-surface flux) is compared with the LDOS
#   (core/ldos.ldos_spectrum on the source-cell Ez series + reconstructed
#   J(t)) at two dipole depths: the two must be PROPORTIONAL -- i.e.
#   total(h1)/total(h2) == LDOS(h1)/LDOS(h2), since both track the same
#   density of radiating states. The dV normalization constant meep uses
#   cancels in the ratio.
#
# SUBSTITUTIONS:
# * lossless metallic ground plane -> dropped (slab in air, as in
#   disc_extraction_efficiency). The closed surface gains a bottom face;
#   extraction is still top/closed. The LDOS-vs-flux proportionality
#   (the actual gate) is independent of the ground plane.
# * 3D cross-check -> dropped (BOR m=0 is the true 3D axisymmetric result;
#   the meep 3D run exists only to confirm the cyl run, which a single
#   BOR m=0 already is).
# * Er in-plane dipole (m = -1) -> on-axis vertical Ez dipole (m = 0),
#   the supported BOR source (disc_extraction uses the same).
#
# Gate (printed): extraction efficiency at two depths, and
# |total_ratio - ldos_ratio| / ldos_ratio <= 10% (proportionality).
# Output: `output/extraction_eff_ldos.png`.
#
# Run: `uv run python examples/05_monitors_analysis/extraction_eff_ldos.py`

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.ldos import ldos_spectrum
from meep_torch.core.near2far_cyl import run_bor_dft
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bor import build_bor_coeffs, init_bor_state, run_bor

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
FCEN, RES = 1.0, 25
N_LAYER, R_DISC, DMAT = 2.4, 1.2, 0.5
N_STEPS_FACTOR = 80.0           # run length in units of 1/dt

# %% [markdown]
# ## Geometry and closed-surface fluxes

# %%
def _geometry(dr):
    """eps grid + slab z-index range for the BOR cell."""
    nr, nz = int(3.0 * RES), int(6.0 * RES)
    eps = torch.ones(nr, nz, dtype=torch.float64)
    jz0 = int((3.0 - DMAT / 2) * RES)
    jz1 = int((3.0 + DMAT / 2) * RES)
    eps[: int(R_DISC * RES), jz0:jz1] = N_LAYER ** 2
    return nr, nz, eps, jz0, jz1


def _fluxes(pl, c, jt, jb, ir):
    """top / bottom / outer-r Poynting flux through the closed surface."""
    dr = c.dr
    r = c.rh.flatten().cpu().numpy()        # (i + 0.5) * dr for i = 0 .. nr-2

    def flux_z(jp, sign):
        er = pl["Er"][0][:, jp]
        hp = 0.5 * (pl["Hphi"][0][:, jp - 1] + pl["Hphi"][0][:, jp])
        return sign * 2 * math.pi * dr * float(
            np.sum(np.real(er * np.conj(hp)) * r))

    def flux_r(ip):
        ez = pl["Ez"][0][ip, :]
        hp = 0.5 * (pl["Hphi"][0][ip - 1, :] + pl["Hphi"][0][ip, :])
        R = ip * dr
        return -2 * math.pi * R * dr * float(np.sum(np.real(ez * np.conj(hp))))

    top = flux_z(jt, +1)
    bottom = flux_z(jb, -1)
    side = flux_r(ir)
    return abs(top), abs(bottom), abs(side)

# %% [markdown]
# ## Per-depth run
# Two runs per depth: DFT planes for the closed-surface flux, and the
# source-cell Ez series (with the reconstructed drive current) for the LDOS.

# %%
def _run(h):
    """Return (extraction_eff, total_closed_flux, ldos) for dipole depth h."""
    dr = 1.0 / RES
    nr, nz, eps, jz0, jz1 = _geometry(dr)
    c = build_bor_coeffs(nr, nz, dr, m=0, eps=eps, n_abs_r=RES, n_abs_z=RES)
    j_src = jz0 + max(1, int(h * (jz1 - jz0)))
    n_steps = int(round(N_STEPS_FACTOR / c.dt))
    profile = GaussianPulse(FCEN, 0.2)

    # (1) DFT planes for the closed-surface Poynting flux.
    _, pl = run_bor_dft(init_bor_state(nr, nz), c, n_steps,
                        src=("Ez", 0, j_src), src_profile=profile, freqs=[FCEN])
    jt = jz1 + RES // 2
    jb = jz0 - RES // 2
    ir = int(2.0 * RES)
    top, bottom, side = _fluxes(pl, c, jt, jb, ir)
    closed = top + bottom + side
    ext = top / closed

    # (2) source-cell Ez time series -> LDOS at fcen (E/J overlap).
    _, series = run_bor(init_bor_state(nr, nz), c, n_steps,
                        src=("Ez", 0, j_src), src_profile=profile,
                        probe=("Ez", 0, j_src))
    e = np.real(series)
    j = np.array([profile.amplitude((t + 0.5) * c.dt) for t in range(len(e))])
    ldos = float(ldos_spectrum(e, j, c.dt, [FCEN])[0])
    return ext, closed, ldos

# %% [markdown]
# ## Gate and plot
# The flux ratio between the two depths must match the LDOS ratio within 10%
# (asserted), then the extraction curve is saved.

# %%
def main():
    depths = [0.3, 0.7]
    rows = [_run(h) for h in depths]
    for h, (ext, closed, ldos) in zip(depths, rows):
        print(f"extraction_eff_ldos  h={h}: extraction={ext:.4f}  "
              f"total_closed={closed:.4g}  LDOS={ldos:.4g}")

    total = [r[1] for r in rows]
    ldos = [r[2] for r in rows]
    ratio_total = total[0] / total[1]
    ratio_ldos = ldos[0] / ldos[1]
    prop_err = abs(ratio_total - ratio_ldos) / abs(ratio_ldos)
    print(f"extraction_eff_ldos  proportionality: total_ratio={ratio_total:.4f} "
          f"ldos_ratio={ratio_ldos:.4f}  err={100*prop_err:.2f}%")
    assert prop_err <= 0.10, \
        f"total-vs-LDOS proportionality gate failed: {100*prop_err:.2f}% > 10%"
    print("extraction_eff_ldos  GATE PASS: total flux proportional to LDOS "
          f"at 2 depths ({100*prop_err:.2f}% <= 10%)")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(depths, [r[0] for r in rows], "bo-")
    ax.set_xlabel("dipole depth (fraction of layer thickness)")
    ax.set_ylabel("top-plane extraction efficiency")
    ax.set_title(f"vertical dipole in n={N_LAYER} slab (m=0, closed-surface norm)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "extraction_eff_ldos.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT / 'extraction_eff_ldos.png'}")


if __name__ == "__main__":
    main()
