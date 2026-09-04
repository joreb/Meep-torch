# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Liquid-crystal polarization grating (geometric phase)
#
# The LC director rotates in the transverse plane as phi(y) = pi y / period,
# with the slab tuned to a half wave ((ne - no) d = lambda / 2). A circularly
# polarized wave converts to the opposite handedness deflected into the -+1
# order — the Pancharatnam-Berry deflector. The y-dependent eps tensor
# couples Ey and Ez, so this runs on the full-vector 2D stepper.

# %%
"""meep-torch port of meep/python/examples/polarization_grating.py.

Liquid-crystal polarization grating: the LC director rotates in the
transverse (y, z) plane with phi(y) = pi y / period, thickness tuned to
a half wave ((ne - no) d = lambda / 2). A circularly polarized normal-
incidence wave converts to the OPPOSITE handedness deflected into the
-+1 order -- the geometric-phase (Pancharatnam-Berry) deflector.

Uses the NEW full-vector 2D stepper (stepper_fv2d): the y-dependent
eps tensor couples Ey and Ez, which the TM/TE split cannot host (this
stub's long-standing blocker).

Run: uv run python examples/10_periodic_mpb/polarization_grating.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.stepper_fv2d import (build_fv2d_coeffs, init_fv2d_state,
                                          step_fv2d)

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
# %% [markdown]
# ## Rotating-director eps tensor and circular drive
#
# The (eyy, ezz, eyz) components encode the rotated uniaxial tensor inside
# the slab; a (cos, sin) two-component source injects one circular state and
# the transmitted Ey/Ez DFT line is split back into circular components.

# %%
FCEN, RES = 1.0, 20
NO, NE = 1.5, 1.7
D = 0.5 / (NE - NO)                                # half-wave thickness = 2.5
PERIOD = 6.0

dx = 1.0 / RES
nx, ny = int(10.0 * RES), int(PERIOD * RES)
ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * dx
phi = math.pi * ys / PERIOD
one = torch.ones(nx, ny, dtype=torch.float64)
exx = one.clone()
eyy = one.clone(); ezz = one.clone(); eyz = torch.zeros(nx, ny, dtype=torch.float64)
i0, i1 = int(4.0 * RES), int((4.0 + D) * RES)
delta = NE ** 2 - NO ** 2
eyy[i0:i1, :] = NO ** 2 + delta * torch.cos(phi)[None, :] ** 2
ezz[i0:i1, :] = NO ** 2 + delta * torch.sin(phi)[None, :] ** 2
eyz[i0:i1, :] = delta * (torch.cos(phi) * torch.sin(phi))[None, :]
exx[i0:i1, :] = NO ** 2
eps6 = (exx, torch.zeros_like(one), torch.zeros_like(one), eyy, eyz, ezz)
c = build_fv2d_coeffs(eps6, dx, npml=RES)
st = init_fv2d_state(nx, ny)
mask = torch.zeros(nx, ny, dtype=torch.float64)
mask[int(2.0 * RES), :] = 1.0
omega = 2 * math.pi * FCEN
n_steps = int(round(50.0 / c.dt))
accY = np.zeros(ny, dtype=complex)
accZ = np.zeros(ny, dtype=complex)
ix_out = nx - int(2.0 * RES)
for t in range(n_steps):
    tt = (t + 0.5) * c.dt
    ramp = min(1.0, tt / 5.0)
    st = step_fv2d(st, c, math.cos(omega * tt) * ramp, mask, "Ey",
                   src2=("Ez", mask, math.sin(omega * tt) * ramp))
    if t >= n_steps // 2:
        ph = np.exp(-2j * math.pi * FCEN * (t + 1.0) * c.dt) * c.dt
        accY += st.Ey[ix_out, :].cpu().numpy() * ph
        accZ += st.Ez[ix_out, :].cpu().numpy() * ph

# %% [markdown]
# ## Order decomposition per handedness
#
# The FFT along the periodic y line gives orders m = -2..2 for the
# input-handed and converted components; the converted power peaks in one
# first order.

# %%
# transmitted order decomposition (periodic y): circular components
# DFT phasors are e^{+iwt}: the (cos, sin) drive is the (1, -i) circular
# state = aR; the CONVERTED handedness is aL
aR = (accY + 1j * accZ) / math.sqrt(2)             # input handedness
aL = (accY - 1j * accZ) / math.sqrt(2)             # converted handedness
orders = np.arange(-2, 3)
powers = {}
for name, a in (("input-handed", aR), ("converted", aL)):
    fa = np.fft.fft(a) / ny
    p = np.abs(np.concatenate([fa[-2:], fa[:3]])) ** 2
    powers[name] = p[np.argsort(np.concatenate([np.arange(-2, 0), np.arange(0, 3)]))]
tot = sum(p.sum() for p in powers.values())
fig, ax = plt.subplots(figsize=(5.8, 3.8))
w = 0.35
ax.bar(orders - w / 2, powers["input-handed"] / tot, w, label="input handedness")
ax.bar(orders + w / 2, powers["converted"] / tot, w, label="converted")
ax.set_xlabel("diffraction order m")
ax.set_ylabel("relative power")
ax.set_title(f"polarization grating (half-wave LC, period {PERIOD})")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "polarization_grating.png", dpi=150)
plt.close(fig)
conv = powers["converted"]
eff = float(conv[np.argmax(conv)] / tot)
m_pk = int(orders[np.argmax(conv)])
print(f"polarization_grating.png  converted-handedness peak at m={m_pk}, "
      f"efficiency {eff:.2f} (geometric-phase deflection)")
