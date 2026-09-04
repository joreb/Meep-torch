# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Asymmetric refraction in tilted hyperbolic calcite
#
# Calcite driven inside its ordinary Reststrahlen band (1470 cm^-1) is
# hyperbolic: Re eps < 0 across the ordinary plane, > 0 along the optic axis.
# Tilting the optic axis 25 deg puts a dispersive off-diagonal eps_xy(f) into
# the lab tensor (oriented Lorentz poles), shearing the iso-frequency
# hyperbola so mirrored +-45 deg beams refract to *different* energy angles.
# Note: this example's measured values are cited by
# tests/examples/test_tensor_poles.py.

# %%
"""meep-torch rebuild of fdtdx/examples/hyperbolic_asymmetric_refraction.py.

Calcite driven inside its ordinary Reststrahlen band (1470 cm^-1,
lambda = 6.8 um) is hyperbolic: Re eps < 0 across the ordinary plane,
Re eps > 0 along the optic axis. Tilting the optic axis 25 deg above
the surface (in the plane of incidence) puts a dispersive off-diagonal
eps_xy(f) into the lab tensor -- this needs ORIENTED Lorentz poles
(build_te_coeffs poles_tensor, P cross-driven by both E components),
which grid-aligned per-axis poles cannot express. The tilt shears the
iso-frequency hyperbola: mirrored +-45 deg beams refract to DIFFERENT
energy angles (one ordinary, one negative) -- asymmetric refraction.

Gates: measured energy-corridor angles (intensity-ridge fit) vs the
analytic group-velocity direction for both beams.

Run: uv run python examples/11_3d/fdtdx_ports/hyperbolic_asymmetric_refraction.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_te_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper_te import init_te_state, step_te

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# calcite, meep units (a = 1 um): f = wavenumber[1/cm] * 1e-4
F0 = 1470.17e-4                                     # lambda = 6.80 um
POLE_O = (1410.0e-4, 10.0e-4, 0.5416)               # E_u ordinary (in-plane part)
POLE_E = (871.0e-4, 3.0e-4, 0.3745)                 # A_2u extraordinary
EPS_INF_O, EPS_INF_E = 2.5898, 2.4691
TILT = math.radians(25.0)                           # optic axis above surface
U_E = np.array([math.cos(TILT), math.sin(TILT)])    # optic axis (lab)
U_O = np.array([-math.sin(TILT), math.cos(TILT)])   # in-plane ordinary axis

DX, NPML = 0.17, 12
NX, NY = 1000, 188                                  # 170 x 32 um
grid = Grid(nx=NX, ny=NY, dx=DX)
J_BOT, J_TOP = 65, 118                              # slab 11..20 um
J_SRC = NY - NPML - 6
WALL = 36                                           # conductor cells at slab x-ends
THETA = math.radians(45.0)
SRC_X = {+1: 31.1, -1: 133.9}                       # beam centers [um]
SIGMA_BEAM, HALF_W = 10.0, 22.0                      # Gaussian envelope [um]
T_END = 600.0                                       # 2 ps: the ordinary-pole
# polariton flat band sits at fn = 1410 cm^-1, only 4% below the drive, with
# a 1/(2 pi gamma) ~ 160 tu ring-down. A fast turn-on excites it and its
# fields pollute a short run. Slow ramp (150 tu) + narrowband DFT phasor
# over the last 250 tu (df ~ 0.004 < band separation 0.006) isolates the
# steady-state propagating branch.


# %% [markdown]
# ## Lab-frame tensor and analytic prediction
#
# The complex 2x2 lab tensor is assembled from oriented ordinary and
# extraordinary Lorentz responses; the analytic energy-refraction angle comes
# from the group velocity on the sheared iso-frequency curve at conserved kx.

# %%
def outer2(u):
    return (u[0] * u[0], u[0] * u[1], u[1] * u[1])


def eps_lab(f):
    """Complex 2x2 lab-frame tensor at frequency f."""
    def chi(p):
        fn, g, s = p
        return s * fn * fn / (fn * fn - f * f - 1j * g * f)
    e = np.zeros((2, 2), dtype=complex)
    for val, u in ((EPS_INF_O + chi(POLE_O), U_O), (EPS_INF_E + chi(POLE_E), U_E)):
        e += val * np.outer(u, u)
    return e


def analytic_energy_angle(theta_inc):
    """Energy-refraction angle from -y (downward), degrees. kx conserved;
    solve inv_xx ky^2 - 2 inv_xy kx ky + inv_yy kx^2 = k0^2 (Hz pol),
    group velocity g ~ (inv_yy kx - inv_xy ky, inv_xx ky - inv_xy kx)."""
    inv = np.linalg.inv(np.real(eps_lab(F0)))
    kx = math.sin(theta_inc)
    a, b, c = inv[0, 0], -2.0 * inv[0, 1] * kx, inv[1, 1] * kx ** 2 - 1.0
    disc = b * b - 4 * a * c
    for sign in (+1.0, -1.0):
        ky = (-b + sign * math.sqrt(disc)) / (2 * a)
        gx = inv[1, 1] * kx - inv[0, 1] * ky
        gy = inv[0, 0] * ky - inv[0, 1] * kx
        if gy < 0:
            return math.degrees(math.atan2(gx, -gy))
    raise ValueError("no downward branch")


# %% [markdown]
# ## Slab coefficients, mirrored beams, and narrowband DFT
#
# The slab carries the static lab tensor plus two oriented Lorentz poles;
# graded conductor walls terminate the slab x-ends (a PML would amplify the
# backward waves of a hyperbolic medium). Two mirrored 45-deg CW beams with
# tilted phase fronts are ramped slowly, and the last 250 time units are
# phase-accumulated at F0 to isolate the steady state.

# %%
# --- coefficients: static lab tensor + oriented poles in the slab interior,
# graded conductor walls at the slab x-ends (a PML would amplify the
# backward waves of a hyperbolic medium) ---
one = torch.ones(NX, NY, dtype=torch.float64, device=grid.device)
slab = torch.zeros(NX, NY, dtype=torch.float64, device=grid.device)
slab[:, J_BOT:J_TOP] = 1.0
exx = one.clone(); exy = torch.zeros_like(one); eyy = one.clone()
e_inf = EPS_INF_O * np.outer(U_O, U_O) + EPS_INF_E * np.outer(U_E, U_E)
exx[slab > 0] = e_inf[0, 0]
exy[slab > 0] = e_inf[0, 1]
eyy[slab > 0] = e_inf[1, 1]
pole_mask = slab.clone()
pole_mask[:WALL, :] = 0.0
pole_mask[NX - WALL:, :] = 0.0
sigma_d = torch.zeros(NX, NY, dtype=torch.float64, device=grid.device)
ramp = torch.linspace(30.0, 0.0, WALL, dtype=torch.float64, device=grid.device)
sigma_d[:WALL, J_BOT:J_TOP] = ramp.flip(0)[:, None]
sigma_d[NX - WALL:, J_BOT:J_TOP] = ramp[:, None]

coeffs = build_te_coeffs(
    grid, eps_inf=one, eps_tensor=(exx, exy, eyy), sigma_d=sigma_d, npml=NPML,
    poles_tensor=[POLE_O[:2] + (outer2(U_O * math.sqrt(POLE_O[2])),),
                  POLE_E[:2] + (outer2(U_E * math.sqrt(POLE_E[2])),)],
    pole_masks_tensor=[pole_mask, pole_mask],
)

# --- two mirrored 45-deg CW beams, drive Ex with a tilted phase front ---
xs = (torch.arange(NX, dtype=torch.float64, device=grid.device) + 0.5) * DX
omega, kx0 = 2 * math.pi * F0, 2 * math.pi * F0 * math.sin(THETA)
masks = []
for sgn in (+1, -1):
    env = torch.exp(-((xs - SRC_X[sgn]) / SIGMA_BEAM) ** 2)
    env[(xs - SRC_X[sgn]).abs() > HALF_W] = 0.0
    ph = sgn * kx0 * xs
    masks.append((env * torch.cos(ph), env * torch.sin(ph)))

st = init_te_state(grid, n_poles=2, n_poles_y=2)
n_steps = int(round(T_END / grid.dt))
accR = torch.zeros(NX - 1, NY - 1, dtype=torch.float64, device=grid.device)
accI = torch.zeros_like(accR)
t_acc = int(round((T_END - 250.0) / grid.dt))
for t in range(n_steps):
    tt = (t + 0.5) * grid.dt
    ramp_t = min(1.0, tt / 150.0)
    c, s = math.cos(omega * tt) * ramp_t, math.sin(omega * tt) * ramp_t
    st = step_te(st, coeffs)
    for mc, ms in masks:
        a = c * mc[:NX - 1] + s * ms[:NX - 1]
        st.Dx[:, J_SRC] += grid.dt * a
        st.Ex[:, J_SRC] += grid.dt * a
    if t >= t_acc:
        ph = (t + 1.0) * grid.dt * omega
        accR += st.Hz * math.cos(ph)
        accI += st.Hz * math.sin(ph)
hz = st.Hz.cpu().numpy()
intens = (accR ** 2 + accI ** 2).cpu().numpy()          # |Hz(F0)|^2, narrowband


# %% [markdown]
# ## Corridor-angle measurement vs analytic gate
#
# The intensity ridge inside the slab is followed row by row and fit
# mid-slab; the two measured angles must differ (asymmetry) and match the
# analytic group-velocity directions.

# %%
def measure_corridor_angle(hit_x_um):
    """Intensity-ridge angle inside the slab, degrees from -y (downward).
    Follows the ridge row by row; fits mid-slab rows only (the first
    1.5 um carry an interface wave, the deepest 2 um little signal)."""
    window = int(round(3.0 / DX))
    skip_face = int(round(1.5 / DX))
    skip_exit = int(round(2.0 / DX))
    x_prev = hit_x_um / DX
    xs_r, js_r = [], []
    for j in range(J_TOP - 3, J_BOT + 3, -1):
        lo = max(int(round(x_prev)) - window, 0)
        hi = min(int(round(x_prev)) + window, intens.shape[0])
        row = intens[lo:hi, j]
        w = np.clip(row - np.median(row), 0.0, None) ** 2
        if w.sum() <= 0:
            continue
        x_c = float(np.average(np.arange(lo, hi), weights=w))
        x_prev = x_c
        if J_TOP - j >= skip_face and j - J_BOT >= skip_exit:
            xs_r.append(x_c)
            js_r.append(float(j))
    slope = np.polyfit(js_r, xs_r, 1)[0]           # dx/dj, j decreasing = down
    return math.degrees(math.atan(-slope))


Y_TOP, Y_BOT, Y_SRC = J_TOP * DX, J_BOT * DX, J_SRC * DX
print(f"hyperbolic_asymmetric_refraction  lab tensor at drive: "
      f"eps_xx={eps_lab(F0)[0, 0]:.2f} eps_yy={eps_lab(F0)[1, 1]:.2f} "
      f"eps_xy={eps_lab(F0)[0, 1]:.2f}")
results = {}
for sgn in (+1, -1):
    th = sgn * 45.0
    hit = SRC_X[sgn] + sgn * (Y_SRC - Y_TOP)
    meas = measure_corridor_angle(hit)
    ana = analytic_energy_angle(math.radians(th))
    results[sgn] = (hit, meas, ana)
    kind = "NEGATIVE" if ana * th < 0 else "positive"
    print(f"  incidence {th:+.0f} deg -> energy {meas:+.1f} deg measured, "
          f"{ana:+.1f} deg analytic ({kind}), diff {abs(meas - ana):.1f} deg")

# --- plots: lab tensor components + time-averaged intensity with rays ---
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 9),
                               gridspec_kw={"height_ratios": [1, 1.4]})
fs_cm = np.linspace(1300, 1600, 400)
eps_arr = np.array([eps_lab(f * 1e-4) for f in fs_cm])
ax0.plot(fs_cm, eps_arr[:, 0, 0].real, label=r"Re $\epsilon_{xx}$")
ax0.plot(fs_cm, eps_arr[:, 1, 1].real, label=r"Re $\epsilon_{yy}$")
ax0.plot(fs_cm, eps_arr[:, 0, 1].real, label=r"Re $\epsilon_{xy}$ (tilt)")
ax0.axhline(0, color="k", lw=0.8)
ax0.axvline(1470.17, color="gray", ls="--", lw=0.8, label="drive")
ax0.set_xlabel(r"frequency [cm$^{-1}$]"); ax0.set_ylabel(r"Re $\epsilon$")
ax0.set_ylim(-15, 15); ax0.legend(fontsize=8)
ax0.set_title("tilted calcite: lab-frame tensor (oriented Lorentz poles)")

ax1.imshow(np.sqrt(intens / intens.max()).T, origin="lower", cmap="magma",
           vmin=0, vmax=1, extent=[0, (NX - 1) * DX, 0, (NY - 1) * DX],
           aspect="equal")
for y in (Y_TOP, Y_BOT):
    ax1.axhline(y, color="w", lw=0.9)
for sgn, color in ((+1, "lime"), (-1, "cyan")):
    hit, meas, ana = results[sgn]
    ax1.plot([SRC_X[sgn], hit], [Y_SRC, Y_TOP], "--", color="0.85", lw=0.9)
    xb = hit + math.tan(math.radians(ana)) * (Y_TOP - Y_BOT)
    ax1.plot([hit, xb], [Y_TOP, Y_BOT], "--", color=color, lw=1.4,
             label=f"{sgn * 45:+d} deg in: {meas:+.1f} meas / {ana:+.1f} analytic")
ax1.legend(loc="lower left", fontsize=9)
ax1.set_xlabel("x [um]"); ax1.set_ylabel("y [um]")
ax1.set_title("time-averaged field: mirrored beams, asymmetric refraction")
fig.tight_layout()
fig.savefig(OUT / "hyperbolic_asymmetric_refraction.png", dpi=150)
plt.close(fig)
d1 = abs(results[+1][1] - results[+1][2])
d2 = abs(results[-1][1] - results[-1][2])
print(f"hyperbolic_asymmetric_refraction.png written; angle errors "
      f"{d1:.1f} / {d2:.1f} deg (gate <= 4)")
