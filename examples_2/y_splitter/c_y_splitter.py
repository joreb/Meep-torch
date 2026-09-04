# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Inverse-designed Y splitter (2D FDTD)
#
# Example (c2): inverse-designed Y splitter -- DEFAULT mmi1x2_with_sbend,
# FULL domain (no mirror symmetry, user spec 2026-08-16), 674 nm.
#
# Geometry: gdsfactory mmi1x2_with_sbend() at DEFAULT parameters (native
# 0.5 um strip; bbox 13.5 x 4.8 um, outputs at +/-2.15 um) + 4 um
# straights on all ports. Rasterized as the base AND the rho seed
# (rho_from_eps). Objective |S21|^2 + |S31|^2 via TE0 mode overlap; the
# optimizer runs on the FULL domain so no symmetry is imposed -- the
# measured imbalance is a real result, not a construction.
#
# Best-iterate selection: after every Adam step the post-step rho is
# re-evaluated at the final BETA=64 (the pre-step J curve ranks iterates
# wrong -- see memory best-iterate-not-last).
#
# Platform: SiN 150 nm / SiO2 (meep materials library at 674 nm), lateral
# effective index. GPU. Outputs land in `examples_2/y_splitter/output/`
# (rho array + summary figure).
#
# Run: `uv run python examples_2/y_splitter/c_y_splitter.py`

# %%
import math
import pathlib

import gdsfactory as gf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.plugins.gdsfactory import component_to_eps

from meep_torch.adjoint import DesignRegion, MaterialGrid, rho_from_eps
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.core.monitors import DFTFieldsMonitor, FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_tm import init_tm_state
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port, validate_ports

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
gf.gpdk.PDK.activate()

# %% [markdown]
# ## Platform and effective index
# SiN 150 nm on SiO2 at 674 nm: a 1D slab solve collapses the vertical
# stack into a lateral effective index for the 2D TM simulation.

# %%
WL = 0.674
FCEN = 1 / WL
N_SIN = Si3N4.n(FCEN).real
N_CLAD = SiO2.n(FCEN).real
eps_line = np.full(400, N_CLAD ** 2)
eps_line[200 - 7: 200 + 8] = N_SIN ** 2
NEFF = float(solve_1d(eps_line, 0.01, FCEN, "te", k=1).neff[0])
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"c_y_splitter  SiN150/SiO2 @{WL*1e3:.0f}nm: n_core={NEFF:.4f} "
      f"n_clad={N_CLAD:.4f} (default 0.5 um strip)")

# %% [markdown]
# ## Seed geometry
# The DEFAULT gdsfactory mmi1x2_with_sbend with 4 um straights on all
# three ports, flattened into one component.

# %%
mmi = gf.components.mmi1x2_with_sbend()             # DEFAULT everything
xs = gf.get_cross_section("strip")
comp = gf.Component("y_splitter_default_sbend")
ref = comp.add_ref(mmi)
for pname in ("o1", "o2", "o3"):
    st = comp.add_ref(gf.components.straight(length=4.0, cross_section=xs))
    st.connect("o1", ref.ports[pname])
    comp.add_port(pname, port=st.ports["o2"])
comp.flatten()

# %% [markdown]
# ## Simulation grid
# Rasterize the component at 30 px/um onto the FDTD grid (GPU when
# available) and size the run to cover ring-down plus transit time.

# %%
RES = 30
dx = 1.0 / RES
grid_c, eps_c, ports = component_to_eps(comp, res=RES, pad=3.0,
                                        n_core=NEFF, n_clad=N_CLAD)
nx, ny = grid_c.nx, grid_c.ny
grid = Grid(nx=nx, ny=ny, dx=dx, device=DEV)
eps_base = eps_c.to(DEV)
NPML = RES
T_RUN = 2 * (5 / 0.2) + (nx * dx) * 1.7 + 50.0
print(f"c_y_splitter  grid {nx}x{ny} ({nx*dx:.1f} x {ny*dx:.1f} um) on {DEV}, "
      f"T_RUN={T_RUN:.0f}")

# %% [markdown]
# ## Design region and rho seed
# The mmi+sbend body between the port straights becomes the design
# region (~80 nm pixels); the rasterized geometry itself seeds rho.

# %%
o1x, o1y, _ = ports["o1"]
o2x, o2y, _ = ports["o2"]
o3x, o3y, _ = ports["o3"]
# design region: the mmi+sbend body (between the port straights)
ix0 = o1x + int(4.0 * RES)
ix1 = o2x - int(4.0 * RES)
iy0 = o3y - int(1.2 * RES)
iy1 = o2y + int(1.2 * RES)
NRX, NRY = int((ix1 - ix0) * dx / 0.08), int((iy1 - iy0) * dx / 0.08)
mg = MaterialGrid((NRX, NRY), eps_min=N_CLAD ** 2, eps_max=NEFF ** 2)
region = DesignRegion(mg, ix0=ix0, ix1=ix1, iy0=iy0, iy1=iy1)
seed = rho_from_eps(eps_base, region)
print(f"c_y_splitter  design {NRX}x{NRY} rho pixels (~80 nm)")

# %% [markdown]
# ## Ports and source pulse
# TE0 overlap ports 2 um inside each straight (kept out of the PML) and
# a Gaussian pulse around FCEN for the S-parameter runs.

# %%
SPAN = int(1.2 * RES)   # 0.25 um half-width + tails (decay ~0.26 um)
p_in = Port(axis="x", index=max(o1x + int(2.0 * RES), NPML + 4),
            lo=o1y - SPAN, hi=o1y + SPAN, direction=+1)
p_up = Port(axis="x", index=min(o2x - int(2.0 * RES), nx - NPML - 4),
            lo=o2y - SPAN, hi=o2y + SPAN, direction=-1)
p_dn = Port(axis="x", index=min(o3x - int(2.0 * RES), nx - NPML - 4),
            lo=o3y - SPAN, hi=o3y + SPAN, direction=-1)
validate_ports([p_in, p_up, p_dn], grid, NPML)
pulse = GaussianPulse(FCEN, 0.2)

# %% [markdown]
# ## Scattering parameters and objective
# One FDTD run per evaluation: TE0 source at the input, flux monitors on
# all three ports, mode decomposition into S21/S31/S11. The sibling
# _3d/_fdfd/video scripts exec this file's text up to the function below,
# so everything above stays importable as a shared setup head.

# %%
def s_params(eps):
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)
    src = ModeSourceTM(grid, coeffs, p_in, pulse, FCEN)
    mons = [FluxMonitorTM(grid, [FCEN], x_normal=True, idx=p.index,
                          lo=p.lo, hi=p.hi) for p in (p_in, p_up, p_dn)]
    _, accs = run_tm(init_tm_state(grid), coeffs, int(round(T_RUN / grid.dt)),
                     source=src, monitors=mons, checkpoint_every=128)
    out = []
    for p, acc in zip((p_in, p_up, p_dn), accs):
        el = eps.detach().cpu()[p.index, p.lo:p.hi]
        out.append(decompose(acc.cpu(), el, dx, [FCEN], "tm", 1,
                             axis_sign=-1.0))
    a_in = out[0][0][0, 0]
    return (out[1][0][0, 0] / a_in, out[2][0][0, 0] / a_in,
            out[0][1][0, 0] / a_in)


def total_T(rho, beta):
    # balance-aware objective: 2*sqrt(T_up*T_dn) equals the total at a
    # perfect 50/50 and penalizes imbalance smoothly -- NO symmetry is
    # imposed on the geometry (a plain total objective measured 9.3%
    # imbalance: the free optimizer trades balance for throughput).
    r = tanh_projection(conic_filter(rho, 0.12, dx), beta)
    s_up, s_dn, _ = s_params(region.paint(eps_base, r))
    Tu = (s_up.conj() * s_up).real
    Td = (s_dn.conj() * s_dn).real
    return 2 * torch.sqrt(Tu * Td + 1e-12)


# %% [markdown]
# ## Optimization loop
# 16 Adam steps with a beta continuation (8 -> 64); each post-step rho is
# re-scored at BETA=64 and the best iterate is kept, not the last.

# %%
rho = seed.clone().requires_grad_(True)
adam = torch.optim.Adam([rho], lr=0.03)
history, post64 = [], []
best = (-1.0, None)
for it in range(16):
    BETA = [8, 16, 32, 64][min(it // 4, 3)]
    adam.zero_grad()
    J = total_T(rho.clamp(0.0, 1.0), BETA)
    (-J).backward()
    adam.step()
    history.append(float(J))
    with torch.no_grad():
        Jp = float(total_T(rho.clamp(0.0, 1.0), 64.0))
    post64.append(Jp)
    if Jp > best[0]:
        best = (Jp, rho.detach().clamp(0, 1).clone())
    print(f"c_y_splitter  it={it} beta={BETA}: J={float(J)*100:.1f}% "
          f"post64={Jp*100:.1f}%")

# %% [markdown]
# ## Final evaluation
# Re-run the best design at BETA=64, record the DFT |Ez| field, and save
# the rho array to `output/c_y_splitter_rho.npy` for the sibling scripts.

# %%
rho_f = best[1]
r = tanh_projection(conic_filter(rho_f, 0.12, dx), 64.0)
eps_fin = region.paint(eps_base, r).detach()
with torch.no_grad():
    s_up, s_dn, s11 = s_params(eps_fin)
    coeffs = build_tm_coeffs(grid, eps_inf=eps_fin, npml=NPML)
    src = ModeSourceTM(grid, coeffs, p_in, pulse, FCEN)
    mon = DFTFieldsMonitor(grid, [FCEN], "Ez")
    _, (acc,) = run_tm(init_tm_state(grid), coeffs,
                       int(round(T_RUN / grid.dt)), source=src,
                       monitors=[mon], checkpoint_every=400)
acc = acc.cpu()
ez = np.abs((acc[0, 0] + 1j * acc[0, 1]).cpu().numpy())
T_up, T_dn = float(abs(s_up) ** 2), float(abs(s_dn) ** 2)
tot = T_up + T_dn
il_db = -10 * math.log10(max(tot, 1e-12))
imb = abs(T_up - T_dn) / max(tot, 1e-12)
np.save(OUT / "c_y_splitter_rho.npy", rho_f.cpu().numpy())
print(f"c_y_splitter  BEST |S21|^2={T_up*100:.1f}% |S31|^2={T_dn*100:.1f}% "
      f"total={tot*100:.1f}% (IL {il_db:.2f} dB), imbalance {imb*100:.2f}% "
      f"(MEASURED, full domain), |S11|^2={float(abs(s11)**2)*100:.2f}%")

# %% [markdown]
# ## Plots
# Convergence curve (pre- vs post-step), seed and optimized permittivity,
# and the |Ez| field of the final design.

# %%
fig, axs = plt.subplots(2, 2, figsize=(13, 7))
ax1, ax2, ax3, ax4 = axs.flat
ax1.plot(np.array(post64) * 100, "bo-", label="post-step @BETA=64")
ax1.plot(np.array(history) * 100, "c.--", lw=0.8, label="pre-step")
i_best = int(np.argmax(post64))
ax1.plot([i_best], [post64[i_best] * 100], "r*", ms=14, label=f"best it{i_best}")
ax1.set_xlabel("iteration"); ax1.set_ylabel("2 sqrt(T21 T31) [%]")
ax1.set_title(f"total {tot*100:.1f}%, IL {il_db:.2f} dB, "
              f"imbalance {imb*100:.2f}% (measured)")
ax1.legend(fontsize=7)
ax2.imshow(eps_c.cpu().numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx*dx, 0, ny*dx], aspect="auto")
ax2.set_title("DEFAULT mmi1x2_with_sbend seed (0.5 um strip)")
ax3.imshow(eps_fin.cpu().numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx*dx, 0, ny*dx], aspect="auto")
ax3.set_title("optimized (full domain, no symmetry imposed)")
ax4.imshow(ez.T, origin="lower", cmap="inferno",
           extent=[0, nx*dx, 0, ny*dx], aspect="auto")
ax4.set_title(r"$|E_z|$ at 674 nm")
for ax in (ax3, ax4):
    ax.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "c_y_splitter.png", dpi=150)
plt.close(fig)
print("c_y_splitter.png written")
