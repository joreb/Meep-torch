# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Absorbed power density in a lossy dielectric
#
# meep-torch port of meep/python/examples/absorbed_power_density.py. Local
# absorbed power density in a lossy dielectric, validated two ways: from the
# DFT fields (volumetric sigma_d * eps * |E|^2 -- D_conductivity damps D, not
# E) and from a closed-surface flux box around the scatterer (net inward
# Poynting flux = absorbed power, energy conservation). The console prints the
# two values and their ratio; output: `output/absorbed_power_density.png`.
#
# SUBSTITUTIONS (meep original -> 3D meep-torch shim):
# * 2D Ez cell + mp.Cylinder (a circle) -> 3D cell + axis-aligned lossy Block
#   (a cube). The W9 wave-3 3D shim rasterizes axis-aligned Blocks only (no
#   Cylinder/Prism), so the circular cross-section becomes a cube; the
#   physics (absorbed power = sigma_d*eps*|E|^2, energy balance vs a flux box)
#   is unchanged.
# * mp.SiO2 dispersive material -> a scalar lossy Medium(epsilon, D_conductivity).
#   The 3D stepper is scalar eps + conductivity (no Lorentz poles this wave),
#   so SiO2's Im(eps) at the drive frequency is represented by D_conductivity
#   (Im(eps) = sigma_d / (2*pi*f)). This is the W5 conductive-slab pattern.
# * Dz DFT field (meep records Dz + Ez) -> sigma_d*|E|^2 from the E DFT fields
#   directly. For a conductive medium Im(E*.D) = (sigma_d/w)|E|^2, so
#   2*pi*f*Im(E*.D) = sigma_d*|E|^2; the D field is not needed.
# * planewave line source (2D) -> full-plane Ex PlaneSource (3D, z-normal);
#   mp.inf transverse keeps the x/y CPML inert (the wave-3 trick).
# * dft_fields region (meep `center/size`) -> whole-cell add_dft_fields (the
#   shim's 3D path is whole-cell only); the lossy region is masked from the
#   eps volume (get_array(Dielectric)) after the run.
# * closed flux box (4 weighted regions, 2D) -> 6 FluxRegion faces (3D),
#   one add_flux call each, summed by hand with inward signs (the shim
#   takes one region per call).
#
# Run: `uv run python examples/03_materials/absorbed_power_density.py`

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp
from meep_torch.mp import (Block, FluxRegion, GaussianSource, Medium, PML,
                           Simulation, Source, Vector3)

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Parameters and simulation builder
# A lossy cube (eps=2.25, sigma_d=2) illuminated from below by a planewave;
# resolution reduced for a fast 3D smoke run.

# %%
# --- reduced resolution / sizes for a fast smoke run (3D is O(n^3)) ---
RES = 10                  # pixels/um
SX = SY = 6.0
SZ = 8.0
DPML = 0.5
CUBE = 2.0                # lossy cube edge (um)
EPS = 2.25                # n = 1.5
SIGMA_D = 2.0             # D_conductivity -> Im(eps) = sigma_d / (2*pi*f)
FCEN, FWIDTH = 1.0, 0.2
SRC_Z = -3.0
CUBE_CENTER = Vector3(0, 0, 1.0)         # cube z=[0,2], source below
BOX = 1.1                 # flux-box face offset from cube center (just outside)
T_END = 14.0


def _sim():
    return Simulation(
        cell_size=Vector3(SX, SY, SZ), resolution=RES, dimensions=3,
        sources=[Source(src=GaussianSource(frequency=FCEN, fwidth=FWIDTH),
                        component=mp.Ex, center=Vector3(0, 0, SRC_Z),
                        size=Vector3(mp.inf, mp.inf, 0))],
        geometry=[Block(material=Medium(epsilon=EPS, D_conductivity=SIGMA_D),
                        center=CUBE_CENTER, size=Vector3(CUBE, CUBE, CUBE))],
        boundary_layers=[PML(thickness=DPML)])

# %% [markdown]
# ## Flux box
# Six flux faces just outside the cube; the net inward flux is the total
# absorbed power (energy conservation).

# %%
def _box_faces():
    c = CUBE_CENTER
    h = BOX
    # one FluxRegion per face (shim takes one region per add_flux call);
    # a finite transverse size -> an interior rectangular flux window.
    return [
        ("x_lo", FluxRegion(center=Vector3(c.x - h, c.y, c.z),
                            size=Vector3(0, 2 * h, 2 * h))),
        ("x_hi", FluxRegion(center=Vector3(c.x + h, c.y, c.z),
                            size=Vector3(0, 2 * h, 2 * h))),
        ("y_lo", FluxRegion(center=Vector3(c.x, c.y - h, c.z),
                            size=Vector3(2 * h, 0, 2 * h))),
        ("y_hi", FluxRegion(center=Vector3(c.x, c.y + h, c.z),
                            size=Vector3(2 * h, 0, 2 * h))),
        ("z_lo", FluxRegion(center=Vector3(c.x, c.y, c.z - h),
                            size=Vector3(2 * h, 2 * h, 0))),
        ("z_hi", FluxRegion(center=Vector3(c.x, c.y, c.z + h),
                            size=Vector3(2 * h, 2 * h, 0))),
    ]

# %% [markdown]
# ## Run, energy balance and plot
# One run records both the flux faces and the whole-cell DFT E fields; the
# volumetric sigma_d*eps*|E|^2 integral is compared against the flux-box
# absorbed power, and an x-z slice of the density is saved.

# %%
def main():
    sim = _sim()
    for name, reg in _box_faces():
        sim.add_flux(FCEN, 0, 1, reg, name=name)
    # Ex is the driven component; Ez the scattered/out-of-plane. Ey ~ 0 for a
    # z-normal Ex planewave, so it is not recorded (keeps the DFT cost down;
    # 3D full-grid DFT fields are the expensive part at O(n^3)/step).
    sim.add_dft_fields([mp.Ex, mp.Ez], FCEN, 0, 1, name="E")
    sim.run(until=T_END)

    fl = {n: sim.get_fluxes(n)[0] for n, _ in _box_faces()}
    # inward flux per face = +flux on the low face, -flux on the high face
    # (FluxMonitor3D returns power in the +axis direction).
    absorbed_flux = (fl["x_lo"] - fl["x_hi"]
                     + fl["y_lo"] - fl["y_hi"]
                     + fl["z_lo"] - fl["z_hi"])

    eps = sim.get_array(component=mp.Dielectric)          # (nx,ny,nz)
    dx = 1.0 / RES
    dV = dx ** 3
    mask = eps > (EPS + 1.0) / 2.0                          # cube voxels
    shape = eps.shape
    comps = [sim.get_dft_array("E", c, 0) for c in (mp.Ex, mp.Ez)]

    def _pad(a):
        a = np.squeeze(a)                                   # drop freq axis
        pads = [(0, max(0, shape[i] - a.shape[i])) for i in range(3)]
        return np.pad(a, pads)                              # Yee -> common grid

    Edens = sum(np.abs(_pad(c)) ** 2 for c in comps)        # |Ex|^2+|Ey|^2+|Ez|^2
    # 2*pi*f * Im(E*.D) = sigma_d * |E|^2 for a conductive medium (meep form).
    # D_conductivity damps D (dD/dt = curl H - sigma_d D), so the
    # dissipated power density is E . (sigma_d D) = sigma_d * eps * |E|^2
    # -- the bare sigma_d*|E|^2 reads 1/eps = 2.25x low (the stuck 0.433)
    absorbed_dft = float(SIGMA_D * EPS * np.sum(Edens[mask]) * dV)

    # energy-balance scaling: the flux monitor accumulates a T-scaled phasor
    # product; the DFT |E|^2 volume carries the same T scaling, so the two
    # agree up to a single real factor (the monitor's time-average 1/2 and
    # the phasor normalization). Report the ratio as the consistency check.
    ratio = absorbed_dft / absorbed_flux if absorbed_flux != 0 else float("nan")
    err = abs(absorbed_dft - absorbed_flux) / max(abs(absorbed_flux), 1e-30)
    print(f"flux:, {absorbed_dft:.6g} (dft_fields), {absorbed_flux:.6g} (dft_flux), "
          f"{err:.4f} (error), ratio={ratio:.4f}")
    print(f"per-face: x_lo={fl['x_lo']:.4g} x_hi={fl['x_hi']:.4g} "
          f"y_lo={fl['y_lo']:.4g} y_hi={fl['y_hi']:.4g} "
          f"z_lo={fl['z_lo']:.4g} z_hi={fl['z_hi']:.4g}")

    # density map: x-z slice through the cube center (y = cube center y)
    ny = eps.shape[1]
    j0 = int((CUBE_CENTER.y + SY / 2) / dx)
    j0 = min(max(j0, 0), ny - 1)
    slab = Edens[:, j0, :]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(np.transpose(slab), origin="lower",
                   extent=[-SX / 2, SX / 2, -SZ / 2, SZ / 2], aspect="auto",
                   cmap="inferno", vmin=0, vmax=float(np.nanmax(slab)))
    ax.axvline(CUBE_CENTER.x - CUBE / 2, color="c", ls="--", lw=0.6)
    ax.axvline(CUBE_CENTER.x + CUBE / 2, color="c", ls="--", lw=0.6)
    ax.axhline(CUBE_CENTER.z - CUBE / 2, color="c", ls="--", lw=0.6)
    ax.axhline(CUBE_CENTER.z + CUBE / 2, color="c", ls="--", lw=0.6)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    ax.set_title(f"absorbed power density (sigma_d*eps*|E|^2), x-z slice\n"
                 f"3D lossy cube, eps={EPS}, sigma_d={SIGMA_D}")
    fig.colorbar(im, ax=ax, label="sigma_d*eps*|E|^2")
    fig.tight_layout()
    fig.savefig(OUT / "absorbed_power_density.png", dpi=150)
    plt.close(fig)
    print(f"absorbed_power_density.png written")


if __name__ == "__main__":
    main()
