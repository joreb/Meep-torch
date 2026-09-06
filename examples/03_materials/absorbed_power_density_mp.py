# Absorbed power density in a lossy dielectric, from the Meep tutorial.
#
# meep-torch port: GEOMETRY CHANGED. Upstream is 2D TM (Ez cylinder); meep-torch
# 3D shim supports only axis-aligned Blocks, so the cylinder becomes a cube.
# Physics validation is unchanged: absorbed power from volumetric sigma_d*eps*|E|^2
# balances net inward Poynting flux through a closed box. Output:
# `output/absorbed_power_density_mp.png`.
#
# Run: uv run python examples/03_materials/absorbed_power_density_mp.py

import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import meep_torch.mp as mp
from meep_torch.mp import Block, FluxRegion, GaussianSource, Medium, PML, Simulation, Source, Vector3

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")

# Parameters (reduced resolution for fast 3D smoke run)
RES = 10                  # pixels/um
SX = SY = 6.0
SZ = 8.0
DPML = 0.5
CUBE = 2.0                # lossy cube edge (um)
EPS = 2.25                # n = 1.5
SIGMA_D = 2.0             # D_conductivity -> Im(eps) = sigma_d / (2*pi*f)
FCEN, FWIDTH = 1.0, 0.2
SRC_Z = -3.0
CUBE_CENTER = Vector3(0, 0, 1.0)
BOX = 1.1                 # flux-box face offset (just outside cube)
T_END = 14.0


def _sim():
    return Simulation(
        cell_size=Vector3(SX, SY, SZ), resolution=RES, dimensions=3,
        sources=[Source(src=GaussianSource(frequency=FCEN, fwidth=FWIDTH),
                        component=mp.Ex, center=Vector3(0, 0, SRC_Z),
                        size=Vector3(mp.inf, mp.inf, 0))],
        geometry=[Block(material=Medium(epsilon=EPS, D_conductivity=SIGMA_D),
                        center=CUBE_CENTER, size=Vector3(CUBE, CUBE, CUBE))],
        boundary_layers=[PML(thickness=DPML)],
        device=DEVICE)


def _box_faces():
    c = CUBE_CENTER
    h = BOX
    return [
        ("x_lo", FluxRegion(center=Vector3(c.x - h, c.y, c.z), size=Vector3(0, 2 * h, 2 * h))),
        ("x_hi", FluxRegion(center=Vector3(c.x + h, c.y, c.z), size=Vector3(0, 2 * h, 2 * h))),
        ("y_lo", FluxRegion(center=Vector3(c.x, c.y - h, c.z), size=Vector3(2 * h, 0, 2 * h))),
        ("y_hi", FluxRegion(center=Vector3(c.x, c.y + h, c.z), size=Vector3(2 * h, 0, 2 * h))),
        ("z_lo", FluxRegion(center=Vector3(c.x, c.y, c.z - h), size=Vector3(2 * h, 2 * h, 0))),
        ("z_hi", FluxRegion(center=Vector3(c.x, c.y, c.z + h), size=Vector3(2 * h, 2 * h, 0))),
    ]


def main():
    sim = _sim()
    for name, reg in _box_faces():
        sim.add_flux(FCEN, 0, 1, reg, name=name)
    # Ex is driven; Ez scattered. Ey ~ 0 for z-normal Ex planewave.
    sim.add_dft_fields([mp.Ex, mp.Ez], FCEN, 0, 1, name="E")
    sim.run(until=T_END)

    fl = {n: sim.get_fluxes(n)[0] for n, _ in _box_faces()}
    # Inward flux: +flux on low face, -flux on high face
    absorbed_flux = (fl["x_lo"] - fl["x_hi"]
                     + fl["y_lo"] - fl["y_hi"]
                     + fl["z_lo"] - fl["z_hi"])

    eps = sim.get_array(component=mp.Dielectric)
    dx = 1.0 / RES
    dV = dx ** 3
    mask = eps > (EPS + 1.0) / 2.0
    shape = eps.shape
    comps = [sim.get_dft_array("E", c, 0) for c in (mp.Ex, mp.Ez)]

    def _pad(a):
        a = np.squeeze(a)
        pads = [(0, max(0, shape[i] - a.shape[i])) for i in range(3)]
        return np.pad(a, pads)

    Edens = sum(np.abs(_pad(c)) ** 2 for c in comps)
    # D_conductivity damps D: P = sigma_d * eps * |E|^2
    absorbed_dft = float(SIGMA_D * EPS * np.sum(Edens[mask]) * dV)

    ratio = absorbed_dft / absorbed_flux if absorbed_flux != 0 else float("nan")
    err = abs(absorbed_dft - absorbed_flux) / max(abs(absorbed_flux), 1e-30)
    print(f"flux:, {absorbed_dft:.6g} (dft_fields), {absorbed_flux:.6g} (dft_flux), "
          f"{err:.4f} (error), ratio={ratio:.4f}")
    print(f"per-face: x_lo={fl['x_lo']:.4g} x_hi={fl['x_hi']:.4g} "
          f"y_lo={fl['y_lo']:.4g} y_hi={fl['y_hi']:.4g} "
          f"z_lo={fl['z_lo']:.4g} z_hi={fl['z_hi']:.4g}")

    # Density map: x-z slice through cube center
    ny = eps.shape[1]
    j0 = int((CUBE_CENTER.y + SY / 2) / dx)
    j0 = min(max(j0, 0), ny - 1)
    slab = Edens[:, j0, :]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(np.transpose(slab), origin="lower",
                   extent=[-SX / 2, SX / 2, -SZ / 2, SZ / 2], aspect="auto",
                   cmap="inferno", vmin=0, vmax=float(np.nanmax(slab)))
    ax.axvline(CUBE_CENTER.x - CUBE / 2, color="c", ls="--", lw=0.8)
    ax.axvline(CUBE_CENTER.x + CUBE / 2, color="c", ls="--", lw=0.8)
    ax.axhline(CUBE_CENTER.z - CUBE / 2, color="c", ls="--", lw=0.8)
    ax.axhline(CUBE_CENTER.z + CUBE / 2, color="c", ls="--", lw=0.8)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    ax.set_title(f"absorbed power density (sigma_d*eps*|E|^2), x-z slice\n"
                 f"3D lossy cube, eps={EPS}, sigma_d={SIGMA_D}")
    fig.colorbar(im, ax=ax, label="sigma_d*eps*|E|^2")
    fig.tight_layout()
    fig.savefig(OUT / "absorbed_power_density_mp.png", dpi=150)
    plt.close(fig)
    print(f"absorbed_power_density_mp.png written  [{DEVICE}]")


if __name__ == "__main__":
    main()
