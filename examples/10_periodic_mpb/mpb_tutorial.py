import math
import os
import pathlib

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar, ridder


from meep_torch import mpb

from _mpb_plots import (band_figure, epsilon_figure, field_figure,
                        save)

import meep_torch.mp as mp
import torch


OUT = pathlib.Path(__file__).parent / "output/mpb_tutorial"
OUT.mkdir(exist_ok=True)

# Rule 3: say which device produced the number, read from the environment
# and NOT detected behind the reader's back. The previous version fell back
# to `"cuda" if torch.cuda.is_available() else "cpu"`, which printed
# "[cuda]" on this machine for a run that was entirely on the CPU -- the
# exact misreport the rule exists to prevent.
#
# It is CPU here whatever the environment says, and that is worth stating
# rather than papering over: every section of this tutorial is a 2D
# lattice, and ModeSolver.device is read only by the 3D planewave backend.
# A 2D solver handed device="cuda" warns and runs on the CPU.
_REQUESTED = os.environ.get("MEEP_TORCH_DEVICE", "cpu")
DEVICE = "cpu"                     # 2D mpb has no device path; see above
if _REQUESTED != "cpu":
    print(f"note: MEEP_TORCH_DEVICE={_REQUESTED!r} is ignored here -- every "
          f"section of this tutorial is 2D, and only the 3D planewave "
          f"backend has a device path.")

def print_heading(h):
    stars = "*" * 10
    print("{0} {1} {0}".format(stars, h))


# Gamma-X-M-Gamma is the path every square-lattice section here walks, and
# the triangular section reuses it (upstream overwrites its own triangular
# k_points immediately after setting them -- quirk preserved).
GXMG = ["$\\Gamma$", "X", "M", "$\\Gamma$"]


# Our First Band Structure

print_heading("Square lattice of rods in air")

num_bands = 8
k_points = [
    mp.Vector3(),  # Gamma
    mp.Vector3(0.5),  # X
    mp.Vector3(0.5, 0.5),  # M
    mp.Vector3(),
]  # Gamma

k_points = mp.interpolate(4, k_points)
geometry = [mp.Cylinder(0.2, material=mp.Medium(epsilon=12))]
geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
resolution = 32

ms = mpb.ModeSolver(
    num_bands=num_bands,
    k_points=k_points,
    geometry=geometry,
    geometry_lattice=geometry_lattice,
    resolution=resolution,
)
# meep writes h5 output to <prefix>-...; point it at output/ instead of cwd.
ms.filename_prefix = str(OUT / "mpb_tutorial")

save(*epsilon_figure(ms, "Square lattice of rods: $\\varepsilon$"), OUT, "eps_sq_rods")

print_heading("Square lattice of rods: TE bands")
ms.run_te()
save(*band_figure(ms, "Square lattice of rods: TE bands", GXMG), OUT, "sq_rods_te")

print_heading("Square lattice of rods: TM bands")
ms.run_tm()
save(*band_figure(ms, "Square lattice of rods: TM bands", GXMG), OUT, "sq_rods_tm")

print_heading("Square lattice of rods: TM, w/efield")
ms.run_tm(mpb.output_efield_z)

print_heading("Square lattice of rods: TE, w/hfield & dpwr")
ms.run_te(mpb.output_at_kpoint(mp.Vector3(0.5), mpb.output_hfield_z, mpb.output_dpwr))

# Bands of a Triangular Lattice

print_heading("Triangular lattice of rods in air")

ms.geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, 1),
    basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
    basis2=mp.Vector3(math.sqrt(3) / 2, -0.5),
)

ms.k_points = [
    mp.Vector3(),  # Gamma
    mp.Vector3(y=0.5),  # M
    mp.Vector3(-1 / 3, 1 / 3),  # K
    mp.Vector3(),
]  # Gamma

ms.k_points = mp.interpolate(4, k_points)
ms.run_tm()
save(*band_figure(ms, "Triangular lattice of rods: TM bands", GXMG), OUT, "tri_rods_tm")
# rectified and tiled: the raw array is indexed in lattice coordinates, so
# without MPBData the rhombic cell is drawn sheared into a square
save(*epsilon_figure(ms, "Triangular lattice of rods: $\\varepsilon$ (rectified, 3x3)", rectify=True, periods=3), OUT, "eps_tri_rods")

# Maximizing the First TM Gap

print_heading("Maximizing the first TM gap")


gap_sweep = []          # (radius, gap %) at every radius the optimiser tries


def first_tm_gap(r):
    ms.geometry = [mp.Cylinder(r, material=mp.Medium(epsilon=12))]
    ms.run_tm()
    gap = ms.retrieve_gap(1)
    gap_sweep.append((float(r), float(gap)))
    return -1 * gap


ms.num_bands = 2
ms.mesh_size = 7

result = minimize_scalar(
    first_tm_gap, method="bounded", bounds=[0.1, 0.5], options={"xatol": 0.1}
)
print(f"radius at maximum: {result.x}")
print(f"gap size at maximum: {result.fun * -1}")

# The optimiser is a bounded scalar search, so the radii it visits are not
# on a grid -- sort them before drawing a line through them.
sweep = np.array(sorted(gap_sweep))
plt.figure(figsize=(5.5, 4), dpi=150)
plt.plot(sweep[:, 0], sweep[:, 1], "o-", color="tab:blue")
plt.plot(result.x, -result.fun, "r*", ms=14, label="maximum")
plt.xlabel("rod radius  (a)")
plt.ylabel("first TM gap  (%)")
plt.title(f"First TM gap vs radius\nmax {-result.fun:.1f}% at r = {result.x:.3f}")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "tm_gap_vs_radius.png", bbox_inches="tight")
plt.close()
np.savetxt(OUT / "tm_gap_vs_radius.csv", sweep, delimiter=",",
           header="radius,gap_percent", comments="")

ms.mesh_size = 3  # Reset to default value of 3

# A Complete 2D Gap with an Anisotropic Dielectric

print_heading("Anisotropic complete 2d gap")

ms.geometry = [mp.Cylinder(0.3, material=mp.Medium(epsilon_diag=mp.Vector3(1, 1, 12)))]

ms.default_material = mp.Medium(epsilon_diag=mp.Vector3(12, 12, 1))
ms.num_bands = 8
ms.run()  # just use run, instead of run_te or run_tm, to find the complete gap
# no-parity, so the bands are the union of TM and TE and a gap here is a
# COMPLETE 2D gap -- which is the point of the anisotropic structure
save(*band_figure(ms, "Anisotropic rods: complete 2D gap (no parity)", GXMG), OUT, "aniso_complete_gap")
# eps_diag (1,1,12) rods in a (12,12,1) background: exx == eyy exactly and
# corr(exx, ezz) = -1.0000, so TM and TE see inverted structures -- which
# is how their gaps can be made to overlap into a COMPLETE gap
save(*epsilon_figure(ms, "Anisotropic rods: the epsilon tensor",
                     rectify=True, periods=3), OUT, "eps_aniso")

# Finding a Point-defect State

print_heading("5x5 point defect")

ms.geometry_lattice = mp.Lattice(size=mp.Vector3(5, 5))
ms.geometry = [mp.Cylinder(0.2, material=mp.Medium(epsilon=12))]

ms.geometry = mp.geometric_objects_lattice_duplicates(ms.geometry_lattice, ms.geometry)
ms.geometry.append(mp.Cylinder(0.2, material=mp.air))

ms.resolution = 16
ms.k_points = [mp.Vector3(0.5, 0.5)]

ms.num_bands = 50
ms.run_tm()

mpb.output_efield_z(ms, 25)
save(*field_figure(ms, 25, "5x5 point defect: $E_z$ of band 25"), OUT, "defect_ez_band25")
save(*epsilon_figure(ms, "5x5 supercell with one rod removed"), OUT, "eps_defect")

ms.get_dfield(25)  # compute the D field for band 25
ms.compute_field_energy()  # compute the energy density from D
c = mp.Cylinder(1.0, material=mp.air)
print(f"energy in cylinder: {ms.compute_energy_in_objects([c])}")

print_heading("5x5 point defect, targeted solver")

ms.num_bands = 1  # only need to compute a single band, now!
ms.target_freq = (0.2812 + 0.4174) / 2
ms.tolerance = 1e-8
ms.run_tm()

# Tuning the Point-defect Mode

print_heading("Tuning the 5x5 point defect")

old_geometry = ms.geometry  # save the 5x5 grid with a missing rod


def rootfun(eps):
    # add the cylinder of epsilon = eps to the old geometry:
    ms.geometry = old_geometry + [mp.Cylinder(0.2, material=mp.Medium(epsilon=eps))]
    ms.run_tm()  # solve for the mode (using the targeted solver)
    print(f"epsilon = {eps} gives freq. =  {ms.get_freqs()[0]}")
    return ms.get_freqs()[0] - 0.314159  # return 1st band freq. - 0.314159


rooteps = ridder(rootfun, 1, 12)
print(f"root (value of epsilon) is at: {rooteps}")

rootval = rootfun(rooteps)
print(f"root function at {rooteps} = {rootval}")

print(f"mpb_tutorial  [{DEVICE}]  backend={ms.active_backend}  "
      f"defect-tuned eps = {rooteps:.6f} -> f = {ms.get_freqs()[0]:.6f}")
