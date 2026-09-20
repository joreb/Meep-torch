# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/polarization_grating.py by scripts/port_upstream_example.py.
# The body below is upstream's, unchanged but for the meep import and the generated ModeSolver lines, so that a difference between
# this file and its twin is a difference between the ENGINES. See that
# script's docstring for why nothing here rewrites a call site.
#
# The literal spellings below (`matplotlib.use("Agg")`, `OUT = ...`,
# `OUT.mkdir`, `DEVICE = os.environ.get("MEEP_TORCH_DEVICE")`) are the ones
# examples/example_template.py greps for. An earlier version used private
# `_mpl` / `_OUT` names, which satisfied the RULES and failed the CHECKER on
# 216 counts -- two dialects for one convention, and the checker was right to
# complain: a rule nothing can verify is a comment.
import os
import pathlib

import matplotlib
import numpy as _np

matplotlib.use("Agg")                # rule 1: headless, before any pyplot import
for _old, _new in (("trapz", "trapezoid"), ("complex_", "complex128"),
                   ("float_", "float64"), ("alltrue", "all"),
                   ("sometrue", "any"), ("product", "prod"),
                   ("cumproduct", "cumprod"), ("round_", "round")):
    if not hasattr(_np, _old) and hasattr(_np, _new):
        setattr(_np, _old, getattr(_np, _new))   # upstream predates numpy 2.0
# rule 2. parents=True because output/ itself may not exist yet -- plain
# exist_ok=True raises FileNotFoundError on the first run of a fresh clone.
OUT = pathlib.Path(__file__).parent / "output/polarization_grating/"
OUT.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402  (device probe only)
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

# polarization grating from C. Oh and M.J. Escuti, Optics Letters, Vol. 33, No. 20, pp. 2287-9, 2008
# note: reference uses z as the propagation direction and y as the out-of-plane direction; this script uses x and z, respectively
import math

import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp

resolution = 50  # pixels/μm

dpml = 1.0  # PML thickness
dsub = 1.0  # substrate thickness
dpad = 1.0  # padding thickness

k_point = mp.Vector3(0, 0, 0)

pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

n_0 = 1.55
delta_n = 0.159
epsilon_diag = mp.Matrix(
    mp.Vector3(n_0**2, 0, 0),
    mp.Vector3(0, n_0**2, 0),
    mp.Vector3(0, 0, (n_0 + delta_n) ** 2),
)

wvl = 0.54  # center wavelength
fcen = 1 / wvl  # center frequency


def pol_grating(d, ph, gp, nmode):
    sx = dpml + dsub + d + d + dpad + dpml
    sy = gp

    cell_size = mp.Vector3(sx, sy, 0)

    # twist angle of nematic director; from equation 1b
    def phi(p):
        xx = p.x - (-0.5 * sx + dpml + dsub)
        if (xx >= 0) and (xx <= d):
            return math.pi * p.y / gp + ph * xx / d
        else:
            return math.pi * p.y / gp - ph * xx / d + 2 * ph

    # return the anisotropic permittivity tensor for a uniaxial, twisted nematic liquid crystal
    def lc_mat(p):
        # rotation matrix for rotation around x axis
        Rx = mp.Matrix(
            mp.Vector3(1, 0, 0),
            mp.Vector3(0, math.cos(phi(p)), math.sin(phi(p))),
            mp.Vector3(0, -math.sin(phi(p)), math.cos(phi(p))),
        )
        lc_epsilon = Rx * epsilon_diag * Rx.transpose()
        lc_epsilon_diag = mp.Vector3(lc_epsilon[0].x, lc_epsilon[1].y, lc_epsilon[2].z)
        lc_epsilon_offdiag = mp.Vector3(
            lc_epsilon[1].x, lc_epsilon[2].x, lc_epsilon[2].y
        )
        return mp.Medium(
            epsilon_diag=lc_epsilon_diag, epsilon_offdiag=lc_epsilon_offdiag
        )

    geometry = [
        mp.Block(
            center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dsub)),
            size=mp.Vector3(dpml + dsub, mp.inf, mp.inf),
            material=mp.Medium(index=n_0),
        ),
        mp.Block(
            center=mp.Vector3(-0.5 * sx + dpml + dsub + d),
            size=mp.Vector3(2 * d, mp.inf, mp.inf),
            material=lc_mat,
        ),
    ]

    # linear-polarized planewave pulse source
    src_pt = mp.Vector3(-0.5 * sx + dpml + 0.3 * dsub, 0, 0)
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=0.05 * fcen),
            component=mp.Ez,
            center=src_pt,
            size=mp.Vector3(0, sy, 0),
        ),
        mp.Source(
            mp.GaussianSource(fcen, fwidth=0.05 * fcen),
            component=mp.Ey,
            center=src_pt,
            size=mp.Vector3(0, sy, 0),
        ),
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        k_point=k_point,
        sources=sources,
        default_material=mp.Medium(index=n_0),
    )

    tran_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad, 0, 0)
    tran_flux = sim.add_flux(
        fcen, 0, 1, mp.FluxRegion(center=tran_pt, size=mp.Vector3(0, sy, 0))
    )

    sim.run(until_after_sources=100)

    input_flux = mp.get_fluxes(tran_flux)

    sim.reset_meep()

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        k_point=k_point,
        sources=sources,
        geometry=geometry,
    )

    tran_flux = sim.add_flux(
        fcen, 0, 1, mp.FluxRegion(center=tran_pt, size=mp.Vector3(0, sy, 0))
    )

    sim.run(until_after_sources=300)

    res1 = sim.get_eigenmode_coefficients(
        tran_flux, range(1, nmode + 1), eig_parity=mp.ODD_Z + mp.EVEN_Y
    )
    res2 = sim.get_eigenmode_coefficients(
        tran_flux, range(1, nmode + 1), eig_parity=mp.EVEN_Z + mp.ODD_Y
    )
    angles = [math.degrees(math.acos(kdom.x / fcen)) for kdom in res1.kdom]

    return input_flux[0], angles, res1.alpha[:, 0, 0], res2.alpha[:, 0, 0]


ph_uniaxial = 0  # chiral layer twist angle for uniaxial grating
ph_twisted = 70  # chiral layer twist angle for bilayer grating
gp = 6.5  # grating period
nmode = 5  # number of mode coefficients to compute
dd = np.arange(0.1, 3.5, 0.1)  # chiral layer thickness

m0_uniaxial = np.zeros(dd.size)
m1_uniaxial = np.zeros(dd.size)
ang_uniaxial = np.zeros(dd.size)

m0_twisted = np.zeros(dd.size)
m1_twisted = np.zeros(dd.size)
ang_twisted = np.zeros(dd.size)

for k in range(len(dd)):
    input_flux, angles, coeffs1, coeffs2 = pol_grating(
        0.5 * dd[k], math.radians(ph_uniaxial), gp, nmode
    )
    tran = (abs(coeffs1) ** 2 + abs(coeffs2) ** 2) / input_flux
    for m in range(nmode):
        print(f"tran (uniaxial):, {m}, {angles[m]:.2f}, {tran[m]:.5f}")
    m0_uniaxial[k] = tran[0]
    m1_uniaxial[k] = tran[1]
    ang_uniaxial[k] = angles[1]

    input_flux, angles, coeffs1, coeffs2 = pol_grating(
        dd[k], math.radians(ph_twisted), gp, nmode
    )
    tran = (abs(coeffs1) ** 2 + abs(coeffs2) ** 2) / input_flux
    for m in range(nmode):
        print(f"tran (twisted):, {m}, {angles[m]:.2f}, {tran[m]:.5f}")
    m0_twisted[k] = tran[0]
    m1_twisted[k] = tran[1]
    ang_twisted[k] = angles[1]


cos_angles = [math.cos(math.radians(t)) for t in ang_uniaxial]
tran = m0_uniaxial + 2 * m1_uniaxial
eff_m0 = m0_uniaxial / tran
eff_m1 = (2 * m1_uniaxial / tran) / cos_angles

phase = delta_n * dd / wvl
eff_m0_analytic = [math.cos(math.pi * p) ** 2 for p in phase]
eff_m1_analytic = [math.sin(math.pi * p) ** 2 for p in phase]

plt.figure(dpi=150)
plt.subplot(1, 2, 1)
plt.plot(phase, eff_m0, "bo-", clip_on=False, label="0th order (meep)")
plt.plot(phase, eff_m0_analytic, "b--", clip_on=False, label="0th order (analytic)")
plt.plot(phase, eff_m1, "ro-", clip_on=False, label="±1 orders (meep)")
plt.plot(phase, eff_m1_analytic, "r--", clip_on=False, label="±1 orders (analytic)")
plt.axis([0, 1.0, 0, 1])
plt.xticks(list(np.arange(0, 1.2, 0.2)))
plt.xlabel("phase delay Δnd/λ")
plt.ylabel("diffraction efficiency @ λ = 0.54 μm")
plt.legend(loc="center")
plt.title("homogeneous uniaxial grating")

cos_angles = [math.cos(math.radians(t)) for t in ang_twisted]
tran = m0_twisted + 2 * m1_twisted
eff_m0 = m0_twisted / tran
eff_m1 = (2 * m1_twisted / tran) / cos_angles

plt.subplot(1, 2, 2)
plt.plot(phase, eff_m0, "bo-", clip_on=False, label="0th order (meep)")
plt.plot(phase, eff_m1, "ro-", clip_on=False, label="±1 orders (meep)")
plt.axis([0, 1.0, 0, 1])
plt.xticks(list(np.arange(0, 1.2, 0.2)))
plt.xlabel("phase delay Δnd/λ")
plt.ylabel("diffraction efficiency @ λ = 0.54 μm")
plt.legend(loc="center")
plt.title("bilayer twisted-nematic grating")

plt.show()
