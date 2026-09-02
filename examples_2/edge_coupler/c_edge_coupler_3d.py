# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Edge coupler 3D validation
#
# device-suite (c1, 3D): true-3D validation of the 674 nm edge coupler.
#
# The 2D optimization collapsed the 150 nm SiN film to a lateral effective index
# (NEFF = 1.6321) and used the beam's 4.0 um LATERAL waist only. This script drops
# both approximations: the design is extruded into a real 150 nm slab with the
# REAL SiN index (n = 2.0347, not NEFF), and the input is the TRUE astigmatic
# laser-diode beam -- 4.0 um lateral x 1.57 um vertical waists.
#
# WHY NOT ModePlaneSourceX: that source solves the local cross-section and injects
# its GUIDED mode, which would answer the wrong question. The whole point of an
# edge coupler is how much FREE-SPACE beam couples into the guide, so the source
# here is a free Gaussian with the diode's two waists, launched in the cladding
# ahead of the facet. PlaneSource only carries a uniform mask, so the profile is
# written onto its mask directly.
#
# RESOLUTION IS SET BY THE FILM, NOT THE WAVELENGTH. At 674 nm in SiN the
# in-medium wavelength is 331 nm, so even RES=20 gives 6.6 cells per wavelength --
# but only 3.0 cells across the 150 nm core, which cannot represent the slab mode.
# RES=40 gives 6 cells across the core (13.3 per wavelength).
#
# REPORTS TWO DIFFERENT THINGS, and conflating them is the trap:
#
#   eta_3d   = P_guided(device) / P_guided(bare 4 um straight)   -- a RATIO
#   eta_abs  = P_guided / P_launched                              -- an EFFICIENCY
#
# eta_3d can exceed 1 and that is unremarkable: the bare facet in the denominator
# is itself lossy (a Gaussian beam and the guide's TE0 mode are different field
# profiles), so beating it by a few percent takes no extra energy. It is the
# direct analogue of the 2D eta_rel = 1.027x butt.
#
# eta_abs is conservation-enforced and MUST be below 1. It comes from a third run
# -- the same beam into homogeneous cladding, integrated over the full non-PML
# cross-section -- which measures the launched power. Both absolutes are asserted
# < 1. This exists because the 2D script's first attempt at an absolute built it
# by multiplying a mode-overlap ratio by a flux ratio and printed 100.28%: two
# individually scale-free quantities measured by different instruments do not
# compose into a valid efficiency.
#
# Run after c_edge_coupler.py:
# `uv run python examples_2/edge_coupler/c_edge_coupler_3d.py`
# (outputs land in `examples_2/edge_coupler/output/`)

# %%
import math
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.core.monitors_3d import FluxMonitor3D, PlaneSource
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d_lean import build_lean, run_3d_lean
from meep_torch.ports.validate3d import extrude

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
DEV = os.environ.get("CEC3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Parameters
# Real material indices (not the 2D NEFF), the two beam waists, and the
# padding/resolution/dtype trade-offs -- PAD_Z must hold the vertical waist in
# non-PML space, and RES=40 is set by the 150 nm core, not the wavelength.

# %%
WL = 0.674
FCEN = 1.0 / WL
N_SIN = float(Si3N4.n(FCEN).real)        # 2.0347 -- REAL index, not the 2D NEFF
N_CLAD = float(SiO2.n(FCEN).real)        # 1.4652
W_WG = 4.0                               # output guide width
W_BEAM_LAT = 4.0                         # diode lateral waist
W_BEAM_VER = 1.57                        # diode vertical waist (astigmatic)
H_CORE = 0.150
# PAD_Z holds the 1.57 um vertical beam waist above/below the film before the
# PML. 1.0 um is +-1.27 waists, which contains the beam while keeping the grid
# affordable: PAD_Z=1.5 costs 6.6 GB and OOMs beside a concurrent 3D run,
# PAD_Z=1.0 costs 4.5 GB. Reducing RES instead would have been the wrong trade
# -- it is the 150 nm core resolution, not the padding, that sets validity.
# PAD_Z must hold the 1.57 um VERTICAL beam waist in NON-PML space. This is the
# number I got wrong first time: at PAD_Z=1.0 with 1 um of PML each side, only
# 15% of the beam power is launched outside the absorber -- the run then
# measures a beam that is not the specified one. PAD_Z=2.0 contains 99%.
PAD_Z = float(os.environ.get("CEC3_PADZ", "2.0"))
# The 3D domain is cropped laterally/longitudinally to pay for that z padding:
# the 2D pad=3.0 was sized for the 2D run, and 20 x 11.5 um is far more x/y than
# a 4 um beam into a 4 um guide needs. 16 x 8 um still contains >99% of the
# lateral beam outside the PML. RES stays 40 -- the 150 nm core resolution is
# the one thing that must not be traded.
PAD_XY = float(os.environ.get("CEC3_PADXY", "1.2"))
RES3 = int(os.environ.get("CEC3_RES", "40"))     # 6 cells across the core
# float32 by default: this is a FORWARD-ONLY validation (no adjoint), it halves
# both memory and step time, and the reported number is a RATIO of two runs on
# the same grid, so the float32 rounding largely cancels. float64 needs ~9.5 GB
# here, which does not fit beside the peer's concurrent 3D run on the GPU
# (measured: one OOM). CEC3_F64=1 forces float64.
DT3 = torch.float64 if os.environ.get("CEC3_F64") == "1" else torch.float32
dx3 = 1.0 / RES3
NPML = RES3

# %% [markdown]
# ## Rebuild the 2D design
# Reconstruct the optimized 2D permittivity from the saved rho and the same
# gds geometry (cheaper than importing the 2D script, which would run its
# validation), plus a bare-straight 2D reference.

# %%
# --- rebuild the 2D design, then extrude ------------------------------------
# Rebuild directly (cheap, and avoids running the 2D validation on import).
import gdsfactory as gf                                    # noqa: E402
from meep_torch.plugins.gdsfactory import component_to_eps                # noqa: E402
from meep_torch.adjoint import DesignRegion, MaterialGrid  # noqa: E402
from meep_torch.modes.fd1d import solve_1d                 # noqa: E402

gf.gpdk.PDK.activate()
_eps_line = np.full(400, N_CLAD ** 2)
_eps_line[200 - 7: 200 + 8] = N_SIN ** 2
NEFF2D = float(solve_1d(_eps_line, 0.01, FCEN, "te", k=1).neff[0])
RES2 = 20
dx2 = 1.0 / RES2
xs = gf.get_cross_section("strip").copy(width=W_WG)
tap = gf.components.taper(length=10.0, width1=5.5, width2=W_WG, cross_section=xs)
comp = gf.Component("edge_coupler_gds_3d")
_ref = comp.add_ref(tap)
_st = comp.add_ref(gf.components.straight(length=4.0, cross_section=xs))
_st.connect("o1", _ref.ports["o2"])
comp.add_port("o1", port=_ref.ports["o1"])
comp.add_port("o2", port=_st.ports["o2"])
comp.flatten()
grid2, eps_base, ports2 = component_to_eps(comp, res=RES2, pad=PAD_XY,
                                           n_core=NEFF2D, n_clad=N_CLAD)
o1x, o1y, _ = ports2["o1"]
o2x, o2y, _ = ports2["o2"]
cy2 = o1y
_h = int(W_WG * RES2 / 2)
eps_base[o2x:, cy2 - _h:cy2 + _h] = NEFF2D ** 2       # guide into the PML
ix0, ix1 = o1x, o2x - int(4.0 * RES2)
iy0, iy1 = cy2 - int(3.0 * RES2), cy2 + int(3.0 * RES2)
mg = MaterialGrid((100, 60), eps_min=N_CLAD ** 2, eps_max=NEFF2D ** 2)
region = DesignRegion(mg, ix0=ix0, ix1=ix1, iy0=iy0, iy1=iy1)
rho = torch.from_numpy(np.load(OUT / "c_edge_coupler_rho.npy")).to(eps_base.dtype)
r2 = tanh_projection(conic_filter(rho.clamp(0, 1), 0.15, dx2), 64.0)
eps2d = region.paint(eps_base, r2).detach().cpu()
# reference: bare 4 um straight spanning the full length, same grid
eps2d_ref = torch.full_like(eps2d, N_CLAD ** 2)
eps2d_ref[:, cy2 - _h:cy2 + _h] = NEFF2D ** 2


# %% [markdown]
# ## Extrude to 3D and place the source
# Upsample to the RES=40 lateral grid, extrude a real 150 nm slab, and place
# the source in non-PML cladding ahead of the facet. The containment guard
# asserts the beam is actually launched outside the absorber on both axes.

# %%
def to3d(e2):
    """Upsample the 2D map to the 3D lateral grid, then extrude a real slab."""
    n3x = int(round(e2.shape[0] * RES3 / RES2))
    n3y = int(round(e2.shape[1] * RES3 / RES2))
    ds = F.interpolate(e2[None, None], size=(n3x, n3y), mode="bilinear",
                       align_corners=False)[0, 0]
    return extrude(ds, n_clad=N_CLAD, n_core=N_SIN, h_core=H_CORE,
                   pad_z=PAD_Z, res=RES3).to(dtype=DT3, device=DEV)


eps3 = to3d(eps2d)
eps3_ref = to3d(eps2d_ref)
nx, ny, nz = eps3.shape
CY, CZ = int(round(cy2 * RES3 / RES2)), nz // 2
# with the tighter pad the facet sits closer to the edge; keep the source in
# non-PML cladding but still ahead of the facet.
IX_SRC = max(int(round((o1x - 0.8 * RES2) * RES3 / RES2)), NPML + 6)
IX_OUT = min(int(round((o2x - 2.0 * RES2) * RES3 / RES2)), nx - NPML - 4)
assert IX_SRC > NPML, "source inside the PML"
# GUARD: the beam must actually be launched into non-PML space. A Gaussian
# whose tails sit in the absorber is not the beam we specified, and the run
# would silently measure something else (first attempt: PAD_Z=1.0 put 85% of
# the vertical beam power inside the PML). Check both transverse axes.
def _contained(waist, extent_cells, center_cell):
    u = (np.arange(extent_cells) + 0.5) * dx3
    inten = np.exp(-2 * ((u - center_cell * dx3) / (waist / 2)) ** 2)
    return inten[NPML:extent_cells - NPML].sum() / inten.sum()


_fy = _contained(W_BEAM_LAT, ny, CY)
_fz = _contained(W_BEAM_VER, nz, CZ)
print(f"c_edge_coupler_3d  beam containment outside PML: "
      f"lateral {_fy*100:.1f}%, vertical {_fz*100:.1f}%")
assert _fy > 0.95 and _fz > 0.95, (
    f"beam is being launched into the PML (lateral {_fy:.1%}, vertical "
    f"{_fz:.1%}) -- raise PAD_Z/PAD_XY; the run would measure a clipped beam")

T_RUN = 2 * (5 / 0.2) + nx * dx3 * 1.7 + 40.0
print(f"c_edge_coupler_3d  grid {nx}x{ny}x{nz} "
      f"({nx*dx3:.1f} x {ny*dx3:.1f} x {nz*dx3:.1f} um) on {DEV}")
print(f"c_edge_coupler_3d  SiN n={N_SIN:.4f} (2D used NEFF={NEFF2D:.4f}), "
      f"core {H_CORE*1e3:.0f} nm = {H_CORE*RES3:.0f} cells, "
      f"{WL/N_SIN*RES3:.1f} cells per in-medium wavelength, T_RUN={T_RUN:.0f}, "
      f"dtype={'float64' if DT3==torch.float64 else 'float32'}")


# %% [markdown]
# ## Astigmatic source and forward runs
# A free-space Gaussian with the diode's two waists written onto a plane
# source mask, and a lean-stepper run helper whose flux plane is apertured to
# the guide (or the full non-PML cross-section for the launched-power run).

# %%
def astigmatic_source(dt):
    """Free-space Gaussian with the diode's two waists, on an x-normal plane.

    Ey (lab) is the in-plane transverse component, matching the quasi-TE guided
    mode the 2D model represented. PlaneSource builds a uniform mask; overwrite
    it with the astigmatic profile.
    """
    src = PlaneSource(nx, ny, nz, dt, "Ey", 0, IX_SRC,
                      GaussianPulse(FCEN, 0.2 * 1.55 / WL))
    comp_name, mask = src.src
    shp = mask.shape                                   # (nx, ny-1, nz) for Ey
    ys = (torch.arange(shp[1], dtype=mask.dtype) + 0.5) * dx3
    zs = (torch.arange(shp[2], dtype=mask.dtype) + 0.5) * dx3
    gy = torch.exp(-((ys - CY * dx3) / (W_BEAM_LAT / 2)) ** 2)
    gz = torch.exp(-((zs - CZ * dx3) / (W_BEAM_VER / 2)) ** 2)
    prof = gy[:, None] * gz[None, :]
    new = torch.zeros_like(mask)
    new[IX_SRC] = prof
    # PlaneSource builds its mask on CPU in float64; the state lives on DEV in
    # DT3, and step_3d does not move it for us.
    src.src = (comp_name, new.to(dtype=DT3, device=DEV))
    return src


def run_case(e3, label, full_aperture=False):
    """Forward-only 3D run; returns power through the measurement plane.

    ``full_aperture=False`` (default) apertures the plane to the guide, i.e.
    GUIDED power. ``full_aperture=True`` integrates the whole non-PML
    cross-section, which on a homogeneous-cladding run is the total power the
    beam launched -- the denominator for an absolute efficiency.

    Uses the memory-lean forward stepper (in-place fields + shell-only CPML):
    the general step_3d peaked at 12.1 GB on this grid, which fights the peer
    session for the GPU. Lean is forward-only, which is all a validation
    needs, and is parity-gated against step_3d in the core tests.
    """
    c = build_lean(nx, ny, nz, dx3, eps=e3, npml=NPML, dtype=DT3, device=DEV)
    src = astigmatic_source(c.dt)
    # aperture the flux plane to the guide + a decay margin, and keep the PML
    # out of the integral, so stray cladding light is not counted as guided.
    if full_aperture:
        win = (NPML, ny - NPML, NPML, nz - NPML)
    else:
        ap = int(round((W_WG / 2 + 1.0) * RES3))
        win = (CY - ap, CY + ap, NPML, nz - NPML)
    mon = FluxMonitor3D(c.dt, [FCEN], normal="x", idx=IX_OUT, win=win,
                        dtype=DT3)
    # FluxMonitor3D builds its DFT frequency tensor on CPU; the fields it
    # samples live on DEV, so move it or every sample() raises a device
    # mismatch. (Same fix applies to any 3D monitor used on GPU.)
    mon.freqs = mon.freqs.to(device=DEV)
    # the lean stepper uses dt = 0.5*dx/sqrt(3) (3D Courant), NOT the 2D-style
    # dt of build_3d_coeffs, so take the step count from ITS dt.
    n_steps = int(round(T_RUN / c.dt))
    with torch.no_grad():
        _, (acc,) = run_3d_lean(c, n_steps, source=src, monitors=[mon],
                                compile=os.environ.get("CEC3_COMPILE") == "1")
    p = float(mon.flux(acc, dx3)[0])
    kind = "P_total " if full_aperture else "P_guided"
    print(f"c_edge_coupler_3d  {label}: {kind} = {p:.6e}")
    return p


# %% [markdown]
# ## Main: three runs, two honest numbers, and the plot
# Device, bare-straight reference, and beam-into-cladding runs give eta_3d (a
# ratio, may exceed 1) and the conservation-enforced absolutes (asserted < 1),
# then a three-panel figure.

# %%
def main():
    p_dev = run_case(eps3, "device (best 2D design, extruded)")
    p_ref = run_case(eps3_ref, "reference (bare 4 um straight)")
    # Absolute efficiency needs the LAUNCHED power, not the reference's output.
    # Same beam into homogeneous cladding, integrated over the full non-PML
    # cross-section, mirrors how the 2D script gets its P_IN.
    eps3_clad = torch.full_like(eps3, N_CLAD ** 2)
    p_in = run_case(eps3_clad, "input (beam in bare cladding)",
                    full_aperture=True)
    eta = p_dev / p_ref
    eta_abs_dev = p_dev / p_in
    eta_abs_ref = p_ref / p_in
    assert p_ref > 0 and p_in > 0, "reference or input run produced no power"

    # THE RATIO IS NOT AN EFFICIENCY. eta_3d > 1 only says the taper beats a
    # bare facet; the bare facet is itself lossy, so the denominator is small.
    # The conservation-enforced numbers are the two absolutes, and BOTH must be
    # below 1 -- that is what makes eta_3d > 1 unremarkable rather than
    # impossible. (2D learned this the hard way: an eta_abs built by
    # multiplying a mode-overlap ratio by a flux ratio printed 100.28%.)
    print(f"c_edge_coupler_3d  eta_3d = P_dev/P_ref = {eta:.4f}x straight "
          f"({10*math.log10(max(eta,1e-12)):+.2f} dB)  <-- RATIO, not efficiency")
    print(f"c_edge_coupler_3d  ABSOLUTE: device {eta_abs_dev*100:.2f}% of "
          f"launched beam ({-10*math.log10(max(eta_abs_dev,1e-12)):.2f} dB), "
          f"bare straight {eta_abs_ref*100:.2f}%")
    print(f"c_edge_coupler_3d  2D for comparison: eta_rel = 1.027x butt, "
          f"flux bound P_arm/P_IN = 98.87% (RES=20) / 98.60% (RES=40)")
    for nm, v in (("device", eta_abs_dev), ("bare straight", eta_abs_ref)):
        assert v < 1.0, (
            f"{nm} absolute efficiency {v:.4f} >= 1 -- unphysical; the guided "
            f"power exceeds the launched beam power, so a monitor window or "
            f"the input normalization is wrong")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    axs[0].imshow(eps3[:, :, CZ].cpu().numpy().T, origin="lower", cmap="viridis",
                  extent=[0, nx * dx3, 0, ny * dx3], aspect="auto")
    axs[0].set_title(f"device, z-midplane (SiN n={N_SIN:.3f})")
    axs[1].imshow(eps3[:, CY, :].cpu().numpy().T, origin="lower", cmap="viridis",
                  extent=[0, nx * dx3, 0, nz * dx3], aspect="auto")
    axs[1].set_title(f"device, y-midplane ({H_CORE*1e3:.0f} nm core)")
    ysx = (np.arange(ny) + 0.5) * dx3
    zsx = (np.arange(nz) + 0.5) * dx3
    beam = (np.exp(-((ysx - CY * dx3) / (W_BEAM_LAT / 2)) ** 2)[:, None]
            * np.exp(-((zsx - CZ * dx3) / (W_BEAM_VER / 2)) ** 2)[None, :])
    axs[2].imshow(beam.T, origin="lower", cmap="inferno",
                  extent=[0, ny * dx3, 0, nz * dx3], aspect="auto")
    axs[2].set_title(f"input beam {W_BEAM_LAT} x {W_BEAM_VER} um (astigmatic)")
    for ax in axs:
        ax.set_xlabel("um")
    fig.suptitle(f"edge coupler 3D @{WL*1e3:.0f} nm: {eta_abs_dev*100:.1f}% of "
                 f"launched beam ({eta:.3f}x bare straight)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "c_edge_coupler_3d.png", dpi=150)
    plt.close(fig)
    print("c_edge_coupler_3d.png written")


if __name__ == "__main__":
    main()
