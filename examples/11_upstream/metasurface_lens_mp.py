# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/metasurface_lens.py by scripts/port_upstream_example.py.
# The body below is upstream's, unchanged but for the meep import and the generated ModeSolver lines, so that a difference between
# this file and its twin is a difference between the ENGINES. See that
# script's docstring for why nothing here rewrites a call site.
#
# The literal spellings below (`matplotlib.use("Agg")`, `OUT = ...`,
# `OUT.mkdir`, `DEVICE`) are the ones
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
OUT = pathlib.Path(__file__).parent / "output/metasurface_lens/"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda"
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/


def _dump_plot_data(_fig, _path):
    """Write the NUMBERS behind a figure next to it, as <name>.plotdata.json.

    Several tutorials compute a result and only plot it -- binary_grating_
    phasemap, gaussian-beam and refl-quartz print nothing numeric and write no
    data file, so a stdout-and-files comparison has nothing to say about them
    at all. The numbers are right there in the axes.

    This lives in the generated preamble and not in the body, which is the
    whole point: it is emitted identically into both twins, so the
    byte-comparability that makes a twin difference mean anything survives. A
    record() call in the body would not.
    """
    import json as _json
    out, n = {}, 0
    for ai, ax in enumerate(_fig.get_axes()):
        for li, ln in enumerate(ax.get_lines()):
            xs, ys = ln.get_xdata(), ln.get_ydata()
            if len(xs):
                out[f"ax{ai}.line{li}.x"] = [float(v) for v in xs]
                out[f"ax{ai}.line{li}.y"] = [float(v) for v in ys]
                n += 1
        for ii, im in enumerate(ax.get_images()):
            a = im.get_array()
            if a is not None and a.size:
                import numpy as _np
                a = _np.asarray(a, dtype=float)
                out[f"ax{ai}.image{ii}.shape"] = list(a.shape)
                out[f"ax{ai}.image{ii}.stats"] = [float(a.min()), float(a.max()),
                                                  float(a.mean()),
                                                  float((a ** 2).mean() ** 0.5)]
                n += 1
    if n:
        # PREFIXED WITH THE SCRIPT STEM, because the figure's name comes from
        # upstream's body and is therefore IDENTICAL on both twins. Without the
        # prefix both write Ez_angle0.plotdata.json into one directory and the
        # second run overwrites the first -- the same defect that made the two
        # mpb twins clobber each other's h5 output.
        _p = pathlib.Path(str(_path))
        _p = _p.with_name("metasurface_lens_mp-" + _p.stem)
        _p.with_suffix(".plotdata.json").write_text(
            _json.dumps(out, indent=1, sort_keys=True))


import matplotlib.figure as _mplfig   # submodule, not an attribute of
                                     # matplotlib until imported
_savefig = _mplfig.Figure.savefig


def _savefig_and_dump(self, fname, *a, **k):
    r = _savefig(self, fname, *a, **k)
    try:
        _dump_plot_data(self, fname)
    except Exception:
        pass          # a figure we cannot read must not kill the tutorial
    return r


_mplfig.Figure.savefig = _savefig_and_dump
# --- end preamble ----------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp

resolution = 50  # pixels/μm

dpml = 1.0  # PML thickness
dsub = 2.0  # substrate thickness
dpad = 2.0  # padding between grating and PML

lcen = 0.5  # center wavelength
fcen = 1 / lcen  # center frequency
df = 0.2 * fcen  # frequency width

focal_length = 200  # focal length of metalens
spot_length = 100  # far field line length
ff_res = 10  # far field resolution (points/μm)

k_point = mp.Vector3(0, 0, 0)

glass = mp.Medium(index=1.5)

pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

symmetries = [mp.Mirror(mp.Y)]


def grating(gp, gh, gdc_list):
    sx = dpml + dsub + gh + dpad + dpml
    src_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub)
    mon_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad)
    geometry = [
        mp.Block(
            material=glass,
            size=mp.Vector3(dpml + dsub, mp.inf, mp.inf),
            center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dsub)),
        )
    ]

    num_cells = len(gdc_list)
    if num_cells == 1:
        sy = gp
        cell_size = mp.Vector3(sx, sy, 0)

        sources = [
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Ez,
                center=src_pt,
                size=mp.Vector3(y=sy),
            )
        ]

        sim = mp.Simulation(
            resolution=resolution,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            k_point=k_point,
            default_material=glass,
            sources=sources,
            symmetries=symmetries,
        )

        flux_obj = sim.add_flux(
            fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy))
        )

        sim.run(until_after_sources=50)

        input_flux = mp.get_fluxes(flux_obj)

        sim.reset_meep()

        geometry.append(
            mp.Block(
                material=glass,
                size=mp.Vector3(gh, gdc_list[0] * gp, mp.inf),
                center=mp.Vector3(-0.5 * sx + dpml + dsub + 0.5 * gh),
            )
        )

        sim = mp.Simulation(
            resolution=resolution,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            k_point=k_point,
            sources=sources,
            symmetries=symmetries,
        )

        flux_obj = sim.add_flux(
            fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy))
        )

        sim.run(until_after_sources=200)

        freqs = mp.get_eigenmode_freqs(flux_obj)
        res = sim.get_eigenmode_coefficients(
            flux_obj, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y
        )
        coeffs = res.alpha

        mode_tran = abs(coeffs[0, 0, 0]) ** 2 / input_flux[0]
        mode_phase = np.angle(coeffs[0, 0, 0])
        if mode_phase > 0:
            mode_phase -= 2 * np.pi

        return mode_tran, mode_phase

    else:
        sy = num_cells * gp
        cell_size = mp.Vector3(sx, sy, 0)

        sources = [
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Ez,
                center=src_pt,
                size=mp.Vector3(y=sy),
            )
        ]

        geometry.extend(
            mp.Block(
                material=glass,
                size=mp.Vector3(gh, gdc_list[j] * gp, mp.inf),
                center=mp.Vector3(
                    -0.5 * sx + dpml + dsub + 0.5 * gh, -0.5 * sy + (j + 0.5) * gp
                ),
            )
            for j in range(num_cells)
        )
        sim = mp.Simulation(
            resolution=resolution,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            k_point=k_point,
            sources=sources,
            symmetries=symmetries,
        )

        n2f_obj = sim.add_near2far(
            fcen, 0, 1, mp.Near2FarRegion(center=mon_pt, size=mp.Vector3(y=sy))
        )

        sim.run(until_after_sources=500)

        return (
            abs(
                sim.get_farfields(
                    n2f_obj,
                    ff_res,
                    center=mp.Vector3(-0.5 * sx + dpml + dsub + gh + focal_length),
                    size=mp.Vector3(spot_length),
                )["Ez"]
            )
            ** 2
        )


gp = 0.3  # grating periodicity
gh = 1.8  # grating height
gdc = np.linspace(0.1, 0.9, 30)  # grating duty cycle

mode_tran = np.empty(gdc.size)
mode_phase = np.empty(gdc.size)
for n in range(gdc.size):
    mode_tran[n], mode_phase[n] = grating(gp, gh, [gdc[n]])

plt.figure(dpi=200)
plt.subplot(1, 2, 1)
plt.plot(gdc, mode_tran, "bo-")
plt.xlim(gdc[0], gdc[-1])
plt.xticks(list(np.linspace(0.1, 0.9, 5)))
plt.xlabel("grating duty cycle")
plt.ylim(0.96, 1.00)
plt.yticks(list(np.linspace(0.96, 1.00, 5)))
plt.title("transmittance")

plt.subplot(1, 2, 2)
plt.plot(gdc, mode_phase, "rs-")
plt.grid(True)
plt.xlim(gdc[0], gdc[-1])
plt.xticks(list(np.linspace(0.1, 0.9, 5)))
plt.xlabel("grating duty cycle")
plt.ylim(-2 * np.pi, 0)
plt.yticks(list(np.linspace(-6, 0, 7)))
plt.title("phase (radians)")

plt.tight_layout(pad=0.5)
plt.show()

gdc_new = np.linspace(0.16, 0.65, 500)
mode_phase_interp = np.interp(gdc_new, gdc, mode_phase)
print(f"phase-range:, {mode_phase_interp.max() - mode_phase_interp.min():.6f}")

phase_tol = 1e-2
num_cells = [100, 200, 400]
ff_nc = np.empty((spot_length * ff_res, len(num_cells)))

for k in range(len(num_cells)):
    gdc_list = []
    for j in range(-num_cells[k], num_cells[k] + 1):
        phase_local = (
            2
            * np.pi
            / lcen
            * (focal_length - ((j * gp) ** 2 + focal_length**2) ** 0.5)
        )  # local phase at the center of the j'th unit cell
        phase_mod = phase_local % (-2 * np.pi)  # restrict phase to [-2*pi,0]
        if phase_mod > mode_phase_interp.max():
            phase_mod = mode_phase_interp.max()
        if phase_mod < mode_phase_interp.min():
            phase_mod = mode_phase_interp.min()
        idx = np.transpose(
            np.nonzero(
                np.logical_and(
                    mode_phase_interp > phase_mod - phase_tol,
                    mode_phase_interp < phase_mod + phase_tol,
                )
            )
        )
        gdc_list.append(gdc_new[idx[0][0]])

    ff_nc[:, k] = grating(gp, gh, gdc_list)

x = np.linspace(
    focal_length - 0.5 * spot_length,
    focal_length + 0.5 * spot_length,
    ff_res * spot_length,
)
plt.figure(dpi=200)
plt.semilogy(x, abs(ff_nc[:, 0]) ** 2, "bo-", label=f"num_cells = {2*num_cells[0] + 1}")
plt.semilogy(x, abs(ff_nc[:, 1]) ** 2, "ro-", label=f"num_cells = {2*num_cells[1] + 1}")
plt.semilogy(x, abs(ff_nc[:, 2]) ** 2, "go-", label=f"num_cells = {2*num_cells[2] + 1}")
plt.xlabel("x coordinate (μm)")
plt.ylabel(r"energy density of far-field electric fields, |E$_z$|$^2$")
plt.title("focusing properties of a binary-grating metasurface lens")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
