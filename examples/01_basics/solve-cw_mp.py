# Calculating driven CW steady-state in a ring, from the Meep tutorial.
#
# meep-torch port: the only change to the physics is the import. The tutorial's
# HDF5 output is kept (mp.output_epsilon / mp.output_efield_z) but pointed at
# output/ via sim.use_output_directory -- without it every output_* call lands
# in the process working directory. The tutorial then says to run
#     h5topng -RZc dkbluered -C ring-eps-*.h5 ring-ez-*.h5
#     convert ring-ez-*.png ring-ez-0.118.gif
# which needs h5utils. The same animation is built here from the field files
# the run just wrote, so it needs no external tools: output/solve-cw_mp-ez.gif.
#
# SUBSTITUTION:
# * meep-torch is time-domain only; there is no solve_cw / CW solver. We
#   substitute a long CWRamp run at fcen=0.118 (two odd-sign Ez sources, the
#   meep odd mode), let the resonant field build up, then:
#   - snapshot the time-domain Ez field (the "CW" field at the last step),
#   - rfft a probe time series to confirm the spectral peak sits at fcen
#     (the DFT-field substitute: a single-frequency DFT bin = the CW field).
# * meep's tolerance-convergence sweep (solve_cw with decreasing tol) has no
#   time-domain analog and is dropped.
#
# Run: uv run python examples/01_basics/solve-cw_mp.py

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from _plotdata import savefig as _savefig  # PNG + .plotdata.json sidecar

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def _load_field(path):
    """One Ez frame from an output_* file, .npz or .h5 alike."""
    if path.suffix == ".npz":
        with np.load(path) as z:
            key = next(k for k in z.files if k.startswith("ez"))
            return z[key]
    import h5py
    with h5py.File(path, "r") as f:
        key = next(k for k in f if k.startswith("ez"))
        return np.array(f[key])


def _animate(out_dir, eps):
    """Field files -> animated GIF."""
    frames = sorted(out_dir.glob("solve-cw_mp-ez-*"))
    if len(frames) < 2:
        print(f"  only {len(frames)} field file(s); skipping the GIF")
        return
    ez = [_load_field(f) for f in frames]
    v = max(float(np.abs(a).max()) for a in ez) or 1.0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_axis_off()
    ax.imshow(eps.T, origin="lower", cmap="binary", alpha=0.5)
    im = ax.imshow(ez[0].T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.85)
    fig.tight_layout(pad=0)

    def step(i):
        im.set_data(ez[i].T)
        return (im,)

    anim = animation.FuncAnimation(fig, step, frames=len(ez), interval=100, blit=True)
    gif = out_dir / "solve-cw_mp-ez.gif"
    anim.save(gif, writer=animation.PillowWriter(fps=10))
    print(f"  {gif.name}  ({len(ez)} frames)")


def main():
    n = 3.4  # index of waveguide
    w = 1  # width of waveguide
    r = 1  # inner radius of ring
    pad = 4  # padding between waveguide and edge of PML
    dpml = 2  # thickness of PML
    sxy = 2 * (r + w + pad + dpml)  # cell size

    c1 = mp.Cylinder(radius=r + w, material=mp.Medium(index=n))
    c2 = mp.Cylinder(radius=r)

    fcen = 0.118  # drive frequency (resonant mode)
    df = 0.1  # pulse width (not used in CWRamp, kept for reference)

    # Two opposite-sign CWRamp sources select the odd mode
    src_pos = mp.Source(mp.CWRamp(fcen, t_ramp=50.0), mp.Ez, mp.Vector3(r + 0.1, 0))
    src_neg = mp.Source(mp.CWRamp(fcen, t_ramp=50.0), mp.Ez, mp.Vector3(-(r + 0.1), 0), amplitude=-1.0)

    sim = mp.Simulation(
        cell_size=mp.Vector3(sxy, sxy),
        geometry=[c1, c2],
        sources=[src_pos, src_neg],
        resolution=10,
        boundary_layers=[mp.PML(dpml)],
    )

    sim.use_output_directory(str(OUT))

    # Long driven run to reach steady state (~700 time units)
    # Note: cannot use step functions / stop conditions with EigenModeSource,
    # so we use a fixed until_after_sources value.
    h = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)
    sim.run(mp.at_beginning(mp.output_epsilon), h, until_after_sources=14000)

    modes = sorted(h.modes, key=lambda m: m.freq)
    print("harminv0:, frequency, imag. freq., Q, |amp|")
    for m in modes:
        print(f"harminv0:, {m.freq:.6f}, {m.decay:.4g}, {m.Q:.4g}, {abs(m.amp):.4g}")

    # Output fields for one period at the end
    best, best_ez = -1.0, None
    for _ in range(20):
        sim.run(mp.at_end(mp.output_efield_z), until=1 / fcen / 20)
        ez = sim.get_array(component=mp.Ez)
        peak = float(np.abs(ez).max())
        if peak > best:
            best, best_ez = peak, ez

    _animate(OUT, sim.get_array(component=mp.Dielectric))

    eps = sim.get_array(component=mp.Dielectric)
    v = best or 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(eps.T, origin="lower", cmap="binary", alpha=0.5)
    ax.imshow(best_ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
    title = "Ring resonator CW steady-state -- Ez"
    if modes:
        top = max(modes, key=lambda m: abs(m.amp))
        title += f"\ndrive f={fcen:.4f}, harminv f={top.freq:.4f}, Q={top.Q:.0f}"
    ax.set_title(title)
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    fig.tight_layout()
    _savefig(fig, OUT / "solve-cw_mp.png", dpi=150)
    n_h5 = len(list(OUT.glob("solve-cw_mp-*")))
    print(f"solve-cw_mp.png  peak|Ez|={best:.4g}  ({len(modes)} modes found, "
          f"{n_h5} field files in {OUT.name}/)")


if __name__ == "__main__":
    main()
