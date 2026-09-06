# Calculating 2d ring-resonator modes, from the Meep tutorial.
#
# meep-torch port: the only change to the physics is the import. The tutorial's
# HDF5 output is kept (mp.output_epsilon / mp.output_efield_z) but pointed at
# output/ via sim.use_output_directory -- without it every output_* call lands
# in the process working directory, so running this from the repo root litters
# it with 21 files. The tutorial then says to run
#     h5topng -RZc dkbluered -C ring-eps-*.h5 ring-ez-*.h5
#     convert ring-ez-*.png ring-ez-0.118.gif
# which needs h5utils. The same animation is built here from the field files
# the run just wrote, so it needs no external tools: output/ring_mp-ez.gif.
#
# Run: uv run python examples/01_basics/ring_mp.py

import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import torch

import meep_torch.mp as mp

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# Examples default to the GPU: they are demonstrations, and the difference is
# not marginal (this cell is 3 s on CUDA against >550 s on CPU). The engine
# itself defaults to CPU -- a library should not pick a device for you -- so
# the script asks explicitly. MEEP_TORCH_DEVICE overrides; falls back to CPU
# on a machine without a card.
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")


def _load_field(path):
    """One Ez frame from an output_* file, .npz or .h5 alike.

    _write picks the format by whether h5py imported, so a script that reads
    its own output back has to handle both or it breaks the moment the
    optional dependency appears.
    """
    if path.suffix == ".npz":
        with np.load(path) as z:
            key = next(k for k in z.files if k.startswith("ez"))
            return z[key]
    import h5py
    with h5py.File(path, "r") as f:
        key = next(k for k in f if k.startswith("ez"))
        return np.array(f[key])


def _animate(out_dir, eps):
    """Field files -> animated GIF, the h5topng + convert step of the tutorial."""
    frames = sorted(out_dir.glob("ring_mp-ez-*"))
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
    gif = out_dir / "ring_mp-ez.gif"
    anim.save(gif, writer=animation.PillowWriter(fps=10))
    plt.close(fig)
    print(f"  {gif.name}  ({len(ez)} frames)")


def main():
    n = 3.4  # index of waveguide
    w = 1  # width of waveguide
    r = 1  # inner radius of ring
    pad = 4  # padding between waveguide and edge of PML
    dpml = 2  # thickness of PML
    sxy = 2 * (r + w + pad + dpml)  # cell size

    # Create a ring waveguide by two overlapping cylinders - later objects
    # take precedence over earlier objects, so we put the outer cylinder first.
    # and the inner (air) cylinder second.

    c1 = mp.Cylinder(radius=r + w, material=mp.Medium(index=n))
    c2 = mp.Cylinder(radius=r)

    # If we don't want to excite a specific mode symmetry, we can just
    # put a single point source at some arbitrary place, pointing in some
    # arbitrary direction.  We will only look for Ez-polarized modes.

    fcen = 0.15  # pulse center frequency
    df = 0.1  # pulse width (in frequency)

    src = mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r + 0.1))

    sim = mp.Simulation(
        cell_size=mp.Vector3(sxy, sxy),
        geometry=[c1, c2],
        sources=[src],
        resolution=10,
        symmetries=[mp.Mirror(mp.Y)],
        boundary_layers=[mp.PML(dpml)],
        device=DEVICE,
    )

    # Every output_* file below goes here instead of the working directory.
    sim.use_output_directory(str(OUT))

    # Keep a handle on the Harminv step function: the tutorial only prints its
    # table, but the modes are also what the plot is annotated with.
    h = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)
    sim.run(mp.at_beginning(mp.output_epsilon), h,
            until_after_sources=300)

    modes = sorted(h.modes, key=lambda m: m.freq)
    print("harminv0:, frequency, imag. freq., Q, |amp|")
    for m in modes:
        print(f"harminv0:, {m.freq:.6f}, {m.decay:.4g}, {m.Q:.4g}, {abs(m.amp):.4g}")
    # meep reports 0.11810 / 0.14716 / 0.17525 for this cell; the frequencies
    # agree to <0.4%, but harminv here has no error estimate to prune with
    # (only Q_thresh), so a few extra low-amplitude modes come through.

    # Output fields for one period at the end.  (If we output
    # at a single time, we might accidentally catch the Ez field when it is
    # almost zero and get a distorted view.)  Stepping a further 1/fcen and
    # keeping the largest |Ez| does the same job as the tutorial's per-period
    # HDF5 dump, without needing to read 20 files back.
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
    title = "Ring resonator -- Ez over one period"
    if modes:
        top = max(modes, key=lambda m: abs(m.amp))
        title += f"\nstrongest mode f={top.freq:.4f}, Q={top.Q:.0f}"
    ax.set_title(title)
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    fig.tight_layout()
    fig.savefig(OUT / "ring_mp.png", dpi=150)
    plt.close(fig)
    n_h5 = len(list(OUT.glob("ring_mp-*")))
    print(f"ring_mp.png  peak|Ez|={best:.4g}  ({len(modes)} modes found, "
          f"{n_h5} field files in {OUT.name}/)")


if __name__ == "__main__":
    main()
