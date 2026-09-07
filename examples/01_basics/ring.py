# Calculating 2d ring-resonator modes, from the Meep tutorial.
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp
import torch

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# Examples default to the GPU: they are demonstrations, and the difference is
# not marginal (this cell is 3 s on CUDA against >550 s on CPU). The engine
# itself defaults to CPU -- a library should not pick a device for you -- so
# the script asks explicitly. MEEP_TORCH_DEVICE overrides; falls back to CPU
# on a machine without a card.
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")


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
    )

    # output_* files go to output/ instead of the working directory
    sim.use_output_directory(str(OUT))

    # Upstream constructs the Harminv inline inside run(); a handle is kept
    # here so the found modes can be exported to the CSV below.
    h = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)

    sim.run(
        mp.at_beginning(mp.output_epsilon),
        mp.after_sources(h),
        until_after_sources=300,
    )

    # Output fields for one period at the end.  (If we output
    # at a single time, we might accidentally catch the Ez field when it is
    # almost zero and get a distorted view.)
    sim.run(mp.at_every(1 / fcen / 20, mp.output_efield_z), until=1 / fcen)

    # Upstream stops at the h5 dump; the twins additionally export the
    # Harminv modes (CSV) and the final Ez over epsilon (png).
    modes = sorted(h.modes, key=lambda m: m.freq)
    np.savetxt(
        OUT / "ring.csv",
        np.array(
            [[m.freq, m.decay, m.Q, abs(m.amp), m.err] for m in modes]
        ),
        delimiter=",",
        header="freq,decay,Q,abs(amp),err",
        comments="",
    )

    eps_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(sxy, sxy),
                             component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(sxy, sxy),
                            component=mp.Ez)
    v = float(np.abs(ez_data).max()) or 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
    ax.imshow(ez_data.transpose(), interpolation="spline36", cmap="RdBu",
              vmin=-v, vmax=v, alpha=0.9)
    ax.set_title("2d ring resonator modes ($E_z$)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "ring.png", dpi=150)
    plt.close(fig)

    freqs = ", ".join(f"{m.freq:.5f}" for m in modes)
    print(f"ring.png  [{DEVICE}]  {len(modes)} modes: {freqs}")


if __name__ == "__main__":
    main()