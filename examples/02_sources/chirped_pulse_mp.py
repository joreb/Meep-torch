# Linearly-chirped Gaussian pulse planewave, from the Meep tutorial.
#
# meep-torch port of meep/python/examples/chirped_pulse.py. The pulse is a
# down-chirp (higher frequencies at the front) launched from the left and
# propagating in +x; the oscillation period visibly grows toward the tail.
# Output: `output/chirped_pulse_mp.png`.
#
# CHANGED from upstream, and why:
#   * 1D via a thin 2D strip with PMC on y, instead of upstream's 2D cell with
#     mp.Mirror(mp.Y). The physics along x is the same; the mirror symmetry is
#     a domain-halving optimisation, not part of the result.
#   * resolution 40 -> 20 for runtime.
# The source itself is NOT changed: it is an mp.CustomSource carrying meep's
# own chirp expression, so this example exercises the real source API rather
# than a shim-only substitute.
#
# Device: GPU when available (MEEP_TORCH_DEVICE overrides).
#
# Run: uv run python examples/02_sources/chirped_pulse_mp.py

import os
import pathlib

import matplotlib

matplotlib.use("Agg")
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


class ChirpedPulse(mp.CustomSource):
    """Linearly-chirped Gaussian, as a genuine ``mp.CustomSource``.

    Subclasses rather than wraps, so it *is* a meep source: it can go
    anywhere a CustomSource goes, and the same file would run under real
    meep. The named arguments are the convenience; the mechanism underneath
    is meep's own.

        E(t) = Re[ exp(i 2 pi v0 (t-t0)) * exp(-a (t-t0)^2 + i b (t-t0)^2) ]

    ``b < 0`` gives a down-chirp (higher frequencies at the front), which is
    what upstream's ``b = -0.5`` selects.
    """

    def __init__(self, v0: float, a: float, b: float, t0: float, **kwargs):
        self.v0, self.a, self.b, self.t0 = v0, a, b, t0
        # meep's chirp expression, unchanged.
        def chirp(t):
            return np.exp(1j * 2 * np.pi * v0 * (t - t0)) * np.exp(
                -a * (t - t0) ** 2 + 1j * b * (t - t0) ** 2
            )

        super().__init__(src_func=chirp, **kwargs)


def main():
    # parameters from the meep original
    v0, a, b, t0 = 1.0, 0.2, -0.5, 15.0
    profile = ChirpedPulse(v0, a, b, t0)

    resolution = 20
    sx, sy = 40.0, 6.0
    dpml = 2.0

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy, 0),
        resolution=resolution,
        sources=[mp.Source(profile, mp.Ez, mp.Vector3(-sx / 2 + dpml, 0),
                           size=mp.Vector3(0, sy, 0))],
        boundary_layers=[mp.PML(dpml)],
        default_material=mp.Medium(epsilon=1.0),
        device=DEVICE,
    )

    # Physical time, exactly as upstream: run(..., until=t0 + 50). Not a step
    # count -- dt = Courant/resolution, so n_steps would need rescaling on
    # every resolution change. (meep has no sim.dt; its timestep is
    # sim.fields.dt.)
    sim.run(until=t0 + 50)

    ez = sim.get_array(component=mp.Ez)
    ez_slice = ez[:, ez.shape[1] // 2]              # centre row
    # x from the returned array and the cell, not sim.grid -- meep exposes no
    # .grid, so reaching for one is a shim-ism that would not run upstream.
    x = (np.arange(ez.shape[0]) + 0.5) * (sx / ez.shape[0])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, ez_slice, lw=0.8)
    ax.set_title("Chirped pulse -- Ez(x) at final time (down-chirp)")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Ez")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "chirped_pulse_mp.png", dpi=150)
    plt.close(fig)

    peak = float(np.max(np.abs(ez_slice)))
    print(f"chirped_pulse_mp.png  [{DEVICE}]  max|Ez| in snapshot = {peak:.6e}")


if __name__ == "__main__":
    main()
