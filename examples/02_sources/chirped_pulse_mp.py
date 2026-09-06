# Linearly-chirped Gaussian pulse planewave, from the Meep tutorial.
#
# meep-torch port of meep/python/examples/chirped_pulse.py. The pulse is a
# down-chirp (higher frequencies at the front) launched from the left and
# propagating in +x; the oscillation period visibly grows toward the tail.
# Output: `output/chirped_pulse_mp.png`.
#
# CHANGED from upstream, and why:
#   * No k_point and no mp.Mirror(mp.Y), so the transverse walls are PEC.
#     That is the whole point of this twin: it isolates the non-Bloch path.
#     PEC clamps Ez at y = 0 and y = sy, so the launched planewave is clipped
#     at the transverse edges and this run is NOT a clean planewave -- expect
#     a visibly larger residual than chirped_pulse.py. Use chirped_pulse.py
#     for the physics; use this one to exercise the plain stepper.
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

    # Identical to chirped_pulse.py so the two are directly comparable: same
    # resolution, same cell, PML on x only, same source plane and stop time.
    # The ONLY difference left is the one this twin exists to isolate --
    # no k_point/Mirror(Y), so the transverse edges are PEC rather than
    # Bloch-periodic.
    resolution = 40
    sx, sy = 40.0, 6.0
    dpml = 2.0

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx + 2 * dpml, sy),
        resolution=resolution,
        sources=[mp.Source(profile, mp.Ez, mp.Vector3(-0.5 * sx),
                           size=mp.Vector3(y=sy))],
        boundary_layers=[mp.PML(thickness=dpml, direction=mp.X)],
        default_material=mp.Medium(epsilon=1.0),
        device=DEVICE,
    )

    # Physical time, exactly as upstream: run(..., until=t0 + 50). Not a step
    # count -- dt = Courant/resolution, so n_steps would need rescaling on
    # every resolution change. (meep has no sim.dt; its timestep is
    # sim.fields.dt.)
    # Snapshot while the pulse is still inside the cell -- see
    # chirped_pulse.py for why the final state is residual, not pulse.
    SNAP = 40.0
    snap = {}

    sim.run(mp.at_time(SNAP, lambda s: snap.update(ez=s.get_array(component=mp.Ez))),
            until=t0 + 50)

    ez = snap["ez"]
    ez_slice = ez[:, ez.shape[1] // 2]              # centre row
    # x from the returned array and the cell, not sim.grid -- meep exposes no
    # .grid, so reaching for one is a shim-ism that would not run upstream.
    # Span the FULL cell (sx + 2*dpml), not sx -- get_array returns the whole
    # grid including the PML, so dividing by sx compresses x by 10% here.
    cell_x = sx + 2 * dpml
    x = (np.arange(ez.shape[0]) + 0.5) * (cell_x / ez.shape[0])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, ez_slice, lw=0.8)
    ax.set_title(f"Chirped pulse -- Ez(x) at t = {SNAP:g} (down-chirp)")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Ez")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "chirped_pulse_mp.png", dpi=150)
    plt.close(fig)

    data=np.hstack((x.reshape(len(x),1),
                          ez_slice.reshape(len(x),1)))

    np.savetxt(OUT / 'chirped_pulse_mp.csv',
               data,
               delimiter=',')
    print(np.shape(data))

    peak = float(np.max(np.abs(ez_slice)))
    print(f"chirped_pulse_mp.png  [{DEVICE}]  max|Ez| at t={SNAP:g} = {peak:.6e}")


if __name__ == "__main__":
    main()
