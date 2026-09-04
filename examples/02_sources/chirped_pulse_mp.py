# Linearly-chirped Gaussian pulse, from the Meep tutorial.
#
# meep-torch port: the only change to the physics is the import. A linearly-chirped
# Gaussian pulse (down-chirp: higher frequencies at the front) launched from
# the left and propagating in +x. The meep original uses a 2D cell with
# Y-mirror symmetry and a CustomSource; meep-torch has no mirror symmetry or
# arbitrary CustomSource, so per the port instructions we model it as 1D (thin
# 2D, x = propagation, PMC on y) and supply the chirp via a custom profile
# class with .amplitude(t). Expect a pulse whose oscillation period visibly
# grows toward the tail; output: `output/chirped_pulse_mp.png`.
#
# Run: uv run python examples/02_sources/chirped_pulse_mp.py

import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


class ChirpedPulse:
    """Linearly-chirped Gaussian pulse.

    E(t) = exp(-a (t-t0)^2) * cos(2 pi v0 (t-t0) + b (t-t0)^2)
    which is Re[ exp(i 2 pi v0 (t-t0)) * exp(-a (t-t0)^2 + i b (t-t0)^2) ],
    matching the meep original (b<0 -> down-chirp: higher freq at the front).
    """

    def __init__(self, v0: float, a: float, b: float, t0: float):
        self.v0, self.a, self.b, self.t0 = v0, a, b, t0

    def amplitude(self, t: float) -> float:
        dt = t - self.t0
        return math.exp(-self.a * dt * dt) * math.cos(
            2 * math.pi * self.v0 * dt + self.b * dt * dt
        )


def main():
    # parameters from the meep original
    v0, a, b, t0 = 1.0, 0.2, -0.5, 15.0
    profile = ChirpedPulse(v0, a, b, t0)

    # resolution reduced 40 -> 20 for runtime (dx 0.025 -> 0.05); 1D via thin 2D + PMC on y
    resolution = 20
    dx = 1.0 / resolution
    sx, sy = 40.0, 6.0
    dpml = 2.0
    nx = int(round(sx / dx))
    ny = 3  # thin 2D strip for 1D propagation
    npml = int(round(dpml / dx))

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy, 0),
        resolution=resolution,
        sources=[mp.Source(profile, mp.Ez, mp.Vector3(-sx / 2 + dpml, 0), size=mp.Vector3(0, sy, 0))],
        boundary_layers=[mp.PML(dpml)],
        default_material=mp.Medium(epsilon=1.0),
    )

    # run until t0+50 = 65 time units
    n_steps = int(round((t0 + 50.0) / sim.dt))
    sim.run(n_steps=n_steps)

    ez = sim.get_array(component=mp.Ez)

    # Plot
    ez_slice = ez[:, 1]  # middle row
    x = (np.arange(sim.grid.nx) + 0.5) * sim.grid.dx

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
    print(f"chirped_pulse_mp.png  max|Ez| in snapshot = {peak:.6e}")


if __name__ == "__main__":
    main()
