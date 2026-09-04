# Absorber vs PML in a lossy background (1D), from the Meep tutorial.
#
# meep-torch port: the only change to the physics is the import. 1D ringdown of a
# Gaussian pulse in a lossy background, terminated by either mp.Absorber
# (matched sigma_d = sigma_b ramp) or PML; the center-probe ringdown
# envelopes are overlaid — the point is that a plain PML misbehaves in a
# conductive background while the matched absorber rings down cleanly.
# Output: `output/absorber-1d_mp.png`.
#
# Run: uv run python examples/04_boundaries/absorber-1d_mp.py

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp
from meep_torch.mp import Absorber, GaussianSource, PML, Simulation, Source, Vector3

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

RES = 40
FCEN, DF = 1.0 / 0.803, 0.1
SIGMA_D = 1.0

cell_size = mp.Vector3(z=10)
boundary_layers_pml = [PML(1.0, direction=mp.Z)]
boundary_layers_absorber = [Absorber(1.0, direction=mp.Z)]

sources = [
    Source(
        src=GaussianSource(FCEN, fwidth=DF),
        center=mp.Vector3(),
        component=mp.Ex,
    )
]


def run_sim(boundary_layers):
    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=RES,
        dimensions=1,
        default_material=mp.Medium(epsilon=1.0, D_conductivity=SIGMA_D),
        boundary_layers=boundary_layers,
        sources=sources,
    )

    ex_series = []

    def probe(sim):
        p = sim.get_field_point(mp.Ex, mp.Vector3())
        ex_series.append(p.real)

    sim.run(
        mp.at_every(1.0, probe),
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(), 1e-6),
    )
    return np.array(ex_series)


def main():
    s_abs = run_sim(boundary_layers_absorber)
    s_pml = run_sim(boundary_layers_pml)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogy(np.abs(s_abs), lw=0.8, label="Absorber(1.0)")
    ax.semilogy(np.abs(s_pml), lw=0.8, label="PML(1.0)")
    ax.set_xlabel("time step")
    ax.set_ylabel("|Ex| at center")
    ax.set_title(f"absorber-1d ringdown (conductive bg sigma_d={SIGMA_D})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "absorber-1d_mp.png", dpi=150)
    plt.close(fig)

    tail = slice(2 * len(s_abs) // 3, None)
    r_abs = float(np.abs(s_abs[tail]).max() / np.abs(s_abs).max())
    r_pml = float(np.abs(s_pml[tail]).max() / np.abs(s_pml).max())
    print(f"absorber-1d_mp.png  residual/peak: absorber={r_abs:.2e}  pml={r_pml:.2e}")


if __name__ == "__main__":
    main()
