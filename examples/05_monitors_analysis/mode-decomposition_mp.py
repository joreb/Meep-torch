# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Taper reflectance via the meep-syntax shim
#
# meep-syntax twin of mode-decomposition.py via `meep_torch.mp`. Linear-taper
# (w1=1 -> w2=2 um) reflectance via the meep-syntax S-param path: an
# EigenModeSource at the input port + mode monitors at input (reflection) and
# output (transmission). The shim routes this to compute_smatrix and returns
# power-normalized S-parameters, matching mode-decomposition.py (|S11|^2 = R)
# within a few 1e-3. See Simulation._run_modes for the measured-incident a_in
# / meep-alpha normalization mapping. Output:
# `output/mode-decomposition_mp.png`.
#
# Run: `uv run python examples/05_monitors_analysis/mode-decomposition_mp.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
FCEN, FWIDTH = 1.0 / 6.67, 0.2 / 6.67
w1, w2 = 1.0, 2.0
LENGTHS = [2, 4, 8]

# %% [markdown]
# ## Reflectance per taper length
# The taper as a meep-style Prism, EigenModeSource drive, mode monitors on
# both sides.

# %%
def taper_reflectance(L):
    # Taper Prism in world (center-origin) coords; cell 34x22, center 0.
    v = [mp.Vector3(-18, -0.5 * w1), mp.Vector3(-L / 2, -0.5 * w1),
         mp.Vector3(L / 2, -0.5 * w2), mp.Vector3(18, -0.5 * w2),
         mp.Vector3(18, 0.5 * w2), mp.Vector3(L / 2, 0.5 * w2),
         mp.Vector3(-L / 2, 0.5 * w1), mp.Vector3(-18, 0.5 * w1)]
    sim = mp.Simulation(
        cell_size=mp.Vector3(34, 22, 0),
        resolution=10,
        geometry=[mp.Prism(vertices=v, material=mp.Medium(epsilon=12))],
        sources=[mp.EigenModeSource(mp.GaussianSource(frequency=FCEN, fwidth=FWIDTH),
                  component=mp.Ez, center=mp.Vector3(-8, 0), size=mp.Vector3(0, 5, 0),
                  direction=1)],
        boundary_layers=[mp.PML(4.0)],
        default_material=mp.Medium(epsilon=1.0),
    )
    m_in = sim.add_mode_monitor(FCEN, 0, 1, mp.FluxRegion(center=mp.Vector3(-8, 0),
                                                          size=mp.Vector3(0, 5, 0)))
    m_out = sim.add_mode_monitor(FCEN, 0, 1, mp.FluxRegion(center=mp.Vector3(8, 0),
                                                          size=mp.Vector3(0, 5, 0), direction=-1))
    sim.run(n_steps=3000)
    R = abs(sim.get_eigenmode_coefficients(m_in, [1]).alpha[0, 0, 0]) ** 2
    T = abs(sim.get_eigenmode_coefficients(m_out, [1]).alpha[0, 0, 0]) ** 2
    return float(R), float(T)

# %% [markdown]
# ## Length sweep and plot

# %%
def main():
    R = [taper_reflectance(L)[0] for L in LENGTHS]
    for L, r in zip(LENGTHS, R):
        print(f"  taper L={L} um: R={r:.3e}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(LENGTHS, R, "o-", color="#1f77b4", label="meep-torch |S11|^2 (mp shim)")
    ax.loglog(LENGTHS, [0.005 / Lt ** 2 for Lt in LENGTHS], "k--", label="1/Lt^2 ref")
    ax.set_xlabel("taper length (um)"); ax.set_ylabel("reflectance")
    ax.set_title("Linear taper reflectance (mp shim)"); ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / "mode-decomposition_mp.png", dpi=150); plt.close(fig)
    print("mode-decomposition_mp.png  R(L=8)=%.3e" % R[-1])


if __name__ == "__main__":
    main()
