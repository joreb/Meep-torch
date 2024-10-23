# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W1 follow-up: meep-syntax S-param path parity vs the raw compute_smatrix twins.

The shim's get_eigenmode_coefficients (EigenModeSource + mode monitors ->
compute_smatrix) must reproduce the raw-port twins' S-parameters to a few 1e-3.
Both paths route to the same compute_smatrix driver with identical geometry/
ports, so the match is bit-identical here; the gate is |S| within 5e-3.
"""
import numpy as np
import pytest

import meep_torch.mp as mp
from meep_torch.core.grid import Grid
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.ports.port import Port
from meep_torch.ports.smatrix import compute_smatrix

FCEN, FWIDTH = 1.0 / 6.67, 0.2 / 6.67
RTOL = 5e-3


def test_mode_decomposition_sparam_parity():
    L = 8.0
    w1, w2 = 1.0, 2.0
    # --- raw twin: corner-coord taper polygon + compute_smatrix ---
    g = Grid(nx=340, ny=220, dx=0.1)
    cx, cy = g.nx * g.dx / 2, g.ny * g.dx / 2
    x0u, x1u = cx - L / 2, cx + L / 2
    xs, xe = -1.0, g.nx * g.dx + 1.0
    poly = Polygon(np.array([
        [xs, cy - 0.5 * w1], [x0u, cy - 0.5 * w1], [x1u, cy - 0.5 * w2], [xe, cy - 0.5 * w2],
        [xe, cy + 0.5 * w2], [x1u, cy + 0.5 * w2], [x0u, cy + 0.5 * w1], [xs, cy + 0.5 * w1],
    ]))
    eps = rasterize_scene([(poly, 12.0)], g, background=1.0).eps_tm
    ports = [Port(axis="x", index=90, lo=85, hi=135, direction=+1),
             Port(axis="x", index=250, lo=85, hi=135, direction=-1)]
    sm = compute_smatrix(eps, g, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                         freqs=[FCEN], n_steps=3000, npml=40)
    R_raw = float(sm.s[0, 0, 0].abs() ** 2)

    # --- shim: meep-syntax Prism + EigenModeSource + mode monitors ---
    v = [mp.Vector3(-18, -0.5 * w1), mp.Vector3(-L / 2, -0.5 * w1),
         mp.Vector3(L / 2, -0.5 * w2), mp.Vector3(18, -0.5 * w2),
         mp.Vector3(18, 0.5 * w2), mp.Vector3(L / 2, 0.5 * w2),
         mp.Vector3(-L / 2, 0.5 * w1), mp.Vector3(-18, 0.5 * w1)]
    sim = mp.Simulation(cell_size=mp.Vector3(34, 22, 0), resolution=10,
        geometry=[mp.Prism(vertices=v, material=mp.Medium(epsilon=12))],
        sources=[mp.EigenModeSource(mp.GaussianSource(frequency=FCEN, fwidth=FWIDTH),
                  component=mp.Ez, center=mp.Vector3(-8, 0), size=mp.Vector3(0, 5, 0), direction=1)],
        boundary_layers=[mp.PML(4.0)], default_material=mp.Medium(epsilon=1.0))
    m_in = sim.add_mode_monitor(FCEN, 0, 1, mp.FluxRegion(center=mp.Vector3(-8, 0),
                                                          size=mp.Vector3(0, 5, 0)))
    sim.add_mode_monitor(FCEN, 0, 1, mp.FluxRegion(center=mp.Vector3(8, 0),
                                                   size=mp.Vector3(0, 5, 0), direction=-1))
    sim.run(n_steps=3000)
    R_shim = float(abs(sim.get_eigenmode_coefficients(m_in)[0]) ** 2)
    assert abs(R_shim - R_raw) < RTOL, f"taper R: shim {R_shim:.3e} vs raw {R_raw:.3e}"


def test_bend_flux_sparam_parity():
    fcen, fwidth = 0.15, 0.1
    freqs = np.linspace(0.105, 0.195, 25)
    # --- raw twin: corner-coord bend polygons + compute_smatrix ---
    g = Grid(nx=160, ny=320, dx=0.1)
    OX, OY = 8.0, 16.0
    horiz = Polygon(np.array([[-8 + OX, -12 + OY], [4 + OX, -12 + OY],
                             [4 + OX, -11 + OY], [-8 + OX, -11 + OY]]))
    vert = Polygon(np.array([[3 + OX, -12 + OY], [4 + OX, -12 + OY],
                            [4 + OX, 16 + OY], [3 + OX, 16 + OY]]))
    eps = rasterize_scene([(horiz, 12.0), (vert, 12.0)], g, background=1.0).eps_tm
    ports = [Port(axis="x", index=20, lo=25, hi=65, direction=+1),
             Port(axis="y", index=280, lo=95, hi=135, direction=-1)]
    sm = compute_smatrix(eps, g, ports, pol="tm", fcen=fcen, fwidth=fwidth,
                         freqs=list(freqs), n_steps=7000, npml=10)
    i = int(np.argmin(np.abs(freqs - fcen)))
    R_raw = float(sm.s[i, 0, 0].abs() ** 2)
    T_raw = float(sm.s[i, 1, 0].abs() ** 2)

    # --- shim: meep-syntax Blocks + EigenModeSource + mode monitors ---
    sim = mp.Simulation(cell_size=mp.Vector3(16, 32, 0), resolution=10,
        geometry=[mp.Block(size=mp.Vector3(12, 1, mp.inf), center=mp.Vector3(-2, -11.5, 0),
                           material=mp.Medium(epsilon=12)),
                  mp.Block(size=mp.Vector3(1, 28, mp.inf), center=mp.Vector3(3.5, 2, 0),
                           material=mp.Medium(epsilon=12))],
        sources=[mp.EigenModeSource(mp.GaussianSource(frequency=fcen, fwidth=fwidth),
                  component=mp.Ez, center=mp.Vector3(-6, -11.5, 0), size=mp.Vector3(0, 4, 0),
                  direction=1)],
        boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))
    df = freqs[1] - freqs[0]
    m_in = sim.add_mode_monitor(fcen, df, len(freqs),
                                mp.FluxRegion(center=mp.Vector3(-6, -11.5, 0), size=mp.Vector3(0, 4, 0)))
    m_out = sim.add_mode_monitor(fcen, df, len(freqs),
                                 mp.FluxRegion(center=mp.Vector3(3.5, 12, 0),
                                               size=mp.Vector3(4, 0, 0), direction=-1))
    sim.run(n_steps=7000)
    R_shim = float(np.abs(sim.get_eigenmode_coefficients(m_in)[i]) ** 2)
    T_shim = float(np.abs(sim.get_eigenmode_coefficients(m_out)[i]) ** 2)
    assert abs(R_shim - R_raw) < RTOL, f"bend R: shim {R_shim:.4f} vs raw {R_raw:.4f}"
    assert abs(T_shim - T_raw) < RTOL, f"bend T: shim {T_shim:.4f} vs raw {T_raw:.4f}"