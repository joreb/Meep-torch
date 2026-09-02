# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Visualization parity: plot2D renders eps+sources+monitors+boundaries of a
waveguide sim to a png with meep's extent/labels; Animate2D follows meep's
step-function protocol (__call__(sim, 'step'/'finish'), co_argcount hack),
collects frames, and exports mp4/gif/jshtml."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import meep_torch.mp as mp
from meep_torch.mp.visualization import (Animate2D, JS_Animation, plot2D,
                                         plot_eps, plot_fields)

FCEN, DF = 0.5, 0.2


def _wg_sim():
    return mp.Simulation(
        cell_size=mp.Vector3(8, 8, 0),
        resolution=10,
        geometry=[mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf),
                           material=mp.Medium(epsilon=12))],
        sources=[mp.Source(mp.GaussianSource(frequency=FCEN, fwidth=DF),
                           component=mp.Ez, center=mp.Vector3(-2, 0),
                           size=mp.Vector3(0, 2, 0))],
        boundary_layers=[mp.PML(1.0)],
    )


def test_plot2d_writes_png(tmp_path):
    from matplotlib import pyplot as plt

    sim = _wg_sim()
    sim.add_flux(FCEN, DF, 3, mp.FluxRegion(center=mp.Vector3(2, 0),
                                            size=mp.Vector3(0, 4)))
    fig, ax = plt.subplots()
    ax = plot2D(sim, ax=ax)
    # eps image spans the meep cell (center-origin extent)
    np.testing.assert_allclose(ax.images[0].get_extent(), [-4, 4, -4, 4])
    assert ax.get_xlabel() == "X" and ax.get_ylabel() == "Y"
    # sources + monitors + boundaries drawn as patches/lines
    assert len(ax.patches) + len(ax.lines) >= 2
    out = tmp_path / "wg.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.stat().st_size > 1000


def test_plot_eps_returns_array_without_ax():
    sim = _wg_sim()
    eps = plot_eps(sim)                      # meep: no ax -> raw eps array
    assert eps.shape == (81, 81)             # (8*10+1) tics per axis
    assert eps.max() == pytest.approx(12.0) and eps.min() == pytest.approx(1.0)


def test_plot_fields_after_run():
    from matplotlib import pyplot as plt

    sim = _wg_sim()
    sim.run(until=30)
    data = plot_fields(sim, fields=mp.Ez)    # no ax -> raw field array
    assert isinstance(data, np.ndarray) and data.ndim == 2
    assert np.abs(data).max() > 0
    fig, ax = plt.subplots()
    ax = plot_fields(sim, ax=ax, fields=mp.Ez)
    assert len(ax.images) == 1
    plt.close(fig)


def test_animate2d_step_protocol_and_export(tmp_path):
    from matplotlib import pyplot as plt

    sim = _wg_sim()
    sim.run(until=30)
    anim = Animate2D(fields=mp.Ez)
    assert anim.__code__.co_argcount == 2    # meep step-function detection hack
    n = 6
    for _ in range(n):
        anim(sim, "step")
    anim(sim, "finish")
    assert len(anim._saved_frames) == n

    mp4 = tmp_path / "anim.mp4"
    anim.to_mp4(fps=5, filename=str(mp4))
    assert mp4.stat().st_size > 10_000

    gif = tmp_path / "anim.gif"
    anim.to_gif(fps=5, filename=str(gif))
    assert gif.stat().st_size > 1000

    js = anim.to_jshtml(fps=5)
    assert isinstance(js, JS_Animation)
    assert "frames[0]" in js.get_jshtml() and js._repr_html_() == js.jshtml
    plt.close(anim.f)


def test_animate2d_normalize(tmp_path):
    from matplotlib import pyplot as plt

    sim = _wg_sim()
    sim.run(until=30)
    anim = Animate2D(fields=mp.Ez, normalize=True)
    for _ in range(3):
        anim(sim, "step")
    assert len(anim.cumulative_fields) == 3 and not anim._saved_frames
    anim(sim, "finish")                      # normalize + grab frames
    assert len(anim._saved_frames) == 3
    plt.close(anim.f)
