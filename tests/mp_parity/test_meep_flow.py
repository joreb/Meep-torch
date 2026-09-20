# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep object-flow parity gate for the rearchitected run path.

Asserts the meep 1.34 simulation lifecycle facts: Simulation.__init__ stores
the spec only (no structure/fields), init_sim builds structure then
fields(structure), run() drives fields.step, and the structure is the single
owner of the stepping coefficients. Physics gate: the step-function run
reproduces the flux values recorded on the pre-rearchitecture code
(scratchpad golden_capture.py, 2026-09-03).
"""
import pytest
import torch

import meep_torch.mp as mp
from meep_torch.core.grid import Grid
from meep_torch.core.run import run_te, run_tm
from meep_torch.core.sources import LineSource
from meep_torch.core.stepper_tm import init_tm_state
from meep_torch.core.stepper_te import init_te_state
from meep_torch.mp.cpp.core import fields as core_fields
from meep_torch.mp.cpp.core import structure as core_structure

# pre-rearchitecture golden fluxes: run(until=40) on the waveguide below.
#
# These are SELF-CONSISTENCY goldens -- captured from this engine, not from
# meep. They guard the run path against silent drift; they are not a parity
# claim. Real meep on the same cell gives 1.031e-06, 1.645e-06, 6.382e-07,
# which is 2.5x to 6.4x below these values with a different spectral shape.
# That gap is pre-existing and unexplained, and an earlier revision of this
# comment asserted a rebaseline had moved these "onto meep's"; it had not.
# Do not read agreement with meep into this gate.
#
# Rebaselined 2026-09-07 (2nd time) for the CPML high-edge anchor fix in
# core/pml.py: the high ramp was anchored in array-index space, so half-offset
# H arrays started it a cell inside the E ramp and the resulting E/H stretch
# mismatch reflected off the layer. This cell's guide runs into the PML, so
# its flux moves. All three values moved TOWARD meep (ratios 2.68/4.51/6.69 ->
# 2.49/4.13/6.39):
#     old  2.7674774850386973e-06  7.415436603754773e-06  4.268356224067546e-06
#     new  2.5635746995635057e-06  6.786305104894549e-06  4.077624008546449e-06
# (-7.4%, -8.5%, -4.5%). See wiki/tasks/fix_cpml-entrance-reflection.md.
#
# 1st rebaseline, same day: exactly 100x, for the source-normalization fix. A
# meep current is a density integrated over its extent, so its field is
# resolution-independent; we injected the bare amplitude, making the field
# scale as dx. Flux goes as amplitude^2 and this is a line source at
# resolution 10, hence 10^2.
GOLDEN_PLAIN = [2.5635746995635057e-06, 6.786305104894549e-06,
                4.0776240085464485e-06]
# same sim through run(at_every(5.0, ...), until=40): chunked accumulation
# differs from the single-chunk plain run only in summation grouping (1 ulp)
GOLDEN_STEPPED = [2.5635746995635057e-06, 6.786305104894549e-06,
                  4.077624008546449e-06]


def _sim():
    return mp.Simulation(
        cell_size=mp.Vector3(10, 6), resolution=10,
        geometry=[mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, 1, mp.inf),
                           material=mp.Medium(epsilon=12))],
        sources=[mp.Source(mp.GaussianSource(0.15, fwidth=0.1), component=mp.Ez,
                           center=mp.Vector3(-3, 0), size=mp.Vector3(0, 2))],
        boundary_layers=[mp.PML(1.0)])


def test_init_defers_structure_and_fields():
    sim = _sim()
    assert sim.structure is None and sim.fields is None
    sim.init_sim()
    assert isinstance(sim.structure, core_structure)
    assert isinstance(sim.fields, core_fields)
    sim.init_sim()                                   # idempotent (meep)
    assert sim.fields is sim.fields


def test_structure_owns_the_coeffs():
    sim = _sim()
    sim.init_sim()
    assert sim._coeffs is sim.structure.coeffs
    assert sim.fields.s is sim.structure


def test_monitor_adder_calls_init_sim_lazily():
    sim = _sim()
    assert sim.fields is None
    sim.add_flux(0.15, 0.1, 3, mp.FluxRegion(center=mp.Vector3(3, 0),
                                             size=mp.Vector3(0, 4)))
    assert sim.fields is not None                    # meep: add_flux inits sim
    assert len(sim.fields._monitors) == 1            # registered on the fields


def test_run_drives_fields_step(monkeypatch):
    sim = _sim()
    sim.init_sim()
    n_calls = [0]
    orig = type(sim.fields).step

    def spy(self):
        n_calls[0] += 1
        return orig(self)

    monkeypatch.setattr(type(sim.fields), "step", spy)
    # a stop condition forces per-step granularity: every step goes
    # through fields.step (meep's _run_until protocol)
    steps = []
    sim.run(until=[lambda s: steps.append(s.round_time()) or len(steps) >= 7])
    assert n_calls[0] == 7 and sim.fields._t == 7


def test_stepped_run_matches_pre_rearchitecture_golden_flux():
    sim = _sim()
    fl = sim.add_flux(0.15, 0.1, 3, mp.FluxRegion(center=mp.Vector3(3, 0),
                                                  size=mp.Vector3(0, 4)))
    hooks = []
    sim.run(mp.at_every(5.0, lambda s: hooks.append(s.round_time())), until=40)
    # first firing time is dT, not 0 -- measured against pymeep, which gives
    # exactly these eight times for at_every(5.0) to t = 40.
    assert hooks == [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    assert mp.get_fluxes(fl) == pytest.approx(GOLDEN_STEPPED, rel=1e-12)


def test_plain_run_matches_pre_rearchitecture_golden_flux():
    sim = _sim()
    fl = sim.add_flux(0.15, 0.1, 3, mp.FluxRegion(center=mp.Vector3(3, 0),
                                                  size=mp.Vector3(0, 4)))
    sim.run(until=40)
    assert mp.get_fluxes(fl) == pytest.approx(GOLDEN_PLAIN, rel=1e-12)


# --- meep fields::step phase decomposition (src/step.cpp order) -------------

def test_step_follows_meep_phase_order(monkeypatch):
    f = core_fields(core_structure(Grid(nx=20, ny=20, dx=0.1), 1.0))
    calls = []
    for name in ("phase_material", "step_db", "step_source", "update_pols",
                 "update_eh", "step_boundaries"):
        orig = getattr(core_fields, name)

        def spy(self, *a, _n=name, _o=orig, **k):
            calls.append((_n, a[0] if a else None))
            return _o(self, *a, **k)

        monkeypatch.setattr(core_fields, name, spy)
    f.step()
    assert calls == [
        ("phase_material", None),
        ("step_db", "B"), ("step_source", "B"),
        ("update_eh", "H"), ("step_boundaries", "H"),
        ("step_db", "D"), ("step_source", "D"),
        ("update_pols", None),
        ("update_eh", "E"), ("step_boundaries", "E"),
    ]


def test_phase_path_matches_fused_tm_kernel_500_steps():
    """Bit-for-bit (float64 torch.equal) state parity: 500 fields.step()
    phase-decomposed steps vs the fused step_tm path (core.run.run_tm) on a
    dispersive + chi3 + PML structure."""
    grid = Grid(nx=60, ny=44, dx=0.05)
    st = core_structure(grid, 2.0, mp.boundary_region(thickness=0.4))
    st.set_chi3(0.01)
    st.add_susceptibility((2.0, 0.1, 1.5))          # engine pole (fn, gamma, sigma)
    f = core_fields(st)
    f.add_point_source(mp.Ez, mp.GaussianSource(frequency=1.0, fwidth=0.3),
                       mp.Vector3())
    for _ in range(500):
        f.step()

    src = LineSource(grid, 30, 22, 23,
                     profile=mp.GaussianSource(frequency=1.0, fwidth=0.3).profile())
    ref, _ = run_tm(init_tm_state(grid, n_poles=1), st.coeffs, 500, source=src)
    assert float(f._state.Ez.abs().max()) > 0
    for name in ("Ez", "Hx", "Hy", "Dz", "P_cur", "P_prev",
                 "psi_Ezy", "psi_Ezx", "psi_Hyx", "psi_Hxy"):
        assert torch.equal(getattr(f._state, name), getattr(ref, name)), name


def test_alloc_f_meep_semantics():
    f = core_fields(core_structure(Grid(nx=20, ny=20, dx=0.1), 1.0))
    assert f.alloc_f("Ez") is False              # constructor already allocated
    assert f.alloc_f("Dz") is False
    f._state = None
    assert f.alloc_f("Hy") is True               # first alloc materializes the set
    assert f.alloc_f("Hy") is False              # second call: no-op (meep)
    with pytest.raises(NotImplementedError):
        f.alloc_f("Hz")                          # outside the TM component set


def test_component_allocation_is_emergent_polarization():
    tm = _sim()
    tm.init_sim()
    st = tm.fields._state
    assert getattr(st, "Hz", None) is None and st.Ez is not None
    te = mp.Simulation(
        cell_size=mp.Vector3(4, 3), resolution=10,
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.3), component=mp.Hz,
                           center=mp.Vector3())],
        boundary_layers=[mp.PML(0.5)])
    te.init_sim()
    st = te.fields._state
    assert getattr(st, "Ez", None) is None and st.Hz is not None


def test_phase_path_matches_fused_te_kernel_500_steps():
    """Same parity gate for the TE component set (duality-framed phase cycle)."""
    sim = mp.Simulation(
        cell_size=mp.Vector3(4, 3), resolution=10,
        geometry=[mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, 1, mp.inf),
                           material=mp.Medium(epsilon=4))],
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.3), component=mp.Hz,
                           center=mp.Vector3())],
        boundary_layers=[mp.PML(0.5)])
    st = core_structure(sim)
    # spec-driven component set (meep require_component flow): the TE set
    # comes from the sim's Hz source, not from a structure polarization flag
    f = core_fields(st, components=sim._component_set())
    src = LineSource(sim.grid, 20, 15, 16,
                     profile=mp.GaussianSource(1.0, fwidth=0.3).profile())
    src.mask = src.mask[:, :-1]                     # Ey-grid injection (TE)
    f.add_source(src)
    for _ in range(500):
        f.step()

    ref, _ = run_te(init_te_state(sim.grid), st.coeffs_for(sim._component_set()), 500, source=src)
    assert float(f._state.Hz.abs().max()) > 0
    for name in ("Hz", "Ex", "Ey", "Dx", "Dy",
                 "psi_Eyx", "psi_Exy", "psi_Hzy", "psi_Hzx"):
        assert torch.equal(getattr(f._state, name), getattr(ref, name)), name
