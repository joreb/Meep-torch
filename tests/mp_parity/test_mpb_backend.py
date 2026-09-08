# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""ModeSolver.backend: the fdfd and planewave band solvers side by side.

Two backends solve the same 2D eigenproblem: ``"fdfd"`` (2nd-order finite
differences, the default, what every committed golden was measured with)
and ``"planewave"`` (spectral). The gates here are about the SWITCH being
faithful -- routing through ModeSolver must equal calling the backend
directly, fdfd numbers must not move, and the shared field/energy/output
machinery must work identically on both.
"""
import json
import math

import numpy as np
import pytest

import meep_torch.mpb as mp_mpb
from meep_torch.mpb import ModeSolver
from meep_torch.mp.geometry import Cylinder, Lattice
from meep_torch.mp.material import Medium
from meep_torch.mp.vec import Vector3
from meep_torch.mpb.planewave import (planewave_bands_2d,
                                      reciprocal_k_to_cartesian)

from .test_mpb_parity import DATA, TRI

RES = 16
ROD = [Cylinder(0.2, material=Medium(epsilon=12))]
SQUARE = Lattice(size=Vector3(1, 1))

# fdfd values read out of commit 36a466b -- the split, before any backend
# work -- in a throwaway worktree, so this is a real before/after and not a
# transcription of the after.
FDFD_GOLDEN = {
    "tm": [0.18041957059455546, 0.4511915177463854, 0.5449221098777607],
    "te": [0.26172728978742593, 0.5330541191560961, 0.6773750118723678],
}
# Agreement is to 1-2 ULP, not bitwise, and the reason is known: the
# difference operators ARE bitwise identical (|Gx - Dxf| = 0.0 is its own
# check), but the mode normalisation now divides by the cell measure
# |a1 x a2|/(nu nv) where it used to divide by dx*dy. Same number,
# different summation order. rtol is set just above that floor so a real
# change in the operator cannot hide under it.
FDFD_RTOL = 1e-13


def _ms(backend, lattice=SQUARE, k=Vector3(0.3), nb=3, res=RES,
        geometry=None, default_material=None, num_bands=None):
    kw = {} if default_material is None else {"default_material": default_material}
    ms = ModeSolver(num_bands=num_bands or nb, resolution=res, k_points=[k],
                    geometry_lattice=lattice,
                    geometry=geometry if geometry is not None else ROD, **kw)
    ms.backend = backend
    return ms


def test_fdfd_numbers_did_not_move():
    """The default backend did not move when the switch went in.

    Not a physics gate -- the oracle gates in test_mpb_parity.py are that.
    This one exists because a backend switch is only worth having if the
    existing backend is untouched by it.
    """
    for pol, freqs in FDFD_GOLDEN.items():
        ms = _ms("fdfd")
        (ms.run_tm if pol == "tm" else ms.run_te)()
        np.testing.assert_allclose(ms.get_freqs(), freqs, rtol=FDFD_RTOL,
                                   atol=0)


@pytest.mark.parametrize("pol,lattice", [("tm", SQUARE), ("te", SQUARE),
                                         ("tm", TRI)])
def test_planewave_through_modesolver_matches_direct(pol, lattice):
    """Routing must not perturb: ModeSolver == the bare backend call."""
    k = Vector3(0.3) if lattice is SQUARE else Vector3(-1 / 3, 1 / 3)
    ms = _ms("planewave", lattice=lattice, k=k)
    (ms.run_tm if pol == "tm" else ms.run_te)()
    a1, a2 = ms._lattice_vectors()
    eps = ms._eps[0].cpu().numpy()
    direct = planewave_bands_2d(
        eps, a1, a2, reciprocal_k_to_cartesian((k.x, k.y), a1, a2), 3, pol,
        eps_te=None, return_vectors=False)
    np.testing.assert_allclose(ms.get_freqs(), np.asarray(direct),
                               rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("pol", ["tm", "te"])
def test_planewave_fields_share_the_energy_identity(pol):
    """The field machinery is backend-neutral: the same normalisation, the
    same energy identity, the same meep shape."""
    ms = _ms("planewave")
    (ms.run_tm if pol == "tm" else ms.run_te)()
    E = ms.get_efield(2)
    assert E.shape == (RES, RES, 1, 3)
    assert E.dtype == np.complex128
    ms.get_dfield(2)
    assert ms.compute_field_energy()[0] == pytest.approx(1.0, abs=1e-8)
    ms.get_bfield(2)
    assert ms.compute_field_energy()[0] == pytest.approx(1.0, abs=1e-8), (
        "eps|E|^2 and |H|^2 must both integrate to 1 on the planewave "
        "backend too")


@pytest.mark.parametrize("backend", ["fdfd", "planewave"])
def test_no_parity_is_the_union_of_the_polarizations(backend):
    """2D run() = TM and TE merged by frequency. They decouple completely
    in a z-invariant structure, so there is no third operator."""
    k = Vector3(0.3, 0.1)
    tm = _ms(backend, k=k, nb=6)
    tm.run_tm()
    te = _ms(backend, k=k, nb=6)
    te.run_te()
    un = _ms(backend, k=k, nb=6)
    un.run()
    want = sorted([float(f) for f in tm.get_freqs()]
                  + [float(f) for f in te.get_freqs()])[:6]
    np.testing.assert_allclose([float(f) for f in un.get_freqs()], want,
                               rtol=0, atol=1e-12)
    assert un.parity == ""            # meep prints "freqs:" with no prefix
    assert set(un._band_pol) <= {"tm", "te"}
    # and a field getter follows the per-band polarization
    un.get_dfield(1)
    assert un.compute_field_energy()[0] == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("backend,tol", [("fdfd", 0.01), ("planewave", 0.01)])
def test_anisotropic_complete_gap_vs_mpb(backend, tol):
    """The tutorial's anisotropic complete 2D gap: rods eps_diag (1,1,12)
    in a (12,12,1) background, solved with run().

    Three things at once -- a diagonal epsilon tensor, 2D no-parity, and
    the union band numbering -- against
    tests/examples/data/aniso_mpb.json (meep-oracle, 2026-09-08).

    Measured worst-band error: fdfd 0.55%, planewave 0.43%; complete gap
    25.64% / 25.58% against MPB's 26.00%.

    This case is why the oblique TE weights are keyed by COMPONENT. Using
    a single 1/eps there (which is invisible for an isotropic structure)
    put this at 60% off, because ezz is the INVERTED structure relative to
    the in-plane pair TE actually sees.
    """
    ref = json.load(open(DATA / "aniso_mpb.json"))
    ms = ModeSolver(
        num_bands=ref["num_bands"], resolution=ref["resolution"],
        geometry_lattice=TRI,
        k_points=[Vector3(*k) for k in ref["k"]],
        geometry=[Cylinder(ref["r"], material=Medium(
            epsilon_diag=Vector3(*ref["rod_eps_diag"])))],
        default_material=Medium(epsilon_diag=Vector3(*ref["bg_eps_diag"])))
    ms.backend = backend
    ms.run()
    want = np.array(ref["all_freqs"])
    got = np.array(ms.all_freqs)
    nz = want > 1e-9
    worst = float((np.abs(got[nz] - want[nz]) / want[nz]).max())
    gap = ms.gap_list[0][0] if ms.gap_list else 0.0
    print(f"{backend}: worst {worst:.4%}, complete gap {gap:.3f}% "
          f"(mpb {ref['gap_list'][0][0]:.3f}%)")
    assert worst < tol, f"{backend} anisotropic bands off by {worst:.2%}"
    assert abs(gap - ref["gap_list"][0][0]) < 1.0, (
        f"{backend} complete gap {gap:.3f}% vs mpb "
        f"{ref['gap_list'][0][0]:.3f}%")


def test_planewave_targeted_above_dense_max_says_how_to_proceed():
    """planewave's targeted solver is dense-only. Above DENSE_MAX it must
    name the way out rather than allocate a gigabyte."""
    ms = ModeSolver(num_bands=1, resolution=80, k_points=[Vector3(0.5, 0.5)],
                    geometry_lattice=SQUARE, geometry=ROD,
                    target_freq=0.35, tolerance=1e-8)
    ms.backend = "planewave"
    with pytest.raises(NotImplementedError, match='backend = "fdfd"'):
        ms.run_tm()


def test_planewave_defect_field_beats_fdfd_against_meep(tmp_path, monkeypatch):
    """Both backends' Ez for the 5x5 defect mode, against meep's own h5.

    Same fixture as
    test_mpb_parity.test_h5_output_matches_meep_byte_for_byte_metadata,
    run on each backend. Fields are compared on a COMMON grid: MPB samples
    at i*d - L/2 and this engine at cell centres, a systematic half cell
    (wiki/tasks/port_mpb-grid-node-convention.md).

    The spectral backend should win here, and by how much is the number
    worth recording -- it is the clearest single measure of what the
    planewave basis buys over 2nd-order differences on a curved boundary.
    """
    import h5py
    from meep_torch.mp.functions import geometric_objects_lattice_duplicates
    from .test_mpb_parity import _to_cell_centres

    lat = Lattice(size=Vector3(5, 5))
    geom = geometric_objects_lattice_duplicates(
        lat, [Cylinder(0.2, material=Medium(epsilon=12))])
    geom.append(Cylinder(0.2, material=Medium(epsilon=1.0)))
    with h5py.File(DATA / "defect-e.k01.b25.z.tm.h5") as b:
        ref = _to_cell_centres(b["z.r"][()] + 1j * b["z.i"][()])

    corr = {}
    for backend in ("fdfd", "planewave"):
        ms = ModeSolver(num_bands=50, resolution=16, geometry_lattice=lat,
                        geometry=geom, k_points=[Vector3(0.5, 0.5)])
        ms.backend = backend
        ms.filename_prefix = backend
        monkeypatch.chdir(tmp_path)
        ms.run_tm()
        mp_mpb.output_efield_z(ms, 25)
        with h5py.File(tmp_path / f"{backend}-e.k01.b25.z.tm.h5") as a:
            f = a["z.r"][()] + 1j * a["z.i"][()]
        corr[backend] = (abs(np.vdot(ref, f))
                         / (np.linalg.norm(f) * np.linalg.norm(ref)))
    print(f"defect Ez vs meep: fdfd {corr['fdfd']:.6f}, "
          f"planewave {corr['planewave']:.6f}")
    for backend, c in corr.items():
        assert c > 0.99, f"{backend} Ez correlation {c:.5f}"
    assert corr["planewave"] >= corr["fdfd"] - 1e-6, (
        "the spectral backend should not be WORSE than 2nd-order "
        "differences on this mode")
