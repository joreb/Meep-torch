# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Mirror symmetry half-domain parity (mp.Mirror wiring, both axes/parities).

Gate: for each of Mirror(Y,+1), Mirror(Y,-1), Mirror(X,+1), Mirror(X,-1) a
TM point-dipole run on the half cell matches the full-cell run: de-mirrored
Ez arrays < 1% (of the field max) and a plane-crossing flux < 1%.

Source convention (mp.Mirror docstring): each physical source is listed
once; an off-plane source implies its mirror image. The full-cell reference
therefore lists the source AND its image explicitly, the image amplitude
carrying the mirror phase (Ez transforms with the user phase).
"""
import numpy as np
import pytest

import meep_torch.mp as mp

RES = 20
DX = 1.0 / RES
FCEN = 1.0
# offsets on cell centers so the mirror image lands exactly on a grid row
OFF = (20 + 0.5) * DX          # 1.025 um off the cut plane
# Half-domain runs reproduce the full cell to roundoff (~4e-15 measured) since
# the CPML high-edge anchor fix; see the comment in the mirror gate below.
TOL = 1e-9


def _run(direction, phase, half):
    if direction == mp.Y:
        src_pos = [mp.Vector3(0, OFF)]
        if not half:
            src_pos.append(mp.Vector3(0, -OFF))
        flux = mp.FluxRegion(center=mp.Vector3(2.5, 0), size=mp.Vector3(0, 4))
    else:
        src_pos = [mp.Vector3(OFF, 0)]
        if not half:
            src_pos.append(mp.Vector3(-OFF, 0))
        flux = mp.FluxRegion(center=mp.Vector3(0, 2.5), size=mp.Vector3(4, 0))
    sources = [mp.Source(mp.GaussianSource(FCEN, fwidth=0.2 * FCEN),
                         component=mp.Ez, center=p,
                         amplitude=(phase if i else 1.0))
               for i, p in enumerate(src_pos)]
    sim = mp.Simulation(
        cell_size=mp.Vector3(8, 8), resolution=RES,
        boundary_layers=[mp.PML(1.0)],
        sources=sources,
        symmetries=[mp.Mirror(direction, phase=phase)] if half else None,
    )
    f = sim.add_flux(FCEN, 0, 1, flux)
    sim.run(until=25)
    return sim.get_array(component=mp.Ez), mp.get_fluxes(f)[0]


@pytest.mark.parametrize("direction,phase", [
    (mp.Y, +1), (mp.Y, -1), (mp.X, +1), (mp.X, -1)])
def test_mirror_half_domain_matches_full(direction, phase):
    ez_full, p_full = _run(direction, phase, half=False)
    ez_half, p_half = _run(direction, phase, half=True)
    assert ez_half.shape == ez_full.shape == (160, 160)
    scale = np.abs(ez_full).max()
    # compare the physical interior: the full cell's LOW-side half-offset
    # CPML profile is half a cell off its high-side mirror (pre-existing
    # cpml_profiles convention), so inside the PML bands the full run is
    # itself asymmetric at the ~1% level; the mirror run is exactly symmetric
    n = RES  # 1 um PML
    d = np.abs(ez_half - ez_full)[n:-n, n:-n]
    rel_ez = d.max() / scale
    rel_p = abs(p_half - p_full) / abs(p_full)
    print(f"Mirror({'XY'[direction == mp.Y]}, {phase:+d}): "
          f"|dEz|/max={rel_ez:.2e} dP/P={rel_p:.2e}")
    # Tightened 2026-09-07 from 1e-2 to EXACT-ish. Before the CPML high-edge
    # anchor fix these ran at 3.6e-03 to 6.2e-03, entirely because the full
    # cell's low edge was aligned and its high edge was not. With both edges
    # aligned the half-domain run reproduces the full cell to roundoff
    # (measured 1.9e-15 to 4.1e-15), so a 1e-2 gate had become vacuous.
    # TOL sits ~6 orders above the observed value and ~6 below the old gate;
    # if a float32 or GPU run ever trips it, raise TOL and say so -- do not
    # read it as a physics regression without checking dtype first.
    assert rel_ez < TOL, f"de-mirrored Ez off by {rel_ez:.4e} (tol {TOL:g})"
    assert rel_p < TOL, f"flux off by {rel_p:.4e} (tol {TOL:g})"
    # the de-mirrored array respects the parity by construction
    flip = ez_half[:, ::-1] if direction == mp.Y else ez_half[::-1, :]
    np.testing.assert_allclose(ez_half, phase * flip, atol=1e-12 * scale)


# --- Rotate2(Z): C2 fold, halves in y while keeping the full x extent ------

def _run_rot2(phase, half):
    """C2 about z through the cell centre. The image of (x, y) is (-x, -y),
    so the full-cell reference lists the dipole AND its rotated partner; the
    half run lists it once and the fold supplies the partner."""
    src_pos = [mp.Vector3(OFF, OFF)]
    if not half:
        src_pos.append(mp.Vector3(-OFF, -OFF))
    sources = [mp.Source(mp.GaussianSource(FCEN, fwidth=0.2 * FCEN),
                         component=mp.Ez, center=p,
                         amplitude=(phase if i else 1.0))
               for i, p in enumerate(src_pos)]
    sim = mp.Simulation(
        cell_size=mp.Vector3(8, 8), resolution=RES,
        boundary_layers=[mp.PML(1.0)],
        sources=sources,
        symmetries=[mp.Rotate2(mp.Z, phase=phase)] if half else None,
    )
    # wholly inside the kept (y >= 0) half: a region crossing the C2 cut
    # would need its partner line at nx-1-i summed in, which _rot2_guard
    # rejects rather than approximates
    f = sim.add_flux(FCEN, 0, 1,
                     mp.FluxRegion(center=mp.Vector3(2.5, 1.0),
                                   size=mp.Vector3(0, 2)))
    sim.run(until=25)
    return sim.get_array(component=mp.Ez), mp.get_fluxes(f)[0]


@pytest.mark.parametrize("phase", [+1, -1])
def test_rotate2_half_domain_matches_full(phase):
    ez_full, p_full = _run_rot2(phase, half=False)
    ez_half, p_half = _run_rot2(phase, half=True)
    assert ez_half.shape == ez_full.shape == (160, 160)
    scale = np.abs(ez_full).max()
    n = RES                                  # skip the PML bands, as above
    rel_ez = np.abs(ez_half - ez_full)[n:-n, n:-n].max() / scale
    rel_p = abs(p_half - p_full) / abs(p_full)
    print(f"Rotate2(Z, {phase:+d}): |dEz|/max={rel_ez:.2e} dP/P={rel_p:.2e}")
    assert rel_ez < TOL, f"de-rotated Ez off by {rel_ez:.4e} (tol {TOL:g})"
    assert rel_p < TOL, f"flux off by {rel_p:.4e} (tol {TOL:g})"
    # C2 invariance of the de-rotated array: Ez(r) = phase * Ez(-r)
    np.testing.assert_allclose(ez_half, phase * ez_half[::-1, ::-1],
                               atol=1e-12 * scale)


def test_rotate2_region_below_cut_raises():
    """A monitor reaching below the C2 plane is rejected, not silently
    folded like a mirror's (the C2 image lands on a different line)."""
    sim = mp.Simulation(
        cell_size=mp.Vector3(8, 8), resolution=RES,
        boundary_layers=[mp.PML(1.0)],
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=0.2 * FCEN),
                           component=mp.Ez, center=mp.Vector3(OFF, OFF))],
        symmetries=[mp.Rotate2(mp.Z)])
    with pytest.raises(NotImplementedError):
        sim.add_flux(FCEN, 0, 1,
                     mp.FluxRegion(center=mp.Vector3(2.5, 0),
                                   size=mp.Vector3(0, 4)))


# --- Rotate4(Z): C4 fold onto one quadrant ---------------------------------

def _run_rot4(phase, quarter):
    """C4 about z. With Ez(r) = p Ez(Rr) the source distribution obeys
    S(r) = p S(Rr), so a dipole at r0 implies partners at R^-1 r0 (amp p),
    R^-2 r0 = -r0 (amp p^2 = 1) and R^-3 r0 (amp p^3 = p); R is the quarter
    turn (u, v) -> (-v, u). The full-cell reference lists all four."""
    src_pos = [(mp.Vector3(OFF, OFF), 1.0)]
    if not quarter:
        src_pos += [(mp.Vector3(OFF, -OFF), phase),
                    (mp.Vector3(-OFF, -OFF), 1.0),
                    (mp.Vector3(-OFF, OFF), phase)]
    sources = [mp.Source(mp.GaussianSource(FCEN, fwidth=0.2 * FCEN),
                         component=mp.Ez, center=c, amplitude=a)
               for c, a in src_pos]
    sim = mp.Simulation(
        cell_size=mp.Vector3(8, 8), resolution=RES,
        boundary_layers=[mp.PML(1.0)],
        sources=sources,
        symmetries=[mp.Rotate4(mp.Z, phase=phase)] if quarter else None,
    )
    f = sim.add_flux(FCEN, 0, 1,
                     mp.FluxRegion(center=mp.Vector3(2.5, 1.5),
                                   size=mp.Vector3(0, 2)))
    sim.run(until=25)
    return sim.get_array(component=mp.Ez), mp.get_fluxes(f)[0]


@pytest.mark.parametrize("phase", [+1, -1])
def test_rotate4_quadrant_matches_full(phase):
    ez_full, p_full = _run_rot4(phase, quarter=False)
    ez_q, p_q = _run_rot4(phase, quarter=True)
    assert ez_q.shape == ez_full.shape == (160, 160)
    scale = np.abs(ez_full).max()
    n = RES
    rel_ez = np.abs(ez_q - ez_full)[n:-n, n:-n].max() / scale
    rel_p = abs(p_q - p_full) / abs(p_full)
    # The full-cell run used to disagree with its own quarter turn at 1.0e-02,
    # which this gate was originally written around ("the cpml half-offset
    # ramp is asymmetric edge to edge"). That was not a quirk: it was the
    # CPML high-edge anchor bug, and with it fixed the reference is C4-exact
    # to roundoff (1.4e-15). The self-asymmetry is still measured, because a
    # reference that stops being C4-exact is itself worth catching, but it no
    # longer sets the tolerance. See wiki/tasks/fix_cpml-entrance-reflection.
    self_err = np.abs(ez_full - phase * ez_full[::-1, :].T)[n:-n, n:-n].max() / scale
    print(f"Rotate4(Z, {phase:+d}): |dEz|/max={rel_ez:.2e} dP/P={rel_p:.2e} "
          f"(full-cell self-C4 asymmetry {self_err:.2e})")
    assert self_err < TOL, \
        f"the full-cell reference is no longer C4-exact ({self_err:.4e}): fix " \
        f"that before reading the comparison below"
    assert rel_ez < TOL, f"de-rotated Ez off by {rel_ez:.4e} (tol {TOL:g})"
    assert rel_p < TOL, f"flux off by {rel_p:.4e} (tol {TOL:g})"
    # C4 invariance of the de-rotated array: a quarter turn is a transpose
    # plus a flip on the grid
    np.testing.assert_allclose(ez_q, phase * ez_q[::-1, :].T,
                               atol=1e-12 * scale)


# --- staggered components unfold too (Hx/Hy under every fold) --------------

def _run_h(symmetries, src_pos):
    """Ez dipole run returning all three TM components on the FULL cell.

    The source list has to obey the symmetry as well as the structure, so the
    full-cell reference carries every partner the fold implies. A lone
    off-centre dipole is a different physical problem from its folded
    counterpart, and comparing the two measures the source layout rather than
    the unfold -- which is how this gate first "failed" while the code was
    right. OFF sits on a cell centre so each image lands exactly on a grid row.
    """
    sim = mp.Simulation(
        cell_size=mp.Vector3(8, 8), resolution=RES,
        boundary_layers=[mp.PML(1.0)],
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=0.2 * FCEN),
                           component=mp.Ez, center=c, amplitude=a)
                 for c, a in src_pos],
        symmetries=symmetries or None)
    sim.run(until=25)
    return {c: np.asarray(sim.get_array(component=c))
            for c in ("Ez", "Hx", "Hy")}


def _h_cases(p):
    V = mp.Vector3
    one = [(V(OFF, OFF), 1.0)]
    return {
        "Mirror(Y)": ([mp.Mirror(mp.Y, phase=p)], one,
                      one + [(V(OFF, -OFF), p)]),
        "Mirror(X)": ([mp.Mirror(mp.X, phase=p)], one,
                      one + [(V(-OFF, OFF), p)]),
        "Rotate2": ([mp.Rotate2(mp.Z, phase=p)], one,
                    one + [(V(-OFF, -OFF), p)]),
        "Rotate4": ([mp.Rotate4(mp.Z, phase=p)], one,
                    one + [(V(OFF, -OFF), p), (V(-OFF, -OFF), 1.0),
                           (V(-OFF, OFF), p)]),
    }


@pytest.mark.parametrize("phase", [+1, -1])
@pytest.mark.parametrize("name", ["Mirror(Y)", "Mirror(X)", "Rotate2", "Rotate4"])
def test_staggered_components_unfold(name, phase):
    """Hx and Hy de-symmetrise onto the full cell, not just Ez.

    These were refused outright before: the fold stores H one short along the
    folded axis (H lives on the gaps between E nodes) and keeps the on-cut
    node aside, so the naive unfold came out a row short. Under C4 it is worse
    than bookkeeping -- a quarter turn maps the Hx grid onto the Hy grid, so
    the full-cell Hx is assembled from folded Hx on two quadrants and folded
    HY on the other two.
    """
    syms, fold_src, full_src = _h_cases(phase)[name]
    full = _run_h([], full_src)
    got = _run_h(syms, fold_src)
    n = RES
    for comp in ("Ez", "Hx", "Hy"):
        a, b = got[comp], full[comp]
        assert a.shape == b.shape, f"{name} {phase:+d} {comp}: {a.shape} vs {b.shape}"
        rel = np.abs(a - b)[n:-n, n:-n].max() / np.abs(b).max()
        assert rel < TOL, f"{name} {phase:+d} {comp} off by {rel:.3e} (tol {TOL:g})"


# --- the structure is checked against the declared symmetry ----------------
#
# meep does not make this check. Its docs: "Meep does not check whether the
# symmetry is obeyed. If you specify a symmetry that does not preserve your
# structure/sources, then the results are undefined." Undefined means silently
# wrong, because the fold never rasterises the half that disagrees.

def _sim_with(geometry, symmetries):
    sim = mp.Simulation(
        cell_size=mp.Vector3(8, 8), resolution=RES, geometry=geometry,
        sources=[mp.Source(mp.ContinuousSource(FCEN), mp.Ez,
                           center=mp.Vector3(OFF, OFF))],
        symmetries=symmetries, boundary_layers=[mp.PML(1.0)])
    sim.init_sim()
    return sim


_CENTRED = [mp.Block(size=mp.Vector3(2, 2), center=mp.Vector3(0, 0),
                     material=mp.Medium(index=3.4))]
_OFFSET_X = [mp.Block(size=mp.Vector3(2, 2), center=mp.Vector3(1.0, 0),
                      material=mp.Medium(index=3.4))]


@pytest.mark.parametrize("geometry,symmetries", [
    ([], [mp.Mirror(mp.Y, phase=+1)]),                 # vacuum
    (_CENTRED, [mp.Mirror(mp.Y, phase=+1)]),
    (_CENTRED, [mp.Mirror(mp.X, phase=-1)]),
    (_CENTRED, [mp.Rotate2(mp.Z, phase=+1)]),
    (_CENTRED, [mp.Rotate4(mp.Z, phase=+1)]),
    # offset in x is genuinely symmetric about y -- the check must not object
    (_OFFSET_X, [mp.Mirror(mp.Y, phase=+1)]),
])
def test_symmetric_structures_are_accepted(geometry, symmetries):
    _sim_with(geometry, symmetries)


@pytest.mark.parametrize("symmetries", [
    [mp.Mirror(mp.X, phase=+1)],
    [mp.Mirror(mp.X, phase=-1)],
    [mp.Rotate2(mp.Z, phase=+1)],
    [mp.Rotate4(mp.Z, phase=+1)],
])
def test_asymmetric_structure_is_refused(symmetries):
    """The half of this gate that makes the other half mean something.

    A block offset in x has none of these symmetries. Before this check the
    run proceeded and quietly solved a different problem -- the fold keeps the
    x >= cut half and the offset block's other side simply never existed.
    """
    with pytest.raises(ValueError, match="not obeyed by the structure"):
        _sim_with(_OFFSET_X, symmetries)
