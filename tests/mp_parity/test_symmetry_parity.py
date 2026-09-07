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
    assert rel_ez < 0.01, f"de-mirrored Ez off by {rel_ez:.4f}"
    assert rel_p < 0.01, f"flux off by {rel_p:.4f}"
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
    assert rel_ez < 0.01, f"de-rotated Ez off by {rel_ez:.4f}"
    assert rel_p < 0.01, f"flux off by {rel_p:.4f}"
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
    # The full-cell run does not obey C4 exactly: the cpml half-offset ramp
    # is half a cell asymmetric edge to edge, and a quarter turn maps every
    # edge onto the next, so all four disagree. Measure that self-asymmetry
    # and require the quadrant run (which is C4-exact by construction) to sit
    # inside it -- a tighter statement than a fixed tolerance, and it does
    # not silently absorb a real fold error, which would show up AT the cut
    # planes rather than decaying inward from the PML bands.
    self_err = np.abs(ez_full - phase * ez_full[::-1, :].T)[n:-n, n:-n].max() / scale
    print(f"Rotate4(Z, {phase:+d}): |dEz|/max={rel_ez:.2e} dP/P={rel_p:.2e} "
          f"(full-cell self-C4 asymmetry {self_err:.2e})")
    assert rel_ez < 1.2 * max(self_err, 1e-3), \
        f"de-rotated Ez off by {rel_ez:.4f}, beyond the reference's own {self_err:.4f}"
    assert rel_p < 0.01, f"flux off by {rel_p:.4f}"
    # C4 invariance of the de-rotated array: a quarter turn is a transpose
    # plus a flip on the grid
    np.testing.assert_allclose(ez_q, phase * ez_q[::-1, :].T,
                               atol=1e-12 * scale)
