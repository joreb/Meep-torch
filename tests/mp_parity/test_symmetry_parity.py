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
