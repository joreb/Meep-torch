# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""TE mirror symmetry half-domain parity (mp.Mirror wiring, both axes/parities).

Gate: for each of Mirror(Y,+1), Mirror(Y,-1), Mirror(X,+1), Mirror(X,-1) a
TE point-dipole run on the half cell matches the full-cell run -- measured
against the full-cell run's OWN symmetry violation, so the gate is a
statement about the mirror machinery, not about an arbitrary tolerance.

Source convention (mp.Mirror docstring / meep parity rules): each physical
source is listed once; an off-plane source implies its mirror image. The
full-cell reference lists the source AND its image explicitly, the image
amplitude carrying the transformation sign of the source component. An Ey
current source transforms as -phase about a Y cut (Ey is normal to the
plane) and +phase about an X cut (Ey tangential); the field comparison is
on Hz, which transforms as -phase about BOTH cuts.

TE staggering: Hz/Ey live on half-offset rows, so a full-cell TE grid is
not node-symmetric about the cut -- the low-side outermost row has no
mirror partner (it sits deep in the PML). The kept half rows map onto full
rows cx..nx-2 / cy..ny-2, so the de-mirror below maps the reconstructed
array onto full rows/cols 1..n-2 and the comparison crops the PML bands.
(The de-mirror is done here on the raw arrays rather than through
get_array(component=Hz), which once passed no mirror parity and had its flip
mapping off by one. That path is fixed and gated separately, below, by
test_te_get_array_hz_demirror.)

Self-asymmetry reference: the full-cell run has no symmetry machinery, but
it is only approximately symmetric about the cut (the low/high CPML ramps
differ at half-offset staggers). Its own violation is measured two ways --
de-mirroring the full run's kept half against itself (fields) and doubling
the kept-half span of its flux plane (flux) -- and the half-vs-full
discrepancy must sit inside 1.2x that reference. A bare 1% assert cannot
discriminate "as symmetric as the unsymmetrized reference" from "broken".
"""
import numpy as np
import pytest

import meep_torch.mp as mp

RES = 20
DX = 1.0 / RES
FCEN = 1.0
OFF_Y = (20 + 0.5) * DX   # 1.025: exactly on an Ey node row (half-y staging)
OFF_X = 20 * DX           # 1.0: exactly on an Ey node column (int-x staging)


def _run(direction, phase, half):
    if direction == mp.Y:
        src_pos = [(0.0, OFF_Y)]
        if not half:
            src_pos.append((0.0, -OFF_Y))
        img_phase = -phase       # Ey normal to a Y cut
        flux = mp.FluxRegion(center=mp.Vector3(2.5, 0), size=mp.Vector3(0, 4))
        flux_h = mp.FluxRegion(center=mp.Vector3(2.5, 1.0), size=mp.Vector3(0, 2))
    else:
        src_pos = [(OFF_X, OFF_Y)]
        if not half:
            src_pos.append((-OFF_X, OFF_Y))
        img_phase = phase        # Ey tangential to an X cut
        flux = mp.FluxRegion(center=mp.Vector3(0, 2.5), size=mp.Vector3(4, 0))
        flux_h = mp.FluxRegion(center=mp.Vector3(1.0, 2.5), size=mp.Vector3(2, 0))
    sources = [mp.Source(mp.GaussianSource(FCEN, fwidth=0.2 * FCEN),
                         component=mp.Ey, center=mp.Vector3(x, y),
                         amplitude=(img_phase if i else 1.0))
               for i, (x, y) in enumerate(src_pos)]
    sim = mp.Simulation(
        cell_size=mp.Vector3(8, 8), resolution=RES,
        boundary_layers=[mp.PML(1.0)],
        sources=sources,
        symmetries=[mp.Mirror(direction, phase=phase)] if half else None,
    )
    f = sim.add_flux(FCEN, 0, 1, flux)
    # kept-half span of the SAME plane: in the full run it doubles into the
    # self-asymmetry reference; in the half run it is the folded plane itself
    fh = sim.add_flux(FCEN, 0, 1, flux_h)
    sim.run(until=25)
    hz = sim._state.Hz.detach().cpu().numpy()
    return sim, hz, mp.get_fluxes(f)[0], mp.get_fluxes(fh)[0]


def _demirror(hz_half, cx, cy, px, py):
    """Kept-half Hz -> reconstructed full-cell rows/cols 1..n-2 (Hz carries
    -phase about both cuts; the mirrored half is the phase-flipped copy)."""
    a = hz_half
    if cy:
        a = np.concatenate([py * a[:, ::-1], a], axis=1)
    if cx:
        a = np.concatenate([px * a[::-1, :], a], axis=0)
    return a


@pytest.mark.parametrize("direction,phase", [
    (mp.Y, +1), (mp.Y, -1), (mp.X, +1), (mp.X, -1)])
def test_mirror_half_domain_matches_full(direction, phase):
    sim_half, hz_half, p_half, ph_half = _run(direction, phase, half=True)
    _, hz_full, p_full, ph_full = _run(direction, phase, half=False)
    # the mirror must actually halve the domain (the old behaviour warned and
    # kept the full cell)
    if direction == mp.Y:
        assert sim_half.grid.ny == RES * 4, "Mirror(Y) did not halve the cell"
    else:
        assert sim_half.grid.nx == RES * 4, "Mirror(X) did not halve the cell"
    # Hz parity: -phase about either cut
    cx = sim_half._sym_cx
    cy = sim_half._sym_cy
    dem = _demirror(hz_half, cx, cy, -phase, -phase)
    # align with the full cell: reconstructed rows/cols k <-> full rows/cols k+1
    full = hz_full[1:] if cx else hz_full
    full = full[:, 1:] if cy else full
    assert dem.shape == full.shape, f"{dem.shape} vs {full.shape}"
    scale = np.abs(full).max()
    # self-asymmetry reference (no symmetry machinery): de-mirror the FULL
    # run's own kept half against itself over the same interior window
    mid = RES * 4
    a = hz_full[mid:, :] if cx else hz_full      # the full run's kept half
    a = a[:, mid:] if cy else a
    self_dem = _demirror(a, cx, cy, -phase, -phase)
    self_full = full                             # hz_full already cropped
    n = RES
    d = np.abs(dem - full)[n:-n, n:-n]
    self_d = np.abs(self_dem - self_full)[n:-n, n:-n]
    rel_hz = d.max() / scale
    self_hz = self_d.max() / scale
    # flux: half run vs full run, referenced against the full run doubling
    # its own kept-half plane span
    rel_p = abs(p_half - p_full) / abs(p_full)
    self_p = abs(2.0 * ph_full - p_full) / abs(p_full)
    # the half run's folded plane must also match its own reference span
    # (p_half carries the crossing-plane mir2 doubling; ph_half does not)
    assert abs(2.0 * ph_half - p_half) < 0.05 * abs(p_full), \
        "folded plane-crossing flux drifted from the kept-half span"
    print(f"TE Mirror({'XY'[direction == mp.Y]}, {phase:+d}): "
          f"|dHz|/max={rel_hz:.2e} (self {self_hz:.2e}) "
          f"dP/P={rel_p:.2e} (self {self_p:.2e})")
    # roundoff floor; a half-vs-full run agreeing to 1e-8 is exact, the ratio
    # only governs above it. Before the CPML high-edge anchor fix these ran at
    # ~7e-04 / ~2e-04 and the "self-asymmetry" they are measured against WAS
    # that bug: the full cell's low edge was aligned and its high edge was
    # not. With both aligned the full cell is symmetric to roundoff, so the
    # ratio would otherwise be dividing two noise-floor quantities.
    assert rel_hz <= 1.2 * self_hz + 1e-7, \
        f"de-mirrored Hz {rel_hz:.4e} exceeds the full run's own " \
        f"asymmetry {self_hz:.4e} by more than 1.2x"
    assert rel_p <= 1.2 * self_p + 1e-7, \
        f"flux {rel_p:.4e} exceeds the full run's own asymmetry " \
        f"{self_p:.4e} by more than 1.2x"
    # the reconstructed array respects the parity by construction
    if cy:
        np.testing.assert_allclose(dem, -phase * dem[:, ::-1],
                                   atol=1e-12 * scale)
    if cx:
        np.testing.assert_allclose(dem, -phase * dem[::-1, :],
                                   atol=1e-12 * scale)


def test_te_offplane_ey_source_fold():
    """An off-plane Ey source under a Y cut implies its image with -phi (Ey is
    normal to the plane). The half run lists the source once, in the KEPT
    half, and the wall supplies the image; the full run lists both, the image
    amplitude carrying -phi. A wrong wall sign flips the image amplitude and
    the two sources cancel/double -- an O(1) mismatch, so a 1% gate is
    decisive (the correct sign sits at the ~1e-3 CPML level).

    The source used to be listed BELOW the cut here, on the assumption that
    the engine folds it up. It does not, and neither does meep: sources.cpp
    intersects source volumes with the untransformed reduced chunk only, so a
    source in the discarded half is DROPPED (oracle: max|Ez| == 0.0 exactly).
    Folding it in instead double-counted the drive."""
    for phase in (+1, -1):
        def run(half):
            src_pos = [(0.0, OFF_Y)]             # in the kept half
            if not half:
                src_pos.append((0.0, -OFF_Y))    # explicit image, sign -phi
            img_phase = -phase                   # Ey normal to a Y cut
            sources = [mp.Source(mp.GaussianSource(FCEN, fwidth=0.2 * FCEN),
                                 component=mp.Ey, center=mp.Vector3(x, y),
                                 amplitude=(img_phase if i else 1.0))
                       for i, (x, y) in enumerate(src_pos)]
            sim = mp.Simulation(
                cell_size=mp.Vector3(8, 8), resolution=RES,
                boundary_layers=[mp.PML(1.0)],
                sources=sources,
                symmetries=[mp.Mirror(mp.Y, phase=phase)] if half else None,
            )
            f = sim.add_flux(FCEN, 0, 1,
                             mp.FluxRegion(center=mp.Vector3(2.5, 0),
                                           size=mp.Vector3(0, 4)))
            sim.run(until=25)
            hz = sim._state.Hz.detach().cpu().numpy()
            return sim, hz, mp.get_fluxes(f)[0]
        sim_half, hz_half, p_half = run(half=True)
        _, hz_full, p_full = run(half=False)
        assert sim_half.grid.ny == RES * 4, "Mirror(Y) did not halve the cell"
        dem = _demirror(hz_half, 0, sim_half._sym_cy, -phase, -phase)
        full = hz_full[:, 1:]
        assert dem.shape == full.shape, f"{dem.shape} vs {full.shape}"
        scale = np.abs(full).max()
        n = RES
        d = np.abs(dem - full)[n:-n, n:-n]
        rel_hz = d.max() / scale
        rel_p = abs(p_half - p_full) / abs(p_full)
        print(f"TE off-plane Ey Mirror(Y, {phase:+d}): "
              f"|dHz|/max={rel_hz:.2e} dP/P={rel_p:.2e}")
        assert rel_hz < 0.01, \
            f"de-mirrored Hz {rel_hz:.4e} off-plane fold (Ey sign wrong?)"
        assert rel_p < 0.01, \
            f"flux {rel_p:.4e} off-plane fold (Ey sign wrong?)"


def test_te_get_array_hz_demirror():
    """get_array(component=Hz) under a mirror uses the TE de-mirror (kept
    half -> full rows/cols cx..nx-2, mirrored half -> 1..cx-1, row/col 0
    zeroed, -phi parity) and matches the full-cell run over the interior.
    The reconstructed plane is (nx, ny) = (160, 160) at RES=20."""
    for direction, phase in [(mp.Y, +1), (mp.Y, -1), (mp.X, +1), (mp.X, -1)]:
        sim_half, _, _, _ = _run(direction, phase, half=True)
        sim_full, _, _, _ = _run(direction, phase, half=False)
        a_half = sim_half.get_array(component=mp.Hz)
        a_full = sim_full.get_array(component=mp.Hz)
        # RES*8, not RES*8-1: get_array reports every component on the
        # CELL-CENTRE grid as meep does, so Hz is the full (nx, ny) rather
        # than its staggered subgrid. The property under test -- the
        # half-domain reconstruction equals the full-domain field -- is
        # unchanged by that.
        assert a_half.shape == a_full.shape == (RES * 8, RES * 8), \
            f"{a_half.shape} vs {a_full.shape}"
        n = RES
        d = np.abs(a_half - a_full)[n:-n, n:-n]
        scale = np.abs(a_full).max()
        rel = d.max() / scale
        print(f"TE get_array(Hz) Mirror({'XY'[direction == mp.Y]}, {phase:+d}): "
              f"|dHz|/max={rel:.2e}")
        assert rel < 0.01, f"get_array(Hz) de-mirror {rel:.4e}"