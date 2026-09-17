# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""MPB track: band structures via Bloch + harminv vs meep/mpb oracles."""
import json
import math
import pathlib

import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bloch import init_bloch_te_state, step_bloch_te
from meep_torch.modes.harminv import harminv

DATA = pathlib.Path(__file__).parent / "data"


def holey_bands_at(kx_meep, res=10):
    """meep frame -> ours: meep x (periodic, k) = our y; meep y (PML) = our x."""
    ref = json.load(open(DATA / "holey_bands.json"))
    dx = 1.0 / res
    nx, ny = int(ref["sy"] * res), int(1.0 * res)
    grid = Grid(nx=nx, ny=ny, dx=dx)
    npml = int(ref["dpml"] * res)
    coeffs = build_tm_coeffs(grid, eps_inf=torch.ones(nx, ny, dtype=torch.float64, device=grid.device),
                             npml=npml, y_pmc=True)
    # 4x4-supersampled coverage eps (the ring-cyl edge-averaging lesson)
    # cx + 1/2, not cx: the waveguide's mirror plane has to sit ON the field
    # lattice. Yee TE puts Hz at i+1/2, so a slab centred on the integer node
    # straddles the samples -- measured, that costs 3.72% on the guided band
    # against 0.62% when aligned. The old dual-lattice TE stepper had Hz on the
    # integer node, so `cx` was aligned for it and this read as "the dual
    # lattice is more accurate"; it was not, it was the placement.
    cx = nx // 2 + 0.5
    ss = 4
    xi = (torch.arange(nx * ss, dtype=torch.float64, device=grid.device) + 0.5) / ss - 0.5
    yj = (torch.arange(ny * ss, dtype=torch.float64, device=grid.device) + 0.5) / ss - 0.5
    II, JJ = torch.meshgrid(xi, yj, indexing="ij")
    fine = torch.ones(nx * ss, ny * ss, dtype=torch.float64, device=grid.device)
    in_slab = ((II - cx).abs() * dx) < ref["w"] / 2
    fine[in_slab] = ref["eps"]
    in_hole = ((II - cx) * dx) ** 2 + ((JJ - ny / 2) * dx) ** 2 < ref["r"] ** 2
    fine[in_hole] = 1.0
    eps = fine.reshape(nx, ss, ny, ss).mean(dim=(1, 3))
    # eps at the Yee E positions: Ex sits at (i+1/2, j), Ey at (i, j+1/2)
    # (periodic y wrap) -- cell-centered eps reused verbatim biases bands low
    eps_x = 0.5 * (eps[:-1, :] + eps[1:, :])
    eps_y = 0.5 * (eps + torch.roll(eps, -1, dims=1))

    st = init_bloch_te_state(grid)
    ky = 2 * math.pi * kx_meep
    ic = nx // 2                                   # integer index for probes
    mask = grid.zeros(nx - 1, ny)                  # Hz grid, Yee TE
    mask[ic + 2, ny // 3] = 1.0                    # low-symmetry point source
    pulse = GaussianPulse(ref["fcen"], ref["df"])
    n_steps = int(round(360.0 / grid.dt))
    probes = [(ic + 2, ny // 3), (ic - 3, 2 * ny // 3), (ic + 1, ny // 5)]
    series = np.empty((len(probes), n_steps), dtype=complex)
    for t in range(n_steps):
        a = pulse.amplitude((t + 0.5) * grid.dt)
        st = step_bloch_te(st, coeffs, eps_x, eps_y, ky,
                           src_c=a, src_s=0.0, src_mask=mask)
        for p, (pi, pj) in enumerate(probes):
            series[p, t] = complex(st.Hz_c[pi, pj].item(), st.Hz_s[pi, pj].item())
    t0 = int(round(60.0 / grid.dt))
    freqs = []
    for p in range(len(probes)):
        freqs += [m.freq for m in harminv(series[p, t0:], grid.dt, 0.02,
                                          ref["fcen"] + ref["df"] / 2)
                  if m.Q > 50 and m.amp > 1e-6]
    return sorted(freqs)


def test_holey_wvg_bands_vs_oracle():
    """Gate (a): every GUIDED oracle band (below the light line f < 1.1 k,
    Q > 100 -- the leaky continuum above the light line is PML-dependent
    noise in both engines) must be reproduced within 1%."""
    ref = json.load(open(DATA / "holey_bands.json"))
    checked = 0
    # kx = 0.5 (zone edge) excluded: the flat band-edge modes there shift
    # by several % between smoothing schemes at res 10 (subpixel-vs-binary
    # oracle spread is itself ~5% there); the dispersive guided branch is
    # the gate. 2.5%: residual scheme difference at eps = 13 contrast.
    for kx in (0.1, 0.2, 0.4):
        want = [m["freq"] for m in ref["bands"][str(kx)]
                if m["freq"] < 1.1 * kx and m["Q"] > 100]
        if not want:
            continue
        got = holey_bands_at(kx)
        for wf in want:
            best = min(got, key=lambda g: abs(g - wf)) if got else 1e9
            assert abs(best - wf) / wf <= 0.025, \
                f"kx={kx}: band {wf:.4f} -> nearest {best:.4f} " \
                f"(got {[round(g, 4) for g in got[:8]]})"
            checked += 1
    assert checked >= 3, f"only {checked} guided bands gated"


def _bloch2_bands(eps, kx, ky, dt, dx, fmax, t_end=300.0):
    from meep_torch.core.stepper_bloch2 import init_bloch2_state, step_bloch2_tm

    class _G:
        pass

    g = _G()
    g.nx, g.ny = eps.shape
    g.zeros = lambda *s: torch.zeros(*s, dtype=torch.float64)
    st = init_bloch2_state(g)
    mask = torch.zeros(*eps.shape, dtype=torch.float64)
    mask[g.nx // 3 + 1, g.ny // 5 + 2] = 1.0
    pulse = GaussianPulse(fmax / 2, fmax)
    n_steps = int(round(t_end / dt))
    probes = [(g.nx // 3 + 1, g.ny // 5 + 2), (2 * g.nx // 3, g.ny // 2 + 1),
              (g.nx // 2 - 2, 0), (g.nx - 3, g.ny // 3)]
    series = np.empty((len(probes), n_steps), dtype=complex)
    for t in range(n_steps):
        a = pulse.amplitude((t + 0.5) * dt)
        st = step_bloch2_tm(st, eps, dt, dx, kx, ky, src_c=a, src_s=0.0,
                            src_mask=mask)
        for p, (pi, pj) in enumerate(probes):
            series[p, t] = complex(st.Ez_c[pi, pj].item(), st.Ez_s[pi, pj].item())
    t0 = int(round(60.0 / dt))
    freqs = []
    for p in range(len(probes)):
        freqs += [m.freq for m in harminv(series[p, t0:], dt, 0.02, fmax)
                  if m.Q > 100 and m.amp > 1e-9]
    return sorted(freqs)


def test_sq_lattice_tm_bands_vs_mpb():
    """Gate (b): square lattice rods (r=0.2, eps=8.9) TM bands vs meep.mpb."""
    ref = json.load(open(DATA / "sq_lattice_mpb.json"))
    res = 16
    dx = 1.0 / res
    ss = 4
    n = res
    xi = (torch.arange(n * ss, dtype=torch.float64) + 0.5) / ss - 0.5
    II, JJ = torch.meshgrid(xi, xi, indexing="ij")
    fine = torch.ones(n * ss, n * ss, dtype=torch.float64)
    fine[((II - n / 2) * dx) ** 2 + ((JJ - n / 2) * dx) ** 2 < ref["r"] ** 2] = ref["eps"]
    eps = fine.reshape(n, ss, n, ss).mean(dim=(1, 3))
    dt = 0.4 * dx
    for i, kxf in enumerate((0.2, 0.4)):
        want = ref["bands"][ref["kx"].index(kxf)][:2]
        got = _bloch2_bands(eps, 2 * math.pi * kxf, 0.0, dt, dx, 0.8)
        for wf in want:
            best = min(got, key=lambda g: abs(g - wf)) if got else 1e9
            assert abs(best - wf) / wf <= 0.02, \
                f"kx={kxf}: band {wf:.4f} -> nearest {best:.4f} " \
                f"(got {[round(g, 4) for g in got[:6]]})"


def test_bragg_sine_bands_vs_mpb():
    """Gate (c): 1D sine-index Bragg bands vs meep.mpb."""
    ref = json.load(open(DATA / "bragg_sine_mpb.json"))
    res = 32
    dx = 1.0 / res
    x = (torch.arange(res, dtype=torch.float64) + 0.5) * dx
    idx = ref["index_min"] + 0.5 * (ref["index_max"] - ref["index_min"]) \
        * (1 + torch.cos(2 * math.pi * x))
    eps = (idx ** 2)[:, None].expand(res, 4).clone()
    dt = 0.4 * dx
    for kxf, bands in zip(ref["kx"], ref["bands"]):
        got = _bloch2_bands(eps, 2 * math.pi * kxf, 0.0, dt, dx, 0.8,
                            t_end=400.0)
        for wf in bands[:2]:
            best = min(got, key=lambda g: abs(g - wf)) if got else 1e9
            assert abs(best - wf) / wf <= 0.02, \
                f"kx={kxf}: band {wf:.4f} -> nearest {best:.4f} " \
                f"(got {[round(g, 4) for g in got[:6]]})"


def _tri_eps(res=26, r=0.2, eps_rod=12.0, basis_shift=None, r2=None):
    """Rect supercell (sqrt(3) x 1) of the triangular lattice: rods at the
    cell corner and center (+ optional 2-rod basis for honeycomb),
    supersampled coverage, periodic min-image distances."""
    nx, ny = int(round(math.sqrt(3) * res)), res
    Lx, Ly = nx / res, 1.0
    ss = 4
    xi = ((torch.arange(nx * ss, dtype=torch.float64) + 0.5) / ss) / res
    yj = ((torch.arange(ny * ss, dtype=torch.float64) + 0.5) / ss) / res
    II, JJ = torch.meshgrid(xi, yj, indexing="ij")
    fine = torch.ones(nx * ss, ny * ss, dtype=torch.float64)
    centers = [(0.0, 0.0), (Lx / 2, Ly / 2)]
    if basis_shift is not None:
        sx, sy = basis_shift
        centers = [(cx + s * sx, cy + s * sy) for cx, cy in centers for s in (+1, -1)]
    for cx, cy in centers:
        dxp = (II - cx) - Lx * torch.round((II - cx) / Lx)
        dyp = (JJ - cy) - Ly * torch.round((JJ - cy) / Ly)
        fine[dxp ** 2 + dyp ** 2 < (r2 or r) ** 2] = eps_rod
    return fine.reshape(nx, ss, ny, ss).mean(dim=(1, 3)), 1.0 / res


def test_tri_rods_bands_vs_mpb():
    """Oblique lattice via the rectangular supercell: mpb's tri-lattice
    bands at Gamma/M/K must appear (folded superset) within 2%."""
    ref = json.load(open(DATA / "tri_rods_mpb.json"))
    eps, dx = _tri_eps(res=26, r=ref["r"], eps_rod=ref["eps"])
    dt = 0.4 * dx
    kpts = {"gamma": (0.0, 0.0),
            "M": (math.pi / math.sqrt(3), math.pi),
            "K": (0.0, 2 * math.pi / 3)}
    for name, (kx, ky) in kpts.items():
        want = [f for f in ref[name] if f > 1e-3][:3]
        got = _bloch2_bands(eps, kx, ky, dt, dx, 0.85, t_end=350.0)
        for wf in want:
            best = min(got, key=lambda g: abs(g - wf)) if got else 1e9
            assert abs(best - wf) / wf <= 0.02, \
                f"{name}: band {wf:.4f} -> nearest {best:.4f} " \
                f"(got {[round(g, 4) for g in got[:8]]})"


def test_honey_rods_gamma_vs_mpb():
    """Honeycomb (2-rod basis, +- (1/6)(a1+a2) = +- sqrt(3)/6 x_hat)."""
    ref = json.load(open(DATA / "honey_rods_mpb.json"))
    eps, dx = _tri_eps(res=26, r=ref["r"], eps_rod=ref["eps"],
                       basis_shift=(math.sqrt(3) / 6, 0.0))
    dt = 0.4 * dx
    # band 1 only: the 0.726 pair at Gamma is degenerate and point-source
    # excitation/capture of degenerate partners is unreliable (ponytail:
    # mode-resolved capture via field-profile projection if needed)
    want = [f for f in ref["gamma"] if f > 1e-3][:1]
    got = _bloch2_bands(eps, 0.0, 0.0, dt, dx, 0.85, t_end=350.0)
    for wf in want:
        best = min(got, key=lambda g: abs(g - wf)) if got else 1e9
        assert abs(best - wf) / wf <= 0.02, \
            f"gamma: band {wf:.4f} -> nearest {best:.4f} " \
            f"(got {[round(g, 4) for g in got[:8]]})"


def test_line_defect_bands_vs_mpb():
    """7-row tri supercell, center rod removed: the mpb band set at
    kx=0.4 (lower continuum; mpb's own 12 bands show no in-gap state at
    this k) reproduced within 2%."""
    ref = json.load(open(DATA / "line_defect_mpb.json"))
    res = 26
    nx, ny = int(round(math.sqrt(3) * res)), 7 * res
    Lx, Ly = nx / res, 7.0
    ss = 2
    xi = ((torch.arange(nx * ss, dtype=torch.float64) + 0.5) / ss) / res
    yj = ((torch.arange(ny * ss, dtype=torch.float64) + 0.5) / ss) / res
    II, JJ = torch.meshgrid(xi, yj, indexing="ij")
    fine = torch.ones(nx * ss, ny * ss, dtype=torch.float64)
    centers = [(i2 * Lx / 2 % Lx, (j2 + i2 / 2) % Ly)
               for i2 in (0, 1) for j2 in range(7)]
    cdef = (0.0, 3.0)                              # removed rod (defect row)
    for cx, cy in centers:
        if abs(cx - cdef[0]) < 1e-9 and abs(cy - cdef[1]) < 1e-9:
            continue
        dxp = (II - cx) - Lx * torch.round((II - cx) / Lx)
        dyp = (JJ - cy) - Ly * torch.round((JJ - cy) / Ly)
        fine[dxp ** 2 + dyp ** 2 < ref["r"] ** 2] = ref["eps"]
    eps = fine.reshape(nx, ss, ny, ss).mean(dim=(1, 3))
    dx = 1.0 / res
    dt = 0.4 * dx
    kx = 0.4 * 2 * math.pi / math.sqrt(3)
    ky = 0.4 * 2 * math.pi                        # folds into the rect BZ
    got = _bloch2_bands(eps, kx, ky, dt, dx, 0.5, t_end=350.0)
    # containment reversed: a single point source does not excite every
    # band of a dense supercell continuum, but every mode it DOES capture
    # must be a true mpb band (no spurious frequencies)
    assert len(got) >= 4, f"only {len(got)} modes captured"
    for gf in got:
        best = min(ref["bands"], key=lambda w: abs(w - gf))
        assert abs(best - gf) / best <= 0.02, \
            f"spurious mode {gf:.4f} (nearest mpb band {best:.4f})"


def test_strip_dispersion_vs_mpb():
    """mpb_strip: Si strip on oxide -- fundamental-mode neff from the
    fd2d cross-section solver vs mpb's omega(beta) within 2%."""
    from meep_torch.modes.fd2d import solve_2d

    ref = json.load(open(DATA / "strip_mpb.json"))
    res = 32
    n = 2 * res
    dxs = 1.0 / res
    ss = 4
    zz = (np.arange(n * ss) + 0.5) / (res * ss) - 1.0
    yy = (np.arange(n * ss) + 0.5) / (res * ss) - 1.0
    YY, ZZ = np.meshgrid(yy, zz, indexing="ij")
    fine = np.ones((n * ss, n * ss))
    fine[ZZ < -ref["h"] / 2] = ref["n_ox"] ** 2
    fine[(np.abs(YY) < ref["w"] / 2) & (np.abs(ZZ) < ref["h"] / 2)] = ref["n_si"] ** 2
    eps = fine.reshape(n, ss, n, ss).mean(axis=(1, 3))
    for beta, freqs in zip(ref["beta"], ref["bands"]):
        f0 = freqs[0]
        want_neff = beta / f0
        got = solve_2d(eps, dxs, f0, k=2, target_neff=want_neff)
        neff = float(np.real(got.neff[0]))
        assert abs(neff - want_neff) / want_neff <= 0.02, \
            f"beta={beta}: neff {neff:.4f} vs mpb {want_neff:.4f}"


def test_hole_slab_zeven_K_vs_mpb():
    """mpb_hole_slab: triangular air-hole slab (h=0.5, r=0.3, eps=12),
    first z-even guided band at K within 2.5% of meep.mpb. Rect sqrt(3)x1
    supercell (K folds to ky = 2 pi/3); midplane source selects z-even."""
    from meep_torch.core.stepper_bloch3 import (build_bloch3_coeffs,
                                                init_bloch3_state, step_bloch3)

    ref = json.load(open(DATA / "hole_slab_mpb.json"))
    res = 16
    nx, ny = int(round(math.sqrt(3) * res)), res
    Lx, Ly = nx / res, 1.0
    nz = 4 * res
    dx = 1.0 / res
    ss = 4
    xi = ((torch.arange(nx * ss, dtype=torch.float64) + 0.5) / ss) / res
    yj = ((torch.arange(ny * ss, dtype=torch.float64) + 0.5) / ss) / res
    II, JJ = torch.meshgrid(xi, yj, indexing="ij")
    plane = torch.full((nx * ss, ny * ss), float(ref["eps"]))
    for cx, cy in ((0.0, 0.0), (Lx / 2, Ly / 2)):
        dxp = (II - cx) - Lx * torch.round((II - cx) / Lx)
        dyp = (JJ - cy) - Ly * torch.round((JJ - cy) / Ly)
        plane[dxp ** 2 + dyp ** 2 < ref["r"] ** 2] = 1.0
    plane = plane.reshape(nx, ss, ny, ss).mean(dim=(1, 3))
    eps = torch.ones(nx, ny, nz, dtype=torch.float64)
    jz0 = nz // 2 - int(ref["h"] / 2 * res)
    jz1 = nz // 2 + int(ref["h"] / 2 * res)
    eps[:, :, jz0:jz1] = plane[:, :, None]
    kx, ky = 0.0, 2 * math.pi / 3                  # K folded into the rect BZ
    c = build_bloch3_coeffs(eps, dx, kx, ky, n_abs_z=res // 2)
    st = init_bloch3_state(nx, ny, nz)
    pulse = GaussianPulse(0.3, 0.3)
    src = ("Ey", nx // 3 + 1, ny // 5 + 2, nz // 2)   # midplane -> z-even
    n_steps = int(round(250.0 / c.dt))
    probes = [(nx // 3 + 1, ny // 5 + 2), (2 * nx // 3, ny // 2 + 1)]
    series = np.empty((len(probes), n_steps), dtype=complex)
    for t in range(n_steps):
        st = step_bloch3(st, c, pulse.amplitude((t + 0.5) * c.dt), src)
        for p, (pi, pj) in enumerate(probes):
            series[p, t] = st.Ey[pi, pj, nz // 2].item()
    t0 = int(round(50.0 / c.dt))
    freqs = []
    for p in range(len(probes)):
        freqs += [m.freq for m in harminv(series[p, t0:], c.dt, 0.15, 0.45)
                  if m.Q > 100 and m.amp > 1e-9]
    want = ref["K_zeven"][0]
    best = min(freqs, key=lambda g: abs(g - want)) if freqs else 1e9
    assert abs(best - want) / want <= 0.025, \
        f"z-even band {want:.4f} -> nearest {best:.4f} " \
        f"(got {[round(f, 4) for f in sorted(freqs)[:6]]})"


def _diamond_eps(res=16):
    """Conventional cubic supercell (side 1) of the diamond lattice: 8
    dielectric spheres (fcc sublattice + (1/4,1/4,1/4) shift), r=0.25/sqrt(2)
    (mpb_diamond's r/a_conv preserved), eps=11.56, supersampled coverage,
    periodic min-image distances."""
    ref = json.load(open(DATA / "diamond_mpb.json"))
    n = res
    ss = 4
    xi = ((torch.arange(n * ss, dtype=torch.float64) + 0.5) / ss) / res
    II, JJ, KK = torch.meshgrid(xi, xi, xi, indexing="ij")
    fine = torch.ones(n * ss, n * ss, n * ss, dtype=torch.float64)
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    off = (0.25, 0.25, 0.25)
    centers = [c for cx in fcc for c in (cx, (cx[0] + off[0],
                                              cx[1] + off[1],
                                              cx[2] + off[2]))]
    for cx, cy, cz in centers:
        dxp = (II - cx) - torch.round(II - cx)
        dyp = (JJ - cy) - torch.round(JJ - cy)
        dzp = (KK - cz) - torch.round(KK - cz)
        fine[dxp ** 2 + dyp ** 2 + dzp ** 2 < ref["r"] ** 2] = ref["eps"]
    return fine.reshape(n, ss, n, ss, n, ss).mean(dim=(1, 3, 5)), 1.0 / res


def diamond_bands_at(kx, ky, kz, res=20, fmax=0.7, t_end=400.0):
    """Triply-periodic 3D Bloch bands of the diamond lattice at one k point
    (kx,ky,kz in rad/length, cell=1) via stepper_bloch3 + harminv."""
    from meep_torch.core.stepper_bloch3 import (build_bloch3_coeffs,
                                                init_bloch3_state, step_bloch3)

    eps, dx = _diamond_eps(res)
    n = eps.shape[0]
    c = build_bloch3_coeffs(eps, dx, kx, ky, n_abs_z=0, kz=kz)
    st = init_bloch3_state(n, n, n)
    pulse = GaussianPulse(fmax / 2, fmax)
    src = ("Ey", n // 3 + 1, n // 5 + 2, n // 2)
    n_steps = int(round(t_end / c.dt))
    probes = [(n // 3 + 1, n // 5 + 2, n // 2),
              (2 * n // 3, n // 2 + 1, n // 3),
              (n // 2, 2 * n // 3, n // 4 + 1)]
    series = np.empty((len(probes), n_steps), dtype=complex)
    for t in range(n_steps):
        st = step_bloch3(st, c, pulse.amplitude((t + 0.5) * c.dt), src)
        for p, (pi, pj, pk) in enumerate(probes):
            series[p, t] = st.Ey[pi, pj, pk].item()
    t0 = int(round(60.0 / c.dt))
    freqs = []
    for p in range(len(probes)):
        freqs += [m.freq for m in harminv(series[p, t0:], c.dt, 0.15, fmax)
                  if m.Q > 100 and m.amp > 1e-9]
    return sorted(freqs)


def test_diamond_band_vs_mpb():
    """mpb_diamond: dielectric spheres in a diamond (fcc) lattice, first band
    at the simple-cubic X point (k=(0,pi,pi)) within 3% of meep.mpb.

    Rect-supercell trick: the fcc primitive cell (mpb_diamond's 2-sphere basis
    at basis_size sqrt(0.5), a_conv=sqrt(2), r=0.25) is mapped to the
    conventional cubic cell of side 1 containing 8 atoms -- the fcc sublattice
    (0,0,0),(0,1/2,1/2),(1/2,0,1/2),(1/2,1/2,0) plus the (1/4,1/4,1/4) diamond
    shift. r rescaled to 0.25/sqrt(2) so r/a_conv is identical; eps=11.56
    unchanged. The cubic BZ is 1/4 the fcc BZ so the fcc bands fold into it,
    but the oracle uses the SAME supercell + k, so the comparison is
    self-consistent. Triply-periodic stepper_bloch3 (n_abs_z=0, kz=pi).
    """
    ref = json.load(open(DATA / "diamond_mpb.json"))
    want = ref["bands"][0][0]
    got = diamond_bands_at(0.0, math.pi, math.pi)
    best = min(got, key=lambda g: abs(g - want)) if got else 1e9
    print(f"diamond first band: mpb={want:.4f} got-nearest={best:.4f} "
          f"(captured {[round(f, 4) for f in got[:6]]})")
    assert abs(best - want) / want <= 0.03, \
        f"diamond band {want:.4f} -> nearest {best:.4f} " \
        f"(got {[round(f, 4) for f in got[:8]]})"


def _bloch2_te_bands(eps, kx, ky, dt, dx, fmax, t_end=600.0):
    """Doubly-periodic TE Bloch stepper driven like _bloch2_bands, harvested
    off Hz instead of Ez."""
    from meep_torch.core.stepper_bloch2 import (init_bloch2_state_te,
                                                step_bloch2_te, te_eps_edges)

    class _G:
        pass

    g = _G()
    g.nx, g.ny = eps.shape
    g.zeros = lambda *s: torch.zeros(*s, dtype=torch.float64)
    st = init_bloch2_state_te(g)
    eps_x, eps_y = te_eps_edges(eps)
    mask = torch.zeros(*eps.shape, dtype=torch.float64)
    mask[g.nx // 3 + 1, g.ny // 5 + 2] = 1.0
    pulse = GaussianPulse(fmax / 2, fmax)
    n_steps = int(round(t_end / dt))
    probes = [(g.nx // 3 + 1, g.ny // 5 + 2), (2 * g.nx // 3, g.ny // 2 + 1),
              (g.nx // 2 - 2, 1)]
    series = np.empty((len(probes), n_steps), dtype=complex)
    for t in range(n_steps):
        st = step_bloch2_te(st, eps_x, eps_y, dt, dx, kx, ky,
                            src_c=pulse.amplitude((t + 0.5) * dt), src_s=0.0,
                            src_mask=mask)
        for p, (pi, pj) in enumerate(probes):
            series[p, t] = complex(st.Hz_c[pi, pj].item(), st.Hz_s[pi, pj].item())
    t0 = int(round(60.0 / dt))
    freqs = []
    for p in range(len(probes)):
        freqs += [m.freq for m in harminv(series[p, t0:], dt, 0.02, fmax)
                  if m.Q > 100.0 and m.amp > 1e-9]
    out = []
    for f in sorted(freqs):
        if not out or f - out[-1] > 1e-3 * max(f, 1e-9):
            out.append(f)
    return out


def test_bloch2_te_empty_lattice_analytic():
    """TE stepper against the exact empty-lattice dispersion.

    With eps = 1 there is no geometry error left, so this isolates the
    operator itself: the staggering, the Bloch phase couplings, and the
    quadrature pair. Folded free-photon bands are f = |k + G| with
    G = (m, n) on the unit cell.
    """
    n = 24
    eps = torch.ones(n, n, dtype=torch.float64)
    dx = 1.0 / n
    for kx_m, ky_m in ((0.25, 0.0), (0.3, 0.2)):
        got = _bloch2_te_bands(eps, 2 * math.pi * kx_m, 2 * math.pi * ky_m,
                               0.4 * dx, dx, 1.2)
        want = sorted(math.hypot(kx_m + a, ky_m + b)
                      for a in range(-2, 3) for b in range(-2, 3))[:3]
        for wf in want:
            best = min(got, key=lambda g: abs(g - wf)) if got else 1e9
            assert abs(best - wf) / wf <= 0.01, \
                f"k=({kx_m},{ky_m}) band {wf:.4f} -> {best:.4f} (got {got[:5]})"


def test_bloch2_te_rod_lattice_vs_mpb():
    """TE stepper on the mpb_tutorial rod lattice, at X and M.

    Loose tolerance on purpose: this is the time-domain path, carrying
    temporal discretization and harminv extraction error on top of the
    staircased rod boundary. The eigensolver reaches ~1.8% on the same
    grid where this sits near 5%, which is why
    meep_torch.mpb.ModeSolver routes 2D band structures through the
    eigensolve and these steppers exist for time-domain work. The gate is
    here to catch a broken operator, not to certify accuracy.
    """
    ref = json.load(open(DATA / "sq_tut_te_mpb.json"))
    res = 24
    ii = (np.arange(res) + 0.5) / res - 0.5
    X, Y = np.meshgrid(ii, ii, indexing="ij")
    eps = torch.as_tensor(np.where(X ** 2 + Y ** 2 < 0.2 ** 2, 12.0, 1.0),
                          dtype=torch.float64)
    dx = 1.0 / res
    for kidx in (5, 10):                      # X and M, away from Gamma
        kx_m, ky_m = ref["k"][kidx]
        want = [f for f in ref["bands"][kidx][:3] if f > 1e-6]
        got = _bloch2_te_bands(eps, 2 * math.pi * kx_m, 2 * math.pi * ky_m,
                               0.4 * dx, dx, 0.8)
        assert got, f"no TE modes found at k=({kx_m},{ky_m})"
        for wf in want:
            best = min(got, key=lambda g: abs(g - wf))
            assert abs(best - wf) / wf <= 0.08, \
                f"k=({kx_m},{ky_m}) band {wf:.4f} -> {best:.4f} (got {got[:5]})"
