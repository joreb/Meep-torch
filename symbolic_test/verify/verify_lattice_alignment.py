"""Does the structure's mirror plane need to sit on the field lattice?

The dual lattice put Hz on the integer node, where the slab centre is. Yee puts
Hz at the half-node, so the waveguide's mirror plane falls between samples.
`shift` moves the structure half a cell to restore the alignment.

RESULT: the holey-waveguide bands are governed by ALIGNMENT, not by which
lattice. Yee with the slab on the integer node gives 3.72% worst error; on the
half-node, 0.62%; the old dual lattice gave 1.33%. Aligned, Yee is twice as
accurate as the dual lattice ever was. The naive reading of "1.33% before, 3.26%
after" says revert, and is wrong.
"""
import json, math, sys, pathlib
import numpy as np
import torch
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bloch import init_bloch_te_state, step_bloch_te
from meep_torch.modes.harminv import harminv

DATA = pathlib.Path("tests/examples/data")


def holey_bands(kx_meep, res=10, shift=0.0):
    ref = json.load(open(DATA / "holey_bands.json"))
    dx = 1.0 / res
    nx, ny = int(ref["sy"] * res), int(1.0 * res)
    grid = Grid(nx=nx, ny=ny, dx=dx)
    npml = int(ref["dpml"] * res)
    coeffs = build_tm_coeffs(grid, eps_inf=torch.ones(nx, ny, dtype=torch.float64),
                             npml=npml, y_pmc=True)
    cx, ss = nx // 2, 4
    xi = (torch.arange(nx * ss, dtype=torch.float64) + 0.5) / ss - 0.5
    yj = (torch.arange(ny * ss, dtype=torch.float64) + 0.5) / ss - 0.5
    II, JJ = torch.meshgrid(xi, yj, indexing="ij")
    fine = torch.ones(nx * ss, ny * ss, dtype=torch.float64)
    c0 = cx + shift
    fine[((II - c0).abs() * dx) < ref["w"] / 2] = ref["eps"]
    fine[((II - c0) * dx) ** 2 + ((JJ - ny / 2) * dx) ** 2 < ref["r"] ** 2] = 1.0
    eps = fine.reshape(nx, ss, ny, ss).mean(dim=(1, 3))
    eps_x = 0.5 * (eps[:-1, :] + eps[1:, :])
    eps_y = 0.5 * (eps + torch.roll(eps, -1, dims=1))

    st = init_bloch_te_state(grid)
    ky = 2 * math.pi * kx_meep
    mask = grid.zeros(nx - 1, ny)
    mask[cx + 2, ny // 3] = 1.0
    pulse = GaussianPulse(ref["fcen"], ref["df"])
    n_steps = int(round(360.0 / grid.dt))
    probes = [(cx + 2, ny // 3), (cx - 3, 2 * ny // 3), (cx + 1, ny // 5)]
    series = np.empty((len(probes), n_steps), dtype=complex)
    for t in range(n_steps):
        a = pulse.amplitude((t + 0.5) * grid.dt)
        st = step_bloch_te(st, coeffs, eps_x, eps_y, ky, src_c=a, src_s=0.0,
                           src_mask=mask)
        for p, (pi, pj) in enumerate(probes):
            series[p, t] = complex(st.Hz_c[pi, pj].item(), st.Hz_s[pi, pj].item())
    t0 = int(round(60.0 / grid.dt))
    freqs = []
    for p in range(len(probes)):
        freqs += [m.freq for m in harminv(series[p, t0:], grid.dt, 0.02,
                                          ref["fcen"] + ref["df"] / 2)
                  if m.Q > 50 and m.amp > 1e-6]
    return sorted(freqs)


ref = json.load(open(DATA / "holey_bands.json"))
for kx in (0.1, 0.2, 0.4):
    want = [m["freq"] for m in ref["bands"][str(kx)]
            if m["freq"] < 1.1 * kx and m["Q"] > 100]
    if not want:
        continue
    print(f"\nkx={kx}  oracle {[round(w,4) for w in want]}")
    for label, sh in (("slab on node (as committed)", 0.0),
                      ("slab on half-node", 0.5)):
        got = holey_bands(kx, shift=sh)
        worst = max(abs(min(got, key=lambda g: abs(g-w)) - w) / w for w in want)
        marks = " ".join(f"{w:.4f}->{min(got, key=lambda g: abs(g-w)):.4f}" for w in want)
        print(f"  {label:<28s} worst {worst*100:5.2f}%  {'PASS' if worst<=0.025 else 'FAIL'}   {marks}")
