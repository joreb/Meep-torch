"""Bitwise reference for the stepper_3d psi-convention change.

Runs four configurations that exercise every branch the E-phase psi touches
(plain CPML, PMC walls, ADE poles, mixed absorber) and records the final
fields. Rerun after the change: the numbers must be bit-identical, because
trimming a pointwise recursion commutes with the recursion.

RESULT: captured at 7df3716, BEFORE stepper_3d's psi arrays were trimmed to
what the step writes. Re-run after any change to the 3D CPML path: the field
sums must be bit-identical. Used to prove the psi-trim and the PSI_NEW/STRETCHED
wiring changed no number.
"""
import math, sys, json
import torch
from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state, step_3d

def run(tag, nx, ny, nz, *, npml=4, pmc=(), n_poles=0, n_abs=0, absorber_axes=(), steps=30):
    kw = dict(npml=npml, pmc_axes=pmc, n_abs=n_abs, absorber_axes=absorber_axes)
    poles = ((1.2, 0.0, 0.3),) if n_poles else ()
    c = build_3d_coeffs(nx, ny, nz, dx=0.05, npml=npml, pmc_axes=pmc,
                        poles=poles, n_abs=n_abs, absorber_axes=absorber_axes)
    try:
        st = init_3d_state(nx, ny, nz, npml=npml, n_poles=n_poles, pmc=pmc)
    except TypeError:                      # pre-change signature
        st = init_3d_state(nx, ny, nz, npml=npml, n_poles=n_poles)
    src = ("Ez", nx // 2, ny // 2, nz // 2)
    for n in range(steps):
        amp = math.sin(0.3 * n) * math.exp(-((n - 10) / 6.0) ** 2)
        st = step_3d(st, c, amp, src)
    out = {"tag": tag}
    for f in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        v = getattr(st, f)
        out[f] = v.sum().item().hex() if hasattr(v.sum().item(), "hex") else float(v.sum())
    out["psi_shapes"] = {k: tuple(getattr(st, k).shape)
                         for k in st._fields if k.startswith("psi_") and getattr(st, k) is not None}
    out["Ez_absmax"] = float(st.Ez.abs().max())
    return out

cases = [
    ("plain",    dict(nx=16, ny=15, nz=14)),
    ("pmc01",    dict(nx=10, ny=10, nz=18, pmc=(0, 1))),
    ("poles",    dict(nx=14, ny=13, nz=12, n_poles=1)),
    ("mixed",    dict(nx=12, ny=12, nz=16, n_abs=4, absorber_axes=(2,))),
]
res = [run(t, **kw) for t, kw in cases]
print(json.dumps(res, indent=1))
