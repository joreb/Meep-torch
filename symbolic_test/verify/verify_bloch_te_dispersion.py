"""Numerical dispersion of the Bloch TE stepper in a UNIFORM medium.

eps placement cannot matter here, so this separates a stagger error in the
stepper from a structure-sampling difference. The Yee TE relation for a
uniform medium with envelope wavevector (kx, ky) is

    (sin(w dt/2)/dt)^2 = (sin(kx dx/2)/dx)^2 + (sin(ky dx/2)/dx)^2

A correct stepper reproduces it; a mis-staggered one does not.

RESULT: the unified Bloch TE stepper reproduces the analytic Yee dispersion
relation to under one FFT bin at ky=0 and ky=4. This is what RULED OUT a stagger
error when the holey-waveguide bands went red after the TE lattice unification --
eps placement cannot matter in a uniform medium, so a passing dispersion check
leaves only the structure/alignment.
"""
import math, sys
import numpy as np
import torch
from meep_torch.core.grid import Grid
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.stepper_bloch import init_bloch_te_state, step_bloch_te

def measure(ky, nx=64, ny=64, dx=0.05, n_steps=6000):
    g = Grid(nx=nx, ny=ny, dx=dx)
    # npml=0 -> inert CPML profiles; walls are plain PEC, x is a resonator
    c = build_tm_coeffs(g, eps_inf=torch.ones(nx, ny, dtype=torch.float64),
                        npml=0, y_pmc=True)
    st = init_bloch_te_state(g)
    eps_x = torch.ones(*st.Ex_c.shape, dtype=torch.float64)
    eps_y = torch.ones(*st.Ey_c.shape, dtype=torch.float64)
    torch.manual_seed(0)
    st = st._replace(Hz_c=torch.randn_like(st.Hz_c) * 1e-3)
    probe = []
    for t in range(n_steps):
        st = step_bloch_te(st, c, eps_x, eps_y, ky)
        probe.append(st.Hz_c[nx // 3, ny // 4].item())
    sig = np.asarray(probe)
    sig -= sig.mean()
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    freqs = np.fft.rfftfreq(len(sig), d=g.dt)
    return g.dt, dx, freqs, spec

def yee_modes(ky, nx, dx, dt, n=6):
    """PEC-resonator kx = m*pi/L, L = (nx-1)*dx, through the Yee relation."""
    L = (nx - 1) * dx
    out = []
    for m in range(1, n + 1):
        kx = m * math.pi / L
        rhs = (math.sin(kx * dx / 2) / dx) ** 2 + (math.sin(ky * dx / 2) / dx) ** 2
        s = math.sqrt(rhs) * dt
        if s > 1:
            break
        out.append(math.asin(s) * 2 / dt / (2 * math.pi))
    return out

for ky in (0.0, 4.0):
    dt, dx, freqs, spec = measure(ky)
    peaks = []
    for i in range(2, len(spec) - 2):
        if spec[i] > spec[i-1] and spec[i] > spec[i+1] and spec[i] > 0.02 * spec.max():
            peaks.append(freqs[i])
    want = yee_modes(ky, 64, dx, dt)
    print(f"ky={ky}")
    print(f"  analytic Yee : {[round(w,4) for w in want]}")
    print(f"  measured     : {[round(p,4) for p in peaks[:len(want)+2]]}")
    for w in want[:4]:
        best = min(peaks, key=lambda p: abs(p - w)) if peaks else 0.0
        print(f"    {w:.4f} -> {best:.4f}   rel {abs(best-w)/w*100:6.3f}%")
