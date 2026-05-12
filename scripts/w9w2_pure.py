import math
import numpy as np
import torch
from meep_torch.core.monitors_3d import FluxMonitor3D, PlaneSource, run_3d
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state

def slab_T(f, eps, sigma_d, d):  # power
    w = 2 * math.pi * f
    eps_c = eps * (1 + 1j * sigma_d / w)
    n = np.sqrt(eps_c)
    r = (1 - n) / (1 + n)
    delta = w * n * d
    t = (1 - r * r) * np.exp(1j * delta) / (1 - r * r * np.exp(2j * delta))
    return float(abs(t) ** 2)

def trans(nxy, nz, dx, npml, fcen, fw, src_z, z0, z1, mon_z, nsteps, sig=None):
    # eps=1 everywhere (vacuum); slab = sigma_d region only.
    sd = torch.zeros(nxy, nxy, nz, dtype=torch.float64)
    if sig is not None:
        sd[:, :, z0:z1] = sig
    c = build_3d_coeffs(nxy, nxy, nz, dx, npml=npml, sigma_d=sd)
    st = init_3d_state(nxy, nxy, nz, npml=npml)
    src = PlaneSource(nxy, nxy, nz, c.dt, "Ex", axis=2, idx=src_z, profile=GaussianPulse(fcen, fw))
    mon = FluxMonitor3D(c.dt, [fcen], normal="z", idx=mon_z, ap=npml + 1)
    _, (acc,) = run_3d(st, c, nsteps, source=src, monitors=[mon])
    return float(mon.flux(acc, dx)[0])

nxy, nz, dx, npml = 28, 64, 0.05, 6
fcen, fw = 1.0, 0.2
src_z, mon_z = 8, 40
ns = int(round(60.0 / (0.5 * dx / math.sqrt(3.0))))
sig = 1.0
for z0, ncells in [(18, 20)]:  # d=1.0
    z1 = z0 + ncells
    d = ncells * dx
    print(f"eps=1 sigma={sig} d={d} (W5-pattern) alignment shifts:")
    for shift in [-1, 0, 1]:
        z0s = z0 + shift; z1s = z1 + shift
        f_inc = trans(nxy, nz, dx, npml, fcen, fw, src_z, z0s, z0s, mon_z, ns)
        f_tr = trans(nxy, nz, dx, npml, fcen, fw, src_z, z0s, z1s, mon_z, ns, sig)
        T = f_tr / f_inc; Ta = slab_T(fcen, 1.0, sig, d)
        print(f"   z0={z0s}: Tsim={T:.4f} Ta={Ta:.4f} err={abs(T-Ta)/Ta:.3%}")
