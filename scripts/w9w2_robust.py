import math
import numpy as np
import torch
from meep_torch.core.monitors_3d import FluxMonitor3D, PlaneSource, run_3d
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state

def slab_T(f, eps, sigma_d, d):
    w = 2 * math.pi * f
    eps_c = eps * (1 + 1j * sigma_d / w)
    n = np.sqrt(eps_c)
    r = (1 - n) / (1 + n)
    delta = w * n * d
    t = (1 - r * r) * np.exp(1j * delta) / (1 - r * r * np.exp(2j * delta))
    return float(abs(t) ** 2)

def trans(nxy, nz, dx, npml, fcen, fw, src_z, z0, z1, mon_z, nsteps, eps_v=None, sig=None):
    if eps_v is not None:
        eps = torch.ones(nxy, nxy, nz, dtype=torch.float64); eps[:, :, z0:z1] = eps_v
        sd = torch.zeros(nxy, nxy, nz, dtype=torch.float64); sd[:, :, z0:z1] = sig
        c = build_3d_coeffs(nxy, nxy, nz, dx, eps=eps, npml=npml, sigma_d=sd)
    else:
        c = build_3d_coeffs(nxy, nxy, nz, dx, npml=npml)
    st = init_3d_state(nxy, nxy, nz, npml=npml)
    src = PlaneSource(nxy, nxy, nz, c.dt, "Ex", axis=2, idx=src_z, profile=GaussianPulse(fcen, fw))
    mon = FluxMonitor3D(c.dt, [fcen], normal="z", idx=mon_z, ap=npml + 1)
    _, (acc,) = run_3d(st, c, nsteps, source=src, monitors=[mon])
    return float(mon.flux(acc, dx)[0])

nxy, nz, dx, npml = 36, 80, 0.04, 7
fcen, fw = 1.0, 0.2
src_z, mon_z = 10, 52
ns = int(round(60.0 / (0.5 * dx / math.sqrt(3.0))))
f_inc = trans(nxy, nz, dx, npml, fcen, fw, src_z, 24, 24, mon_z, ns)
print("neighbor sigmas (d=0.4, z0=24):")
for sig in [3.0, 4.0, 5.0]:
    f_tr = trans(nxy, nz, dx, npml, fcen, fw, src_z, 24, 34, mon_z, ns, 4.0, sig)
    T = f_tr / f_inc; Ta = slab_T(fcen, 4.0, sig, 0.4)
    print(f"  sig={sig}: Tsim={T:.4f} Ta={Ta:.4f} err={abs(T-Ta)/Ta:.3%}")
print("alignment shifts (sig=4, d=0.4=10 cells):")
for z0 in [23, 24, 25, 26]:
    z1 = z0 + 10
    f_inc_i = trans(nxy, nz, dx, npml, fcen, fw, src_z, z0, z0, mon_z, ns)
    f_tr = trans(nxy, nz, dx, npml, fcen, fw, src_z, z0, z1, mon_z, ns, 4.0, 4.0)
    T = f_tr / f_inc_i; Ta = slab_T(fcen, 4.0, 4.0, 0.4)
    print(f"  z0={z0}: Tsim={T:.4f} Ta={Ta:.4f} err={abs(T-Ta)/Ta:.3%}")
