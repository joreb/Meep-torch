import math, time
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


def trans(nxy, nz, dx, npml, fcen, fw, src_z, z0, z1, mon_z, nsteps,
          eps_val=None, sigma=None, cutoff=5.0):
    if eps_val is not None:
        eps = torch.ones(nxy, nxy, nz, dtype=torch.float64)
        eps[:, :, z0:z1] = eps_val
        sd = torch.zeros(nxy, nxy, nz, dtype=torch.float64)
        sd[:, :, z0:z1] = sigma
        c = build_3d_coeffs(nxy, nxy, nz, dx, eps=eps, npml=npml, sigma_d=sd)
    else:
        c = build_3d_coeffs(nxy, nxy, nz, dx, npml=npml)
    st = init_3d_state(nxy, nxy, nz, npml=npml)
    src = PlaneSource(nxy, nxy, nz, c.dt, "Ex", axis=2, idx=src_z,
                      profile=GaussianPulse(fcen, fw, cutoff=cutoff))
    mon = FluxMonitor3D(c.dt, [fcen], normal="z", idx=mon_z, ap=npml + 1)
    _, (acc,) = run_3d(st, c, nsteps, source=src, monitors=[mon])
    # acc: (nfreq, 8, nx', ny') = [Re Ex, Im Ex, Re Hy, Im Hy, Re Ey, Im Ey, Re Hx, Im Hx]
    ReEx, ImEx = acc[0, 0], acc[0, 1]
    ReHy, ImHy = acc[0, 2], acc[0, 3]
    Ex = (ReEx + 1j * ImEx)
    Hy = (ReHy + 1j * ImHy)
    fwd = 0.25 * (Ex + Hy).abs() ** 2      # +z forward (vacuum, eta=1)
    bwd = 0.25 * (Ex - Hy).abs() ** 2
    cell = dx * dx
    fwd_pow = float(fwd.sum() * cell)
    bwd_pow = float(bwd.sum() * cell)
    net = float(mon.flux(acc, dx)[0])
    return net, fwd_pow, bwd_pow


# config B geometry; eps4 lossless (the failing case)
nxy, nz, dx, npml = 36, 80, 0.04, 7
fcen, fw = 1.0, 0.2
src_z, z0, z1, mon_z = 10, 24, 34, 52
ns = int(round(60.0 / (0.5 * dx / math.sqrt(3.0))))
d = (z1 - z0) * dx

print(f"eps4 lossless d={d} Ta={slab_T(fcen,4,0,d):.4f}  ns={ns}")
net_v, fwd_v, bwd_v = trans(nxy, nz, dx, npml, fcen, fw, src_z, z0, z1, mon_z, ns)
net_s, fwd_s, bwd_s = trans(nxy, nz, dx, npml, fcen, fw, src_z, z0, z1, mon_z, ns, 4.0, 0.0)
print(f" vac: net={net_v:.5f} fwd={fwd_v:.5f} bwd={bwd_v:.5f}")
print(f" slab: net={net_s:.5f} fwd={fwd_s:.5f} bwd={bwd_s:.5f}")
print(f" T_net  ={net_s/net_v:.4f}  (err {abs(net_s/net_v - slab_T(fcen,4,0,d))/slab_T(fcen,4,0,d):.3%})")
print(f" T_fwd  ={fwd_s/fwd_v:.4f}  (err {abs(fwd_s/fwd_v - slab_T(fcen,4,0,d))/slab_T(fcen,4,0,d):.3%})")
print(f" back/forward (vac)={bwd_v/fwd_v:.4f}  (slab)={bwd_s/fwd_s:.4f}")