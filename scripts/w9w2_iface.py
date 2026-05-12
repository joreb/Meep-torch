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


def trans(nxy, nz, dx, npml, fcen, fw, src_z, z0, z1, mon_z, nsteps,
          eps_val=None, sigma=None):
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
                      profile=GaussianPulse(fcen, fw))
    mon = FluxMonitor3D(c.dt, [fcen], normal="z", idx=mon_z, ap=npml + 1)
    _, (acc,) = run_3d(st, c, nsteps, source=src, monitors=[mon])
    return float(mon.flux(acc, dx)[0])


# --- single-interface reflection: half-space eps=4 for z>=z0 ---
def interface_refl(nxy, nz, dx, npml, fcen, fw, src_z, z0, nsteps, eps_v):
    eps = torch.ones(nxy, nxy, nz, dtype=torch.float64)
    eps[:, :, z0:] = eps_v
    c = build_3d_coeffs(nxy, nxy, nz, dx, eps=eps, npml=npml)
    st = init_3d_state(nxy, nxy, nz, npml=npml)
    src = PlaneSource(nxy, nxy, nz, c.dt, "Ex", axis=2, idx=src_z,
                      profile=GaussianPulse(fcen, fw))
    # monitor BEFORE the interface (between src and iface) -> net flux = inc - refl
    mon = FluxMonitor3D(c.dt, [fcen], normal="z", idx=src_z + 2, ap=npml + 1)
    _, (acc,) = run_3d(st, c, nsteps, source=src, monitors=[mon])
    return float(mon.flux(acc, dx)[0])


nxy, nz, dx, npml = 28, 64, 0.05, 6
fcen, fw = 1.0, 0.2
ns = int(round(60.0 / (0.5 * dx / math.sqrt(3.0))))
# single interface: src at 8, iface at 30, monitor at 10. inc flux = same-plane vacuum
src_z, z0 = 8, 30
mon_iface = src_z + 2
f_vac = trans(nxy, nz, dx, npml, fcen, fw, src_z, 60, 60, mon_iface, ns)  # vacuum placeholder
# proper: run vacuum, net at mon_iface = forward (no refl). run slab half-space, net = fwd - refl.
# do vacuum as no-slab:
c0 = build_3d_coeffs(nxy, nxy, nz, dx, npml=npml)
st0 = init_3d_state(nxy, nxy, nz, npml=npml)
s0 = PlaneSource(nxy, nxy, nz, c0.dt, "Ex", axis=2, idx=src_z, profile=GaussianPulse(fcen, fw))
m0 = FluxMonitor3D(c0.dt, [fcen], normal="z", idx=mon_iface, ap=npml + 1)
_, (a0,) = run_3d(st0, c0, ns, source=s0, monitors=[m0])
inc_flux = float(m0.flux(a0, dx)[0])
net_iface = interface_refl(nxy, nz, dx, npml, fcen, fw, src_z, z0, ns, 4.0)
# refl = inc - net ; |r|^2 = (inc-net)/inc  (net = inc - refl, flux-based)
Rsim = (inc_flux - net_iface) / inc_flux
n = 2.0
Rfres = ((1 - n) / (1 + n)) ** 2
print(f"single interface eps4: R_sim={Rsim:.4f} R_fresnel={Rfres:.4f} err={abs(Rsim-Rfres)/Rfres:.3%}")

# --- eps4 sig=4 robustness: config A vs B, and d=0.4 vs d=0.8 ---
print("\neps4 sig=4 robustness:")
for (nxy2, nz2, dx2, npml2, s_z, z0s, z1s, mz) in [
    (28, 64, 0.05, 6, 8, 18, 26, 40),
    (36, 80, 0.04, 7, 10, 24, 34, 52),
]:
    nss = int(round(60.0 / (0.5 * dx2 / math.sqrt(3.0))))
    d = (z1s - z0s) * dx2
    f_inc = trans(nxy2, nz2, dx2, npml2, fcen, fw, s_z, z0s, z1s, mz, nss)
    f_tr = trans(nxy2, nz2, dx2, npml2, fcen, fw, s_z, z0s, z1s, mz, nss, 4.0, 4.0)
    T = f_tr / f_inc
    Ta = slab_T(fcen, 4.0, 4.0, d)
    print(f"  dx={dx2} d={d}: Tsim={T:.4f} Ta={Ta:.4f} err={abs(T-Ta)/Ta:.3%}")
# thick heavily-lossy slab, config A
for z0s, z1s in [(18, 34), (16, 36)]:  # d=0.8, 1.0
    d = (z1s - z0s) * dx
    f_inc = trans(nxy, nz, dx, npml, fcen, fw, src_z, z0s, z1s, 40 if z1s <= 40 else 44, ns)
    f_tr = trans(nxy, nz, dx, npml, fcen, fw, src_z, z0s, z1s, 40 if z1s <= 40 else 44, ns, 4.0, 4.0)
    T = f_tr / f_inc
    Ta = slab_T(fcen, 4.0, 4.0, d)
    print(f"  thick dx=0.05 d={d}: Tsim={T:.4f} Ta={Ta:.4f} err={abs(T-Ta)/Ta:.3%}")