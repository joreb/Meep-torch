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


def trans(nxy, nz, dx, npml, fcen, fwidth, src_z, z0, z1, mon_z, nsteps,
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
                      profile=GaussianPulse(fcen, fwidth))
    mon = FluxMonitor3D(c.dt, [fcen], normal="z", idx=mon_z, ap=npml + 1)
    t0 = time.time()
    _, (acc,) = run_3d(st, c, nsteps, source=src, monitors=[mon])
    dt = time.time() - t0
    return float(mon.flux(acc, dx)[0]), dt


def run(cfg):
    nxy, nz, dx, npml = cfg["nxy"], cfg["nz"], cfg["dx"], cfg["npml"]
    fcen, fw = cfg["fcen"], cfg["fw"]
    src_z, z0, z1, mon_z = cfg["src_z"], cfg["z0"], cfg["z1"], cfg["mon_z"]
    ns = cfg["ns"]
    d = (z1 - z0) * dx
    f_inc, t_inc = trans(nxy, nz, dx, npml, fcen, fw, src_z, z0, z1, mon_z, ns)
    out = {"t_inc": t_inc, "f_inc": f_inc}
    for label, (ev, sg) in cfg["cases"].items():
        f_tr, t_tr = trans(nxy, nz, dx, npml, fcen, fw, src_z, z0, z1, mon_z, ns,
                          ev, sg)
        Tsim = f_tr / f_inc
        Ta = slab_T(fcen, ev, sg, d)
        out[label] = dict(Tsim=Tsim, Ta=Ta, err=abs(Tsim - Ta) / Ta, t=t_tr,
                          f_tr=f_tr)
    return out


# config A: current baseline, longer run
cfgA = dict(nxy=28, nz=64, dx=0.05, npml=6, fcen=1.0, fw=0.2,
            src_z=8, z0=18, z1=26, mon_z=40,
            ns=int(round(60.0 / (0.5 * 0.05 / math.sqrt(3.0)))),
            cases=dict(eps4_lossless=(4.0, 0.0), eps2_lossless=(2.0, 0.0),
                       eps4_lossy=(4.0, 2.0)))

# config B: finer dx=0.04, ~10-cell/0.4 slab
cfgB = dict(nxy=36, nz=80, dx=0.04, npml=7, fcen=1.0, fw=0.2,
            src_z=10, z0=24, z1=34, mon_z=52,
            ns=int(round(60.0 / (0.5 * 0.04 / math.sqrt(3.0)))),
            cases=dict(eps4_lossless=(4.0, 0.0), eps2_lossless=(2.0, 0.0),
                       eps4_lossy=(4.0, 2.0)))

for name, cfg in [("A", cfgA), ("B", cfgB)]:
    print(f"=== config {name} ===")
    r = run(cfg)
    print(f"  inc: flux={r['f_inc']:.4g} {r['t_inc']:.1f}s")
    for k, v in r.items():
        if not isinstance(v, dict):
            continue
        print(f"  {k}: Tsim={v['Tsim']:.4f} Ta={v['Ta']:.4f} err={v['err']:.3%} ({v['t']:.1f}s)")