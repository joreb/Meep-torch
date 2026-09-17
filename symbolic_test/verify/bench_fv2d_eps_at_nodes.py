"""bench_fv2d_eps_at_nodes.py

RESULT: moving eps onto each E component's own Yee node makes the polarization
grating WORSE on all three rows (0.000656 -> 0.002821, 0.026098 -> 0.031292,
0.005405 -> 0.020085) with an arithmetic average, and changes NOTHING measurable
with the Kottke tau blend. Reason: it only touches the Ex row, and x is the
propagation direction there, so Ex ~ 0. Ey and Ez already sit where eps is
sampled. Two dead hypotheses for the half-wave leakage.
"""
# Run from the repository root: these read fixtures under tests/ by relative
# path. `cd` elsewhere and they will not find them.

import json, math, sys, pathlib
import numpy as np, torch
from meep_torch.core.stepper_fv2d import build_fv2d_coeffs, init_fv2d_state, step_fv2d

REF = json.load(open("tests/examples/data/polarization_grating.json"))

def eff(d, res, eps_at_nodes, t_end=120.0):
    dpml = dsub = dpad = 1.0
    n0, dn, gp = REF["n_0"], REF["delta_n"], REF["grating_period"]
    fcen = 1.0 / REF["wvl"]
    sx = dpml + dsub + d + d + dpad + dpml
    dx = 1.0 / res
    nx, ny = int(round(sx*res)), int(round(gp*res))
    xs = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * dx
    a, b = n0**2, (n0+dn)**2
    one = torch.ones(nx, ny, dtype=torch.float64)
    exx, eyy, ezz = one.clone(), one.clone(), one.clone()
    eyz = torch.zeros(nx, ny, dtype=torch.float64)
    sub = xs <= (dpml+dsub); exx[sub,:] = eyy[sub,:] = ezz[sub,:] = a
    lc = (xs > (dpml+dsub)) & (xs <= (dpml+dsub+2*d))
    phi = math.pi*ys/gp
    c2, s2, cs = torch.cos(phi)**2, torch.sin(phi)**2, torch.cos(phi)*torch.sin(phi)
    exx[lc,:] = a
    eyy[lc,:] = (a*c2 + b*s2)[None,:]
    ezz[lc,:] = (a*s2 + b*c2)[None,:]
    eyz[lc,:] = ((b-a)*cs)[None,:]
    eps6 = (exx, torch.zeros_like(one), torch.zeros_like(one), eyy, eyz, ezz)
    c = build_fv2d_coeffs(eps6, dx, npml=int(dpml*res), eps_at_nodes=eps_at_nodes)
    st = init_fv2d_state(nx, ny)
    mask = torch.zeros(nx, ny, dtype=torch.float64)
    mask[int(round((dpml+0.3*dsub)*res)), :] = 1.0
    omega = 2*math.pi*fcen
    n_steps = int(round(t_end/c.dt))
    ix_out = int(round((sx-dpml-0.5*dpad)*res))
    accY = np.zeros(ny, dtype=complex); accZ = np.zeros(ny, dtype=complex)
    for t in range(n_steps):
        tt = (t+0.5)*c.dt
        amp = math.cos(omega*tt)*min(1.0, tt/10.0)
        st = step_fv2d(st, c, amp, mask, "Ey", src2=("Ez", mask, amp))
        if t >= n_steps//2:
            ph = np.exp(-2j*math.pi*fcen*(t+1.0)*c.dt)*c.dt
            accY += st.Ey[ix_out,:].cpu().numpy()*ph
            accZ += st.Ez[ix_out,:].cpu().numpy()*ph
    p = np.abs(np.fft.fft(accY))**2 + np.abs(np.fft.fft(accZ))**2
    m0, m1 = p[0], 0.5*(p[1]+p[-1])
    sin_t = (1.0/gp)/fcen; cos_t = math.sqrt(1.0-sin_t**2)
    tot = m0 + 2*m1
    return m0/tot, (2*m1/tot)/cos_t

print(f"{'phase':>6} {'meep m0':>9} {'old m0':>9} {'NEW m0':>9} | {'old err':>9} {'NEW err':>9}")
for r in REF["rows"]:
    o0, _ = eff(0.5*r["d"], REF["resolution"], False)
    n0_, _ = eff(0.5*r["d"], REF["resolution"], True)
    print(f"{r['phase']:6.2f} {r['eff_m0']:9.4f} {o0:9.4f} {n0_:9.4f} | "
          f"{abs(o0-r['eff_m0']):9.4f} {abs(n0_-r['eff_m0']):9.4f}")
