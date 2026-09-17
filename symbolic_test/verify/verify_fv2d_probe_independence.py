"""verify_fv2d_probe_independence.py

RESULT: m0 is flat to 0.5% across five probe planes (0.0479-0.0481). This RULED
OUT near-field and evanescent contamination of the Fourier-order extraction --
the transmitted field genuinely carries 4.8% in order 0 wherever it is measured.
"""
# Run from the repository root: these read fixtures under tests/ by relative
# path. `cd` elsewhere and they will not find them.

import json, math, sys, pathlib
import numpy as np, torch
from meep_torch.core.stepper_fv2d import build_fv2d_coeffs, init_fv2d_state, step_fv2d

REF = json.load(open("tests/examples/data/polarization_grating.json"))

def eff(d, res, eps_at_nodes, t_end=120.0, probe_off=0.5):
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
    ix_out = int(round((sx-dpml-probe_off*dpad)*res))
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


row = [r for r in REF["rows"] if r["phase"] == 0.5][0]
print(f'half-wave: meep m0={row["eff_m0"]:.5f}')
print(f'{"probe_off":>10} {"ix":>5} {"m0":>12} {"m1":>12}')
for off in (0.2, 0.35, 0.5, 0.65, 0.8):
    m0, m1 = eff(0.5*row["d"], REF["resolution"], False, probe_off=off)
    print(f'{off:10.2f} {"":>5} {m0:12.6f} {m1:12.6f}')
