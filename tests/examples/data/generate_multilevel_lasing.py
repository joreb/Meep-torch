# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle reference: multilevel-atom 1D laser (reduced resolution).

Mirrors meep/python/examples/multilevel-atom.py at resolution 100,
endt 2000; dumps the steady-state field tail and its spectral peak.

Run: mamba run -n meep-oracle python tests/examples/data/generate_multilevel_lasing.py
-> tests/examples/data/multilevel_lasing.json
"""
import json
import math
import os

import meep as mp
import numpy as np

RESOLUTION = 100
ENDT = 2000
TAIL_T = 250.0

ncav, Lcav, dpad, dpml = 1.5, 1.0, 1.0, 1.0
sz = Lcav + dpad + dpml
omega_a = 40
freq_21 = omega_a / (2 * math.pi)
gamma_perp = 4
gamma_21 = (2 * gamma_perp) / (2 * math.pi)
theta = 1
sigma_21 = 2 * theta * theta * omega_a
rate_21 = 0.005
N0 = 37
Rp = 0.0051

transitions = [
    mp.Transition(1, 2, pumping_rate=Rp, frequency=freq_21, gamma=gamma_21,
                  sigma_diag=mp.Vector3(sigma_21, 0, 0)),
    mp.Transition(2, 1, transition_rate=rate_21),
]
ml_atom = mp.MultilevelAtom(sigma=1, transitions=transitions,
                            initial_populations=[N0])
two_level = mp.Medium(index=ncav, E_susceptibilities=[ml_atom])

sim = mp.Simulation(
    cell_size=mp.Vector3(z=sz), resolution=RESOLUTION, dimensions=1,
    boundary_layers=[mp.PML(dpml, side=mp.High)],
    geometry=[mp.Block(center=mp.Vector3(z=-0.5 * sz + 0.5 * Lcav),
                       size=mp.Vector3(mp.inf, mp.inf, Lcav), material=two_level)])
sim.init_sim()
sim.fields.initialize_field(
    mp.Ex, lambda p: 1 if abs(p.z - (-0.5 * sz + 0.5 * Lcav)) < 0.5 / RESOLUTION else 0)

series = []
probe = mp.Vector3(z=-0.5 * sz + Lcav + 0.5 * dpad)


def rec(sim):
    series.append(sim.get_field_point(mp.Ex, probe).real)


sim.run(mp.after_time(ENDT - TAIL_T, rec), until=ENDT)

s = np.array(series)
dt_out = TAIL_T / len(s)
spec = np.abs(np.fft.rfft(s))
freqs = np.fft.rfftfreq(len(s), dt_out)
band = (freqs > 5.0) & (freqs < 8.0)
f_peak = float(freqs[band][spec[band].argmax()])

out = {"resolution": RESOLUTION, "endt": ENDT, "tail_t": TAIL_T,
       "params": {"ncav": ncav, "Lcav": Lcav, "omega_a": omega_a,
                  "gamma_21": gamma_21, "sigma_21": sigma_21,
                  "rate_21": rate_21, "N0": N0, "Rp": Rp},
       "f_peak": f_peak, "tail_max": float(np.abs(s).max()),
       "tail": s[::4].tolist()}
path = os.path.join(os.path.dirname(__file__), "multilevel_lasing.json")
with open(path, "w") as f:
    json.dump(out, f)
print("wrote", path, "f_peak=", f_peak, "tail_max=", out["tail_max"])
