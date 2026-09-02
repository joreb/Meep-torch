# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Si strip waveguide dispersion (vs mpb)
#
# Fundamental-mode dispersion neff(f) of a Si strip on oxide (w = 0.3,
# h = 0.25 um) from the fd2d full-vector cross-section mode solver, compared
# with meep.mpb's omega(beta) points. The mode is guided where neff exceeds
# the oxide light line n_ox = 1.45.

# %%
"""meep-torch port of meep/python/examples/mpb_strip.py.

Si strip on oxide (w=0.3, h=0.25 um): fundamental-mode dispersion
neff(f) from the fd2d full-vector cross-section solver, vs meep.mpb's
omega(beta) points (strip_mpb.json). Guided where neff > n_ox = 1.45.

Run: uv run python examples/10_periodic_mpb/mpb_strip.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meep_torch.modes.fd2d import solve_2d

OUT = pathlib.Path(__file__).parent / "output"
ref = json.load(open(pathlib.Path(__file__).parents[2] / "tests" / "examples" / "data" / "strip_mpb.json"))

# %% [markdown]
# ## Cross-section eps and frequency sweep
#
# The strip cross-section is supersampled and averaged; `solve_2d` returns
# the fundamental neff at each of 8 frequencies.

# %%
res, n, ss = 32, 64, 4
zz = (np.arange(n * ss) + 0.5) / (res * ss) - 1.0
yy = zz.copy()
YY, ZZ = np.meshgrid(yy, zz, indexing="ij")
fine = np.ones((n * ss, n * ss))
fine[ZZ < -ref["h"] / 2] = ref["n_ox"] ** 2
fine[(np.abs(YY) < ref["w"] / 2) & (np.abs(ZZ) < ref["h"] / 2)] = ref["n_si"] ** 2
eps = fine.reshape(n, ss, n, ss).mean(axis=(1, 3))

fs = np.linspace(0.7, 1.2, 8)
neffs = [float(np.real(solve_2d(eps, 1.0 / res, f, k=1, target_neff=2.5).neff[0]))
         for f in fs]
fig, ax = plt.subplots(figsize=(5.4, 4))
ax.plot(fs, neffs, "b.-", label="meep-torch fd2d")
for beta, freqs in zip(ref["beta"], ref["bands"]):
    ax.plot(freqs[0], beta / freqs[0], "rx", ms=9, mew=1.5,
            label="meep.mpb" if beta == ref["beta"][0] else None)
ax.axhline(ref["n_ox"], color="g", ls=":", lw=0.8, label="oxide light line")
ax.set_xlabel("frequency (um/lambda)")
ax.set_ylabel("$n_{eff}$")
ax.set_title("Si strip on oxide, fundamental mode")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "mpb_strip.png", dpi=150)
plt.close(fig)
print("mpb_strip.png  neff range:", round(neffs[0], 3), "-", round(neffs[-1], 3))
