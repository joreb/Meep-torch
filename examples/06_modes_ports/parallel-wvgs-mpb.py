# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Supermode splitting of two parallel Si strips
#
# meep-torch port of meep/python/examples/parallel-wvgs-mpb.py. Even/odd
# supermode splitting of two parallel Si strips vs separation via the fd2d
# full-vector cross-section solver (the mpb ModeSolver's role here).
# Presented as the top-2 neff per separation (unsigned pair; rigorous
# even/odd labeling needs mode-profile parity classification -- the
# shift-invert eigensolver mixes near-degenerate pairs, noted ceiling).
# Expect the pair to converge toward each other as the separation grows.
# Output: `output/parallel-wvgs-mpb.png`.
#
# Run: `uv run python examples/06_modes_ports/parallel-wvgs-mpb.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meep_torch.modes.fd2d import solve_2d

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
F, N_SI, W, H = 1.0 / 1.55, 3.45, 0.5, 0.22
res = 25

# %% [markdown]
# ## Per-separation solve
# Rasterize the two-strip cross-section (2x supersampled averaging) and take
# the two largest neff from the full-vector solver.

# %%
def split(sep):
    n = int(3.0 * res)
    ss = 2
    y = (np.arange(n * ss) + 0.5) / (res * ss) - 1.5
    z = y.copy()
    YY, ZZ = np.meshgrid(y, z, indexing="ij")
    fine = np.ones((n * ss, n * ss))
    for cy in (-(sep + W) / 2, (sep + W) / 2):
        fine[(np.abs(YY - cy) < W / 2) & (np.abs(ZZ) < H / 2)] = N_SI ** 2
    eps = fine.reshape(n, ss, n, ss).mean(axis=(1, 3))
    modes = solve_2d(eps, 1.0 / res, F, k=4, target_neff=2.3)
    ne = np.sort(np.real(np.asarray(modes.neff)))[::-1]
    return float(ne[0]), float(ne[1])

# %% [markdown]
# ## Separation sweep and plot

# %%
seps = [0.1, 0.2, 0.3, 0.4]
rows = [split(s) for s in seps]
fig, ax = plt.subplots(figsize=(5.4, 3.8))
ax.plot(seps, [a for a, b in rows], "bo-", label="mode 1")
ax.plot(seps, [b for a, b in rows], "rs-", label="mode 2")
ax.set_xlabel("separation (um)")
ax.set_ylabel("$n_{eff}$")
ax.set_title("parallel Si strips: top supermode pair (fd2d)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "parallel-wvgs-mpb.png", dpi=150)
plt.close(fig)
for s, (ne, no) in zip(seps, rows):
    print(f"parallel-wvgs-mpb  sep={s}: pair = {ne:.4f} / {no:.4f}")
