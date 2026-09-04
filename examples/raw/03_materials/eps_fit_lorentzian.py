# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Fitting eps(f) with Lorentzian poles
#
# meep-torch port of meep/python/examples/eps_fit_lorentzian.py. The meep
# original fits a complex eps profile (read from a CSV) to a sum of Lorentzian
# polarizability terms using NLopt gradient-based optimization. Per the port
# instructions, this is a standalone fit with no FDTD: we synthesize a target
# eps(f) from meep_torch's analytic_eps with known LorentzPole parameters
# (here the cSi material), then fit a 2-pole Lorentzian sum to it via
# scipy.optimize.least_squares (complex residual split into real + imag), and
# plot actual vs fit. No CSV file and no NLopt dependency. Output:
# `output/eps_fit_lorentzian.png`.
#
# Run: `uv run python examples/03_materials/eps_fit_lorentzian.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from meep_torch.core.materials import LorentzPole, analytic_eps

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Target material
# cSi's 3-pole model sampled over its valid band is the synthetic "measured"
# data; the fit uses only 2 poles.

# %%
# target: cSi (3-pole) over its valid band; fit with 2 Lorentzian poles
TARGET = (
    LorentzPole(3.64, 0.0, 8.0),
    LorentzPole(2.76, 2 * 0.063, 2.85),
    LorentzPole(1.73, 2 * 2.5, -0.107),
)
EPS_INF_TRUE = 1.0
N_POLES = 2

freqs = np.linspace(1.0, 2.5, 120)
eps_true = np.array([analytic_eps(EPS_INF_TRUE, TARGET, f) for f in freqs])

# %% [markdown]
# ## Fit
# Complex residual split into real+imag halves; 40 random multi-starts guard
# against local minima of the nonconvex pole fit.

# %%
def lorentz_sum(p, f):
    """p = [A0, w0, g0, A1, w1, g1, ...]; sum of A_n/(w_n^2 - f^2 - i f g_n)."""
    y = np.zeros_like(f, dtype=complex)
    for n in range(len(p) // 3):
        A, w, g = p[3 * n], p[3 * n + 1], p[3 * n + 2]
        y = y + A / (w * w - f * f - 1j * f * g)
    return y


def residual(p):
    model = EPS_INF_TRUE + lorentz_sum(p, freqs)
    diff = model - eps_true
    return np.concatenate([diff.real, diff.imag])


rng = np.random.default_rng(0)
best_p, best_err = None, np.inf
for _ in range(40):
    p0 = 10 ** rng.random(3 * N_POLES)
    try:
        sol = least_squares(residual, p0, method="lm", max_nfev=5000)
        err = float(np.sum(sol.fun ** 2))
        if err < best_err:
            best_err, best_p = err, sol.x
    except Exception:
        continue

poles_fit = tuple(
    LorentzPole(best_p[3 * n + 1], best_p[3 * n + 2], best_p[3 * n + 0]) for n in range(N_POLES)
)
eps_fit = np.array([analytic_eps(EPS_INF_TRUE, poles_fit, f) for f in freqs])

# %% [markdown]
# ## Plot
# Real and imaginary parts of the target vs the 2-pole fit; the title reports
# the final L2 residual.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(freqs, eps_true.real, "bo-", ms=3, label="target (cSi, 3 poles)")
ax1.plot(freqs, eps_fit.real, "r--", label=f"fit ({N_POLES} poles)")
ax1.set_xlabel("frequency (1/um)")
ax1.set_ylabel(r"Re($\epsilon$)")
ax1.legend()
ax1.grid(alpha=0.3)
ax2.plot(freqs, eps_true.imag, "bo-", ms=3, label="target")
ax2.plot(freqs, eps_fit.imag, "r--", label="fit")
ax2.set_xlabel("frequency (1/um)")
ax2.set_ylabel(r"Im($\epsilon$)")
ax2.legend()
ax2.grid(alpha=0.3)
fig.suptitle(f"Lorentzian eps fit (least_squares), final L2 = {best_err:.3e}")
fig.tight_layout()
fig.savefig(OUT / "eps_fit_lorentzian.png", dpi=150)
plt.close(fig)

print(f"eps_fit_lorentzian.png  N_poles={N_POLES}  final L2 error = {best_err:.3e}")
