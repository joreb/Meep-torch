# SPDX-License-Identifier: MIT
# Investigating whether the whole-curl derived form (discrete_curl, see
# meep_torch/core/symbolic.py) describes different physics from meep's
# per-difference composition, or only conditions worse.
#
# Background: wiring stepper_tm's D phase to the whole-curl form failed both
# solve_cw parity gates -- on "BiCGSTAB(2) did not converge" at tol=1e-10, NOT
# on a field value. So the question is open. See
# wiki/concepts/finite-difference-deviation.md.
#
# Run each against whichever meep_torch is on sys.path, e.g. a worktree at the
# whole-curl commit vs the current checkout:
#   PYTHONPATH=<checkout> uv run python symbolic_test/<script>.py <tag>
"""Analytic 2D Green's function vs the two converged solve_cw solutions.

Form choices (dataviz procedure):
  * Re(Ez) is a SIGNED oscillating field -> diverging blue<->red with a neutral
    gray midpoint. Never a rainbow; no hue at the midpoint.
  * Error magnitudes are unsigned -> sequential, one hue light->dark.
  * The radial comparison is three identities -> categorical slots 1/2/3
    (#2a78d6, #eb6834, #1baf7a), validated: all checks PASS, one contrast WARN
    on aqua which is relieved by direct labels rather than a legend box.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.special import hankel1

RES, FCEN, SXY, DPML = 20, 0.3, 12.0, 2.0
k = 2 * np.pi * FCEN

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#e3e2de"
SURFACE = "#fcfcfb"
# diverging: blue <-> red, neutral gray midpoint (palette.md)
DIVERGING = LinearSegmentedColormap.from_list(
    "bl_gy_rd", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec",
                 "#f5a3a2", "#e34948", "#8f1f1e"])
# sequential: single hue, light -> dark
SEQ = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"])

wc = np.load("symbolic_test/output/cw_ez_wholecurl.npy")
pd_ = np.load("symbolic_test/output/cw_ez_perdiff.npy")

ny, nx = wc.shape
x = (np.arange(nx) - nx / 2 + 0.5) / RES
y = (np.arange(ny) - ny / 2 + 0.5) / RES
X, Y = np.meshgrid(x, y, indexing="ij")
R = np.hypot(X, Y)

inner, outer = 1.0, SXY / 2 - DPML - 0.5
mask = (R > inner) & (R < outer)
ana_full = 0.25j * hankel1(0, k * np.maximum(R, 1e-9))


def scale_to(num):
    """Best complex scale taking the numerical solve onto the analytic."""
    a, n = ana_full[mask], num[mask]
    return (a @ n.conj()) / (n @ n.conj())


s_wc, s_pd = scale_to(wc), scale_to(pd_)
wc_s, pd_s = wc * s_wc, pd_ * s_pd

err_pd = np.abs(pd_s - ana_full)
err_wc = np.abs(wc_s - ana_full)
form_gap = np.abs(wc_s - pd_s)

rel = lambda e: np.linalg.norm(e[mask]) / np.linalg.norm(ana_full[mask])
rel_pd, rel_wc = rel(err_pd), rel(err_wc)
rel_gap = np.linalg.norm(form_gap[mask]) / np.linalg.norm(ana_full[mask])

fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.2), facecolor=SURFACE)
fig.suptitle("solve_cw against the analytic 2D Green's function  "
             r"$E_z=\frac{i}{4}H_0^{(1)}(kr)$",
             fontsize=14, color=INK, y=0.975)

vmax = float(np.abs(ana_full[mask]).max())
extent = [x[0], x[-1], y[0], y[-1]]


def field_panel(ax, data, title):
    im = ax.imshow(np.real(data).T, origin="lower", extent=extent,
                   cmap=DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.set_xlabel("x (µm)", fontsize=9, color=MUTED)
    ax.set_ylabel("y (µm)", fontsize=9, color=MUTED)
    return im


im0 = field_panel(axes[0, 0], ana_full, r"analytic  Re $E_z$")
field_panel(axes[0, 1], pd_s, r"per-difference curl  Re $E_z$")
cb = fig.colorbar(im0, ax=axes[0, :2].tolist(), fraction=0.035, pad=0.02)
cb.set_label(r"Re $E_z$ (a.u.)", fontsize=9, color=MUTED)
cb.outline.set_visible(False)

ax = axes[0, 2]
im = ax.imshow(np.where(mask, err_pd, np.nan).T, origin="lower", extent=extent,
               cmap=SEQ, interpolation="nearest")
ax.set_title(f"|numerical − analytic|   rel-L2 = {rel_pd:.3e}",
             fontsize=11, color=INK, pad=8)
ax.set_xlabel("x (µm)", fontsize=9, color=MUTED)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.outline.set_visible(False)

# --- radial cut along +x, direct-labelled (relief for the contrast WARN) ---
ax = axes[1, 0]
row = ny // 2
sel = (x > inner) & (x < outer)
ax.plot(x[sel], np.real(ana_full[sel, row]), color=BLUE, lw=2.0, zorder=3)
ax.plot(x[sel], np.real(pd_s[sel, row]), color=ORANGE, lw=2.0, ls="--", zorder=4)
ax.plot(x[sel], np.real(wc_s[sel, row]), color=AQUA, lw=2.0, ls=":", zorder=5)
ax.annotate("analytic", (x[sel][-1], np.real(ana_full[sel, row])[-1]),
            xytext=(4, 6), textcoords="offset points", color=INK, fontsize=9)
ax.annotate("per-difference", (x[sel][len(x[sel]) // 3],
            np.real(pd_s[sel, row])[len(x[sel]) // 3]),
            xytext=(4, 10), textcoords="offset points", color=INK, fontsize=9)
ax.annotate("whole-curl", (x[sel][2 * len(x[sel]) // 3],
            np.real(wc_s[sel, row])[2 * len(x[sel]) // 3]),
            xytext=(4, -14), textcoords="offset points", color=INK, fontsize=9)
ax.set_title(r"radial cut along $+x$", fontsize=11, color=INK, pad=8)
ax.set_xlabel("r (µm)", fontsize=9, color=MUTED)
ax.set_ylabel(r"Re $E_z$ (a.u.)", fontsize=9, color=MUTED)

# --- the decisive panel: the two forms against each other ----------------
ax = axes[1, 1]
finite = form_gap[mask]
im = ax.imshow(np.where(mask, form_gap, np.nan).T, origin="lower", extent=extent,
               cmap=SEQ, interpolation="nearest",
               norm=LogNorm(vmin=max(finite[finite > 0].min(), 1e-20),
                            vmax=finite.max()))
ax.set_title(f"|whole-curl − per-difference|   rel-L2 = {rel_gap:.3e}",
             fontsize=11, color=INK, pad=8)
ax.set_xlabel("x (µm)", fontsize=9, color=MUTED)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.outline.set_visible(False)

# --- error vs radius, both forms ----------------------------------------
ax = axes[1, 2]
bins = np.linspace(inner, outer, 40)
idx = np.digitize(R[mask], bins)
rc = 0.5 * (bins[1:] + bins[:-1])
prof = lambda e: np.array([np.sqrt((e[mask][idx == i] ** 2).mean())
                           if (idx == i).any() else np.nan
                           for i in range(1, len(bins))])
ax.semilogy(rc, prof(err_pd), color=ORANGE, lw=2.0)
ax.semilogy(rc, prof(err_wc), color=AQUA, lw=2.0, ls=":")
ax.semilogy(rc, prof(form_gap), color=BLUE, lw=2.0)
ax.annotate("per-difference vs analytic", (rc[len(rc) // 2], prof(err_pd)[len(rc) // 2]),
            xytext=(-30, 12), textcoords="offset points", color=INK, fontsize=9)
ax.annotate("whole-curl vs analytic", (rc[len(rc) // 3], prof(err_wc)[len(rc) // 3]),
            xytext=(-20, -18), textcoords="offset points", color=INK, fontsize=9)
ax.annotate("the two forms vs each other", (rc[len(rc) // 2], prof(form_gap)[len(rc) // 2]),
            xytext=(-60, 10), textcoords="offset points", color=INK, fontsize=9)
ax.set_title("RMS error vs radius", fontsize=11, color=INK, pad=8)
ax.set_xlabel("r (µm)", fontsize=9, color=MUTED)
ax.set_ylabel("RMS |error| (a.u.)", fontsize=9, color=MUTED)

for a in axes.ravel():
    a.set_facecolor(SURFACE)
    a.tick_params(colors=MUTED, labelsize=8)
    for sp in a.spines.values():
        sp.set_color(GRID)
    a.grid(True, color=GRID, lw=0.6, alpha=0.7)
    a.set_axisbelow(True)

out = "symbolic_test/output/solve_cw_analytic_comparison.png"
fig.savefig(out, dpi=140, facecolor=SURFACE, bbox_inches="tight")
print(f"saved {out}")
print(f"  rel-L2 per-difference vs analytic : {rel_pd:.6e}")
print(f"  rel-L2 whole-curl     vs analytic : {rel_wc:.6e}")
print(f"  rel-L2 the two forms vs each other: {rel_gap:.6e}")
