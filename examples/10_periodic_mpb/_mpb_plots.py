# Shared plot builders for the mpb examples in this directory.
"""Band-diagram and epsilon figures, built but NOT saved.

Upstream's mpb examples print numbers and draw nothing, so every figure
here is a deviation from the script being ported (rule 8). Keeping the
drawing in one place means the deviation is one reviewable file instead of
ten near-copies.

These functions RETURN ``(fig, table, header)`` and never touch the disk.
Saving stays in the example, next to its own ``OUT``, because that is what
makes each example self-contained and what the rules check for.
"""
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["band_figure", "epsilon_figure"]


def band_figure(ms, title, corner_labels=None, gap_floor=1.0):
    """Frequency against k-point INDEX, gaps shaded.

    The index axis is MPB's own convention and the honest one: the path is
    a piecewise walk between symmetry points and the segments are not the
    same length in k, so plotting against |k| would stretch some segments
    and squash others.
    """
    a = np.asarray(ms.all_freqs, dtype=float)
    x = np.arange(a.shape[0])
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    ax.plot(x, a, "-", color="tab:blue", lw=1.1)
    for pct, lo, hi in ms.gap_list:
        if pct > gap_floor:      # thinner than the line below this
            ax.fill_between(x, lo, hi, color="tab:orange", alpha=0.35, lw=0)
            ax.text(x[-1], 0.5 * (lo + hi), f" {pct:.1f}% ", va="center",
                    ha="right", fontsize=8, color="darkorange")
    if corner_labels:
        ticks = np.linspace(0, a.shape[0] - 1, len(corner_labels))
        ax.set_xticks(ticks)
        ax.set_xticklabels(corner_labels)
        for t in ticks[1:-1]:
            ax.axvline(t, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("k-point index")
    ax.set_ylabel("frequency  (c/a)")
    ax.set_title(title)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    header = "k_index," + ",".join(f"band{b + 1}" for b in range(a.shape[1]))
    return fig, np.column_stack((x, a)), header


def epsilon_figure(ms, title):
    """The dielectric the solver runs on, one panel per distinct component.

    ``get_epsilon()`` is the AXIAL component, which is what TM sees; TE
    runs on the in-plane pair. They are rarely the same array -- Kottke
    subpixel smoothing is directional, so even a scalar material gives
    three different boundary treatments -- and for an anisotropic material
    they are different structures entirely.
    """
    comps = [np.asarray(ms.get_epsilon_tensor(c, c)) for c in range(3)]
    xx, yy, zz = comps
    eq = lambda p, q: np.allclose(p, q, rtol=0, atol=1e-12)
    shown = [(r"$\varepsilon_{zz}$  (TM)", zz)]
    if not eq(xx, zz):
        shown.append((r"$\varepsilon_{xx}$  (TE)", xx))
        if not eq(yy, xx):
            shown.append((r"$\varepsilon_{yy}$  (TE)", yy))

    fig, axes = plt.subplots(1, len(shown), figsize=(4.2 * len(shown), 4.0),
                             dpi=150, squeeze=False)
    vmin = min(a.min() for _, a in shown)
    vmax = max(a.max() for _, a in shown)
    for ax, (lab, a) in zip(axes[0], shown):
        im = ax.imshow(np.atleast_2d(a).T, interpolation="nearest",
                       cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(lab)
        ax.axis("off")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.035, pad=0.02,
                 label="relative permittivity")
    fig.suptitle(title)
    lab, arr = shown[-1]
    return fig, np.atleast_2d(arr), f"{lab.strip()} on the lattice grid"
