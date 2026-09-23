# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Write the NUMBERS behind a figure next to it, so twins can be compared.

A PNG is not diffable. `examples/11_upstream` already solves this: its
generated preamble carries `_dump_plot_data`, and the oracle diff compares the
resulting `<name>.plotdata.json` sidecars. The hand-written examples under
`01_basics/`, `02_sources/` and the rest had no equivalent, so a `_mp`/`_meep`
pair that agrees numerically and disagrees VISUALLY -- or the reverse -- said
nothing either way.

This is that same dump, in one importable place rather than pasted into each
script. The schema is deliberately identical to the preamble's
(`axN.lineN.{x,y}`, `axN.imageN.{shape,stats}`) so one comparison tool reads
both families.

Images get shape and (min, max, mean, rms) rather than every pixel: a field
map is ~10^5 floats whose exact values differ between engines by construction
(different grids), while those four numbers move when the PHYSICS moves. Lines
get every point, because a spectrum has few of them and its shape is the
result.

Usage from an example (two lines, since examples live one directory down):

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from _plotdata import savefig
    ...
    savefig(fig, OUT / "bent-waveguide_mp.png")   # writes PNG + .plotdata.json
"""
from __future__ import annotations

import json
import pathlib

__all__ = ["dump", "savefig"]


def dump(fig, path) -> int:
    """`<path stem>.plotdata.json` beside the figure. Returns series written.

    Returns 0 and writes nothing for a figure with no lines and no images --
    a colorbar-only or text-only axes has no numbers to compare, and an empty
    sidecar would read as "compared, agreed" to a tool that only checks
    whether the file is there.
    """
    out: dict[str, object] = {}
    n = 0
    for ai, ax in enumerate(fig.get_axes()):
        for li, ln in enumerate(ax.get_lines()):
            xs, ys = ln.get_xdata(), ln.get_ydata()
            if len(xs):
                out[f"ax{ai}.line{li}.x"] = [float(v) for v in xs]
                out[f"ax{ai}.line{li}.y"] = [float(v) for v in ys]
                n += 1
        for ii, im in enumerate(ax.get_images()):
            a = im.get_array()
            if a is not None and a.size:
                import numpy as np
                a = np.asarray(a, dtype=float)
                out[f"ax{ai}.image{ii}.shape"] = list(a.shape)
                out[f"ax{ai}.image{ii}.stats"] = [
                    float(a.min()), float(a.max()), float(a.mean()),
                    float((a ** 2).mean() ** 0.5)]
                n += 1
    if n:
        p = pathlib.Path(str(path))
        p.with_suffix(".plotdata.json").write_text(json.dumps(out, indent=1))
    return n


def savefig(fig, path, *, close=True, **kw):
    """`fig.savefig(path)` plus the sidecar, then close.

    Closing by default because these scripts draw in a loop often enough that
    leaving figures open runs matplotlib out of memory before the script runs
    out of sections -- the same reason `10_periodic_mpb/_mpb_plots.save` closes.
    """
    kw.setdefault("dpi", 150)
    fig.savefig(path, **kw)
    n = dump(fig, path)
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return n
