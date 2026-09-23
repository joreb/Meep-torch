# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Plot the tutorials that print numbers instead of drawing figures.

40 of the 81 upstream twins never plot: they print a table to stdout and the
comparison is numeric. Their `_mp`/`_meep` JSON sidecars already hold that
table as `label[row][col]` keys, so the figures can be RENDERED from what the
runs already captured -- no edit to any example, and no re-run.

That is the whole design argument. The alternative was adding a plot to 40
scripts AND their 40 meep twins (a plot only one side draws compares against
nothing), which is 80 hand edits of generated files that
`scripts/port_upstream_example.py` would overwrite on the next regeneration.

Output per table, beside the sidecars in `output/<example>/`:

    <example>_mp-<label>.stdout.png          the figure
    <example>_mp-<label>.stdout.plotdata.json   its data, in the SAME
    <example>_meep-<label>.stdout.png           `axN.lineN.{x,y}` schema the
    <example>_meep-<label>.stdout.plotdata.json porter's `_dump_plot_data`
                                                writes, so the existing oracle
                                                diff compares them with no
                                                changes of its own

plus one manifest at `output/stdout_plots.json` listing every figure, which
side produced it, and whether both sides did -- a one-sided plot is a finding,
not a plot.

Usage:
    python scripts/plot_from_stdout.py [--all] [--example NAME] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "11_upstream" / "output"
MANIFEST = OUT / "stdout_plots.json"

#: `label[row][col]` -- the shape `_dump_stdout` writes for a printed table.
KEY = re.compile(r"^(?P<label>[A-Za-z_][\w.-]*)\[(?P<row>\d+)\]\[(?P<col>\d+)\]$")

#: A table needs at least this many rows to be worth a line plot; below it the
#: figure says less than the numbers it came from.
MIN_ROWS = 3


def tables(stdout: dict) -> dict[str, list[list[float]]]:
    """`{label: [[col0, col1, ...] per row]}` from flat `label[r][c]` keys.

    Rows are ordered by their numeric index, NOT by dict order: the sidecar is
    written by a dict comprehension over stdout lines and `refl[10]` sorts
    before `refl[2]` as a string. Sorting on the parsed int is what stops the
    plotted curve from zig-zagging back on itself.
    """
    cells: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for k, v in stdout.items():
        m = KEY.match(k)
        if not m or not isinstance(v, (int, float)):
            continue
        g = m.groupdict()
        cells[g["label"]][int(g["row"])][int(g["col"])] = float(v)
    out = {}
    for label, rows in cells.items():
        idx = sorted(rows)
        width = max(len(rows[i]) for i in idx)
        # A ragged table is a parse artefact, not data; keep only full rows.
        full = [[rows[i][c] for c in range(width)]
                for i in idx if len(rows[i]) == width]
        if len(full) >= MIN_ROWS and width >= 2:
            out[label] = full
    return out


def plot(table: list[list[float]], label: str, title: str, png: pathlib.Path):
    """Columns 1..N against column 0, which is the independent variable by
    convention in every one of these printed tables (frequency, angle, chi3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r[0] for r in table]
    fig, ax = plt.subplots(figsize=(6, 4))
    data = {}
    for c in range(1, len(table[0])):
        ys = [r[c] for r in table]
        ax.plot(xs, ys, marker=".", lw=1, label=f"col{c}")
        data[f"ax0.line{c - 1}.x"] = xs
        data[f"ax0.line{c - 1}.y"] = ys
    ax.set_xlabel("col0")
    ax.set_ylabel(label)
    ax.set_title(title, fontsize=9)
    if len(table[0]) > 2:
        ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)
    png.with_suffix(".plotdata.json").write_text(json.dumps(data, indent=1))
    return data


def body_plots(example: str) -> bool:
    """Does the example already draw its own figures?

    Checked against the SOURCE rather than against the presence of a PNG: a
    run that crashed before its savefig leaves no PNG and would otherwise be
    mistaken for a script that never plots.
    """
    src = ROOT / "examples" / "11_upstream" / f"{example}_mp.py"
    if not src.is_file():
        return False
    text = src.read_text(errors="replace")
    end = text.find("# --- end preamble")
    body = text[end:] if end != -1 else text
    return bool(re.search(r"\b(plt\.(plot|imshow|pcolormesh|contour|scatter|semilogy|"
                          r"loglog|matshow)|\.plot2D|subplots\()", body))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="include examples that already draw their own figures")
    ap.add_argument("--example", help="one example directory only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    dirs = [OUT / a.example] if a.example else sorted(
        d for d in OUT.iterdir() if d.is_dir())
    manifest, made, skipped = [], 0, 0
    for d in dirs:
        name = d.name
        if not a.all and body_plots(name):
            skipped += 1
            continue
        sides = {}
        for engine in ("mp", "meep"):
            f = d / f"{name}_{engine}.json"
            if not f.is_file():
                continue
            try:
                sides[engine] = tables(json.loads(f.read_text()).get("stdout", {}))
            except json.JSONDecodeError:
                continue
        if not sides:
            continue
        for label in sorted(set().union(*(s.keys() for s in sides.values()))):
            row = {"example": name, "label": label,
                   "engines": sorted(k for k in sides if label in sides[k])}
            # A table only one engine printed is a PARITY finding: either the
            # port dropped output or it invented some. Recorded, not drawn.
            row["both"] = row["engines"] == ["meep", "mp"]
            for engine in row["engines"]:
                png = d / f"{name}_{engine}-{label}.stdout.png"
                row[f"{engine}_rows"] = len(sides[engine][label])
                row[f"{engine}_cols"] = len(sides[engine][label][0])
                row[f"{engine}_png"] = str(png.relative_to(OUT))
                if not a.dry_run:
                    plot(sides[engine][label], label,
                         f"{name} [{engine}] {label}", png)
                    made += 1
            manifest.append(row)

    if not a.dry_run:
        MANIFEST.write_text(json.dumps(
            {"generated_from": "stdout tables in the *_mp.json / *_meep.json sidecars",
             "schema": "axN.lineN.{x,y}, matching the porter's _dump_plot_data",
             "plots": manifest}, indent=1))
    both = sum(1 for r in manifest if r["both"])
    print(f"{len(manifest)} tables across {len({r['example'] for r in manifest})} examples")
    print(f"  both engines: {both}   one-sided (a parity finding): {len(manifest) - both}")
    print(f"  {made} figures written, {skipped} examples skipped (they plot already)")
    if not a.dry_run:
        print(f"  manifest -> {MANIFEST}")
    for r in manifest:
        if not r["both"]:
            print(f"    ONE-SIDED {r['example']}:{r['label']} only {r['engines']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
