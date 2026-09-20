# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Generate a twin pair from an upstream meep tutorial script.

Usage
-----
    uv run python scripts/port_upstream_example.py examples_meep/examples/pw-source.py
    uv run python scripts/port_upstream_example.py examples_meep/examples --all -n
    uv run python scripts/port_upstream_example.py <script> --into examples/11_upstream

For ``<name>.py`` upstream this writes two files into the target directory:

    <name>_mp.py     the same script, with `import meep` resolved to meep_torch
    <name>_meep.py   the same script, on real pymeep

THE BODY IS NOT REWRITTEN. That is the whole design. ``example_template.py``
rule 9 asks for twins "byte-comparable apart from the import and the output
names", and every transformation applied to a body is a way for the two to
stop being comparable -- at which point a difference between them stops being
evidence about the engine. So the only edit to upstream's own lines is the
import, in one of the two files.

HOW THE OUTPUT RULE IS SATISFIED WITHOUT TOUCHING CALL SITES. Rule 2 says
everything lands under ``output/``. Upstream writes with bare relative paths
(``plt.savefig("foo.png")``, meep's own h5 writers, ``np.savetxt``). Rewriting
each call site means parsing and editing 76 scripts' string arguments, and
each edit is a place the twins can diverge. Instead the preamble chdirs into
``output/<name>/`` before upstream's first line runs, so every relative write
lands there -- matplotlib, numpy and meep's writers alike, with no knowledge of
which is which.

The cost is named rather than hidden: a script that READS a data file by
relative path will not find it after the chdir. It fails loudly and gets
reported, which is the right outcome -- rule 11 already says a raise is an
acceptable ending. It is not silently wrong.

``plt.show()`` is left alone. Under Agg it is a no-op with a warning, and
deleting it would be a body edit for no gain.

THE numpy-2 SHIM IS IN THE PREAMBLE FOR THE SAME REASON EVERYTHING ELSE IS.
numpy 2.0 removed a set of long-deprecated aliases and both environments here
are past it (2.4.3 and 2.4.6), so upstream tutorials written before the
removal die on import: ``np.trapz`` in antenna-radiation, ``np.complex_`` in
solve-cw. Each dies on BOTH sides identically, because the twins share a body,
so it yields no evidence in either direction -- an environment mismatch, not a
parity finding. Aliasing them in the preamble fixes both twins with the same
lines, so the byte-comparability that makes a twin difference mean anything
survives. Putting it in the body would not.

The list covers the removals upstream actually uses plus the neighbours in the
same family. It was extended once already: the first version handled only
``trapz``, because the scan that produced it guessed at which aliases to look
for instead of enumerating them, and ``complex_`` surfaced two hours later as
a fresh failure in the oracle run.
"""
import argparse
import ast
import pathlib
import re
import sys

PREAMBLE = '''# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from {upstream} by scripts/port_upstream_example.py.
# The body below is upstream's, unchanged{note}, so that a difference between
# this file and its twin is a difference between the ENGINES. See that
# script's docstring for why nothing here rewrites a call site.
#
# The literal spellings below (`matplotlib.use("Agg")`, `OUT = ...`,
# `OUT.mkdir`, `DEVICE`) are the ones
# examples/example_template.py greps for. An earlier version used private
# `_mpl` / `_OUT` names, which satisfied the RULES and failed the CHECKER on
# 216 counts -- two dialects for one convention, and the checker was right to
# complain: a rule nothing can verify is a comment.
import os
import pathlib

import matplotlib
import numpy as _np

matplotlib.use("Agg")                # rule 1: headless, before any pyplot import
for _old, _new in (("trapz", "trapezoid"), ("complex_", "complex128"),
                   ("float_", "float64"), ("alltrue", "all"),
                   ("sometrue", "any"), ("product", "prod"),
                   ("cumproduct", "cumprod"), ("round_", "round")):
    if not hasattr(_np, _old) and hasattr(_np, _new):
        setattr(_np, _old, getattr(_np, _new))   # upstream predates numpy 2.0
OUT = pathlib.Path(__file__).parent / "output/{experiment}/"
OUT.mkdir(exist_ok=True)
{device_line}
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/


def _dump_plot_data(_fig, _path):
    """Write the NUMBERS behind a figure next to it, as <name>.plotdata.json.

    Several tutorials compute a result and only plot it -- binary_grating_
    phasemap, gaussian-beam and refl-quartz print nothing numeric and write no
    data file, so a stdout-and-files comparison has nothing to say about them
    at all. The numbers are right there in the axes.

    This lives in the generated preamble and not in the body, which is the
    whole point: it is emitted identically into both twins, so the
    byte-comparability that makes a twin difference mean anything survives. A
    record() call in the body would not.
    """
    import json as _json
    out, n = {{}}, 0
    for ai, ax in enumerate(_fig.get_axes()):
        for li, ln in enumerate(ax.get_lines()):
            xs, ys = ln.get_xdata(), ln.get_ydata()
            if len(xs):
                out[f"ax{{ai}}.line{{li}}.x"] = [float(v) for v in xs]
                out[f"ax{{ai}}.line{{li}}.y"] = [float(v) for v in ys]
                n += 1
        for ii, im in enumerate(ax.get_images()):
            a = im.get_array()
            if a is not None and a.size:
                import numpy as _np
                a = _np.asarray(a, dtype=float)
                out[f"ax{{ai}}.image{{ii}}.shape"] = list(a.shape)
                out[f"ax{{ai}}.image{{ii}}.stats"] = [float(a.min()), float(a.max()),
                                                  float(a.mean()),
                                                  float((a ** 2).mean() ** 0.5)]
                n += 1
    if n:
        # PREFIXED WITH THE SCRIPT STEM, because the figure's name comes from
        # upstream's body and is therefore IDENTICAL on both twins. Without the
        # prefix both write Ez_angle0.plotdata.json into one directory and the
        # second run overwrites the first -- the same defect that made the two
        # mpb twins clobber each other's h5 output.
        _p = pathlib.Path(str(_path))
        _p = _p.with_name("{stem}-" + _p.stem)
        _p.with_suffix(".plotdata.json").write_text(
            _json.dumps(out, indent=1, sort_keys=True))


import matplotlib.figure as _mplfig   # submodule, not an attribute of
                                     # matplotlib until imported
_savefig = _mplfig.Figure.savefig


def _savefig_and_dump(self, fname, *a, **k):
    r = _savefig(self, fname, *a, **k)
    try:
        _dump_plot_data(self, fname)
    except Exception:
        pass          # a figure we cannot read must not kill the tutorial
    return r


_mplfig.Figure.savefig = _savefig_and_dump
# --- end preamble ----------------------------------------------------------

'''

#: Does this file import meep at all? ``(?!_)`` keeps it off ``meep_torch``.
MEEP_IMPORT = re.compile(r"^\s*(?:import|from)\s+meep\b(?!_)", re.M)

#: The four forms upstream actually uses, counted across examples_meep/examples:
#: 81x ``import meep as mp``, 13x ``from meep import ...``,
#: 6x ``from meep.materials import ...``, 1x ``import meep.adjoint as mpa``.
#: The submodule path and the ``as`` clause are CAPTURED AND KEPT -- an earlier
#: version substituted the head of the statement and left upstream's trailing
#: ``as mp`` behind, producing ``import meep_torch.mp as mp as mp``, which is a
#: SyntaxError on every one of the 81.
#: ``[ \t]*$``, NOT ``\s*$``. Under re.M, ``\s`` matches a newline, so a
#: trailing ``\s*$`` swallows the blank line after the import and the two twins
#: differ by one line -- destroying the byte-comparability the whole comparison
#: rests on, for a reason invisible in the diff of the import itself.
_IMPORT_AS = re.compile(r"^([ \t]*)import[ \t]+meep\b(?!_)((?:\.[\w.]+)?)"
                        r"((?:[ \t]+as[ \t]+\w+)?)[ \t]*$", re.M)
_FROM = re.compile(r"^([ \t]*)from[ \t]+meep\b(?!_)((?:\.[\w.]+)?)[ \t]+import\b", re.M)


#: rule 3: the script REPORTS the device, the engine chooses it.
#: MEEP_TORCH_DEVICE overrides. torch is importable on this side by
#: construction; the oracle side has no torch and no device concept, so the two
#: preambles differ here exactly as they differ on the import.
TORCH_DEVICE = 'DEVICE = "cuda"'
MEEP_DEVICE = 'DEVICE = "meep-cpu"       # real meep: CPU only, no device concept'

#: ``ms.device = DEVICE`` goes in after every ModeSolver, on the torch side only.
#: READ ONLY BY THE 3D PLANEWAVE BACKEND -- meep_torch/mpb/backends.py:173-187
#: warns and stays on the CPU for every other backend -- so the line is a
#: request, not a guarantee, and that warning is the engine being honest.
#: Real meep's mpb.ModeSolver has no such attribute, which is why it can only
#: ever exist in the _mp twin.
MODESOLVER = re.compile(r"ModeSolver$")


def _modesolver_lines(src, stem, device_expr=None):
    """Insert the generated ModeSolver settings after each construction.

    ``filename_prefix`` goes in on BOTH sides -- it is real mpb API, and it is
    how ``example_template.py`` rule 2 says an mpb script is told where to
    write -- but it carries the SCRIPT STEM, not the experiment name.

    An earlier version used the experiment name on both twins, on the reasoning
    that identical filenames are what lets the h5 outputs be compared. That is
    exactly backwards: both twins then wrote `mpb_bragg-h.k01.b01.y.tm.h5` into
    one directory and the second run silently overwrote the first, leaving one
    file where a comparison needs two. The stem gives `mpb_bragg_mp-...` and
    `mpb_bragg_meep-...`, which is also what meep's own FDTD writers already do
    (`ring_mp-ez-*` against `ring_meep-ez-*`).

    ``device`` goes in on the TORCH SIDE ONLY, because real mpb's ModeSolver
    has no such attribute.

    By AST, not regex: upstream builds these across many lines
    (``ms = mpb.ModeSolver(\n    num_bands=8,\n    ...)``) and a line-oriented
    match would land the assignment inside the call.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, 0
    inserts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        fn = node.value.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if not MODESOLVER.search(name or ""):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        # col_offset, because a ModeSolver can be built inside a function --
        # parallel-wvgs-mpb.py does -- and an unindented insertion there is an
        # "unexpected indent" SyntaxError on the NEXT line, not on this one.
        inserts.append((node.end_lineno, node.targets[0].id, node.col_offset))
    if not inserts:
        return src, 0
    lines = src.splitlines(keepends=True)
    for lineno, var, col in sorted(inserts, reverse=True):
        pad = " " * col
        block = f'{pad}{var}.filename_prefix = str(OUT / "{stem}")\n'
        if device_expr:
            block = f'{pad}{var}.device = "cuda"\n' + block
        lines.insert(lineno, block)
    return "".join(lines), len(inserts)


#: ``from meep import mpb`` is the ONE import that does not go to meep_torch.mp.
#: mpb is a sibling package, ``meep_torch.mpb``, exactly as the hand-written
#: examples in examples/10_periodic_mpb import it. Routing it through .mp gave
#: ``ImportError: cannot import name 'mpb' from 'meep_torch.mp'`` on all nine
#: mpb tutorials, and that read as nine shim gaps in the first results table
#: when it was one porter bug.
_FROM_MPB = re.compile(r"^([ \t]*)from[ \t]+meep[ \t]+import[ \t]+mpb\b", re.M)


def to_meep_torch(src):
    """Resolve upstream's meep imports to ``meep_torch``, and nothing else."""
    src = _FROM_MPB.sub(r"\1from meep_torch import mpb", src)
    src = _IMPORT_AS.sub(
        lambda m: f"{m.group(1)}import meep_torch.mp{m.group(2)}"
                  f"{m.group(3) or ' as meep'}", src)
    return _FROM.sub(lambda m: f"{m.group(1)}from meep_torch.mp{m.group(2)} import",
                     src)


def _split_head(src):
    """Return ``(head, body)`` where the preamble may be inserted between them.

    ``head`` holds a shebang, an encoding line, the module docstring and any
    ``from __future__`` imports -- everything the language requires to come
    first. Getting this wrong produces a SyntaxError on 76 files at once.
    """
    lines = src.splitlines(keepends=True)
    i = 0
    while i < len(lines) and (lines[i].startswith("#!")
                              or re.match(r"^#.*coding[:=]", lines[i])):
        i += 1
    rest = "".join(lines[i:])
    m = re.match(r'\s*(?P<q>"""|\'\'\')', rest)
    if m:                                    # module docstring
        end = rest.find(m.group("q"), m.end())
        if end != -1:
            consumed = end + 3
            upto = rest[:consumed].count("\n") + 1
            i += upto
    while i < len(lines) and (re.match(r"^\s*from\s+__future__\s+import", lines[i])
                              or not lines[i].strip()
                              or lines[i].lstrip().startswith("#")):
        if re.match(r"^\s*from\s+__future__\s+import", lines[i]):
            i += 1
        elif i + 1 < len(lines) and re.match(r"^\s*from\s+__future__",
                                             lines[i + 1]):
            i += 1
        else:
            break
    return "".join(lines[:i]), "".join(lines[i:])


def port(upstream, into, force=False, dry_run=False):
    """Write the twin pair. Returns a list of ``(path, action)``."""
    src = upstream.read_text()
    experiment = upstream.stem
    if not MEEP_IMPORT.search(src):
        return [(upstream, "skipped: imports no meep")]
    head, body = _split_head(src)

    out = []
    for suffix, note, transform, device_line in (
            ("_mp", " but for the meep import and the generated ModeSolver lines",
             lambda s: _modesolver_lines(to_meep_torch(s), experiment + "_mp",
                                         "cuda")[0], TORCH_DEVICE),
            ("_meep", " but for the generated ModeSolver lines",
             lambda s: _modesolver_lines(s, experiment + "_meep")[0],
             MEEP_DEVICE)):
        dest = into / f"{experiment}{suffix}.py"
        if dest.exists() and not force:
            out.append((dest, "exists: left alone (use --force)"))
            continue
        text = head + PREAMBLE.format(upstream=upstream.as_posix(),
                                      experiment=experiment, note=note,
                                      stem=f"{experiment}{suffix}",
                                      device_line=device_line) + transform(body)
        if not dry_run:
            into.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
        out.append((dest, "would write" if dry_run else "written"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=pathlib.Path)
    ap.add_argument("--into", type=pathlib.Path,
                    default=pathlib.Path("examples/11_upstream"))
    ap.add_argument("--all", action="store_true",
                    help="port every .py under a directory")
    ap.add_argument("--force", action="store_true",
                    help="overwrite twins that already exist")
    ap.add_argument("-n", "--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.path.is_dir() and not args.all:
        print("path is a directory; pass --all to port all of it", file=sys.stderr)
        return 2
    scripts = sorted(args.path.glob("*.py")) if args.path.is_dir() else [args.path]

    # An upstream directory holds our own already-ported files too; skip them.
    scripts = [s for s in scripts if not s.stem.endswith(("_mp", "_meep"))]

    n_written = n_skipped = 0
    for s in scripts:
        for dest, action in port(s, args.into, force=args.force,
                                 dry_run=args.dry_run):
            if action.startswith(("written", "would write")):
                n_written += 1
            else:
                n_skipped += 1
                print(f"  --    {dest.name}: {action}")
    print(f"\n{len(scripts)} upstream script(s): {n_written} twin file(s) "
          f"{'planned' if args.dry_run else 'written'}, {n_skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
