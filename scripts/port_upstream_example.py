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
``np.trapz`` was removed in numpy 2.0 and both environments here are past it
(2.4.3 and 2.4.6), so one upstream tutorial dies on import with
``AttributeError: module 'numpy' has no attribute 'trapz'``. It dies on BOTH
sides identically, so it yields no evidence in either direction -- it is an
environment mismatch with code written before the removal, not a parity
finding. Aliasing it in the preamble fixes both twins with the same line, so
the byte-comparability that makes a twin difference mean anything survives.
Putting it in the body would not.
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
# `OUT.mkdir`, `DEVICE = os.environ.get("MEEP_TORCH_DEVICE")`) are the ones
# examples/example_template.py greps for. An earlier version used private
# `_mpl` / `_OUT` names, which satisfied the RULES and failed the CHECKER on
# 216 counts -- two dialects for one convention, and the checker was right to
# complain: a rule nothing can verify is a comment.
import os
import pathlib

import matplotlib
import numpy as _np

matplotlib.use("Agg")                # rule 1: headless, before any pyplot import
if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
    _np.trapz = _np.trapezoid        # upstream predates numpy 2.0, which removed it
# rule 2. parents=True because output/ itself may not exist yet -- plain
# exist_ok=True raises FileNotFoundError on the first run of a fresh clone.
OUT = pathlib.Path(__file__).parent / "output/{experiment}/"
OUT.mkdir(parents=True, exist_ok=True)
{device_line}
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
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
TORCH_DEVICE = ('import torch  # noqa: E402  (device probe only)\n'
                'DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (\n'
                '    "cuda" if torch.cuda.is_available() else "cpu")')
MEEP_DEVICE = 'DEVICE = "meep-cpu"       # real meep: CPU only, no device concept'

#: ``ms.device = DEVICE`` goes in after every ModeSolver, on the torch side only.
#: READ ONLY BY THE 3D PLANEWAVE BACKEND -- meep_torch/mpb/backends.py:173-187
#: warns and stays on the CPU for every other backend -- so the line is a
#: request, not a guarantee, and that warning is the engine being honest.
#: Real meep's mpb.ModeSolver has no such attribute, which is why it can only
#: ever exist in the _mp twin.
MODESOLVER = re.compile(r"ModeSolver$")


def _modesolver_lines(src, experiment, device_expr=None):
    """Insert the generated ModeSolver settings after each construction.

    ``filename_prefix`` goes in on BOTH sides: it is real mpb API, and setting
    it is how ``example_template.py`` rule 2 says an mpb script is told where
    to write. Naming both engines' outputs identically is also what lets the
    h5 files be compared by name rather than by guesswork.

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
        block = f'{pad}{var}.filename_prefix = str(OUT / "{experiment}")\n'
        if device_expr:
            block = (f"{pad}{var}.device = {device_expr}"
                     f"   # generated: 3D planewave backend only\n") + block
        lines.insert(lineno, block)
    return "".join(lines), len(inserts)


def to_meep_torch(src):
    """Resolve upstream's meep imports to ``meep_torch.mp``, and nothing else."""
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
             lambda s: _modesolver_lines(to_meep_torch(s), experiment,
                                         "DEVICE")[0], TORCH_DEVICE),
            ("_meep", " but for the generated ModeSolver lines",
             lambda s: _modesolver_lines(s, experiment)[0], MEEP_DEVICE)):
        dest = into / f"{experiment}{suffix}.py"
        if dest.exists() and not force:
            out.append((dest, "exists: left alone (use --force)"))
            continue
        text = head + PREAMBLE.format(upstream=upstream.as_posix(),
                                      experiment=experiment, note=note,
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
