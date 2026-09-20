# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Run an example and its oracle twin, and turn both stdouts into comparable JSON.

Usage
-----
    uv run python scripts/run_example_pair.py examples/01_basics/bend-flux.py
    uv run python scripts/run_example_pair.py examples/01_basics --all
    uv run python scripts/run_example_pair.py examples/01_basics/ring.py --only meep

WHY STDOUT IS THE RESULT, AND NOT A HAND-WRITTEN record() CALL. There are 82
upstream scripts under ``examples_meep/examples`` and 6 are ported. Writing a
per-script ``record(...)`` for the other 76 means deciding, 76 times, which
numbers matter -- and getting it wrong silently, because a number nobody
recorded is a number nobody compares.

meep's own tutorials already solved this. They print machine-readable rows::

    print(f"refl:, {nm:2d}, {math.degrees(r_angle):6.2f}, {Rmode:.8f}")
    print(f"ldos:, {ldos_1[j]}, {ldos_2[2]}")

The documentation tells you to grep for the tag. 25 of the 82 use it. So the
twin pair's stdout IS the result, already in a comparable form, and -- this is
the part that matters -- the two scripts are byte-comparable apart from the
import, so they print the SAME LINES IN THE SAME ORDER BY CONSTRUCTION. No
mapping table, nothing to keep in sync, and no way for a port to quietly stop
reporting a number.

``record()`` is still there for examples that want richer data than they print
(arrays, fields). This merges into whatever the script already wrote rather
than competing with it.

WHAT IS FILTERED OUT AND WHY THAT IS NOT CHEATING. Engine chatter carries
numbers that are guaranteed to differ and mean nothing: wall-clock timings,
step counters, progress widgets, output paths. Comparing those would make every
pair fail for a reason that is not physics. The denylist is ``NOISE`` below --
explicit, short, and every entry is engine bookkeeping rather than a result. A
line that is filtered is COUNTED and reported, so a script whose entire output
gets swallowed shows up as "no comparable output" instead of as a pass.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

#: Lines that are engine bookkeeping, not results. Each carries numbers that
#: differ between engines by construction (wall clock, step counters, paths).
NOISE = (
    re.compile(r"^\s*Elapsed run time"),
    re.compile(r"^\s*run \d+ finished at t ="),
    re.compile(r"^\s*Meep\s*:"),
    re.compile(r"^\s*FloatProgress"),
    re.compile(r"^\s*creating output file"),
    re.compile(r"^\s*time for "),
    re.compile(r"^\s*on time step "),
    re.compile(r"^\s*Working in \d+D"),
    # meep echoes the geometry it built, one line per object, and those lines
    # carry 1e+20 for mp.inf. Left in, that single value used to wreck a
    # peak-scaled tolerance floor for every other key in the file.
    re.compile(r"^\s*(block|cylinder|sphere|cone|ellipsoid|prism|wedge)\b"),
    re.compile(r"^\s*(size|center|axes|subpixel|geometric object)\b"),
    re.compile(r"^\s*\(1,0,0\)|^\s*\(0,1,0\)|^\s*\(0,0,1\)"),
    re.compile(r"^\s*field decay\(t ="),
    re.compile(r"^\s*halving dt"),
    # meep's startup banner. These carry numbers -- cell size, resolution, the
    # eps diagonal -- and meep-torch does not print them, so left in they show
    # up as one-sided keys on every single pair. Worse, on a script that prints
    # nothing of its own they were the ONLY keys, which made an empty
    # comparison look like a populated one: bent-waveguide reported "10
    # numbers from 3 lines" on the oracle side and 0 on ours, when the truth is
    # that the script has no print() in it at all.
    re.compile(r"^\s*Computational cell is "),
    re.compile(r"^\s*dielectric constant epsilon diagonal"),
    re.compile(r"^\s*subpixel-averaging"),
    re.compile(r"^\s*autocorrelation"),
    re.compile(r"^\s*estimated time"),
    re.compile(r"^\s*-+\s*$"),
    re.compile(r"^\s*$"),
)

#: meep's machine-readable row: ``tag:, v1, v2, ...``
TAGGED = re.compile(r"^\s*([A-Za-z][\w-]*):,\s*(.*)$")

#: A float, including exponent and sign. Deliberately does NOT match the digits
#: inside an identifier (``ldos_1``), because those are names, not results.
NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.])")

#: Upstream scripts that take REQUIRED positional arguments and exit 2 without
#: them. Five of the 82 do. Both twins are given the SAME arguments, which is
#: not a detail: a pair run with different parameters compares two different
#: experiments and any agreement or disagreement between them is meaningless.
#:
#: The values are upstream's own first choice where there is a choices= list,
#: and a representative position otherwise. They are written out rather than
#: derived, because "pick the first choice" is a rule that silently picks
#: something else the day upstream reorders the list.
SCRIPT_ARGS = {
    "dipole_in_vacuum_1D": ["x"],                     # choices x, y
    "dipole_in_vacuum_cyl_on_axis": ["x"],            # choices x, z
    "dipole_in_vacuum_cyl_off_axis": ["x", "0.5"],    # pol, radial position
    "edge_emitter_2D": ["x", "0.5"],                  # pol, position from facet
    "edge_emitter_3D": ["x", "0.5"],
}

ORACLE_PYTHON = os.environ.get(
    "MEEP_ORACLE_PYTHON",
    str(pathlib.Path.home() / "miniforge3/envs/meep-oracle/bin/python"))


def parse_stdout(text):
    """``{"refl[0][2]": 0.31, "line0[0]": 1.5}`` from a run's stdout.

    Tagged rows are keyed by their tag and occurrence, so they stay aligned
    even if an untagged line appears on one side only. Untagged numeric lines
    fall back to their ordinal among untagged numeric lines -- weaker, and the
    reason the tagged idiom is worth keeping when porting.
    """
    out, seen, kept, dropped = {}, {}, 0, 0
    for line in text.splitlines():
        if any(p.search(line) for p in NOISE):
            dropped += 1
            continue
        m = TAGGED.match(line)
        body = m.group(2) if m else line
        nums = [float(x) for x in NUMBER.findall(body)]
        if not nums:
            continue
        if m:
            key = m.group(1)
        else:
            # KEY AN UNTAGGED LINE BY ITS OWN TEXT, not by its ordinal.
            # The twins are byte-comparable, so they print the same f-string
            # templates; blanking the numbers out of a line therefore yields
            # the SAME key on both engines. An ordinal cannot do this: meep
            # prints geometry and progress chatter that meep-torch does not,
            # every later line shifts, and the comparison reports phantom
            # differences. Measured on straight-waveguide: 22 of them.
            key = "~" + NUMBER.sub("#", body).strip()
        i = seen.get(key, 0)
        seen[key] = i + 1
        for j, v in enumerate(nums):
            out[f"{key}[{i}][{j}]"] = v
        kept += 1
    return out, kept, dropped


def _interp(script):
    if script.stem.endswith("_meep"):
        return [ORACLE_PYTHON]
    return ["uv", "run", "python"]


def _experiment(stem):
    for suffix in ("_meep", "_mp"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def run_one(script, timeout=1800, device="cpu"):
    """Run one script and merge its stdout numbers into its experiment JSON."""
    script = script.resolve()
    env = dict(os.environ, MPLBACKEND="Agg", MEEP_TORCH_DEVICE=device)
    argv = SCRIPT_ARGS.get(_experiment(script.stem), [])
    try:
        p = subprocess.run(_interp(script) + [script.name] + argv, cwd=script.parent,
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"script": script.name, "status": "timeout", "kept": 0}
    rows, kept, dropped = parse_stdout(p.stdout)
    status = "ok" if p.returncode == 0 else f"exit {p.returncode}"

    stem = script.stem
    experiment = _experiment(stem)
    dest = script.parent / "output" / experiment / f"{stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(dest.read_text()) if dest.exists() else {
        "engine": "meep" if stem.endswith("_meep") else "meep-torch",
        "script": stem}
    doc["stdout"] = rows
    doc["status"] = status
    if argv:
        doc["argv"] = argv          # recorded so the pair can be shown identical
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return {"script": script.name, "status": status, "kept": kept,
            "dropped": dropped, "nums": len(rows), "json": dest,
            "stderr_tail": p.stderr.strip().splitlines()[-1:] if p.returncode else []}


def targets(path, want):
    """The scripts to run: an experiment's flavours, or a whole directory's."""
    if path.is_file():
        stem = path.stem
        for suffix in ("_meep", "_mp"):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        cands = [path.parent / f"{stem}{s}.py" for s in ("", "_mp", "_meep")]
    else:
        seen, cands = set(), []
        for f in sorted(path.glob("*.py")):
            s = f.stem
            for suffix in ("_meep", "_mp"):
                if s.endswith(suffix):
                    s = s[:-len(suffix)]
                    break
            if s in seen:
                continue
            seen.add(s)
            cands += [path / f"{s}{x}.py" for x in ("", "_mp", "_meep")]
    for c in cands:
        if not c.exists():
            continue
        if want == "meep" and not c.stem.endswith("_meep"):
            continue
        if want == "torch" and c.stem.endswith("_meep"):
            continue
        yield c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=pathlib.Path)
    ap.add_argument("--only", choices=("meep", "torch"), default=None,
                    help="run only the oracle side or only the meep-torch side")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--device", default=os.environ.get("MEEP_TORCH_DEVICE", "cpu"))
    args = ap.parse_args(argv)

    if not pathlib.Path(ORACLE_PYTHON).exists() and args.only != "torch":
        print(f"oracle interpreter not found: {ORACLE_PYTHON}\n"
              f"set MEEP_ORACLE_PYTHON, or pass --only torch", file=sys.stderr)
        return 2

    n_bad = 0
    for script in targets(args.path, args.only):
        r = run_one(script, timeout=args.timeout, device=args.device)
        mark = "ok  " if r["status"] == "ok" else "FAIL"
        extra = f"{r.get('nums', 0)} number(s) from {r.get('kept', 0)} line(s)"
        if r["status"] == "ok" and not r.get("nums"):
            mark, extra = "----", "no comparable output (every line was noise or textless)"
            n_bad += 1
        elif r["status"] != "ok":
            n_bad += 1
            extra = f"{r['status']}" + (f"  {r['stderr_tail'][0][:90]}"
                                        if r.get("stderr_tail") else "")
        print(f"  {mark}  {script.name:46s} {extra}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
