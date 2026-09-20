# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Diff every example against its oracle twin, from the JSONs both wrote.

Usage
-----
    uv run python scripts/example_oracle_diff.py                  # all of examples/
    uv run python scripts/example_oracle_diff.py examples/02_sources
    uv run python scripts/example_oracle_diff.py --rtol 5e-3 --verbose

Reads ``examples/**/output/<experiment>/`` and compares whatever it finds
against ``<experiment>_meep.json``, the real-pymeep run. All of them are
written by :func:`examples._report.record`.

TWO COMPARISONS, NOT ONE, AND THEY ARE NOT EQUALLY STRONG:

    <experiment>.json     vs  _meep.json    THE TWIN. These two scripts are
                                            byte-comparable apart from the
                                            import, the DEVICE block and the
                                            output names -- that is rule 9 of
                                            example_template.py, and it is
                                            checked, not assumed. A difference
                                            here is the ENGINE's, because
                                            nothing else differs.

    <experiment>_mp.json  vs  _meep.json    A separately written notebook-style
                                            variant on the same shim. It can
                                            differ because it was written
                                            differently -- straight-waveguide_mp
                                            is 63 lines against the twin's 86 --
                                            so it localises a bug less sharply.

They are reported separately and never averaged: collapsing them would let a
clean twin hide a broken variant, or a divergent variant cast doubt on a twin
that is exact.

Both scripts in a twin pair say ``meep_torch.mp`` / ``meep`` -- the port is
NOT a different API, it is the same API resolved to a different engine. That is
what makes the comparison worth anything.

WHY A SEPARATE SCRIPT AND NOT A TEST. A test would have to run both engines,
and the oracle lives in a different interpreter (``~/miniforge3/envs/
meep-oracle``). The two runs are therefore separate acts and the comparison is
a third; making it a script is what lets the oracle side be re-run only when
the fixture is suspect, which is what the 34 committed fixtures under
``tests/examples/data`` already do for the unit tests.

WHAT COUNTS AS A DIFFERENCE. The rule is numpy's::

    close  iff  |a - b| <= atol + rtol * max(|a|, |b|)

with ``atol`` an ABSOLUTE floor (default 1e-12). A relative difference alone
explodes when both values are near zero -- exactly where a field sits outside
its support -- so "a small difference between two small numbers" needs the
absolute term to read as small.

AN EARLIER VERSION SCALED THAT FLOOR FROM THE ORACLE RUN'S OWN PEAK, and that
was wrong in a way worth recording, because it failed silently in the safe
direction. meep echoes the geometry it built and prints ``1e+20`` for
``mp.inf``. One such value made the peak 1e20, the floor 1e14, and every real
field value of order 1e-1 then compared as identical: the first run of the
straight-waveguide pair reported a worst relative difference of 1.3e-10 on
data that had not been checked at all. A tolerance derived from the data it
judges can be moved by the data. An absolute floor cannot.

A SECOND TRAP, RECORDED BECAUSE IT COST A DAY HERE. Comparing engines on a
final field snapshot compares PML residual if the pulse has already left the
cell -- ratios there are enormous and meaningless. This script cannot detect
that; it is a property of when the example snapshots. See
``wiki/bugs/source-normalization-extent.md`` and rule 6 of
``examples/example_template.py``.
"""
import argparse
import json
import math
import pathlib
import sys

# Keys that describe the run rather than its result. Comparing them is not a
# physics check and `engine` is guaranteed to differ, which would make every
# pair fail.
METADATA = {"engine", "script", "device", "path"}

#: Which comparison each file makes against the oracle, and how much it proves.
#: Ordered STRONGEST FIRST, which is the order findings are printed in: the
#: byte-comparable twin before the separately written variant.
PATHS = (("", "twin"), ("mp", "variant"))


def _flatten(obj, prefix=""):
    """``{"a": {"b": [1, 2]}}`` -> ``{"a.b[0]": 1, "a.b[1]": 2}``.

    Flattening is what lets a nested result be compared entry by entry and, more
    importantly, lets a SHAPE difference be reported as a set of missing keys
    rather than as a crash: two arrays of different length simply have keys the
    other lacks, which is a finding rather than a traceback.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(obj, (list, tuple)):
        out = {}
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
        return out
    return {prefix: obj}


def _leaf(key):
    """``"raw_ez.im[3]"`` -> ``"raw_ez"``: the recorded name, without the path."""
    return key.split(".")[0].split("[")[0]


def _numeric(d):
    """The comparable numbers: floats, minus metadata, minus ``raw_``.

    A ``raw_`` PREFIX MEANS "KEEP THIS, DO NOT COMPARE IT". It exists because
    the first pair this harness ran reported a 196% disagreement that was not
    one: comparing an instantaneous CW field across engines compares the PHASE
    of the snapshot. Measured on straight-waveguide, the two agreed to 0.08% on
    the guided envelope and 0.6% on rms, and correlated at 0.995 once meep was
    rolled 18 cells -- 0.058 at zero shift. The field was right; sampling it at
    an instant was the mistake.

    So an example records the instantaneous array as ``raw_`` (a human reading
    the JSON still wants it) and records a phase-insensitive summary beside it
    for the machine. This is the rule-6 trap of example_template.py in a second
    costume: there the run had ended and the comparison saw PML residual, here
    the run is fine and the comparison sees phase. Both are the comparison
    being wrong about WHEN, not the engine being wrong.
    """
    return {k: float(v) for k, v in d.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and _leaf(k) not in METADATA and not _leaf(k).startswith("raw_")}


def _is_result(key):
    """True if a one-sided key is a RESULT the script printed, not chatter.

    A tagged row (``refl:, ...``) or a recorded value is something the script
    asked for, so its absence on one engine is a finding. An untagged stdout
    line -- keyed ``stdout.~<template>`` by the runner -- is usually the
    engine talking to itself, and meep talks a great deal more than meep-torch
    does. Counting those as findings buried the real ones 22 deep on the very
    first pair, so they are summarised instead.
    """
    return not key.startswith("stdout.~")


def compare(port, oracle, rtol=2e-2, atol=1e-12, strict_missing=True):
    """Return ``(worst, findings, n_chatter)`` for one pair of loaded JSONs.

    ``strict_missing`` decides whether a result present on one side only is a
    finding. It is TRUE for a twin and FALSE for a variant, and that is not a
    convenience: the twin is byte-comparable with the oracle apart from the
    import, so a number it fails to produce is the engine failing to produce
    it. A variant is a separately written script -- of course it records
    different things -- so the same signal there means nothing, and reporting
    it buries the twin's real findings under the variant's 179 non-findings.
    """
    a, b = _numeric(_flatten(port)), _numeric(_flatten(oracle))
    findings, worst, chatter = [], 0.0, 0
    for side, keys in (("only-in-port", set(a) - set(b)),
                       ("only-in-oracle", set(b) - set(a))):
        for k in sorted(keys):
            if _is_result(k) and strict_missing:
                findings.append((side, k, None))
            else:
                chatter += 1
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if math.isnan(x) or math.isnan(y):
            findings.append(("nan", k, f"port={x} oracle={y}"))
            continue
        scale = max(abs(x), abs(y))
        if abs(x - y) <= atol + rtol * scale:
            worst = max(worst, abs(x - y) / scale if scale else 0.0)
            continue
        rel = abs(x - y) / scale if scale else float("inf")
        worst = max(worst, rel)
        findings.append(("diff", k, f"port={x:.6e} oracle={y:.6e} rel={rel:.3e}"))
    return worst, findings, chatter


def pairs(root):
    """Yield ``(experiment_dir, label, port_json, oracle_json_or_None)``.

    One experiment can yield two rows -- the shim run and the native run --
    against the same oracle file.
    """
    for d in sorted(root.glob("**/output/*/")):
        if not d.is_dir():
            continue
        twin = d / f"{d.name}_meep.json"
        twin = twin if twin.exists() else None
        for suffix, label in PATHS:
            port = d / (f"{d.name}_{suffix}.json" if suffix else f"{d.name}.json")
            if port.exists():
                yield d, label, port, twin


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default="examples", type=pathlib.Path)
    ap.add_argument("--rtol", type=float, default=2e-2,
                    help="relative tolerance (default 2e-2, matching "
                         "scripts/tutorial_parity.py)")
    ap.add_argument("--verbose", action="store_true",
                    help="also print pairs that agree")
    ap.add_argument("--max-findings", type=int, default=12,
                    help="findings printed per pair before the rest are counted")
    args = ap.parse_args(argv)

    n_pair = n_lonely = n_empty = 0
    bad = {"twin": 0, "variant": 0}
    for d, label, port, twin in pairs(args.root):
        rel = d.relative_to(args.root.parent if args.root.parent != pathlib.Path(".")
                            else args.root)
        if twin is None:
            n_lonely += 1
            if args.verbose:
                print(f"  --    {rel} [{label}]  (no oracle twin JSON)")
            continue
        n_pair += 1
        pdoc, odoc = json.loads(port.read_text()), json.loads(twin.read_text())
        worst, findings, chatter = compare(pdoc, odoc, rtol=args.rtol,
                                           strict_missing=(label == "twin"))
        n_cmp = len(set(_numeric(_flatten(pdoc))) & set(_numeric(_flatten(odoc))))
        note = f" ({chatter} chatter key(s) ignored)" if chatter else ""
        if not n_cmp:
            n_empty += 1
            print(f"  ----  {rel} [{label}]  NOTHING COMPARABLE -- the two runs "
                  f"share no numeric key{note}")
            continue
        if not findings:
            if args.verbose:
                print(f"  ok    {rel} [{label}]  {n_cmp} key(s), "
                      f"worst rel = {worst:.3e}{note}")
            continue
        bad[label] += 1
        print(f"  ----  {rel} [{label}]  {n_cmp} key(s), worst rel = {worst:.3e}"
              f"{note}")
        for kind, key, detail in findings[:args.max_findings]:
            print(f"        [{kind}] {key}" + (f"  {detail}" if detail else ""))
        if len(findings) > args.max_findings:
            print(f"        ... {len(findings) - args.max_findings} more "
                  f"(raise --max-findings to see them)")

    print(f"\n{n_pair} comparison(s) at rtol={args.rtol:g}: "
          f"{bad['twin']} twin-vs-oracle and {bad['variant']} variant-vs-oracle "
          f"disagree; {n_empty} with nothing comparable; "
          f"{n_lonely} run(s) with no oracle twin JSON")
    if bad["twin"]:
        print("A twin disagreement is the ENGINE's: that script is "
              "byte-comparable with the oracle's apart from the import.")
    return 1 if (bad["twin"] or bad["variant"] or n_empty) else 0


if __name__ == "__main__":
    sys.exit(main())
