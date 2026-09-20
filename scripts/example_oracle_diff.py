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
import re
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


ROW = re.compile(r"^(?P<tag>.+)\[(?P<i>\d+)\]\[(?P<j>\d+)\]$")


def realign(a: dict, b: dict) -> tuple[dict, dict, list]:
    """Re-key ``tag[i][j]`` rows by their FIRST column instead of their ordinal.

    meep's tutorials print rows whose first field is the independent variable:
    a time, a frequency, an angle. Keying the rest of the row by its position
    in the output means one extra or missing row shifts everything after it,
    and a single behavioural difference is reported as dozens of numeric ones.

    Measured on absorber-1d: meep-torch prints 16 rows from t=0, meep prints 15
    from t=10, because our `at_every` fires at t=0 and meep's does not. That one
    fact produced 23 phantom differences -- every later row compared against its
    neighbour. Re-keyed by t, it produces exactly what it is: one row present on
    our side only.

    Falls back to the ordinal key for any tag whose first column is not a usable
    label (duplicated or missing on either side), because a bad join is worse
    than an honest ordinal.
    """
    groups: dict[str, dict[str, dict[int, float]]] = {}
    for side, d in (("a", a), ("b", b)):
        for k, v in d.items():
            m = ROW.match(k)
            if not m:
                continue
            groups.setdefault(m["tag"], {}).setdefault(side, {})
    out_a, out_b, notes = dict(a), dict(b), []
    for tag in groups:
        rows = {}
        for side, d in (("a", a), ("b", b)):
            byi: dict[int, dict[int, float]] = {}
            for k, v in d.items():
                m = ROW.match(k)
                if m and m["tag"] == tag:
                    byi.setdefault(int(m["i"]), {})[int(m["j"])] = v
            rows[side] = byi
        labels = {}
        for side in ("a", "b"):
            ls = [r.get(0) for r in rows[side].values()]
            if not ls or any(l is None for l in ls):
                labels = None
                break
            # FORMAT HERE TOO, so the overlap gate below and the join further
            # down agree on what "the same label" means. Raw floats made the
            # gate stricter than the join: two engines printing 0.3 and
            # 0.30000000000000004 share no raw value, the gate finds no overlap
            # and bails to ordinal keys -- reviving the row-shift phantom
            # diffs realign exists to kill, on exactly the rounding the %g join
            # was chosen to absorb.
            ls = [f"{l:g}" for l in ls]
            labels[side] = ls
        # A REPEATED LABEL IS NOT A REASON TO GIVE UP. cyl-ellipsoid runs both
        # at_every(0.25, print_stuff) and at_end(print_stuff), so its last time
        # prints twice on BOTH engines. Demanding unique labels threw away the
        # alignment of all 94 rows to avoid ambiguity in one. Occurrences of the
        # same label are numbered instead, which keeps the join exact and lines
        # the repeats up with each other.
        if labels is None:
            continue                       # keep ordinal keys for this tag
        # A first column is only an INDEPENDENT VARIABLE if it behaves like
        # one: several rows, and the two engines sweeping the same values. On a
        # single-row tag the first column is the RESULT -- perturbation_theory_2d
        # prints one `dwdR:` line -- and re-keying by it guarantees the sides
        # never meet, turning a real 54% difference into "2 rows on one side
        # only". Overlap is the test that separates the two cases.
        shared = set(labels["a"]) & set(labels["b"])
        if len(labels["a"]) < 2 or len(labels["b"]) < 2 or not shared:
            continue
        seen_counts = {}
        for side, dst in (("a", out_a), ("b", out_b)):
            nth = {}
            for i in sorted(rows[side]):
                cols = rows[side][i]
                # COUNT UNDER THE KEY THAT IS ACTUALLY WRITTEN. %g keeps six
                # significant digits, so two labels differing past that print
                # the same; counting occurrences under the raw float gave both
                # of them n=0 and the second row silently overwrote the first,
                # dropping it from the diff entirely. Formatting first makes
                # the occurrence number and the key agree by construction.
                lab = f"{cols[0]:g}"
                n = nth.get(lab, 0)
                nth[lab] = n + 1
                for j, v in cols.items():
                    dst.pop(f"{tag}[{i}][{j}]", None)
                    dst[f"{tag}@{lab}#{n}[{j}]"] = v
            seen_counts[side] = {f"{l}#{k}"
                                 for l, c in nth.items() for k in range(c)}
        only = seen_counts["a"] ^ seen_counts["b"]
        if only:
            notes.append(f"{tag}: {len(only)} row(s) on one side only "
                         f"(first column {sorted(only)[:4]})")
    return out_a, out_b, notes


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
    a, b, notes = realign(a, b)
    findings, worst, chatter = [], 0.0, 0
    for n in notes:
        findings.append(("row-count", n, None))
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


def _arrays_in(path):
    """``{name: ndarray}`` for a data file the twins wrote, or {} if unreadable.

    Covers what the tutorials actually produce: meep's and mpb's h5 writers,
    and numpy text exports. A file that cannot be read returns {} rather than
    raising -- a comparison that dies on one unreadable artefact reports
    nothing about the other forty.
    """
    import numpy as _np
    try:
        if path.suffix == ".h5":
            import h5py
            out = {}
            with h5py.File(path, "r") as f:
                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset) and obj.size:
                        out[name] = _np.asarray(obj[()])
                f.visititems(visit)
            return out
        if path.name.endswith(".plotdata.json"):
            # Numbers lifted out of a matplotlib figure by the generated
            # preamble, for tutorials that compute a result and only plot it.
            import json as _json
            d = _json.loads(path.read_text())
            return {k: _np.asarray(v, dtype=float) for k, v in d.items()
                    if isinstance(v, list) and v}
        if path.suffix in (".csv", ".txt", ".dat"):
            a = _np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
            return {"": _np.asarray(a)}
    except Exception:
        return {}
    return {}


def _envelope_note(x, y, live_frac=0.02):
    """For a time-series dataset, say whether a big pointwise gap is PHASE.

    A dataset written by ``to_appended`` has time as its LAST axis. Comparing
    such a thing element by element compares the phase of the snapshot, and on
    a continuous-wave source that is meaningless: measured on bent-waveguide,
    the two engines sat at a worst relative difference of exactly 2.000 -- a
    sign flip -- while agreeing to 1.1% on the envelope and 1.6% on rms, with
    the last frame correlating at -0.36.

    So the ratio is reported beside the pointwise number rather than left for
    the reader to discover. Exactly 2.000 means antiphase, and an envelope
    ratio near 1 with a large pointwise gap means the fields agree and the
    clocks do not. This is the file-level twin of the ``raw_`` prefix.
    """
    import numpy as _np
    if x.ndim < 3 or x.shape[-1] < 8:
        return ""
    n = min(40, x.shape[-1])
    wx, wy = x[..., -n:], y[..., -n:]
    ex, ey = _np.abs(wx).max(-1), _np.abs(wy).max(-1)
    live = ey > live_frac * ey.max() if ey.max() else None
    if live is None or not live.any():
        return ""
    with _np.errstate(divide="ignore", invalid="ignore"):
        env = float(_np.median(ex[live] / ey[live]))
    rms = float(_np.sqrt((wx ** 2).mean()) / _np.sqrt((wy ** 2).mean() or 1))
    return (f"; envelope ratio={env:.4f} rms ratio={rms:.4f} over the last "
            f"{n} frames -- close to 1 means the fields agree and the PHASE "
            f"differs")


def file_pairs(d: pathlib.Path):
    """Yield ``(suffix, mp_path, meep_path)`` for data files BOTH twins wrote.

    This is why the two twins must not share a filename prefix. They write into
    one directory, so identical names mean the second run overwrites the first
    and there is nothing left to pair -- which is exactly what the mpb twins
    did until the porter was fixed to carry the script stem.
    """
    name = d.name
    def side(tag):
        # Excludes the run JSON itself (exactly `<name>_<tag>.json`) but keeps
        # the `<name>_<tag>-<figure>.plotdata.json` sidecars, which are data.
        return {f.name[len(name) + len(tag) + 1:]: f for f in d.iterdir()
                if f.is_file() and f.name.startswith(f"{name}_{tag}")
                and f.name != f"{name}_{tag}.json"}
    a, b = side("mp"), side("meep")
    for suffix in sorted(set(a) & set(b)):
        yield suffix, a[suffix], b[suffix]


def compare_files(d, rtol=2e-2, atol=1e-12):
    """One finding per dataset that disagrees, not one per element."""
    import numpy as _np
    findings, n_cmp = [], 0
    for suffix, pa, pb in file_pairs(d):
        A, B = _arrays_in(pa), _arrays_in(pb)
        for key in sorted(set(A) & set(B)):
            x, y = A[key], B[key]
            if x.shape != y.shape:
                findings.append(("file-shape", f"{suffix}:{key}",
                                 f"port{list(x.shape)} oracle{list(y.shape)}"))
                continue
            n_cmp += 1
            scale = _np.maximum(_np.abs(x), _np.abs(y))
            bad = _np.abs(x - y) > atol + rtol * scale
            if bad.any():
                with _np.errstate(divide="ignore", invalid="ignore"):
                    rel = _np.where(scale > 0, _np.abs(x - y) / scale, 0.0)
                detail = (f"{bad.sum()}/{x.size} elements, "
                          f"worst rel={float(rel.max()):.3e}")
                detail += _envelope_note(x, y)
                findings.append(("file-diff", f"{suffix}:{key}", detail))
    return findings, n_cmp


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

    n_pair = n_lonely = n_empty = n_failed = 0
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
        # A RUN THAT CRASHED IS NOT A RUN THAT DISAGREED. Without this, a
        # meep-torch side that exited 1 reported as "NOTHING COMPARABLE, 3959
        # chatter key(s) ignored" -- indistinguishable from a script that
        # simply printed different things, and the 3959 were the ORACLE's
        # perfectly good results being counted as noise. binary_grating is
        # exactly that case: it dies on get_eigenmode_coefficients(bands=[...]).
        bad_status = [(w, d.get("status")) for w, d in
                      (("port", pdoc), ("oracle", odoc))
                      if d.get("status") not in (None, "ok")]
        if bad_status:
            n_failed += 1
            who = ", ".join(f"{w} {st}" for w, st in bad_status)
            print(f"  ----  {rel} [{label}]  DID NOT RUN: {who}")
            continue
        worst, findings, chatter = compare(pdoc, odoc, rtol=args.rtol,
                                           strict_missing=(label == "twin"))
        n_cmp = len(set(_numeric(_flatten(pdoc))) & set(_numeric(_flatten(odoc))))
        note = f" ({chatter} chatter key(s) ignored)" if chatter else ""
        # File datasets count as comparable output. Deciding "nothing to
        # compare" from stdout alone hid ring and bent-waveguide, which print
        # nothing numeric and write h5 field snapshots.
        if label == "twin" or not any(True for _ in file_pairs(d)):
            ff, fn = [], 0
        else:
            ff, fn = compare_files(d, rtol=args.rtol)
        findings += ff
        if fn:
            note += f", {fn} dataset(s) from paired files"
        if not n_cmp and not fn:
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
          f"disagree; {n_failed} where a side did not run; "
          f"{n_empty} with nothing comparable; "
          f"{n_lonely} run(s) with no oracle twin JSON")
    if bad["twin"]:
        print("A twin disagreement is the ENGINE's: that script is "
              "byte-comparable with the oracle's apart from the import.")
    return 1 if (bad["twin"] or bad["variant"] or n_empty or n_failed) else 0


if __name__ == "__main__":
    sys.exit(main())
