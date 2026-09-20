# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Rank the meep_torch.mp gaps the upstream tutorials found.

Usage
-----
    uv run python scripts/shim_gap_priority.py --scan            # gaps only
    uv run python scripts/shim_gap_priority.py                   # ranked
    uv run python scripts/shim_gap_priority.py --markdown        # wiki table

Running the 81 generated tutorials on ``meep_torch.mp`` produced a list of
errors, each of which is a script that runs on real meep and does not run
here. This turns that list into an order of work.

WHAT IS COUNTED IN CODE, because it is arithmetic and must be exact:
  - how many tutorials each gap blocks, read from the run log rather than
    remembered;
  - which tutorials they are;
  - whether the gap is a hard stop (the script exits) or a numeric difference.

WHAT IS JUDGED, because it needs reading:
  - how deep the change is -- wiring an argument through an existing path is
    not the same job as adding a physical model;
  - whether the work is already described somewhere in the tree.

THE RANK IS COMPUTED HERE, not asked for. Value is the number of tutorials
unblocked; cost is the judged depth. Asking a model to "prioritise" collapses
those two into one number nobody can argue with afterwards, and the whole
point of a priority list is that the next person can disagree with the weights
without redoing the judgments.
"""
import argparse
import collections
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNLOG = pathlib.Path("/tmp/claude-1000/shim_run.txt")

#: Collapse an error message to the capability it is about. The message text
#: varies per script (band lists, component names); the gap does not.
NORMALISE = [
    (r"is_integrated=True", "source: is_integrated (integrated current)"),
    (r"get_eigenmode_coefficients\(bands=DiffractedPlanewave",
     "mode decomposition: DiffractedPlanewave bands"),
    (r"get_eigenmode_coefficients\(bands=\[", "mode decomposition: multiple bands"),
    (r"get_eigenmode_coefficients kpoint_func", "mode decomposition: kpoint_func"),
    (r"EigenModeSource eig_kpoint", "EigenModeSource: eig_kpoint"),
    (r"EigenModeSource transverse \(y\) parity", "EigenModeSource: y-parity filter"),
    (r"step function '.*' in a cylindrical run", "cylindrical: step functions"),
    (r"component '.*' for this run", "cylindrical: component access"),
    (r"PML\.direction must be", "PML: direction R / non-cartesian"),
    (r"Mirror\(.\) needs an even cell", "symmetry: Mirror on odd cell count"),
    (r"run_k_points", "mpb: run_k_points beyond the homogeneous case"),
    (r"gyrotropic media exist only in the 1D", "materials: gyrotropic beyond 1D"),
    (r"MultilevelAtom exists only in the 1D", "materials: MultilevelAtom beyond 1D"),
    (r"phase_in_material", "fields: phase_in_material"),
    (r"k_point\.z .*kz_2d", "k_point: kz_2d not wired to Simulation"),
    (r"AttributeError: set_boundary", "Simulation: set_boundary"),
    (r"SVD did not converge", "numerics: SVD non-convergence"),
    (r"cannot import name 'mpb'", "PORTER BUG (fixed in efa9b6e)"),
    (r"timeout", "slow: exceeded the run budget"),
]

QUESTIONS = {
    "depth": {
        "type": "score",
        "instructions": (
            "`gap.capability` is a capability meep has and meep-torch does not, "
            "found by running meep's own tutorials against the meep-torch "
            "compatibility shim. `gap.message` is the error the engine raised "
            "and `gap.blocks` lists the tutorials that stop on it. How deep a "
            "change would supporting it be in an FDTD engine that already has a "
            "working core, given that the shim's API surface already exposes the "
            "name?"
        ),
        "criteria": [
            "Shallow: plumbing. An argument or attribute that the underlying "
            "engine already supports has not been passed through, or a guard is "
            "stricter than it needs to be.",
            "Moderate: real but bounded work inside one subsystem -- extending a "
            "mode solver to several bands, or letting an existing step-function "
            "mechanism run in another coordinate system.",
            "Deep: new physics or a new numerical capability -- a material model "
            "that exists in only one dimensionality, or a solver that does not "
            "exist yet in this engine at all.",
        ],
    },
    "kind": {
        "type": "choice",
        "instructions": (
            "What KIND of work is closing `gap.capability`? Judge from the error "
            "message and the capability name."
        ),
        "criteria": {
            "wiring": "An existing capability is not connected to the API path "
                      "that needs it.",
            "extension": "An existing subsystem must be generalised to a case it "
                         "does not yet cover.",
            "new_capability": "A model or solver that does not exist in this "
                              "engine must be written.",
            "not_engine_work": "Not an engine gap at all -- a tooling bug, a "
                               "timeout, or a numerical robustness issue.",
        },
    },
}


def scan(path: pathlib.Path):
    """Gaps read from the per-run JSONs, which are the durable record.

    NOT from a run log. A log in /tmp goes stale the moment a gap is fixed, and
    a stale ranking keeps recommending work that is already done -- this tool
    listed the mpb porter bug, set_boundary and Mirror as open after all three
    had been fixed, because it was parsing the log of the run that found them.
    The JSONs live beside the outputs and are rewritten by every run.
    """
    gaps = collections.defaultdict(lambda: {"blocks": [], "message": ""})
    root = path if path.is_dir() else ROOT / "examples/11_upstream"
    # A JSON written before 0e5993c records `status` but no `error`, so its gap
    # cannot be named. The run log can fill that in -- but ONLY as a message for
    # a script the JSONs already say is failing. The set of failures always
    # comes from the JSONs, so a stale log can add a wrong description and can
    # never resurrect a gap that has since been fixed. That asymmetry is the
    # whole point: staleness that cannot change the conclusion is tolerable.
    from_log = {}
    if RUNLOG.exists():
        for line in RUNLOG.read_text().splitlines():
            m = re.match(r"\s*(?:FAIL|----)\s+(\S+?)_mp\.py\s+(.*)", line)
            if m:
                from_log.setdefault(m.group(1),
                                    re.sub(r"^exit \d+\s+", "", m.group(2).strip()))
    not_ours, unverified = [], []
    for j in sorted((root / "output").glob("*/*_mp.json")):
        doc = json.loads(j.read_text())
        if doc.get("status") in (None, "ok"):
            continue
        script = j.stem[:-len("_mp")]
        # A failure is only a GAP once the oracle passes the same script. The
        # runner writes meep's status in the sibling JSON, and four tutorials
        # fail there too -- cavity_arrayslice calls
        # get_array(Volume(...), component=...) which meep itself rejects
        # ("The first argument must be the component"), cherenkov-radiation
        # wants the h5topng binary, coupler cannot open its input, and
        # differential_cross_section wants PyMieScatt. Ranking a missing
        # Python module as a shim gap keeps recommending work that is not ours.
        oracle = j.with_name(j.name.replace("_mp.json", "_meep.json"))
        if oracle.exists():
            if json.loads(oracle.read_text()).get("status") != "ok":
                not_ours.append(script)
                continue
        else:
            unverified.append(script)
        msg = doc.get("error") or from_log.get(script) or doc.get("status", "")
        if j.stat().st_mtime < _engine_mtime():
            # This script has not run since the engine last changed, so its
            # failure may already be fixed. Per-script, not per-tree: a global
            # check passes the moment ANY script is re-run, which is exactly
            # when the stale ones are easiest to believe.
            script += "  [STALE: not re-run since the engine changed]"
        cap = next((name for pat, name in NORMALISE if re.search(pat, msg)),
                   None)
        if cap is None:
            cap = "other: " + re.sub(r"^exit \d+\s+", "", msg)[:60]
        gaps[cap]["blocks"].append(script)
        if not gaps[cap]["message"]:
            gaps[cap]["message"] = re.sub(r"^exit \d+\s+", "", msg)[:200]
    if not_ours:
        gaps["NOT OURS: the oracle fails this script too"] = {
            "blocks": sorted(not_ours),
            "message": "real meep exits non-zero on the same script, so this "
                       "is upstream or environment, not a shim gap"}
    if unverified:
        gaps["UNVERIFIED: no oracle result to compare against"] = {
            "blocks": sorted(unverified),
            "message": "no _meep.json next to it; run the oracle side before "
                       "counting this as ours"}
    return dict(gaps)


_ENGINE_MTIME = None


def _engine_mtime():
    """Commit time of the newest change under meep_torch/, cached."""
    global _ENGINE_MTIME
    if _ENGINE_MTIME is None:
        import subprocess as _sp
        try:
            out = _sp.run(["git", "log", "-1", "--format=%ct", "--",
                           "meep_torch"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
            _ENGINE_MTIME = float(out) if out else 0.0
        except Exception:
            _ENGINE_MTIME = 0.0
    return _ENGINE_MTIME


def staleness_banner(path):
    """Warn when the engine is newer than the runs this ranking is built on.

    The JSONs record what the last RUN found, never what the code now does. A
    gap fixed after its script last ran still appears here, and that is not a
    bug in the tool -- it is the tool being unable to know. Saying so is the
    difference between a stale list and a misleading one: this ranking already
    listed set_boundary and Mirror-on-odd-cells after both were fixed in
    f457f5c, because neither script had been re-run.
    """
    import subprocess as _sp
    root = path if path.is_dir() else ROOT / "examples/11_upstream"
    jsons = list((root / "output").glob("*/*_mp.json"))
    if not jsons:
        return
    newest_run = max(j.stat().st_mtime for j in jsons)
    try:
        code = _sp.run(["git", "log", "-1", "--format=%ct", "--",
                        "meep_torch/mp", "meep_torch/core"],
                       cwd=ROOT, capture_output=True, text=True).stdout.strip()
        newest_code = float(code) if code else 0.0
    except Exception:
        return
    if newest_code > newest_run:
        import datetime as _dt
        fmt = lambda t: _dt.datetime.fromtimestamp(t).strftime("%H:%M")
        print(f"!! engine changed at {fmt(newest_code)}, newest run is "
              f"{fmt(newest_run)}.\n"
              f"!! Gaps fixed since then still appear below. Re-run the shim "
              f"pass before trusting this order.\n")


def load_dotenv(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def judge(cap, info):
    from typesafe_sdk import Choice, Score, TypeSafeClient
    kind = {"choice": Choice, "score": Score}
    qs = {q: kind[d["type"]](instructions=d["instructions"], criteria=d["criteria"])
          for q, d in QUESTIONS.items()}
    state = {"gap": {"capability": cap, "message": info["message"],
                     "blocks": info["blocks"], "n_blocked": len(info["blocks"])}}
    with TypeSafeClient() as client:
        return client.system_one(state=state, questions=qs)


def unpack(ans, qid):
    a = ans.answers[qid] if hasattr(ans, "answers") else ans[qid]
    if hasattr(a, "score"):
        return float(a.score), None
    return getattr(a, "choice", None), getattr(a, "confidence", None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", type=pathlib.Path,
                    default=ROOT / "examples/11_upstream",
                    help="experiment tree (reads output/*/*_mp.json)")
    ap.add_argument("--scan", action="store_true", help="list gaps, do not judge")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)

    staleness_banner(args.log)
    gaps = scan(args.log)
    if not gaps:
        print(f"no gaps parsed from {args.log}", file=sys.stderr)
        return 2
    if args.scan:
        for cap, info in sorted(gaps.items(), key=lambda kv: -len(kv[1]["blocks"])):
            print(f"{len(info['blocks']):3d}  {cap}")
            print(f"     {' '.join(info['blocks'])}")
        return 0

    load_dotenv(ROOT / ".env")
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set", file=sys.stderr)
        return 2

    rows = []
    for cap, info in gaps.items():
        if cap.startswith("PORTER BUG"):
            continue
        ans = judge(cap, info)
        depth, _ = unpack(ans, "depth")
        kindv, kconf = unpack(ans, "kind")
        n = len(info["blocks"])
        # value over cost. depth runs 0..1 across the three levels; +0.35 keeps
        # a shallow gap from dividing by ~0 and dominating on rounding alone.
        rows.append({"capability": cap, "blocks": info["blocks"], "n": n,
                     "depth": depth, "kind": kindv, "kind_conf": kconf,
                     "score": n / (depth + 0.35)})
    rows.sort(key=lambda r: -r["score"])

    if args.markdown:
        print("| # | gap | blocks | depth | kind | tutorials |")
        print("| --- | --- | --- | --- | --- | --- |")
        for i, r in enumerate(rows, 1):
            print(f"| {i} | {r['capability']} | {r['n']} | {r['depth']:.2f} | "
                  f"{r['kind']} | {' '.join(r['blocks'])} |")
    else:
        for i, r in enumerate(rows, 1):
            print(f"{i:2d}. [{r['score']:5.1f}] {r['capability']}")
            print(f"     blocks {r['n']}: {' '.join(r['blocks'])}")
            print(f"     depth {r['depth']:.2f}  kind {r['kind']} "
                  f"(conf {r['kind_conf']})")
    json.dump(rows, open("/tmp/claude-1000/shim_gap_rank.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
