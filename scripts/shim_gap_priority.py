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
    """Gaps found in a run log, with the tutorials each one blocks."""
    gaps = collections.defaultdict(lambda: {"blocks": [], "message": ""})
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(FAIL|----)\s+(\S+?)_mp\.py\s+(.*)", line)
        if not m:
            continue
        script, msg = m.group(2), m.group(3).strip()
        cap = next((name for pat, name in NORMALISE if re.search(pat, msg)),
                   None)
        if cap is None:
            cap = "other: " + re.sub(r"^exit \d+\s+", "", msg)[:60]
        gaps[cap]["blocks"].append(script)
        if not gaps[cap]["message"]:
            gaps[cap]["message"] = re.sub(r"^exit \d+\s+", "", msg)[:200]
    return dict(gaps)


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
    ap.add_argument("--log", type=pathlib.Path, default=RUNLOG)
    ap.add_argument("--scan", action="store_true", help="list gaps, do not judge")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)

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
