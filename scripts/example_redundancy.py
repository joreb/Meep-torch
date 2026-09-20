# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Which hand-written examples are now redundant against the generated twins?

Usage
-----
    uv run python scripts/example_redundancy.py --dry-run      # show the requests
    uv run python scripts/example_redundancy.py                # judge
    uv run python scripts/example_redundancy.py --apply        # and act on it

WHY A JUDGMENT AND NOT A DIFF. ``scripts/port_upstream_example.py`` now
generates a twin pair for all 81 upstream tutorials, and roughly 40 files under
``examples/01_basics`` ... ``examples/10_periodic_mpb`` cover the same
tutorials. Some of those are faithful ports that the generator now supersedes.
Others are not: they demonstrate the meep-torch NATIVE api rather than the
shim, or carry analysis, plots and commentary upstream never had, or share a
name with an upstream script while running a different experiment.

A textual diff cannot separate those. Two files can overlap ninety percent and
differ in the ten percent that is the whole reason one of them exists. So the
mechanical facts stay in code and the reading stays in the model.

IN CODE, because they are lookups and must not be guessed:
  - which upstream script a file corresponds to, and whether a generated twin
    exists for it;
  - whether any test imports or names the file -- a referenced example is not
    removable whatever it contains;
  - line counts and a similarity ratio.

IN THE MODEL, because they need to be read:
  - what relationship the file has to the upstream script;
  - whether replacing it with the generated twin would lose anything;
  - how much of its content is its own.

THE POLICY IS EXPLICIT AND LIVES HERE, not in the model. Nothing is removed
unless the model says redundant AND says faithful port AND code says no test
refers to it. Any one of those failing keeps the file. That ordering is
deliberate: the cost of keeping a redundant example is a duplicate; the cost of
deleting a load-bearing one is a hole in the evidence nobody notices until a
gate goes missing.

Credentials: TYPESAFE_API_KEY, from the environment or the repo-root .env.
Never hardcoded, never printed.
"""
import argparse
import difflib
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "examples_meep" / "examples"
GENERATED = ROOT / "examples" / "11_upstream"
EXAMPLE_DIRS = sorted(
    d for d in (ROOT / "examples").iterdir()
    if d.is_dir() and re.match(r"^\d\d_", d.name) and d.name != "11_upstream"
)

#: Enough source for the model to read the file's intent without paying for
#: the whole thing. Head plus every comment block that explains a deviation.
EXCERPT_CHARS = 3500


def excerpt(text: str) -> str:
    """Head of the file, plus any line that reads like a stated deviation.

    The head carries the docstring and the geometry; the deviation lines are
    where a port says what it changed and why, which is exactly the content
    that makes a file NOT redundant. Taking the head alone would systematically
    hide the evidence against removal.
    """
    head = text[:EXCERPT_CHARS]
    marks = [ln.strip() for ln in text.splitlines()
             if re.search(r"#.*\b(differs?|deviat|unlike upstream|instead of|"
                          r"changed|native|not upstream|we |ours?)\b", ln, re.I)]
    if marks:
        head += "\n\n# lines stating a deviation:\n" + "\n".join(marks[:25])
    return head


def referenced_by_tests(stem: str) -> list[str]:
    """Tests that name this example FILE. A lookup, never a judgment.

    Matches `<stem>.py` or the stem as a whole word, never as a substring. The
    first version grepped the bare stem and reported ring.py as referenced by
    93 test files -- it was matching "string", "measuring", "during". A lookup
    that over-reports here is not harmless: every false hit becomes a KEEP for
    a reason that does not exist, and the whole point of doing this part in
    code rather than in the model is that it is supposed to be exact.
    """
    pattern = rf"{re.escape(stem)}\.py|\b{re.escape(stem)}\b"
    try:
        out = subprocess.run(
            ["grep", "-rlE", "-e", pattern, str(ROOT / "tests")],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    return [str(pathlib.Path(p).relative_to(ROOT)) for p in out.split() if p]


def pairs():
    """Yield the hand-written files that have an upstream counterpart."""
    for d in EXAMPLE_DIRS:
        for f in sorted(d.glob("*.py")):
            stem = f.stem
            for suffix in ("_meep", "_mp"):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            up = UPSTREAM / f"{stem}.py"
            if up.exists():
                yield f, up, stem


QUESTIONS = {
    "relationship": {
        "type": "choice",
        "instructions": (
            "`local.source` is a file in this project's curated examples tree. "
            "`upstream.source` is the meep tutorial of the same name that it was "
            "derived from, and `generated.exists` says whether an automatically "
            "generated twin of that upstream tutorial now exists in the project. "
            "What relationship does the LOCAL file have to the UPSTREAM one?"
        ),
        "criteria": {
            "faithful_port": (
                "The local file runs the same experiment as upstream with the "
                "same geometry and parameters, differing only in the engine "
                "import and in housekeeping such as output paths, headless "
                "plotting or a closing print. A generated twin would serve the "
                "same purpose."
            ),
            "adds_analysis": (
                "The local file runs upstream's experiment and adds something a "
                "generated twin would not have: extra analysis or validation, "
                "commentary explaining a deviation from upstream, or additional "
                "figures and data exports. Using meep-torch rather than meep is "
                "NOT such an addition -- see the note in `api_standard`."
            ),
            "different_experiment": (
                "Despite the shared name, the local file studies a different "
                "configuration, geometry, or quantity from the upstream script."
            ),
            "unclear": "There is not enough in the excerpts to tell.",
        },
    },
    "redundant": {
        "type": "noul",
        "instructions": (
            "Would deleting `local.source` and relying on the automatically "
            "generated twin of `upstream.source` lose nothing that a reader or a "
            "maintainer of this project would want? Answer for the CONTENT only; "
            "ignore whether tests reference the file, which is checked separately "
            "in code."
        ),
        "criteria": {
            "true": "Nothing of value is lost; the generated twin covers it.",
            "false": (
                "Something is lost: analysis, an explanation of a deviation "
                "from upstream, or a different experiment."
            ),
        },
    },
    "own_content": {
        "type": "score",
        "instructions": (
            "How much of `local.source` is content of its own, rather than "
            "upstream's script with the engine import swapped and output paths "
            "adjusted?"
        ),
        "criteria": [
            "Almost none: upstream's script with the import and output paths "
            "changed, and nothing else of substance.",
            "Some: upstream's experiment, plus a modest amount of added "
            "commentary, export or plotting that upstream did not have.",
            "A great deal: substantial original code -- a different API, "
            "additional analysis or validation, or a materially different "
            "experiment.",
        ],
    },
}


def build_state(local: pathlib.Path, up: pathlib.Path, stem: str) -> dict:
    lt, ut = local.read_text(), up.read_text()
    ratio = difflib.SequenceMatcher(None, lt, ut).quick_ratio()
    return {
        "local": {
            "path": str(local.relative_to(ROOT)),
            "lines": len(lt.splitlines()),
            "source": excerpt(lt),
        },
        "upstream": {
            "path": str(up.relative_to(ROOT)),
            "lines": len(ut.splitlines()),
            "source": excerpt(ut),
        },
        "generated": {
            "exists": (GENERATED / f"{stem}_mp.py").exists(),
            "note": (
                "A generated twin is upstream's body unmodified apart from the "
                "engine import, wrapped in a preamble that sets the output "
                "directory and the device. It carries no commentary of its own."
            ),
        },
        "api_standard": (
            "Examples in this project are written against MEEP's API, reached "
            "through meep_torch.mp, so that each stays line-comparable with the "
            "upstream tutorial it came from. Using meep-torch's own internal API "
            "instead would be a defect to correct, never a feature to preserve. "
            "Every file being judged already meets this standard, so it is not a "
            "point of difference between any local file and its upstream."
        ),
        "textual_similarity": round(ratio, 3),
    }


def load_dotenv(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def call_typesafe(state: dict) -> dict:
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
    kind = {"noul": Noul, "choice": Choice, "score": Score}
    questions = {
        qid: kind[q["type"]](instructions=q["instructions"], criteria=q["criteria"])
        for qid, q in QUESTIONS.items()
    }
    with TypeSafeClient() as client:
        return client.system_one(state=state, questions=questions)


def _answer(answers, qid):
    """(value, probability) for one question, whatever shape the SDK returns.

    SystemOneResponse carries `.answers`; an individual answer may expose its
    payload as `.value`/`.answer` and a probability under one of a few names.
    Read defensively rather than pin a shape -- the alternative is a crash on
    the next SDK release, in a script whose whole job is a slow batch.
    """
    a = answers.answers[qid] if hasattr(answers, "answers") else answers[qid]
    if isinstance(a, dict):
        val = a.get("value", a.get("answer"))
        prob = a.get("probability", a.get("confidence"))
    else:
        # The SDK names the payload after the primitive: Noul carries `.noul`
        # (the probability of true), Choice `.choice` plus `.confidence`, Score
        # `.score`. Reading only `.value`/`.probability` silently yielded None
        # for every Noul, so the redundancy threshold could never fire and one
        # file sitting exactly at 0.75 was reported as "not confidently
        # redundant". A missing attribute that degrades to None is worse than a
        # crash: it looks like an answer.
        if hasattr(a, "noul"):
            return bool(a.noul >= 0.5), float(a.noul)
        val = getattr(a, "choice", getattr(a, "score",
                      getattr(a, "value", getattr(a, "answer", a))))
        prob = getattr(a, "confidence", getattr(a, "probability", None))
    return val, (float(prob) if isinstance(prob, (int, float)) else None)


def verdict(answers, tests: list[str]) -> tuple[str, str]:
    """The POLICY, in code. Three independent conditions, all required."""
    rel, _ = _answer(answers, "relationship")
    red, red_p = _answer(answers, "redundant")
    if tests:
        return "KEEP", f"referenced by {len(tests)} test file(s)"
    choice = getattr(rel, "choice", rel)
    if choice != "faithful_port":
        return "KEEP", f"relationship = {choice}"
    if not (red is True or str(red).lower() == "true"
            or (isinstance(red_p, float) and red_p >= 0.75)):
        return "KEEP", f"not confidently redundant (redundant={red}, p={red_p})"
    return "REPLACE", f"faithful port, redundant (p={red_p}), no test refers to it"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the request that would be sent, and stop")
    ap.add_argument("--apply", action="store_true",
                    help="act on REPLACE verdicts (otherwise report only)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    load_dotenv(ROOT / ".env")
    if not args.dry_run and not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set (environment or repo-root .env).",
              file=sys.stderr)
        return 2

    rows = list(pairs())
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} hand-written file(s) have an upstream counterpart\n")

    decisions = []
    for local, up, stem in rows:
        state = build_state(local, up, stem)
        tests = referenced_by_tests(local.stem)
        rel = local.relative_to(ROOT)
        if args.dry_run:
            print(f"=== {rel} ===")
            print(json.dumps({"state": state, "questions": QUESTIONS},
                             indent=2)[:2000])
            continue
        answers = call_typesafe(state)
        what, why = verdict(answers, tests)
        decisions.append((rel, what, why, answers))
        print(f"  {what:8s} {str(rel):52s} {why}")

    if args.apply:
        n = sum(1 for _, w, _, _ in decisions if w == "REPLACE")
        print(f"\n--apply: {n} file(s) marked REPLACE. Review the list above "
              f"before acting; this script does not delete on its own.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
