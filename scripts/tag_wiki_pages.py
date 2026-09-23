# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Semantically tag and group the wiki's tasks and bugs with TypeSafe.

The wiki's own `tags:` taxonomy answers "what subsystem" (fdtd, pml, torch).
It does NOT answer the two questions that decide what to work on next:

  * which pages share a ROOT CAUSE, so fixing one likely fixes its neighbours
  * which pages block meep parity, as opposed to adding new capability

Those are semantic judgments over prose, which is what System One is for. Code
owns the workflow here: it finds the pages, builds the state, applies the
thresholds and writes the grouping. The model supplies only the three
judgments, and every one of them is a closed set the code already knows how to
consume.

Usage:

    python scripts/tag_wiki_pages.py [--limit N] [--dry-run] [--force]

Credentials, transport and retries come from `wiki_coherence`
(`check_credentials`, `call_typesafe`) and the typesafe_sdk underneath it, so
the key never has to be typed at a prompt and this file owns no HTTP at all.
An earlier version hand-rolled a dotenv reader and a urllib client here; both
already existed two files over.
Results land in `wiki/summaries/semantic-grouping.json` (raw, re-usable) and
a rendered `wiki/summaries/semantic-grouping.md`.

Re-running with unchanged pages is wasted spend: the raw JSON is keyed by
page path and content hash, and a page whose hash is unchanged is skipped
unless `--force`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from wiki_coherence import call_typesafe, check_credentials  # noqa: E402

MODEL = "jev-latest"
ROOT = pathlib.Path(__file__).resolve().parent.parent
WIKI = ROOT / "wiki"
OUT_JSON = WIKI / "summaries" / "semantic-grouping.json"
OUT_MD = WIKI / "summaries" / "semantic-grouping.md"

# --- the judgments ------------------------------------------------------
# One narrow question each. `criteria` describe concrete situations so the
# options stand on their own, per the Choice guidance, and each closed set
# carries an explicit no-match outcome so the model can decline rather than
# be forced into the nearest label.

CLUSTER = {
    "source-injection": {
        "what": "How a current source is injected into the grid: its waveform, "
                "amplitude, measure, placement, polarization or sign.",
        "not_for": "What a monitor does with the resulting fields.",
        "examples": "source carries no measure; injected the envelope not the "
                    "current; point source snapped instead of interpolated.",
    },
    "monitors-and-dft": {
        "what": "Accumulating fields into a DFT and turning them into a flux, "
                "an energy, a force, a far field or a mode coefficient. "
                "Includes Fourier conventions and surface measures.",
        "not_for": "How the source that produced the fields was injected.",
        "examples": "DFT convention is plain where meep is unitary; flux read "
                    "zero; eigenmode coefficients ignore parity.",
    },
    "pml-and-boundaries": {
        "what": "Absorbing boundaries, CPML profiles, absorbers, PEC/PMC walls "
                "and symmetry folding at the domain edge.",
        "examples": "CPML high-edge anchor; alpha_max reflects propagating waves.",
    },
    "cylindrical-bor": {
        "what": "The cylindrical / body-of-revolution path specifically: the "
                "(r,z) grid, the m azimuthal order, axis regularity.",
        "examples": "CW source silenced in cylindrical runs; radial source "
                    "measure; cylindrical flux read zero.",
    },
    "mode-solver-and-mpb": {
        "what": "Eigenmode and band-structure machinery: the FD mode solver, "
                "the MPB port, plane-wave expansion, Kottke smoothing.",
        "examples": "MPB object centre in lattice coordinates; two TE lattices.",
    },
    "device-and-runtime": {
        "what": "Where tensors live and how the code runs rather than what it "
                "computes: CPU/GPU device, dtype, memory, threads, speed.",
        "examples": "CUDA as the default device; thread oversubscription; "
                    "monitor accumulation OOM.",
    },
    "adjoint-and-autodiff": {
        "what": "Gradients: the adjoint solve, checkpointing, reversibility, "
                "differentiable design parameters and inverse design.",
        "examples": "checkpoint boundary state accumulation; reversible adjoint.",
    },
    "geometry-and-materials": {
        "what": "Turning shapes and materials into an eps array: rasterizing, "
                "subpixel smoothing, dispersion poles, chi2/chi3, layout import.",
        "examples": "full-tensor eps smoothing; 3D chi2/chi3; gdstk backend.",
    },
    "symbolic-and-codegen": {
        "what": "The sympy-derived stepper factory: deriving coefficients and "
                "curl compositions symbolically and compiling them.",
        "examples": "sympy stepper kernels; two curl compositions in one package.",
    },
    "tooling-and-oracle": {
        "what": "The scaffolding around the physics: the example porter, the "
                "oracle diff, test fixtures, the knowledge graph, the wiki.",
        "not_for": "A physics defect that a tool happened to surface.",
        "examples": "oracle diff tolerance floor scaled by its own data; "
                    "porter regex swallowed the blank line.",
    },
    "run-control-and-api": {
        "what": "Driving a simulation rather than computing physics: the run "
                "loop, step functions, at_every / at_beginning scheduling, "
                "stopping conditions, and the shape of the mp-facade API.",
        "not_for": "What a monitor computes once the run has produced fields.",
        "examples": "at_every fires at t = 0 here and not in meep; a shim "
                    "method that raises instead of dispatching.",
    },
    "research-roadmap": {
        "what": "Exploratory or forward-looking work with no current defect: "
                "new solver families, ML methods, quantum layers, surveys.",
        "examples": "diffusion inverse design; PINN verdict; entropy methods.",
    },
    "other": "None of the above genuinely fits this page.",
}

ROOT_CAUSE = {
    "convention-mismatch": "Our sign, phase, Fourier convention, coordinate "
        "order or normalization differs from meep's. The code runs and "
        "produces a plausible wrong number.",
    "measure-or-normalization": "A missing or wrong factor from a cell volume, "
        "surface element, resolution or extent. Usually a clean constant "
        "ratio against the reference.",
    "off-by-one-or-indexing": "A stencil, slice, grid offset or half-cell "
        "placement is one index out.",
    "missing-wiring": "The physics exists somewhere in the codebase but this "
        "path does not reach it: a raise, an unhandled branch, a shim gap.",
    "stale-test-or-golden": "The engine is right and the TEST is wrong: a "
        "pinned constant, tolerance or fixture that predates a deliberate "
        "change.",
    "environment-or-dependency": "An external library, driver, device or build "
        "is at fault rather than this repo's logic.",
    "new-capability": "Nothing is broken; this is work to add something that "
        "does not exist yet.",
    "unknown": "The page does not say clearly enough to classify the cause.",
}

QUESTIONS = {
    "cluster": {
        "type": "choice",
        "instructions": "Which part of this differentiable-FDTD codebase does "
                        "this page belong to? Judge by the SUBJECT of the work, "
                        "not by which file happened to be edited.",
        "criteria": CLUSTER,
    },
    "root_cause": {
        "type": "choice",
        "instructions": "What KIND of problem is this, underneath the symptom? "
                        "For a task with no defect, say new-capability.",
        "criteria": ROOT_CAUSE,
    },
    # The wiki's recurring theme is not "is this broken" -- the page says so --
    # but "would it have told us". A silent wrong number survives a green
    # suite; a raise does not. Ordered levels, so Score rather than Noul.
    "silence": {
        "type": "score",
        "instructions": "If this defect were present today, how loudly would "
                        "it announce itself? Judge the FAILURE MODE described "
                        "on the page, not how hard it was to fix.",
        # Score criteria are an ORDERED LIST, not a map -- position is the
        # level, 0 upward, and the API rejects a dict outright. The order here
        # runs quiet-to-loud inverted: level 0 is the loudest failure, level 4
        # the one nothing could have caught, so a HIGHER score means a more
        # dangerous defect and the ranking sorts the right way by default.
        "criteria": [
            {"what": "Not a defect at all: this page describes work to do, "
                     "not something producing a wrong result.",
             "examples": ["port a new capability", "a research direction"]},
            {"what": "It stops the run. An exception, a crash, a shape error. "
                     "Impossible to miss.",
             "examples": ["NotImplementedError from a shim gap",
                          "tensors on two devices"]},
            {"what": "The run completes but the output is visibly absurd: "
                     "zeros, NaNs, a field many orders out, an empty plot.",
             "examples": ["a field of 1e-48", "flux reads 0.00000"]},
            {"what": "The run completes and produces a number that looks "
                     "reasonable and is not. Only a comparison against an "
                     "external reference reveals it.",
             "examples": ["a constant factor against meep",
                          "a sign or phase convention that differs"]},
            {"what": "No existing check could have caught it. The tests are "
                     "normalised, self-consistent, or do not cover the "
                     "quantity at all, so finding it required inventing a new "
                     "measurement.",
             "examples": ["a scale error under a test that divides by its own "
                          "maximum",
                          "an invariant stated in prose but never asserted"]},
        ],
    },
}


def page_state(path: pathlib.Path) -> dict | None:
    """Frontmatter plus the opening prose -- the part that says what it IS."""
    text = path.read_text(errors="replace")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return None
    fm, body = m.group(1), m.group(2)

    def field(name):
        got = re.search(rf"^{name}:\s*(.+)$", fm, re.M)
        return got.group(1).strip().strip("\"'") if got else None

    # Drop headings and code fences: they are structure, not meaning, and the
    # fences in particular are long and carry little signal per token.
    prose = re.sub(r"```.*?```", " ", body, flags=re.S)
    prose = re.sub(r"^#.*$", "", prose, flags=re.M)
    prose = re.sub(r"\s+", " ", prose).strip()
    return {
        "path": str(path.relative_to(WIKI)),
        "title": field("title") or path.stem,
        "description": field("description") or "",
        "declared_tags": field("tags") or "",
        "declared_status": field("status") or "",
        "opening": prose[:1800],
    }


# --- redundancy ---------------------------------------------------------
# Pairwise over 119 pages is 7021 comparisons. The grouping already narrows
# it: two pages can only be redundant if they share BOTH a cluster and a root
# cause, which is 195 pairs. That is code doing the elimination and the model
# doing only the judgment code cannot make.

PAIR_QUESTIONS = {
    "relation": {
        "type": "choice",
        "instructions": "Two pages from the same engineering wiki, A and B. "
                        "How do they relate? Judge the UNDERLYING issue, not "
                        "whether the prose overlaps -- two pages about the "
                        "same subsystem are not redundant unless they are "
                        "about the same thing.",
        "criteria": {
            "duplicate": {
                "what": "The same underlying issue written up twice. Closing "
                        "one closes the other; a reader learns nothing new "
                        "from the second.",
                "not_for": "Two DIFFERENT defects that happen to share a "
                           "cause or a file.",
            },
            "a-subsumes-b": "A is the broader page and fully contains B's "
                            "content. B could be a section of A.",
            "b-subsumes-a": "B is the broader page and fully contains A's "
                            "content. A could be a section of B.",
            "same-cause-different-symptom": {
                "what": "Genuinely different defects or pieces of work sharing "
                        "one underlying cause, so one fix plausibly closes "
                        "both. Worth cross-linking, NOT worth merging.",
                "examples": "two monitors both missing the same measure; two "
                            "call sites of one wrong convention.",
            },
            "overlapping-scope": "Different work whose scopes intersect, so "
                                 "doing one changes what the other means. "
                                 "Neither contains the other.",
            "unrelated": "Same neighbourhood of the codebase, different "
                         "matters. No action.",
        },
    },
    "one_fix_closes_both": {
        "type": "noul",
        "instructions": "Would a single change plausibly resolve BOTH pages at "
                        "once? Yes only if the same edit, in the same place, "
                        "settles each of them.",
    },
}


def pair_state(a: dict, b: dict) -> dict:
    keep = lambda r: {"path": r["path"], "title": r["title"],
                      "status": r.get("declared_status", ""),
                      "opening": r.get("opening", "")[:900]}
    return {"A": keep(a), "B": keep(b)}


def find_redundant(limit: int = 0) -> None:
    rows = [r for r in json.loads(OUT_JSON.read_text())["pages"]
            if r.get("answers")]
    state_by_path = {}
    for p in sorted(list((WIKI / "bugs").glob("*.md"))
                    + list((WIKI / "tasks").glob("*.md"))):
        st = page_state(p)
        if st:
            state_by_path[st["path"]] = st

    buckets: dict[tuple, list] = {}
    for r in rows:
        k = (r["answers"].get("cluster", {}).get("choice"),
             r["answers"].get("root_cause", {}).get("choice"))
        buckets.setdefault(k, []).append(r)

    pairs = []
    for k, v in buckets.items():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                pairs.append((v[i], v[j], k))
    if limit:
        pairs = pairs[:limit]
    print(f"{len(pairs)} candidate pairs (same cluster AND same root cause)\n")

    found = []
    for a_row, b_row, k in pairs:
        sa = state_by_path.get(a_row["path"])
        sb = state_by_path.get(b_row["path"])
        if not (sa and sb):
            continue
        ans = call_typesafe({
            "state": pair_state({**a_row, **sa}, {**b_row, **sb}),
            "questions": PAIR_QUESTIONS})
        rel = ans.get("relation", {})
        row = {"a": a_row["path"], "b": b_row["path"], "bucket": list(k),
               "relation": rel.get("choice"),
               "relation_confidence": rel.get("confidence"),
               "one_fix": ans.get("one_fix_closes_both", {}).get("noul")}
        found.append(row)
        if row["relation"] != "unrelated":
            print(f"  {row['relation']:30s} conf {row['relation_confidence']:.2f} "
                  f"fix-both {row['one_fix']:.2f}\n"
                  f"      A {row['a']}\n      B {row['b']}")
    out = WIKI / "summaries" / "semantic-redundancy.json"
    out.write_text(json.dumps({"model": MODEL, "pairs": found}, indent=1))
    print(f"\n{len(found)} pairs judged -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N pages only")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the state and print it, call nothing")
    ap.add_argument("--dupes", action="store_true",
                    help="pairwise redundancy pass over the existing grouping")
    a = ap.parse_args()

    pages = sorted(list((WIKI / "bugs").glob("*.md"))
                   + list((WIKI / "tasks").glob("*.md")))
    if a.limit:
        pages = pages[:a.limit]

    cache = {}
    if OUT_JSON.exists() and not a.force:
        cache = {r["path"]: r for r in json.loads(OUT_JSON.read_text())["pages"]}

    if not a.dry_run:
        check_credentials()

    out, reused, called = [], 0, 0
    for p in pages:
        st = page_state(p)
        if st is None:
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        prev = cache.get(st["path"])
        if prev and prev.get("hash") == h:
            out.append(prev)
            reused += 1
            continue
        if a.dry_run:
            print(json.dumps(st, indent=1)[:900])
            out.append({**st, "hash": h, "answers": {}})
            continue
        ans = call_typesafe({"state": st, "questions": QUESTIONS})
        called += 1
        row = {"path": st["path"], "title": st["title"], "hash": h,
               "declared_tags": st["declared_tags"],
               "declared_status": st["declared_status"],
               "answers": ans.get("answers", {}),
               "usage": ans.get("usage", {})}
        out.append(row)
        print(f"  {st['path']:58s} "
              f"{row['answers'].get('cluster', {}).get('choice', '?'):24s} "
              f"{row['answers'].get('root_cause', {}).get('choice', '?'):24s} "
              f"{row['answers'].get('silence', {}).get('score', '?')}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"model": MODEL, "pages": out}, indent=1))
    print(f"\n{called} classified, {reused} reused from cache -> {OUT_JSON}")
    render(out)
    print(f"rendered -> {OUT_MD}")
    return 0


def render(rows: list[dict]) -> None:
    """The grouping, as a page. Policy lives HERE and not in the questions.

    The judgments are reusable raw signal; which of them to sort by, and where
    to put the cut, is a decision this file can change without re-spending a
    single token.
    """
    def a(r, q, field):
        return r.get("answers", {}).get(q, {}).get(field)

    live = [r for r in rows if r.get("answers")]
    by_cluster: dict[str, list] = {}
    for r in live:
        by_cluster.setdefault(a(r, "cluster", "choice") or "?", []).append(r)

    open_rows = [r for r in live
                 if r.get("declared_status") in ("investigating", "open",
                                                 "in progress")]
    risky = sorted(open_rows, key=lambda r: -(a(r, "silence", "score") or 0))

    L = ["---",
         "title: Semantic grouping of tasks and bugs",
         "description: Every bug and task page clustered by subject and by root "
         "cause, and ranked by how quietly it would fail. Generated by "
         "scripts/tag_wiki_pages.py.",
         "created: 2026-09-23", f"updated: {date.today()}", "type: summary",
         "tags: [validation, roadmap]", "sources: []", "confidence: medium",
         "---", "",
         "# Semantic grouping of tasks and bugs", "",
         f"{len(live)} pages, judged by TypeSafe `{MODEL}` and regenerated by "
         "`scripts/tag_wiki_pages.py`. The wiki's own `tags:` answer *which "
         "subsystem*; these answer *which root cause* and *how quietly it "
         "would fail*, which is what decides the order of work.", "",
         "`confidence: medium` is deliberate: these are model judgments over "
         "page prose, useful for grouping and ordering, and NOT evidence about "
         "the code. Where a row here disagrees with the page, the page wins.",
         "",
         "## Ranked by how quietly it fails (open pages only)", "",
         "Level 4 means no existing check could have caught it. Level 0 means "
         "it is not a defect at all. This repo's expensive bugs cluster at the "
         "top, which is the argument for reading the list this way rather than "
         "by subsystem.", "",
         "| silence | page | cluster | root cause |", "| --- | --- | --- | --- |"]
    for r in risky[:20]:
        L.append(f"| {a(r,'silence','score') or 0:.2f} | [{r['title'][:64]}]"
                 f"(../{r['path']}) | {a(r,'cluster','choice')} "
                 f"| {a(r,'root_cause','choice')} |")

    L += ["", "## By cluster", ""]
    for c, rs in sorted(by_cluster.items(), key=lambda kv: -len(kv[1])):
        causes: dict[str, int] = {}
        for r in rs:
            causes[a(r, "root_cause", "choice") or "?"] = causes.get(
                a(r, "root_cause", "choice") or "?", 0) + 1
        top = ", ".join(f"{k} ({v})" for k, v in
                        sorted(causes.items(), key=lambda kv: -kv[1])[:3])
        L += [f"### {c} — {len(rs)} pages", "", f"Dominant causes: {top}.", ""]
        for r in sorted(rs, key=lambda r: -(a(r, "silence", "score") or 0)):
            st = r.get("declared_status") or "-"
            L.append(f"- `{st:13s}` {a(r,'silence','score') or 0:.2f} "
                     f"[{r['title'][:70]}](../{r['path']})")
        L.append("")
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
