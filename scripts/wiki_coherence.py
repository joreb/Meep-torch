#!/usr/bin/env python3
"""Wiki coherence judgments via TypeSafe (System One / Jev).

Reads wiki/{bugs,tasks,concepts} pages, builds state in code (stdlib only:
frontmatter parsing, link resolution, excerpting), batches independent Jev
judgments per page:
  a) contradiction  -- linked page supports/contradicts/unrelated  (Choice)
  b) stale premise  -- body shows opening premise since corrected    (Noul)
  c) status drift   -- bug pages: resolution state per the BODY    (Choice)
  d) type fit       -- SCHEMA.md page type per the CONTENT         (Choice)
  e) confidence fit -- how well-supported the claims read           (Score)

The frontmatter value under test is withheld from `state` for (c)/(d)/(e):
Jev judges the body blind, code compares to the withheld ground truth after.
File I/O, link resolution, excerpting and that comparison are code's job;
Jev supplies semantic judgment only.

Credentials: TYPESAFE_API_KEY, from the environment only. Never hardcoded.

Usage:
    python3 scripts/wiki_coherence.py --dry-run
    python3 scripts/wiki_coherence.py --dry-run --pages tasks/port_full-tensor-eps-smoothing.md
    python3 scripts/wiki_coherence.py                      # real run, needs the SDK + key
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

WIKI_DIR = Path(__file__).resolve().parent.parent / "wiki"
DEFAULT_SCOPE = ["bugs", "tasks", "concepts"]
MAX_RELATED = 3       # ponytail: bounds request size; raise if related pages are being missed
MAX_EXCERPT = 1800     # chars; head+tail, see excerpt()
MAX_CLAIM = 500        # chars; the atomic claim relates_* compares

TYPE_CRITERIA = {
    "entity": "Answers 'who/what IS X' -- a named person, org, product, model, or place; overview, key facts/dates, relationships.",
    "concept": "Answers 'what does X MEAN, and why' -- describes an idea or mechanism: definition, state of knowledge, open questions. Descriptive, not instructional -- text phrased as 'rules govern X' still stays concept if it explains how something works rather than telling an operator what to do.",
    "comparison": "Answers 'how does X COMPARE TO Y' -- side-by-side dimensions plus an explicit verdict.",
    "summary": "Answers 'what is the OVERVIEW of topic X' -- topic-organized digest of multiple sources.",
    "query": "Answers 'what is the answer to THIS SPECIFIC QUESTION' -- organized around one verbatim question.",
    "procedure": "Answers 'HOW should X be done' -- prescribes steps an operator follows: a trigger ('when this applies') plus a rule or numbered steps. Prescriptive, not descriptive.",
    "bug": "Answers 'what BROKE and how was it FIXED' -- incident record: symptom, root cause, fix/commit, resolution status.",
    "unclear": "Does not cleanly match one type, mixes two, or there isn't enough content to tell.",
}

STATUS_CRITERIA = {
    "fixed": "States a specific commit/change landed and resolved the incident; no open work remains on the core symptom.",
    "investigating": "Still open: root cause not fully pinned down, or a fix is proposed/queued/not yet landed, or the text says no fix yet.",
    "wontfix": "Framed as an accepted, permanent limitation nobody intends to change.",
    "unclear": "The text does not state a clear resolution state.",
}

# Judges the TEXT only. `sources_count` used to be fed in as a supporting
# signal and weighted by these levels -- but `sources: []` is a convention on
# this wiki, where the evidence is in-body measurement tables (a 1.421e-14 diff,
# an 85x solve_cw regression). Telling the rubric support was zero and then
# asking how well-supported the text reads is a poisoned input, and it scored
# 0.69/2 and 0.66/2 on two pages that carry hard numbers.
CONFIDENCE_CRITERIA = [
    "Speculative or thin: claims stated as guesses, open questions, or assertions with no measurement, commit, or citation behind them.",
    "Reasonably supported: at least one concrete measurement, commit hash or cited source, but a caveat or untested edge remains.",
    "Strongly supported: several independent numbers/sources agree, or the claim is checked against a ground truth -- an oracle, a passing gate, a primary source quoted directly.",
]

RELATION_CRITERIA = {
    "supports": "The related page states or implies the same claim.",
    "contradicts": "The related page states or implies something incompatible with the claim.",
    "unrelated": "The related page says nothing that bears on this specific claim.",
}

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#]+)\)")
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Minimal YAML-subset parser for this wiki's frontmatter: scalars and
    single-line `[a, b]` lists only -- exactly what SCHEMA.md declares."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.strip().startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            meta[key] = [v.strip() for v in inner.split(",") if v.strip()] if inner else []
        else:
            meta[key] = value.strip('"').strip("'")
    return meta, text[m.end():]


def excerpt(text: str, max_chars: int = MAX_EXCERPT) -> str:
    """Head, then EVERY section heading with its opening lines, then the tail.

    A flat head-truncate cut `two-curl-compositions` at 1191 of 3935 chars,
    stopping BEFORE the "## Moved, 2026-09-20" section that carries the commit,
    the measured deviation and "full suite green". The model was then asked what
    resolution the page describes, having been shown no resolution: it answered
    `unclear`, correctly, about text that says a fix landed. Two of the three
    live mismatches were that, not the model.

    Head+tail alone does not fix it either -- the tail of that page is its
    "## See also" links, so the resolution in the MIDDLE is still invisible.
    What a reader uses to answer "what does this page conclude" is the section
    structure, so that is what gets kept: every `##` heading survives, each with
    enough text under it to say what the section concluded.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text

    lines = text.splitlines()
    head_budget = max_chars // 3
    out, used = [], 0
    for line in lines:                                   # head, verbatim
        if used + len(line) > head_budget:
            break
        out.append(line)
        used += len(line) + 1

    seen_head = len(out)
    per_section = max(160, (max_chars - used) // max(1, text.count("\n## ") + 1))
    section, taking = None, 0
    for line in lines[seen_head:]:
        if line.startswith("## "):
            section, taking = line, 0
            out.append("\n" + line)
            used += len(line) + 2
        elif section is not None and taking < per_section and line.strip():
            out.append(line)
            used += len(line) + 1
            taking += len(line)
    return "\n".join(out)[:max_chars]


def related_sections(related: list[dict], max_sections: int = 8) -> list[dict]:
    """Flatten related pages into SECTIONS, one judgment each.

    Replaces lexical retrieval, which structurally cannot find a refutation.
    Measured on the known-bad pair: the section that refutes the claim shares
    exactly ONE token with it ("meep"). The claim says "meep's grouping is
    per-quantity"; the refutation says "meep's curl is a flat four-term sum
    scaled once by the Courant number". Semantically opposite, lexically
    disjoint -- a contradiction asserts the opposite IN DIFFERENT WORDS, so
    overlap scoring ranks the refutation below three unrelated sections and no
    amount of tuning the scorer fixes that.

    So code stops choosing. It splits, and the model judges each piece: the
    questions are independent, so they go in ONE request, and code takes the
    strongest contradiction across sections. Whole-page excerpts were measured
    at `unrelated 0.75` on a pair that genuinely contradicts, purely because the
    refuting section sat outvoted beside three irrelevant ones.
    """
    out = []
    for rel in related:
        cur: list[str] = []
        chunks: list[str] = []
        for line in rel["body"].strip().splitlines():
            if line.startswith("## ") and cur:
                chunks.append("\n".join(cur))
                cur = [line]
            else:
                cur.append(line)
        if cur:
            chunks.append("\n".join(cur))
        for chunk in chunks:
            text = chunk.strip()
            if len(text) < 80:                    # a heading with no substance
                continue
            head = text.splitlines()[0].lstrip("# ").strip()
            out.append({
                "page": rel["path"],
                "section": head[:80] or rel["title"],
                "text": excerpt(text, 700),
            })
    return out[:max_sections]


def lead_claim(text: str, max_chars: int = MAX_CLAIM) -> str:
    """The page's ATOMIC claim: its first substantive paragraph.

    `relates_*` must compare ONE claim, not a whole excerpt. Measured on the
    pre-correction `finite-difference-deviation` against a page that refutes it:
    the whole 1211-char excerpt scored `supports 0.95`, the single false clause
    scored `contradicts 0.88`. Same disagreement, opposite verdicts -- topical
    agreement on the surrounding claims drowns the one clause that is wrong.
    That is exactly how the meep-attribution error survived: it sat inside text
    that was otherwise correct.
    """
    paras, buf = [], []
    for line in text.strip().splitlines():
        st = line.strip()
        if st.startswith(("#", "|", "```", ">", "-", "*")) or not st:
            if buf:
                paras.append(" ".join(buf))
                buf = []
            continue
        buf.append(st)
    if buf:
        paras.append(" ".join(buf))
    for para in paras:
        if len(para) > 80:                      # skip one-line stubs
            return para[:max_chars]
    return excerpt(text, max_chars)


CORRECTION_MARKERS = ("NARROWED", "CORRECTED", "UPDATE:", "UPDATED:", "REVISED")


def original_premise(description: str) -> str:
    """The premise as first written, with any self-correction stripped off.

    `stale_premise` asks whether the body shows the opening premise has since
    become false. Feeding it a description that ALREADY carries its correction
    -- "...NARROWED 2026-09-20: the engine CAN accept a tensor eps..." -- asks
    whether an already-corrected premise has since become false, which has no
    clean answer. The live run returned 0.56, dead centre: the right answer to a
    malformed question, not a misread of the page.

    So: cut at the first correction marker and test what remains. If nothing
    remains, the description IS the correction and the question is not asked.
    """
    text = (description or "").strip()
    for marker in CORRECTION_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break
    text = text.rstrip(" .;:-\u2014")
    return text if len(text) > 40 else ""


def relates_question(idx: int) -> dict:
    """The `relates_i` question payload -- shared by `assemble_request` (per-page
    run) and `--pair` (diagnostic run) so neither reimplements it."""
    return {
        "type": "choice",
        "instructions": (
            f"`claim_under_test` is ONE specific claim. Compare ONLY that claim to the "
            f"single section at `related_sections[{idx}]`. Does that section support the "
            "claim, contradict it, or say nothing bearing on it? Judge the claim itself, "
            "not whether the two texts are broadly about the same subject -- a section "
            "may agree with everything around the claim and still contradict it, and it "
            "may contradict it in entirely different words rather than by denying it."
        ),
        "criteria": RELATION_CRITERIA,
    }


def resolve_links(page_path: Path, body: str) -> list[Path]:
    out: list[Path] = []
    for raw in LINK_RE.findall(body):
        if raw.startswith("http"):
            continue
        candidate = (page_path.parent / raw).resolve()
        if candidate.suffix == ".md" and candidate.is_file() and candidate not in out:
            out.append(candidate)
        if len(out) >= MAX_RELATED:
            break
    return out


def load_page(path: Path) -> dict:
    meta, body = parse_frontmatter(path.read_text())
    return {"path": path, "meta": meta, "body": body}


def assemble_request(page: dict) -> dict:
    """Build the (state, questions) payload for one page. Frontmatter values
    under test (type, status, confidence) are withheld from state."""
    meta, body, path = page["meta"], page["body"], page["path"]

    related = []
    for rel_path in resolve_links(path, body):
        rmeta, rbody = parse_frontmatter(rel_path.read_text())
        related.append({
            "path": str(rel_path.relative_to(WIKI_DIR)),
            "title": rmeta.get("title", rel_path.stem),
            "body": rbody,
        })

    sections = related_sections(related)
    # Two states, deliberately. relation_state carries ONLY the atomic claim and
    # the candidate sections -- handing `relates_*` the whole `page_excerpt` as
    # well dilutes a single-clause contradiction, measured at `unrelated 0.77`
    # against `contradicts 0.96` once both that and the whole-page related
    # excerpt were removed. The two compound; neither alone was enough.
    relation_state = {"claim_under_test": lead_claim(body),
                      "related_sections": sections}
    state = {"page_excerpt": excerpt(body)}
    premise = original_premise(meta.get("description", ""))
    if premise:
        state["declared_premise"] = premise

    questions: dict = {}
    ground_truth: dict = {}  # what code will compare the answers against; not sent to the model

    if premise:
        questions["stale_premise"] = {
            "type": "noul",
            "instructions": (
                "`declared_premise` is the claim this page opened with. Does `page_excerpt` "
                "-- this page's own body -- show that premise has since become false, been "
                "narrowed, or been corrected (e.g. a later paragraph says the capability now "
                "exists, the gap has closed, or an earlier claim on this page was wrong)?"
            ),
            "criteria": {
                "true": "The body itself documents that the opening premise no longer holds.",
                "false": "The body is consistent with the opening premise throughout.",
            },
        }

    relation_questions = {f"relates_{i}": relates_question(i)
                          for i in range(len(sections))}

    # Only ask what the frontmatter can score. `type_fit` used to fire on task
    # pages, which carry no `type:` field -- an answer, a probability, and
    # nothing to compare it against.
    if meta.get("type"):
        questions["type_fit"] = {
        "type": "choice",
        "instructions": (
            "Judging only the CONTENT of `page_excerpt` (ignore any type already implied by "
            "its folder or filename), which page type from this wiki's schema best fits it?"
            ),
        "criteria": TYPE_CRITERIA,
        }
        ground_truth["type"] = meta.get("type")

    if meta.get("type") == "bug":
        questions["status_drift"] = {
            "type": "choice",
            "instructions": (
                "Based only on `page_excerpt`, what resolution state does this bug incident "
                "describe -- fixed, still investigating, wontfix, or unclear?"
            ),
            "criteria": STATUS_CRITERIA,
        }
        ground_truth["status"] = meta.get("status")

    if "confidence" in meta:
        questions["confidence_fit"] = {
            "type": "score",
            "instructions": (
                "How well-supported do the claims in `page_excerpt` read? Judge the TEXT -- "
                "the measurements, commits and citations it actually contains -- not the "
                "declared confidence label and not the number of entries in any sources field."
            ),
            "criteria": CONFIDENCE_CRITERIA,
        }
        ground_truth["confidence"] = meta.get("confidence")

    return {"state": state, "questions": questions,
            "relation_state": relation_state, "relation_questions": relation_questions,
            "ground_truth": ground_truth}


def iter_pages(paths: list[str] | None) -> list[Path]:
    if paths:
        return [WIKI_DIR / p for p in paths]
    found: list[Path] = []
    for sub in DEFAULT_SCOPE:
        found.extend(sorted((WIKI_DIR / sub).glob("*.md")))
    return found


def load_dotenv(path: Path) -> None:
    """Read `KEY=value` lines from a .env into os.environ, without overriding.

    stdlib rather than python-dotenv: three lines against a dependency. An
    already-exported variable WINS, so a shell that has the key does not get it
    silently replaced by a stale file.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def check_credentials() -> str:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit(
            "TYPESAFE_API_KEY is not set. Export it, or put it in .env at the repo "
            "root (see https://console.typesafe.ai/), before running without --dry-run."
        )
    try:
        import typesafe_sdk  # noqa: F401
    except ImportError:
        sys.exit(
            "typesafe_sdk is not installed. Run `pip install typesafe-sdk` "
            "before running without --dry-run."
        )
    return key


def call_typesafe(payload: dict) -> dict:
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    kind_cls = {"noul": Noul, "choice": Choice, "score": Score}
    questions = {
        qid: kind_cls[q["type"]](instructions=q["instructions"], criteria=q["criteria"])
        for qid, q in payload["questions"].items()
    }
    with TypeSafeClient() as client:
        return client.system_one(state=payload["state"], questions=questions)


def run(pages: list[Path], dry_run: bool) -> None:
    if not dry_run:
        check_credentials()

    for path in pages:
        if not path.is_file():
            print(f"-- skip (not found): {path}")
            continue
        page = load_page(path)
        payload = assemble_request(page)
        rel = path.relative_to(WIKI_DIR)

        print(f"\n=== {rel} ===")
        print(f"declared frontmatter (withheld from model): {payload['ground_truth']}")
        print(f"questions asked: {list(payload['questions'].keys())}")

        if dry_run:
            print(json.dumps(
                {"state": payload["state"], "questions": payload["questions"]},
                indent=2, default=str,
            ))
            continue

        answers = call_typesafe(payload)
        print(f"answers: {answers}")
        if payload["relation_questions"]:
            rel_answers = call_typesafe(
                {"state": payload["relation_state"],
                 "questions": payload["relation_questions"]})
            print(f"relations: {rel_answers}")


def run_pair(claim_path: Path, related_path: Path, dry_run: bool) -> None:
    """Diagnostic entry point, not a second code path: runs the SAME
    `lead_claim`/`excerpt`/`relates_question`/`call_typesafe` production
    functions `assemble_request` uses, on two files given directly (need not
    be under WIKI_DIR -- unlike the per-page path, whose `relative_to` would
    raise on them).

    ARM A = claim_under_test is the whole excerpt (pre-fix behaviour).
    ARM B = claim_under_test is lead_claim (production behaviour today).
    Same related excerpt for both arms.
    """
    if not dry_run:
        check_credentials()

    claim_page = load_page(claim_path)
    related_page = load_page(related_path)
    try:
        rel_label = str(related_path.relative_to(WIKI_DIR))
    except ValueError:
        rel_label = str(related_path)
    related = {
        "path": rel_label,
        "title": related_page["meta"].get("title", related_path.stem),
        "excerpt": excerpt(related_page["body"], 600),
    }

    body = claim_page["body"]
    arms = {
        "A (old behaviour: claim = excerpt(body))": excerpt(body),
        "B (new behaviour: claim = lead_claim(body))": lead_claim(body),
    }
    for label, claim in arms.items():
        print(f"\n=== ARM {label} ===")
        print(f"claim_under_test ({len(claim)} chars): {claim!r}")
        payload = {
            "state": {"claim_under_test": claim, "related_pages": [related]},
            "questions": {"relates_0": relates_question(0)},
        }
        if dry_run:
            print(json.dumps(payload, indent=2, default=str))
            continue
        answers = call_typesafe(payload)
        print(f"answers: {answers}")


def demo() -> None:
    """Self-check: build state/questions for one real page of each shape and
    assert the state actually contains what each question references."""
    bug = load_page(WIKI_DIR / "bugs" / "two-curl-compositions-in-one-package.md")
    task = load_page(WIKI_DIR / "tasks" / "port_full-tensor-eps-smoothing.md")

    assert bug["meta"]["type"] == "bug"
    assert bug["meta"]["status"] == "fixed"

    bug_payload = assemble_request(bug)
    assert "status_drift" in bug_payload["questions"], "bug pages must get a status_drift question"
    assert bug_payload["ground_truth"]["status"] == "fixed"
    assert "page_excerpt" in bug_payload["state"]
    assert len(bug_payload["state"]["page_excerpt"]) > 0

    task_payload = assemble_request(task)
    assert "description" in task["meta"]
    assert "stale_premise" in task_payload["questions"], "a page with description: must get stale_premise"
    # the premise is the description with any self-correction stripped (fix 3)
    assert task_payload["state"]["declared_premise"] == original_premise(
        task["meta"]["description"])
    assert "NARROWED" not in task_payload["state"]["declared_premise"]
    assert "status_drift" not in task_payload["questions"], "tasks are not bugs"

    # relations live in their own state and their own request: one question per
    # SECTION of each related page, and no page_excerpt to dilute the claim.
    for payload in (bug_payload, task_payload):
        rs = payload["relation_state"]
        assert "page_excerpt" not in rs, "relation state must not carry the whole page"
        assert rs["claim_under_test"], "relations need an atomic claim"
        assert len(payload["relation_questions"]) == len(rs["related_sections"])
        for i in range(len(rs["related_sections"])):
            assert f"relates_{i}" in payload["relation_questions"]
        for sec in rs["related_sections"]:
            assert sec["text"] and sec["section"], sec

    print("demo(): state-building self-check passed.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", nargs="+", help="wiki-relative paths to check (default: all of bugs/, tasks/, concepts/)")
    ap.add_argument("--dry-run", action="store_true", help="print request payloads, call nothing")
    ap.add_argument("--pair", nargs=2, metavar=("CLAIM_FILE", "RELATED_FILE"),
                     help="diagnostic: run ARM A (excerpt) vs ARM B (lead_claim) of "
                          "CLAIM_FILE's relates_0 judgment against RELATED_FILE. Files "
                          "may be outside wiki/.")
    args = ap.parse_args()

    if args.pair:
        run_pair(Path(args.pair[0]), Path(args.pair[1]), args.dry_run)
        return

    pages = iter_pages(args.pages)
    if not pages:
        sys.exit("no pages found")
    run(pages, args.dry_run)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo()
    else:
        main()
