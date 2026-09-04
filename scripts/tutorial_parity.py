# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Run a meep tutorial page under meep_torch and report every deviation.

meep's tutorials are prose with the script dribbled out in fragments -- a
section's ```py blocks concatenate into one runnable program, and the prose
between them is the explanation. So the unit of execution is a *section*,
not a block.

For each section this builds two programs from the same fragments, one
importing ``meep`` and one importing ``meep_torch.mp``, runs both, and
compares the numbers they print. That makes the report measured rather than
asserted: a section is only "matches" if real meep ran it too and the
numbers agree.

Run:
    uv run python scripts/tutorial_parity.py <tutorial.md> [-o out.md]
        [--timeout 300] [--oracle ~/miniforge3/envs/meep-oracle/bin/python]

Without a working oracle interpreter the meep side is skipped and sections
are reported as runs/raises only, with no deviation column -- stated in the
output rather than quietly implied.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

# A tutorial fragment is not always a program. These never are.
_SKIP_BLOCK = re.compile(
    r"^\s*(\$|\%|!|pip |conda |mpirun |python |meep )", re.M)

# numbers in printed output, for the deviation comparison
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# meep narrates its own progress on stdout and meep-torch does not, so a raw
# stdout diff compares banner text, not results: the straight-waveguide
# section prints nothing itself, yet meep emitted 32 numbers (chunk timings,
# cell dimensions, geometry dump, step counts). Strip meep's narration so
# what remains is what the tutorial script actually chose to print.
_MEEP_NOISE = re.compile(
    r"^(?:-{5,}"
    r"|Initializing structure"
    r"|time for \w+ ="
    r"|Working in .* dimensions"
    r"|Computational cell is"
    r"|Halving computational cell"
    r"|FloatProgress"
    r"|Meep progress:"
    r"|on time step"
    r"|run \d+ finished at t"
    r"|field decay\(t ="
    r"|Elapsed run time"
    r"|creating output file"
    r"|Padding computational cell"
    r"|\s+(?:block|cylinder|sphere|prism|ellipsoid|cone)\b"
    r"|\s+(?:size|axes|center|radius|height|dielectric constant"
    r"|epsilon diagonal|nonlinear susceptibilities)\b"
    r")", re.I)


def strip_noise(out: str) -> str:
    return "\n".join(l for l in out.splitlines() if not _MEEP_NOISE.match(l))

_HEADING = re.compile(r"^(#{2,4})\s+(.*?)\s*$")
# Cylindrical_Coordinates.md uses setext headings (text underlined with ---
# or ===) and carries exactly one ATX heading for the whole page, so an
# ATX-only parser collapses it into a single 766-line "section".
_SETEXT = re.compile(r"^(-{3,}|={3,})\s*$")
_FENCE = re.compile(r"^```(\w*)\s*$")

# matplotlib must not try to open a window, and meep's own scripts often
# call plt.show(); Agg makes that a no-op instead of a hang.
_PREAMBLE = (
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as _plt\n"
    "_plt.show = lambda *a, **k: None\n"
    "_plt.savefig = lambda *a, **k: None\n"
)


@dataclass
class Section:
    title: str
    level: int
    blocks: list[str] = field(default_factory=list)

    @property
    def source(self) -> str:
        return "\n".join(self.blocks)

    @property
    def runnable(self) -> bool:
        src = self.source
        return bool(src.strip()) and "mp.Simulation" in src


def parse_sections(path: str) -> list[Section]:
    """Split a tutorial into sections, accumulating each one's ```py blocks.

    Fragments accumulate *within* a section only. Meep's tutorial pages
    restart the script at each example heading, so carrying blocks across a
    heading would splice two unrelated simulations together.
    """
    sections: list[Section] = [Section("(preamble)", 1)]
    lang, buf = None, []
    prev = ""
    for line in open(path, encoding="utf-8"):
        fence = _FENCE.match(line)
        if fence:
            if lang is None:
                lang = fence.group(1) or "text"
                buf = []
            else:
                if lang in ("py", "python"):
                    code = "".join(buf)
                    if not _SKIP_BLOCK.match(code):
                        sections[-1].blocks.append(code.rstrip())
                lang = None
            continue
        if lang is not None:
            buf.append(line)
            continue
        head = _HEADING.match(line)
        if head:
            sections.append(Section(head.group(2), len(head.group(1))))
            prev = ""
            continue
        if (_SETEXT.match(line) and prev.strip()
                and not prev.startswith(("#", "```", "|"))):
            # the underlined line was the title, not body text
            sections.append(Section(prev.strip(), 2 if line[0] == "-" else 1))
            prev = ""
            continue
        prev = line
    return [s for s in sections if s.blocks]


def _program(src: str, torch: bool) -> str:
    if torch:
        src = re.sub(r"^import meep as mp$", "import meep_torch.mp as mp",
                     src, flags=re.M)
        src = re.sub(r"^from meep import mpb$", "from meep_torch import mpb",
                     src, flags=re.M)
        src = re.sub(r"^from meep\.materials import",
                     "from meep_torch.mp.materials import", src, flags=re.M)
    return _PREAMBLE + src + "\n"


def run_one(src: str, interpreter: list[str], timeout: int, workdir: str):
    """(status, stdout) for one program. status: ok | raises:<...> | timeout."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=workdir,
                                     delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        p = subprocess.run(interpreter + [path], capture_output=True,
                           text=True, timeout=timeout, cwd=workdir)
    except subprocess.TimeoutExpired:
        return f"timeout({timeout}s)", ""
    if p.returncode == 0:
        return "ok", p.stdout
    last = ""
    for m in re.finditer(r"^(\w*(?:Error|Exception)): (.*)$", p.stderr, re.M):
        last = f"{m.group(1)}: {m.group(2)}"
    return f"raises: {last or 'exit ' + str(p.returncode)}", p.stdout


def compare(a: str, b: str, rtol: float = 0.02):
    """Worst relative deviation between the numbers two runs printed.

    Compares position by position and only when both printed the same count
    -- a differing count means the runs did different things, which is a
    deviation to report, not a number to average away.
    """
    xa = [float(x) for x in _NUM.findall(strip_noise(a))]
    xb = [float(x) for x in _NUM.findall(strip_noise(b))]
    if not xa and not xb:
        return None, "no numbers printed"
    if len(xa) != len(xb):
        return None, f"printed {len(xa)} numbers vs {len(xb)}"
    worst = 0.0
    for u, v in zip(xa, xb):
        d = abs(u - v) / max(abs(v), 1e-12) if abs(v) > 1e-12 else abs(u - v)
        worst = max(worst, d)
    return worst, ("matches" if worst <= rtol else f"deviates {worst:.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tutorial")
    ap.add_argument("-o", "--out")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--oracle",
                    default=os.path.expanduser(
                        "~/miniforge3/envs/meep-oracle/bin/python"))
    ap.add_argument("--only", help="substring filter on section titles")
    args = ap.parse_args()

    have_oracle = os.path.exists(args.oracle) and subprocess.run(
        [args.oracle, "-c", "import meep"], capture_output=True).returncode == 0

    sections = [s for s in parse_sections(args.tutorial) if s.runnable]
    if args.only:
        sections = [s for s in sections if args.only.lower() in s.title.lower()]

    rows = []
    workdir = tempfile.mkdtemp(prefix="tutparity-")
    for s in sections:
        t_status, t_out = run_one(_program(s.source, True),
                                  [sys.executable], args.timeout, workdir)
        if have_oracle:
            m_status, m_out = run_one(_program(s.source, False),
                                      [args.oracle], args.timeout, workdir)
        else:
            m_status, m_out = "skipped", ""
        if t_status == "ok" and m_status == "ok":
            _, verdict = compare(t_out, m_out)
        elif t_status == "ok" and m_status != "ok":
            verdict = f"meep-torch ok; meep {m_status}"
        else:
            verdict = "-"
        rows.append((s.title, t_status, m_status, verdict))
        print(f"{s.title[:44]:46s} torch={t_status[:34]:36s} verdict={verdict}",
              flush=True)

    name = os.path.basename(args.tutorial)
    lines = [f"# {name} under meep-torch", ""]
    if not have_oracle:
        lines += [f"> No meep oracle at `{args.oracle}`, so the meep column is "
                  "**skipped** and nothing here is a parity claim -- these are "
                  "runs/raises only.", ""]
    lines += ["| section | meep-torch | meep | verdict |", "|---|---|---|---|"]
    lines += [f"| {t} | {a} | {b} | {v} |" for t, a, b, v in rows]
    ok = sum(1 for _, a, _, _ in rows if a == "ok")
    lines += ["", f"{ok}/{len(rows)} runnable sections execute under "
                  f"meep-torch (timeout {args.timeout}s)."]
    text = "\n".join(lines) + "\n"
    if args.out:
        open(args.out, "w").write(text)
        print(f"\nwrote {args.out}")
    else:
        print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
