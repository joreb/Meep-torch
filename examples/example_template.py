# How to write a meep-torch example. Run it to see the shape; run it with
# --check DIR to hold a directory of examples to the rules below.
"""Template and rule-checker for meep-torch examples.

The rules are not invented here. They are read off the three examples that
between them cover the range -- a one-shot field plot, a time-domain
snapshot, and a multi-section study:

    examples/01_basics/straight-waveguide.py   simplest: run, plot, export
    examples/02_sources/chirped_pulse.py       snapshot mid-run, 1D slice
    examples/10_periodic_mpb/mpb_tutorial.py   many sections, plot helpers

Usage
-----
    uv run python examples/example_template.py            # run the template
    uv run python examples/example_template.py --check examples/10_periodic_mpb

The checker is deliberately shallow: it catches the mistakes that were
actually made in this repo, not everything that could go wrong. Each rule
below names the example that earned it.


THE RULES
=========

1.  HEADLESS. `matplotlib.use("Agg")` before `import matplotlib.pyplot`,
    and never `plt.show()`. Examples run in CI and over ssh; a blocking
    window is a hang, not a plot.

2.  EVERYTHING GOES TO output/. `OUT = pathlib.Path(__file__).parent /
    "output"` and `OUT.mkdir(exist_ok=True)`, then every `savefig`,
    `savetxt` and engine writer under it. `examples/**/output/` is
    gitignored, so anything written elsewhere shows up as a dirty tree --
    a run from the repo root once dropped 144 h5 files beside the source.
    meep's own writers need telling:

        sim.use_output_directory(str(OUT))        # mp.Simulation
        ms.filename_prefix = str(OUT / "name")    # mpb.ModeSolver

3.  SAY WHICH DEVICE PRODUCED THE NUMBER. `DEVICE` is read from
    `MEEP_TORCH_DEVICE`, never hard-coded and never detected behind the
    reader's back, and it goes in the closing line (rule 7) so a number
    can be told from a CPU number.

    Note what DEVICE is NOT: none of these examples passes it to
    `mp.Simulation`. The engine reads `MEEP_TORCH_DEVICE` itself and
    defaults to CPU -- a library should not pick a device for a caller --
    so the block is there to REPORT the choice, not to make it. That is
    why an mpb-only example does not need one: `mpb.ModeSolver` takes no
    device, and its FDFD path is scipy on the CPU unless
    `differentiable=True` puts it in torch.

4.  PLOT AND EXPORT THE SAME DATA. Every figure gets a CSV beside it with
    the numbers behind it: a PNG is not reviewable and not diffable. Give
    the CSV a header row (`header=..., comments=""`). Name both after the
    script, or after the section in a multi-section script.

5.  AXES COME FROM THE RETURNED SHAPE AND THE CELL. Never from `sim.grid`
    or any other private attribute -- those are shim-isms that would not
    run against upstream meep, which is the whole point of a twin. And
    `get_array` returns the FULL grid INCLUDING the PML: chirped_pulse
    divided by `sx` instead of `sx + 2*dpml` and compressed x by 10%.

6.  SNAPSHOT WHILE THE PHYSICS IS IN THE CELL. A run that ends after the
    pulse has left leaves PML residual, and comparing final states across
    engines then compares leftovers. chirped_pulse grabs at t=40 with
    `mp.at_time`, not at the end. Say in a comment WHY the moment was
    chosen.

7.  END WITH ONE LINE THAT CARRIES A NUMBER.

        print(f"{name}.png  [{DEVICE}]  max|Ez| after t=200 = {peak:.6e}")

    It is what a reader diffs against the twin, and what tells you the run
    did something. No bare `print(np.shape(data))` debris.

8.  KEEP UPSTREAM'S NAMES AND STRUCTURE. Same variable names, same order,
    same numbers as the meep tutorial being ported. Deviations get a
    comment saying what changed and why -- "upstream shows two figures
    with plt.show(); the twins are headless, so both go in one png".
    A reader should be able to diff this against the upstream script.

9.  PAIR IT WITH AN ORACLE TWIN. `<name>.py` is meep-torch; `<name>_meep.py`
    is the same script against real pymeep, run under the oracle env:

        MPLBACKEND=Agg ~/miniforge3/envs/meep-oracle/bin/python <name>_meep.py

    The twin is what turns "it ran" into "it is right". Keep the two
    byte-comparable apart from the import and the output names.

    An mpb example may gate differently: the band tracks are checked
    against committed oracle JSONs by tests/examples/test_mpb_track.py
    rather than by a `_meep.py` twin. That counts, and the warning here is
    a prompt to say which of the two you did -- not an order to add a file.

10. IMPORT THE ENGINE THE FILENAME PROMISES. Checked both ways, because
    both directions have a failure mode:

      - `<name>.py` must import `meep_torch`. One that imports `meep` is
        an unported upstream script sitting in the tree for reference, not
        an example. The checker says so rather than holding it to
        conventions it was never written for, and the count is a useful
        backlog number -- examples/10_periodic_mpb is 10 upstream copies
        against 2 ports.

      - `<name>_meep.py` must import `meep` and must NOT import
        `meep_torch`. A twin that reaches for meep_torch compares the
        engine with itself: it will agree perfectly, prove nothing, and
        look like the strongest evidence in the directory. That is the
        worse of the two mistakes, so it is the one rated an error.

11. A RAISE IS AN ACCEPTABLE ENDING. If the port does not cover a section
    yet, let it raise where it raises and say so in the report. An example
    that silently skips the unported half is worse than one that stops.
"""
import argparse
import os
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).parent / "output"

# Examples default to the GPU: they are demonstrations, and the difference is
# not marginal (this cell is 3 s on CUDA against >550 s on CPU). The engine
# itself defaults to CPU -- a library should not pick a device for you -- so
# the script asks explicitly. MEEP_TORCH_DEVICE overrides; falls back to CPU
# on a machine without a card.
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or "cpu"


# --- the rule checker -----------------------------------------------------

def _code_only(src: str) -> str:
    """Source with comments and string literals blanked to spaces.

    Line and column structure is preserved exactly, so reported line
    numbers are the file's own. The first version wrote every token back
    into a grid, which put NEWLINE tokens' "\n" INSIDE a row and doubled
    the line count -- it reported a debug print at line 167 of an 86-line
    file. Blanking only the spans that need blanking cannot do that.

    Why blank at all: without it the checker reads its own advice as a
    violation. chirped_pulse.py carries the comment "x from the returned
    array and the cell, not sim.grid", and a raw regex flags it for using
    sim.grid. A checker that fires on a comment telling you not to do the
    thing trains you to ignore the checker.
    """
    import io
    import tokenize
    grid = [list(ln) for ln in src.splitlines()]
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src                       # unparseable: fall back to raw
    for tok in toks:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (r0, c0), (r1, c1) = tok.start, tok.end
        for r in range(r0, min(r1, len(grid)) + 1):
            row = grid[r - 1]
            lo = c0 if r == r0 else 0
            hi = c1 if r == r1 else len(row)
            for i in range(lo, min(hi, len(row))):
                row[i] = " "
    return "\n".join("".join(r) for r in grid)


def _engine(src: str):
    """Which engine the file imports: "meep_torch", "meep", or None."""
    code = _code_only(src)
    torch_ = re.search(r"^\s*(?:import|from)\s+meep_torch\b", code, re.M)
    plain = re.search(r"^\s*(?:import\s+meep\b|from\s+meep\s+import)", code, re.M)
    if torch_:
        return "meep_torch"
    return "meep" if plain else None


def _engine_rule(path: pathlib.Path, src: str):
    """Rule 10, both directions. Returns (level, msg) or None."""
    eng = _engine(src)
    if path.stem.endswith("_meep"):
        if eng == "meep_torch":
            return ("error", "rule 10: an oracle twin that imports "
                             "meep_torch compares the engine with itself")
        if eng is None:
            return ("warn", "rule 10: _meep twin imports no engine")
        return None
    if eng == "meep_torch":
        return None
    return ("skip", "upstream copy, not ported" if eng == "meep"
            else "no engine imported")


def _rules(path: pathlib.Path, src: str, twin: bool = False):
    """Yield (level, message) for each rule this file breaks.

    ``twin`` marks a `<name>_meep.py` oracle: it runs real pymeep, so the
    device rule does not apply -- its DEVICE is the literal "meep-cpu",
    which is the correct thing for it to report -- and it does not need a
    twin of its own. Everything else (headless, output/, CSV beside every
    figure, a closing line) applies equally, because the two scripts are
    meant to be diffable.
    """
    name = path.stem
    lines = src.splitlines()
    code = _code_only(src)
    has_plt = "matplotlib.pyplot" in src
    builds_sim = bool(re.search(r"\bmp\.Simulation\(|mpb\.ModeSolver\(", src))

    if has_plt:
        use = src.find('matplotlib.use(')
        pyp = src.find("import matplotlib.pyplot")
        if use == -1:
            yield "error", 'rule 1: no matplotlib.use("Agg")'
        elif use > pyp:
            yield "error", "rule 1: matplotlib.use() comes after pyplot import"
        if re.search(r"^\s*plt\.show\(", code, re.M):
            yield "error", "rule 1: plt.show() in a headless example"

    writes = re.findall(r"(savefig|savetxt)\(\s*([^,)]+)", src)
    if writes:
        if 'OUT = pathlib.Path(__file__).parent / "output"' not in src:
            yield "error", "rule 2: OUT is not the standard output/ path"
        if "OUT.mkdir(" not in src:
            yield "error", "rule 2: OUT.mkdir(exist_ok=True) missing"
        for fn, arg in writes:
            if "OUT" not in arg:
                yield "error", f"rule 2: {fn}({arg.strip()}) does not go to OUT"

    builds_fdtd = bool(re.search(r"\bmp\.Simulation\(", code))
    if not twin:
        if "DEVICE" in code and "MEEP_TORCH_DEVICE" not in src:
            yield "error", "rule 3: DEVICE is hard-coded, not from the env var"
        if builds_fdtd and "MEEP_TORCH_DEVICE" not in src:
            yield "error", "rule 3: torch engine run with no DEVICE block"
    elif "DEVICE" not in code:
        yield "warn", "rule 3: twin does not report a DEVICE tag"

    n_png = len(re.findall(r"savefig\(", src))
    n_csv = len(re.findall(r"savetxt\(", src))
    if n_png and not n_csv:
        yield "error", f"rule 4: {n_png} figure(s) and no CSV export"
    for m in re.finditer(r"savetxt\((?:[^()]|\([^()]*\))*\)", src):
        if "header=" not in m.group(0):
            yield "warn", "rule 4: savetxt without header="

    for pat, msg in ((r"\bsim\.grid\b", "sim.grid"),
                     (r"\b(?:sim|ms)\._[a-z]", "a private engine attribute")):
        if re.search(pat, code):
            yield "error", f"rule 5: reaches for {msg} -- not upstream API"

    for i, ln in enumerate(code.splitlines(), 1):
        if re.match(r"\s*print\(\s*np\.shape\(", ln):
            yield "warn", f"rule 7: bare debug print at line {i}"
    if n_png and not re.search(r'print\(f?"[^"]*\[\{DEVICE\}\]', src):
        yield "warn", ("rule 7: no closing report line"
                       + (" carrying [{DEVICE}]" if builds_fdtd else ""))

    if not twin and not name.endswith("_mp") and builds_sim:
        if not (path.parent / f"{name}_meep.py").exists():
            yield "warn", f"rule 9: no oracle twin {name}_meep.py"


def check(target: pathlib.Path) -> int:
    files = sorted(target.glob("*.py")) if target.is_dir() else [target]
    files = [f for f in files if f.name != pathlib.Path(__file__).name]
    if not files:
        print(f"no .py files under {target}")
        return 1
    n_err = n_warn = n_upstream = n_twin = 0
    for f in files:
        src = f.read_text()
        eng = _engine_rule(f, src)
        if eng and eng[0] == "skip":
            print(f"  --    {f.name}  ({eng[1]})")
            n_upstream += 1
            continue
        twin = f.stem.endswith("_meep")
        n_twin += twin
        found = ([eng] if eng else []) + list(_rules(f, src, twin=twin))
        if not found:
            print(f"  ok    {f.relative_to(target.parent if target.is_dir() else f.parent)}")
            continue
        print(f"  ----  {f.name}")
        for level, msg in found:
            print(f"        [{level}] {msg}")
            n_err += level == "error"
            n_warn += level == "warn"
    print(f"\n{len(files)} file(s): {len(files) - n_upstream - n_twin} port(s), "
          f"{n_twin} oracle twin(s), {n_upstream} upstream; "
          f"{n_err} error(s), {n_warn} warning(s)")
    return 1 if n_err else 0


# --- the template itself, run when no --check is given --------------------

def demo():
    """The smallest example that obeys every rule, so it can be copied."""
    OUT.mkdir(exist_ok=True)
    x = np.linspace(0, 4 * np.pi, 400)
    y = np.exp(-0.1 * x) * np.sin(x)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, y, lw=1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("example_template: a damped sine standing in for a field")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "example_template.png", dpi=150)
    plt.close(fig)

    np.savetxt(OUT / "example_template.csv", np.column_stack((x, y)),
               delimiter=",", header="x,y", comments="")

    peak = float(np.max(np.abs(y)))
    print(f"example_template.png  [{DEVICE}]  max|y| = {peak:.6e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", metavar="PATH",
                    help="check a file or a directory of examples")
    args = ap.parse_args()
    if args.check:
        sys.exit(check(pathlib.Path(args.check)))
    demo()
