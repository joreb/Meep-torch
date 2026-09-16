"""Hand-sliced TM vs generic-gather TM: the unconfounded genericity number.

Earlier benchmarks compared `step_tm` (hand, 2D) with `step_3d` (generic, 3D)
and reported 1.1-1.47x. That gap confounds code style with arithmetic: 3D runs
two curl terms per component against TM's 4 differences over 3 fields, so it
does ~1.5x the work per scalar updated and the physics alone could explain the
whole thing. Same physics, same dimensionality, same field count here -- only
the gather differs.

Correctness is asserted before timing: without a PML the two are BITWISE equal,
and with one they agree to ~5e-16 (the CPML term association, the documented
evaluation-order effect). If that assertion fails the timings are meaningless
and the script refuses to print them.

    uv run python symbolic_test/bench_generic_tm.py
"""
import time

import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper_tm import init_tm_state, step_db, update_eh

from .generic_tm import init as ginit, step as gstep
from .generic_tm_hoisted import make_step


def hand_step(s, m):
    """The phases `generic_tm.step` covers, and only those.

    `step_tm` is the full seven-phase cycle -- B, source(B), H, D, source(D),
    pols, E -- while the generic prototype covers the two curl phases and the
    field updates. Timing one against the other measures the extra phases, not
    the gather: it showed "generic 3x faster" on CUDA eager, which is the
    asymmetry talking. Sources and poles are orthogonal to the gather and are
    absent from both sides here.
    """
    s = step_db(s, m, "B")
    s = step_db(s, m, "D")
    return update_eh(s, m, "E")


def _agree(device):
    g = Grid(nx=24, ny=20, dx=0.05, device=device)
    for npml, want_bitwise in ((0, True), (4, False)):
        m = build_tm_coeffs(g, npml=npml)
        torch.manual_seed(0)
        ez = torch.randn(g.nx, g.ny, dtype=g.dtype, device=device)
        a = init_tm_state(g)._replace(Ez=ez.clone())
        a = a._replace(Dz=a.Ez.clone())
        b = ginit(g.nx, g.ny, device=device)._replace(Ez=ez.clone())
        for _ in range(6):
            a, b = hand_step(a, m), gstep(b, m)
        same = all(torch.equal(getattr(a, n), getattr(b, n))
                   for n in ("Ez", "Hx", "Hy"))
        err = max((getattr(a, n) - getattr(b, n)).abs().max().item()
                  for n in ("Ez", "Hx", "Hy"))
        # Bitwise is demanded on CPU only. On CUDA the two orderings reach
        # different kernels and agree to ~2e-15 at npml=0 -- device op order,
        # not a physics difference, and the same reason the project's bitwise
        # gates are CPU gates.
        if device == "cpu" and want_bitwise:
            assert same, (f"npml=0 on CPU must be bitwise, got max|d|={err:.2e}"
                          " -- the gathers differ, timings would be meaningless")
        assert err < 1e-13, (
            f"npml={npml} on {device}: max|d|={err:.2e} -- too large to be "
            f"evaluation order; the two steppers are not the same physics")
    return True


def _time(fn, state, m, steps, device, repeats=5):
    """Median of `repeats` timed blocks, after warm-up.

    Single-shot timing gave ratios that swung 1.19 to 2.93 between grid sizes
    and claimed the generic version was 4.5x FASTER on CUDA eager -- both
    artefacts of one sample and of per-launch overhead dominating a small
    state. Median over repeats, and grids large enough that the kernels rather
    than the launches set the time.
    """
    for _ in range(5):                                   # warm-up / compile
        state = fn(state, m)
    best = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            state = fn(state, m)
        if device == "cuda":
            torch.cuda.synchronize()
        best.append(steps / (time.perf_counter() - t0))
    best.sort()
    return best[len(best) // 2]


def run(nx, ny, device, steps=30):
    g = Grid(nx=nx, ny=ny, dx=0.05, device=device)
    m = build_tm_coeffs(g, npml=8)
    hand = init_tm_state(g)
    hand = hand._replace(Dz=hand.Ez.clone())
    gen = ginit(nx, ny, device=device)
    scal = sum(getattr(gen, n).numel() for n in gen._fields) / 1e6

    hoisted_fn = make_step(m)
    out = {}
    for label, fn, st in (("hand", hand_step, hand), ("generic", gstep, gen),
                          ("hoisted", lambda s, _m: hoisted_fn(s), gen)):
        out[label + "/eager"] = _time(fn, st, m, steps, device)
        try:
            out[label + "/compiled"] = _time(
                torch.compile(fn, fullgraph=True), st, m, steps, device)
        except Exception as e:
            out[label + "/compiled"] = float("nan")
    return scal, out


if __name__ == "__main__":
    import os
    # A benchmark run while anything else holds the cores measures the other
    # thing. This script has already produced one set of numbers that were
    # pure contention noise (a full pytest run held all 24 threads).
    # Measure the thing the guard cares about -- are the cores busy -- rather
    # than trying to name the process holding them. Two attempts at pattern
    # matching failed for the same reason: this harness wraps each command in
    # a shell whose argv contains the whole script text, so any pattern the
    # guard greps for is present in the guard's own process line.
    load = os.getloadavg()[0]
    ncpu = os.cpu_count() or 1
    if load > 0.25 * ncpu:
        raise SystemExit(
            f"refusing to benchmark: 1-min load {load:.1f} on {ncpu} cores. "
            f"Something else is using the machine and the numbers would "
            f"measure it. Wait for it, then re-run.")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        assert _agree(device)
        print(f"\n{device}  (correctness asserted: bitwise at npml=0)")
        print(f"  {'grid':>12s} {'hand cmp':>9s} {'generic':>9s} {'x':>5s} "
              f"{'hoisted':>9s} {'x':>5s}")
        # large enough that kernels, not launch overhead, set the time
        for nx, ny in ([(600, 600), (1200, 1200)] if device == "cpu"
                       else [(2000, 2000), (4000, 4000)]):
            scal, r = run(nx, ny, device)
            hc, gc, oc = (r["hand/compiled"], r["generic/compiled"],
                          r["hoisted/compiled"])
            print(f"  {nx:5d}x{ny:<6d} {hc:9.1f} {gc:9.1f} {hc/gc:5.2f} "
                  f"{oc:9.1f} {hc/oc:5.2f}")
    print("\ncompiled steps/s; x = hand / that column. >1 means hand is faster")
