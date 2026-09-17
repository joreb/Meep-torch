"""Does the 2D parity result survive in 3D?

The 2D TM answer was 0.95-0.99x hand at equal work, once the spec is resolved
at build time. 3D is the case that matters for runtime and the one the 2D
measurement could not speak for. Same methodology, and the same four ways it
went wrong before:

  1. confounded physics  -> generic_3d.check() asserts bitwise vs step_3d
  2. core contention     -> refuses to run when 1-minute load exceeds cores/4
  3. phase asymmetry     -> both sides run the full step_3d CPML path
  4. single-shot noise   -> medians of 5, grids large enough to be kernel-bound

`hoist_eps` is reported separately rather than folded in: step_3d recomputes
three eps averages every step though eps never changes. Hoisting is a win the
spec knows about and hand code forgets -- the same shape as the 2D allocator
result -- so it must not be smuggled into the genericity number.
"""
import os
import time

import torch

from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state, step_3d

from .generic_3d import check, make_step_3d


def _guard():
    load1 = os.getloadavg()[0]
    cores = os.cpu_count() or 1
    if load1 > cores / 4:
        raise SystemExit(f"load {load1:.1f} too high for {cores} cores -- "
                         "timings would be noise (this happened before)")


def _time(fn, state, steps, device, repeats=5):
    for _ in range(5):                                   # warm-up / compile
        state = fn(state)
    best = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            state = fn(state)
        if device == "cuda":
            torch.cuda.synchronize()
        best.append(steps / (time.perf_counter() - t0))
    best.sort()
    return best[len(best) // 2]


def run(n, device, npml=8, steps=20):
    c = build_3d_coeffs(n, n, n, dx=0.05, npml=npml, device=device)
    st = init_3d_state(n, n, n, npml=npml, device=device)
    scal = sum(v.numel() for v in st if v is not None) / 1e6

    variants = (
        ("hand", lambda s: step_3d(s, c)),
        ("generic", make_step_3d(c, hoist_eps=False)),
        ("generic+hoisted_eps", make_step_3d(c, hoist_eps=True)),
    )
    out = {}
    for label, fn in variants:
        out[label + "/eager"] = _time(fn, st, steps, device)
        try:
            out[label + "/compiled"] = _time(
                torch.compile(fn, fullgraph=True), st, steps, device)
        except Exception as e:
            out[label + "/compiled"] = f"FAILED {type(e).__name__}"
    return scal, out


if __name__ == "__main__":
    _guard()
    bad = check()
    if bad:
        raise SystemExit(f"NOT bitwise vs step_3d ({bad}) -- refusing to time")
    print("bitwise vs step_3d: OK\n")

    cases = [(48, "cpu"), (64, "cpu")]
    if torch.cuda.is_available():
        cases += [(96, "cuda"), (144, "cuda")]

    for n, device in cases:
        scal, out = run(n, device)
        print(f"{device} {n}^3  ({scal:.1f} Mscalars of state)")
        for mode in ("eager", "compiled"):
            h = out.get("hand/" + mode)
            for label in ("hand", "generic", "generic+hoisted_eps"):
                v = out.get(label + "/" + mode)
                if isinstance(v, str):
                    print(f"  {mode:<9s} {label:<20s} {v}")
                else:
                    rel = f"{v / h:.2f}x hand" if isinstance(h, float) else ""
                    print(f"  {mode:<9s} {label:<20s} {v:8.1f} steps/s   {rel}")
        print()
