"""step_3d (full-volume psi) vs stepper_3d_lean (psi in the PML shells only).

This is the 3D ALLOCATOR number, the thing the 2D 13% turned out to be. It
bundles two effects and says so: psi-in-shells (allocator) and in-place buffer
reuse (implementation). Both come from knowing the configuration at build time,
but only the first is the allocator.

RESULT: after fixing the shell predicate (a cell is inert iff a == 0 AND k == 1,
not b != 0), stepper_3d_lean's psi memory falls 5-9x -- 81.9 -> 13.1 MB at 96^3,
279.7 -> 29.7 MB at 144^3 -- and total state 2.4x. Before the fix every row read
"1x less" and lean ran at 0.41-0.52x.
"""
import os, time, torch
from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state, step_3d
from meep_torch.core.stepper_3d_lean import build_lean, run_3d_lean

assert os.getloadavg()[0] < (os.cpu_count() or 1) / 4, "load too high"


def psi_bytes(obj, lean):
    if lean:
        return sum(p.numel() * p.element_size()
                   for pair in obj.psi.values() for p in pair)
    return sum(getattr(obj, f).numel() * getattr(obj, f).element_size()
               for f in obj._fields if f.startswith("psi_") and getattr(obj, f) is not None)


def time_full(n, npml, device, steps, repeats=5):
    c = build_3d_coeffs(n, n, n, dx=0.05, npml=npml, device=device)
    st = init_3d_state(n, n, n, npml=npml, device=device)
    fn = torch.compile(lambda s: step_3d(s, c), fullgraph=True)
    for _ in range(5):
        st = fn(st)
    best = []
    for _ in range(repeats):
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            st = fn(st)
        if device == "cuda": torch.cuda.synchronize()
        best.append(steps / (time.perf_counter() - t0))
    best.sort()
    return best[len(best) // 2], psi_bytes(st, False)


def time_lean(n, npml, device, steps, repeats=5):
    eps = torch.ones(n, n, n, dtype=torch.float64, device=device)
    c = build_lean(n, n, n, 0.05, eps, npml, dtype=torch.float64, device=device)
    pb = psi_bytes(c, True)
    run_3d_lean(c, 5, compile=True)                     # warm-up / compile
    best = []
    for _ in range(repeats):
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_3d_lean(c, steps, compile=True)
        if device == "cuda": torch.cuda.synchronize()
        best.append(steps / (time.perf_counter() - t0))
    best.sort()
    return best[len(best) // 2], pb


for n, npml, device, steps in ((96, 8, "cuda", 20), (144, 8, "cuda", 20),
                               (144, 16, "cuda", 20)):
    f, fb = time_full(n, npml, device, steps)
    l, lb = time_lean(n, npml, device, steps)
    print(f"{device} {n}^3 npml={npml}")
    print(f"  step_3d (full psi)  {f:8.1f} steps/s   psi {fb/1e6:8.1f} MB")
    print(f"  lean (shell psi)    {l:8.1f} steps/s   psi {lb/1e6:8.1f} MB"
          f"   -> {l/f:.2f}x faster, {fb/lb:.0f}x less psi memory")
