"""Cylindrical Yee table (A) vs transformation optics (B), on the one axis
where they differ: the CONSTITUTIVE update.

Everything else is shared -- same curl, same CPML, same sources -- so the
comparison is exactly:

  A  scalar eps and mu, plus 1/r metric coefficients (1-D, negligible).
     E from D is one divide per component.
  B  the metric absorbed into a full 3x3 symmetric eps AND mu, so E from D is
     a 3x3 apply per component, and the same again for H from B.

Both shapes are already in the tree: A is `kernels.EH_FROM_D`, B is
`stepper_fv2d`'s tensor solve.

RESULT (2026-09-18, CUDA float64): route A (cylindrical Yee table, scalar
eps/mu + 1/r coefficients) beats route B (transformation optics, metric absorbed
into tensor eps AND mu) on both axes -- B costs 2.44-2.50x the time and 6x the
coefficient memory at 1024^2/1536^2/2048^2. A 512^2 row read "B is 2.3x FASTER"
and was DISCARDED as launch-bound; it contradicts both larger sizes and the
arithmetic. This is the measurement behind choosing a cylindrical Yee table.
"""
import time
import torch

DEV = "cuda"
DT = torch.float64


def _time(fn, n=50, repeats=9):
    for _ in range(5):
        fn()
    best = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        best.append(n / (time.perf_counter() - t0))
    best.sort()
    return best[len(best) // 2]


def bench(n):
    g = (n, n)
    D = [torch.randn(*g, dtype=DT, device=DEV) for _ in range(3)]
    P = [torch.randn(*g, dtype=DT, device=DEV) for _ in range(3)]
    eps = torch.rand(*g, dtype=DT, device=DEV) + 1.0
    inv6 = [torch.rand(*g, dtype=DT, device=DEV) for _ in range(6)]

    def route_a():                      # scalar: one divide per component
        return [(D[i] - P[i]) / eps for i in range(3)]

    def route_b():                      # 3x3 symmetric apply, per component
        ixx, ixy, ixz, iyy, iyz, izz = inv6
        return [ixx * D[0] + ixy * D[1] + ixz * D[2],
                ixy * D[0] + iyy * D[1] + iyz * D[2],
                ixz * D[0] + iyz * D[1] + izz * D[2]]

    a, b = _time(route_a), _time(route_b)
    # coefficient memory: A carries eps + mu (2 scalar fields); B carries a
    # symmetric tensor for each, 6 + 6.
    mb = n * n * 8 / 1e6
    return a, b, 2 * mb, 12 * mb


print(f"{'grid':>10} {'A upd/s':>10} {'B upd/s':>10} {'B/A time':>9} "
      f"{'A coeff':>9} {'B coeff':>9} {'ratio':>6}")
for n in (512, 1024, 1536, 2048, 2048):
    a, b, ma, mb_ = bench(n)
    print(f"{n:>5}^2 {a:10.1f} {b:10.1f} {a/b:9.2f}x {ma:8.1f}M {mb_:8.1f}M "
          f"{mb_/ma:5.1f}x")
