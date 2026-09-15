# SPDX-License-Identifier: MIT
# Run the straight-waveguide scenario three ways and compare:
#   1. meep_torch.core reference (controlled numerics, the test golden)
#   2. devito symbolic-PDE stencil
#   3. pylops linear-operator stencil
# Optionally also the high-level meep_torch.mp straight-waveguide (the actual
# examples/01_basics/straight-waveguide.py path) for the plot. Output png + csv
# to symbolic_test/output/. Run:  uv run python -m symbolic_test.run_straight_waveguide
import argparse
import os
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:                       # package import (pytest, -m) ...
    from . import scenario as S
    from .fdtd_devito import run_devito
    from .fdtd_pylops import run_pylops
    from .fdtd_sympy import run_sympy
except ImportError:        # ... or plain script run (path invocation)
    import scenario as S
    from fdtd_devito import run_devito
    from fdtd_pylops import run_pylops
    from fdtd_sympy import run_sympy


def _timed(fn):
    """Wall clock around fn() -> (result, seconds).

    No explicit device sync needed: the torch reference ends in .cpu(), the
    cupy path in .get(), and devito is host-side -- each already blocks on the
    device before returning."""
    t0 = time.perf_counter()
    res = fn()
    return res, time.perf_counter() - t0


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="devito nvidiaX / pylops cupy")
    ap.add_argument("--with-mp", action="store_true",
                    help="also run the high-level mp straight-waveguide for the plot")
    args = ap.parse_args()

    out = pathlib.Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    dev = "cuda" if args.gpu else S.device()
    n = S.n_steps(S.DX * 0.5)
    print(f"straight-waveguide  grid {S.NX}x{S.NY}, {n} steps, "
          f"{'gpu' if args.gpu else 'cpu'} requested")
    times = []

    print(f"[1/3] meep_torch.core reference ({dev}) ...")
    ref, t_ref = _timed(lambda: S.reference_ez(dev))
    print(f"      max|Ez| = {np.max(np.abs(ref)):.6e}   {t_ref:8.2f} s")
    times.append(("meep_torch step_tm", dev, t_ref))

    # Honest backend labels: devito GPU needs the NVIDIA HPC SDK (nvc) and
    # pylops GPU needs cupy. Neither falls back silently any more -- a missing
    # backend raises here and is reported as skipped WITH the reason, rather
    # than quietly producing a CPU number under a GPU heading.
    stencils = []
    for name, fn, backend in (("devito", run_devito, "openmp/nvidiaX"),
                              ("pylops", run_pylops, "scipy/cupy"),
                              ("sympy", run_sympy, "lambdify/numpy")):
        print(f"[?] {name} stencil ({backend}) ...")
        try:
            data, t = _timed(lambda: fn(gpu=args.gpu))
        except Exception as exc:   # backend missing (devito nvc, pylops cupy)
            print(f"      skipped: {type(exc).__name__}: {exc}")
            times.append((name, "unavailable", float("nan")))
            continue
        print(f"      max|Ez| = {np.max(np.abs(data)):.6e}  "
              f"corr={_correlation(ref, data):.4f}   {t:8.2f} s")
        # devito never reaches the GPU here: its offload runtime cannot share
        # a process with torch's (see fdtd_devito.run_devito), so it is CPU
        # whatever --gpu says. Label it for what it is.
        dev_used = {"devito": "cpu (openmp)", "sympy": "cpu (numpy)"}.get(
            name, "cuda" if args.gpu else "cpu")
        times.append((name, dev_used, t))
        stencils.append((name, data))

    print(f"\n{'backend':22s} {'device':14s} {'seconds':>10s}  {'vs ref':>8s}")
    for name, d, t in times:
        rel = "-" if t != t else f"{t / t_ref:.2f}x"
        print(f"{name:22s} {d:14s} {t:10.2f}  {rel:>8s}")

    panels = [("meep_torch.core", ref)] + stencils
    if args.with_mp:
        print("[+] meep_torch.mp straight-waveguide ...")
        mp_ez = S.mp_straight_waveguide_ez()
        panels.append(("mp straight-waveguide", mp_ez))
        print(f"      max|Ez| = {np.max(np.abs(mp_ez)):.6e}  "
              f"corr={_correlation(ref, mp_ez):.4f}")

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, data) in zip(axes, panels):
        ax.imshow(data.T, interpolation="spline36", cmap="RdBu",
                  vmin=-np.max(np.abs(ref)), vmax=np.max(np.abs(ref)))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "symbolic_straight_waveguide.png", dpi=150)
    plt.close(fig)
    print(f"saved {out / 'symbolic_straight_waveguide.png'}")

    np.savetxt(out / "reference_ez.csv", ref, delimiter=",")
    print(f"saved {out / 'reference_ez.csv'}")


if __name__ == "__main__":
    main()