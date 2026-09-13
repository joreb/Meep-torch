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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import scenario as S
from .fdtd_devito import run_devito
from .fdtd_pylops import run_pylops


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

    print("[1/3] meep_torch.core reference ...")
    ref = S.reference_ez()
    print(f"      max|Ez| = {np.max(np.abs(ref)):.6e}")

    stencils = []
    for name, fn in (("devito", run_devito), ("pylops", run_pylops)):
        print(f"[?] {name} stencil ...")
        try:
            data = fn(gpu=args.gpu)
        except Exception as exc:   # backend missing (devito on Windows) or GPU
            print(f"      skipped: {exc!r}")
            continue
        print(f"      max|Ez| = {np.max(np.abs(data)):.6e}  "
              f"corr={_correlation(ref, data):.4f}")
        stencils.append((name, data))

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