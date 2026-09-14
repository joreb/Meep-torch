# SPDX-License-Identifier: MIT
# Adjoint optimization runner for the 2D Y splitter at 674 nm. Runs the
# canonical examples_2/y_splitter/c_y_splitter.py optimization (16 Adam steps,
# beta continuation 8->64, balance-aware objective 2*sqrt(T21*T31)) byte-
# identically -- the only patch is the device line, so the result (rho.npy +
# c_y_splitter.png in examples_2/y_splitter/output/) is exactly what the
# original script produces, just with a working device.
#
# Why a runner: c_y_splitter.py hardcodes `DEV = "cuda" if
# torch.cuda.is_available() else "cpu"`, which lies on cards like a GTX 1050
# (sm_61): cuda is "available" but the cu128 kernels have no image for it, so
# every op crashes. --device auto probes a real cuda op and falls back to cpu.
#
# Run on a Linux+CUDA box (fast, minutes):
#   uv run python -m symbolic_test.optimize_splitter --device cuda
# Short / probe runs:
#   uv run python -m symbolic_test.optimize_splitter --device cpu --max-steps 4
#   uv run python -m symbolic_test.optimize_splitter --dry-run
import argparse
import pathlib

import torch

SPLITTER = pathlib.Path(__file__).resolve().parent.parent / "examples_2" / "y_splitter"
SRC = SPLITTER / "c_y_splitter.py"
DEV_LINE = 'DEV = "cuda" if torch.cuda.is_available() else "cpu"'


def _cuda_works() -> bool:
    """torch.cuda.is_available() can be True on incompatible cards (e.g. sm_61
    vs a cu128 build). Probe an actual op and trust that instead."""
    if not torch.cuda.is_available():
        return False
    try:
        (torch.zeros(1, device="cuda") + 1.0).cpu()
        return True
    except Exception:
        return False


def pick_device(req: str) -> str:
    if req == "cpu":
        return "cpu"
    if req == "cuda":
        return "cuda"
    return "cuda" if _cuda_works() else "cpu"   # req == "auto"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--max-steps", type=int, default=16,
                    help="cap the Adam iterations (default 16, the full run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="exec only the setup head, then exit (no optimization)")
    args = ap.parse_args()

    dev = pick_device(args.device)
    print(f"optimize_splitter  device={dev} (requested {args.device})")

    src = SRC.read_text()
    assert DEV_LINE in src, f"DEV line not found in {SRC}"
    src = src.replace(DEV_LINE, f'DEV = "{dev}"')
    if args.max_steps < 16:
        src = src.replace("for it in range(16):",
                          f"for it in range({args.max_steps}):")

    if args.dry_run:
        head = src.split("def s_params")[0]
        g = {"__file__": str(SRC)}
        exec(compile(head, "splitter_head", "exec"), g)
        print(f"dry-run OK  grid {g['grid'].nx}x{g['grid'].ny}, "
              f"DEV={dev}, head exec'd cleanly")
        return

    g = {"__file__": str(SRC)}
    exec(compile(src, str(SRC), "exec"), g)   # runs the optimization + plots


if __name__ == "__main__":
    main()