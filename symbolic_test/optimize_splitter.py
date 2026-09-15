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
# --stepper swaps the Yee curl kernel for the pylops or devito drop-in, keeping
# everything else (CPML, source, monitors, objective, Adam) identical. Both
# drop-ins are differentiable -- their numpy/devito round-trips live inside
# torch.autograd.Functions whose backward applies the adjoint operator -- so the
# same adjoint optimization runs on all three. Hard truth about --device cuda
# with a foreign stepper: the curl still runs on the CPU (cupy is not installed,
# so pylops has no GPU matvec; nvc is not installed, so devito emits OpenMP CPU
# code), and the torch state has to cross PCIe 8x per step to reach it. GPU
# helps the torch reference; it makes the drop-ins slower. --time-only measures
# exactly that instead of guessing.
#
# Run on a Linux+CUDA box (fast, minutes):
#   uv run python -m symbolic_test.optimize_splitter --device cuda
# Time one objective evaluation + one gradient, per stepper:
#   uv run python -m symbolic_test.optimize_splitter --time-only --device cuda
#   uv run python -m symbolic_test.optimize_splitter --time-only --stepper pylops
# Short / probe runs:
#   uv run python -m symbolic_test.optimize_splitter --device cpu --max-steps 4
#   uv run python -m symbolic_test.optimize_splitter --dry-run
import argparse
import pathlib
import time

import torch

from meep_torch.core.run import _make_runner
from meep_torch.core.stepper_tm import TMState, step_tm

SPLITTER = pathlib.Path(__file__).resolve().parent.parent / "examples_2" / "y_splitter"
SRC = SPLITTER / "c_y_splitter.py"
DEV_LINE = 'DEV = "cuda" if torch.cuda.is_available() else "cpu"'
# The drop-in steppers import splitter_setup, which exec's this same head to
# build its own reference -- so by the time we exec the real script, gdsfactory's
# global KCLayout already holds a cell by this name and Component() raises
# "Cellname ... already exists". Rename the cell in our copy: it is local to the
# head (only add_ref/add_port/component_to_eps touch it) and never reaches the
# rasterized eps, so the two builds stay identical -- _check_grid proves it.
CELL_LINE = 'comp = gf.Component("y_splitter_default_sbend")'
# A drop-in run must not clobber the torch reference's rho.npy / png, and
# splitter_setup reads output/c_y_splitter_rho.npy to pick its device geometry
# -- so only the foreign steppers get redirected; --stepper torch keeps the
# canonical output/ the docs and sibling scripts point at.
OUT_LINE = 'OUT = pathlib.Path(__file__).parent / "output"'
# Resolution knob. NPML = RES in the head, so the PML stays 1 um thick, and
# T_RUN is set from nx*dx, so the physical run length is unchanged -- doubling
# RES costs ~4x the cells and ~2x the steps (dt follows dx), i.e. ~8x the work
# and ~8x the adjoint checkpoint memory.
RES_LINE = "RES = 30"
# Conic-filter radius, the knob that sets the minimum fabricable feature.
# c_y_splitter hardcodes 0.12 um = 120 nm in both the objective and the final
# re-evaluation; with a 200 nm DRC rule that lets the optimiser create features
# it cannot fabricate (measured on the res-30 result at 10 nm sampling: 0.09%
# of solid and 0.24% of void area violate a 200 nm rule). Both call sites must
# move together or the design is optimised at one length scale and scored at
# another.
FILT_LINES = ("conic_filter(rho, 0.12, dx)", "conic_filter(rho_f, 0.12, dx)")


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


def pick_step_fn(name: str):
    """Import the requested stepper lazily -- the drop-in modules exec the
    c_y_splitter setup head at import, which is slow and needs gdsfactory."""
    if name == "torch":
        return step_tm
    if name == "sympy":
        # the only drop-in that imports no meep_torch at all: it consumes
        # TMState/TMCoeffs by attribute and derives every phase itself.
        from .stepper_sympy import step_tm_sympy
        return step_tm_sympy
    if name == "pylops":
        from .fdtd_pylops_2d import step_pylops
        return step_pylops
    from .fdtd_devito_2d import step_devito
    return step_devito


def _check_grid(g) -> None:
    """The drop-in modules bind NX/NY/DX at import from splitter_setup's own
    CPU head. If that head ever drifts from the one exec'd here, the two would
    silently be different problems -- and the sparse/devito operators would be
    built for the wrong shape."""
    from . import splitter_setup as SS
    assert (SS.grid.nx, SS.grid.ny) == (g["grid"].nx, g["grid"].ny), (
        f"grid mismatch: splitter_setup {SS.grid.nx}x{SS.grid.ny} vs "
        f"this head {g['grid'].nx}x{g['grid'].ny}")
    assert abs(SS.grid.dx - g["grid"].dx) < 1e-15, "dx mismatch"


def _time_it(fn, dev: str) -> float:
    """Wall clock around fn(), with a cuda sync so async kernels are counted."""
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


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
    ap.add_argument("--stepper", choices=("torch", "sympy", "pylops", "devito"),
                    default="torch",
                    help="Yee curl kernel: meep_torch's own, or a drop-in")
    ap.add_argument("--res", type=int, default=30,
                    help="pixels per um (default 30, the canonical run); "
                         "doubling it is ~8x the work and ~8x the memory")
    ap.add_argument("--filter-radius", type=float, default=0.12,
                    help="conic filter radius in um; sets the minimum feature "
                         "size (default 0.12 = 120 nm). Use 0.2 for a 200 nm "
                         "DRC rule.")
    ap.add_argument("--time-only", action="store_true",
                    help="time one objective eval and one gradient, then exit")
    args = ap.parse_args()

    dev = pick_device(args.device)
    print(f"optimize_splitter  device={dev} (requested {args.device}), "
          f"stepper={args.stepper}")

    src = SRC.read_text()
    assert DEV_LINE in src, f"DEV line not found in {SRC}"
    src = src.replace(DEV_LINE, f'DEV = "{dev}"')
    assert CELL_LINE in src, f"cell line not found in {SRC}"
    src = src.replace(CELL_LINE, CELL_LINE[:-2] + '_run")')
    if args.res != 30:
        assert RES_LINE in src, f"RES line not found in {SRC}"
        src = src.replace(RES_LINE, f"RES = {args.res}")
    if args.filter_radius != 0.12:
        for line in FILT_LINES:
            assert line in src, f"filter call not found in {SRC}: {line}"
            src = src.replace(line, line.replace("0.12", str(args.filter_radius)))
    if args.max_steps < 16:
        src = src.replace("for it in range(16):",
                          f"for it in range({args.max_steps}):")

    if args.stepper != "torch":
        # one hook: every run_tm call site goes through the drop-in runner.
        # _make_runner gives the foreign stepper the same chunked-checkpoint
        # machinery run_tm has, so the adjoint memory profile is unchanged.
        src = src.replace("run_tm(", "_RUN(")
        assert OUT_LINE in src, f"OUT line not found in {SRC}"
        suffix = args.stepper
        if args.res != 30:
            suffix += f"_res{args.res}"
        if args.filter_radius != 0.12:
            suffix += f"_r{int(round(args.filter_radius * 1000))}"
        src = src.replace(OUT_LINE, OUT_LINE[:-1] + f'_{suffix}"')

    if args.dry_run:
        head = src.split("def s_params")[0]
        g = {"__file__": str(SRC)}
        exec(compile(head, "splitter_head", "exec"), g)
        print(f"dry-run OK  grid {g['grid'].nx}x{g['grid'].ny}, "
              f"DEV={dev}, head exec'd cleanly")
        return

    g = {"__file__": str(SRC)}
    if args.stepper != "torch":
        g["_RUN"] = _make_runner(pick_step_fn(args.stepper), TMState)

    if args.time_only:
        # exec the head + s_params/total_T definitions, stopping just before
        # the Adam loop, then time one forward and one forward+backward.
        head = src.split("rho = seed.clone().requires_grad_(True)")[0]
        exec(compile(head, str(SRC), "exec"), g)
        if args.stepper != "torch":
            _check_grid(g)
        rho = g["seed"].clone().requires_grad_(True)

        def fwd():
            with torch.no_grad():
                g["total_T"](rho.clamp(0.0, 1.0), 64.0)

        def fwd_bwd():
            J = g["total_T"](rho.clamp(0.0, 1.0), 8.0)
            (-J).backward()

        t_f = _time_it(fwd, dev)
        t_fb = _time_it(fwd_bwd, dev)
        n_steps = int(round(g["T_RUN"] / g["grid"].dt))
        print(f"time-only  stepper={args.stepper} device={dev} "
              f"grid={g['grid'].nx}x{g['grid'].ny} steps={n_steps}")
        # NB: the two are not the same code path. Under no_grad, _make_runner
        # sees no grad leaves and skips the checkpoint wrapper entirely; the
        # grad call pays forward + per-chunk recompute + backward. So the ratio
        # below is "graph-building run vs plain run", not "cost of backward".
        print(f"  no_grad forward (rescore)  {t_f:8.2f} s")
        print(f"  grad step (fwd+recompute+bwd) {t_fb:8.2f} s")
        # the real loop does one fwd+bwd and one post-beta-64 rescore per step
        print(f"  projected 16 Adam steps    {16 * (t_fb + t_f) / 60:8.1f} min "
              f"(projection, not measured)")
        return

    exec(compile(src, str(SRC), "exec"), g)   # runs the optimization + plots


if __name__ == "__main__":
    main()