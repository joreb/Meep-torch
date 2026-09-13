# SPDX-License-Identifier: MIT
# Shared 2D Y-splitter setup for the devito + pylops drop-in integrators.
# exec's the head of examples_2/y_splitter/c_y_splitter.py (everything before
# `def s_params`, the same trick c_y_splitter_3d.py uses) so the grid, ports,
# design region and base permittivity are byte-identical to the optimization
# run -- then builds the meep_torch reference (ModeSourceTM + FluxMonitorTM x3
# + decompose) and a stepper-agnostic run loop. Only the Yee curl kernel is
# swapped by a foreign step_fn; CPML coefficients, source, monitors and mode
# decomposition are reused verbatim, so the comparison is tight.
#
# No optimized rho on this box -> uses eps_base (the rasterized mmi1x2_with_sbend
# seed geometry) as the device. A valid Y-splitter with real S-params; the
# stencils are still exercised against the meep_torch reference on the SAME eps.
import pathlib

import torch

from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import _run_chunk
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_tm import init_tm_state, step_tm
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM

HERE = pathlib.Path(__file__).resolve().parent
SPLITTER = HERE.parent / "examples_2" / "y_splitter"


def _load_head(device: str):
    """exec the c_y_splitter.py setup head with DEV forced to `device`.

    The head hardcodes `DEV = "cuda" if torch.cuda.is_available() else "cpu"`;
    on boxes where cuda is "available" but the GPU is unusable (e.g. sm_61 vs a
    cu128 build) that picks cuda and every kernel fails. Rewriting that one
    line forces CPU (or a working cuda device) deterministically."""
    src = open(SPLITTER / "c_y_splitter.py").read()
    head = src.split("def s_params")[0]
    head = head.replace(
        'DEV = "cuda" if torch.cuda.is_available() else "cpu"',
        f'DEV = "{device}"')
    g = {"__file__": str(SPLITTER / "c_y_splitter.py")}
    exec(compile(head, "splitter_head", "exec"), g)
    return g


# one-time setup on CPU (the test golden); device stays CPU so the GTX-1050
# incompatibility on this box is a non-issue. A working cuda box can pass
# device="cuda" to build() for the --gpu runner path.
_HEAD = _load_head("cpu")
grid = _HEAD["grid"]
dx = _HEAD["dx"]
FCEN = _HEAD["FCEN"]
NPML = _HEAD["NPML"]
T_RUN = _HEAD["T_RUN"]
ports = _HEAD["ports"]
eps_base = _HEAD["eps_base"].detach().to("cpu")
# Port objects (built in the head at the validate_ports block), not the
# `ports` dict (those are (x, y, w) tuples from component_to_eps).
p_in, p_up, p_dn = _HEAD["p_in"], _HEAD["p_up"], _HEAD["p_dn"]
# device geometry: the rasterized seed. paint the design region with the seed
# rho projected at BETA=64 (same threshold the optimization evaluates at) so the
# geometry matches what c_y_splitter.py would simulate given its own seed.
region = _HEAD["region"]
seed = _HEAD["seed"]
_eps_fin = region.paint(
    eps_base, tanh_projection(conic_filter(seed, 0.12, dx), 64.0)
).detach()
coeffs = build_tm_coeffs(grid, eps_inf=_eps_fin, npml=NPML)
src = ModeSourceTM(grid, coeffs, p_in, GaussianPulse(FCEN, 0.2), FCEN)
mons = [
    FluxMonitorTM(grid, [FCEN], x_normal=True, idx=p.index,
                  lo=p.lo, hi=p.hi)
    for p in (p_in, p_up, p_dn)
]
N_STEPS = int(round(T_RUN / grid.dt))


def _s_params_from_accs(accs):
    """decompose each port's accumulated DFT -> (S21, S31, S11) complex."""
    out = []
    for p, acc in zip((p_in, p_up, p_dn), accs):
        el = _eps_fin.detach().cpu()[p.index, p.lo:p.hi]
        out.append(decompose(acc.cpu(), el, dx, [FCEN], "tm", 1, axis_sign=-1.0))
    a_in = out[0][0][0, 0]              # forward mode amp at input
    s21 = out[1][0][0, 0] / a_in
    s31 = out[2][0][0, 0] / a_in
    s11 = out[0][1][0, 0] / a_in       # backward at input
    return s21, s31, s11


def _metrics(s21, s31, s11):
    t_up = float(abs(s21) ** 2)
    t_dn = float(abs(s31) ** 2)
    tot = t_up + t_dn
    imb = abs(t_up - t_dn) / max(tot, 1e-12)
    return t_up, t_dn, tot, imb


def meep_reference():
    """Golden: meep_torch's own step_tm through the same loop. Returns
    (s21, s31, s11) and (t_up, t_dn, total, imbalance)."""
    _, accs = _run_chunk(step_tm, init_tm_state(grid), coeffs, 0, N_STEPS,
                         src, mons)
    s21, s31, s11 = _s_params_from_accs(accs)
    return (s21, s31, s11), _metrics(s21, s31, s11)


def run_with_step(step_fn):
    """Same loop, foreign integrator. Returns (s21, s31, s11), metrics."""
    _, accs = _run_chunk(step_fn, init_tm_state(grid), coeffs, 0, N_STEPS,
                         src, mons)
    s21, s31, s11 = _s_params_from_accs(accs)
    return (s21, s31, s11), _metrics(s21, s31, s11)