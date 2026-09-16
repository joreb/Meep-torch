"""Does each stepper survive torch.compile, and stay bitwise under it?

Written for a specific finding: `NamedTuple._replace` raises a bare
AssertionError inside dynamo (torch 2.9.1, side_effects.codegen_save_tempvars),
while direct construction and `_make` compile fine. stepper_tm used _replace
ten times and stepper_te six, so the two hand-written 2D steppers were the only
ones that could not compile -- which inverted the usual intuition, because the
compiled generic steppers then beat the uncompiled hand-written ones.

Run it directly:  uv run python symbolic_test/check_stepper_compile.py
It reports, per stepper, whether compile succeeds and whether the compiled
result is bitwise-equal to eager. Bitwise is the bar, not close: these steppers
are gated against meep bit for bit.
"""
import torch

from meep_torch.core.coeffs import build_tm_coeffs, build_te_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper_te import init_te_state, step_te
from meep_torch.core.stepper_tm import init_tm_state, step_tm


def _seeded(state, g, names):
    torch.manual_seed(0)
    fields = {n: torch.randn_like(getattr(state, n)) for n in names}
    return state._replace(**fields)


def case_tm():
    g = Grid(nx=24, ny=20, dx=0.05)
    m = build_tm_coeffs(g, npml=4)
    s = _seeded(init_tm_state(g), g, ("Ez", "Hx", "Hy"))
    return s._replace(Dz=s.Ez.clone()), m, step_tm


def case_te():
    g = Grid(nx=24, ny=20, dx=0.05)
    m = build_te_coeffs(g, npml=4)
    s = _seeded(init_te_state(g), g, ("Hz", "Ex", "Ey"))
    return s._replace(Dx=s.Ex.clone(), Dy=s.Ey.clone()), m, step_te


def run(label, factory, steps=3):
    state, m, step = factory()
    eager = state
    for _ in range(steps):
        eager = step(eager, m)
    try:
        compiled = torch.compile(step, fullgraph=True)
        out = state
        for _ in range(steps):
            out = compiled(out, m)
    except Exception as exc:
        print(f"  {label:6s} compile FAILED -> {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:60]}")
        return False
    bad = [n for n in eager._fields
           if isinstance(getattr(eager, n), torch.Tensor)
           and not torch.equal(getattr(eager, n), getattr(out, n))]
    print(f"  {label:6s} compile OK, bitwise vs eager: "
          f"{'yes' if not bad else 'NO -- ' + ', '.join(bad)}")
    return not bad


if __name__ == "__main__":
    print(f"torch {torch.__version__}")
    ok = all([run("tm", case_tm), run("te", case_te)])
    raise SystemExit(0 if ok else 1)
