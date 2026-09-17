"""Does one parameterised stepper survive torch.compile across many configs?

The unification question has one unmeasured part. Twelve steppers today are
twelve code objects, so dynamo gives each its own cache entry and they cannot
interfere. A unified stepper is ONE code object specialised by a factory, so
every config's cache entry lands on the same code object -- and dynamo's
`cache_size_limit` is per code object.

Two things are measured separately because they fail differently:

  A. varying the captured CONSTANTS (slice bounds, axis ints) -- the shape a
     unified stepper takes, since the factory bakes the spec into closure cells
  B. varying the captured TENSORS (coefficients) and the input SHAPES -- the
     adjoint's rebuild loop, already known safe for coefficients alone

If A recompiles per config and the limit is 8, then twelve steppers behind one
factory fall off a cliff at the ninth and silently run eager.
"""
import torch
import torch._dynamo

LIMIT = torch._dynamo.config.cache_size_limit


def _graphs():
    return torch._dynamo.utils.counters["stats"]["unique_graphs"]


def _report(tag, n_cfg, before, fell_back):
    got = _graphs() - before
    print(f"  {tag:<34s} configs={n_cfg:<3d} graphs={got:<3d} "
          f"{'EAGER FALLBACK' if fell_back else 'all compiled'}")
    return got


def probe_constants(n_cfg):
    """A: one code object, closures differing only in a baked-in int."""
    torch._dynamo.reset()
    before = _graphs()
    fell_back = False

    def make(trim):
        sl = slice(trim, -trim if trim else None)

        def step(x):                      # ONE code object for every trim
            return (x[sl] * 2.0).sum()
        return step

    for trim in range(1, n_cfg + 1):
        fn = torch.compile(make(trim), fullgraph=True)
        x = torch.randn(64, dtype=torch.float64)
        try:
            fn(x)
        except Exception:
            fell_back = True
    return _report("A: baked-in constants differ", n_cfg, before, fell_back)


def probe_structure(n_cfg, limit=None):
    """C: one code object whose BODY differs per config -- the real cliff.

    A and B vary a number dynamo can make dynamic. Structure cannot be made
    dynamic: a reduction with three curl terms and one with two are different
    graphs, full stop. If a unified stepper puts all twelve reductions on one
    code object, twelve graphs land on one cache entry, and the limit is 8.
    """
    torch._dynamo.reset()
    if limit is not None:
        torch._dynamo.config.cache_size_limit = limit
    before = _graphs()
    fell_back = False

    def make(n_terms):
        def step(x):                      # ONE code object, n_terms ops
            acc = x
            for _ in range(n_terms):
                acc = acc * 1.5 + x
            return acc.sum()
        return step

    for n in range(1, n_cfg + 1):
        fn = torch.compile(make(n), fullgraph=True)
        try:
            fn(torch.randn(64, dtype=torch.float64))
        except Exception as e:
            fell_back = True
            print(f"    config {n}: {type(e).__name__}")
    got = _report(f"C: body differs (limit={torch._dynamo.config.cache_size_limit})",
                  n_cfg, before, fell_back)
    torch._dynamo.config.cache_size_limit = LIMIT
    return got


def probe_tensors_and_shapes(n_cfg):
    """B: one code object, closures over different tensors, different shapes."""
    torch._dynamo.reset()
    before = _graphs()

    def make(coeff):
        def step(x):
            return (x * coeff).sum()
        return step

    for i in range(n_cfg):
        fn = torch.compile(make(torch.tensor(float(i))), fullgraph=True)
        fn(torch.randn(64 + i, dtype=torch.float32))
    return _report("B: tensors + shapes differ", n_cfg, before, False)


# Twelve literal defs: the control has to be genuinely distinct code objects.
# Building them with exec() does NOT work -- every exec'd function reports
# ("<string>", 1), so dynamo keys them to a single cache entry and the control
# hits the recompile limit, measuring the harness instead of torch.
def _s0(x):  return (x[1:] * 2.0).sum()
def _s1(x):  return (x[2:] * 2.0).sum()
def _s2(x):  return (x[3:] * 2.0).sum()
def _s3(x):  return (x[4:] * 2.0).sum()
def _s4(x):  return (x[5:] * 2.0).sum()
def _s5(x):  return (x[6:] * 2.0).sum()
def _s6(x):  return (x[7:] * 2.0).sum()
def _s7(x):  return (x[8:] * 2.0).sum()
def _s8(x):  return (x[9:] * 2.0).sum()
def _s9(x):  return (x[10:] * 2.0).sum()
def _s10(x): return (x[11:] * 2.0).sum()
def _s11(x): return (x[12:] * 2.0).sum()
_TWELVE = [_s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7, _s8, _s9, _s10, _s11]


def probe_separate_code_objects(n_cfg):
    """Control: twelve steppers today -- a distinct code object each."""
    torch._dynamo.reset()
    before = _graphs()
    for f in _TWELVE[:n_cfg]:
        torch.compile(f, fullgraph=True)(torch.randn(64, dtype=torch.float64))
    return _report("control: one code object each", n_cfg, before, False)


def probe_real_factory(n_cfg):
    """The real thing: make_step closures over different grids and npml."""
    from meep_torch.core.grid import Grid
    from meep_torch.core.coeffs import build_tm_coeffs
    from .generic_tm import init as ginit
    from .generic_tm_hoisted import make_step

    torch._dynamo.reset()
    before = _graphs()
    for i in range(n_cfg):
        nx = ny = 40 + 4 * i
        g = Grid(nx=nx, ny=ny, dx=0.05)
        fn = torch.compile(make_step(build_tm_coeffs(g, npml=4 + i)),
                           fullgraph=True)
        s = ginit(nx, ny)
        for _ in range(2):
            s = fn(s)
    return _report("real make_step, varied grid+npml", n_cfg, before, False)


if __name__ == "__main__":
    print(f"torch {torch.__version__}  cache_size_limit = {LIMIT}\n")
    for n in (4, 8, 12):
        print(f"n_configs = {n}")
        probe_constants(n)
        probe_structure(n)
        probe_tensors_and_shapes(n)
        probe_separate_code_objects(n)
        print()
    print("structure with the limit raised:")
    probe_structure(12, limit=32)
    print()
    print("real factory (slowest, run once at 12):")
    probe_real_factory(12)
