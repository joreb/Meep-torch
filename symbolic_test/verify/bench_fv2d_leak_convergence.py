"""Half-wave m0 leakage vs resolution and run length.

meep gives 0.0219 at res=20; fv2d gives 0.0480. Analytic says 0. If ours is a
discretisation artefact it falls with resolution; if it is a staggering error
it plateaus. Run length is swept too, in case the DFT window still holds
transient.

STATUS: written, NOT YET RUN -- it was killed twice for starving a full-suite
run. This is the decisive outstanding test for the fv2d half-wave leakage: if m0
falls like 1/res^2 toward meep's 0.0219 it is discretisation, if it plateaus
something structural survives the six exclusions on
wiki/tasks/fix_fv2d-halfwave-leakage.md. Run it with `python -u` on an IDLE box.
"""
# Run from the repository root: these read fixtures under tests/ by relative
# path. `cd` elsewhere and they will not find them.

import sys, time
sys.path.insert(0, "tests/examples")
from test_fv2d_polgrating_vs_meep import _efficiencies, REF

row = [r for r in REF["rows"] if r["phase"] == 0.5][0]
d_half = 0.5 * row["d"]
print(f"half-wave: d={row['d']:.4f}  meep m0={row['eff_m0']:.4f}  analytic 0")
print(f"{'res':>4} {'t_end':>6} {'m0':>9} {'m1':>9} {'secs':>6}")
for res in (20, 30, 40):
    for t_end in (120.0, 240.0):
        t0 = time.time()
        e0, e1 = _efficiencies(d_half, res, t_end=t_end)
        print(f"{res:4d} {t_end:6.0f} {e0:9.4f} {e1:9.4f} {time.time()-t0:6.0f}")
