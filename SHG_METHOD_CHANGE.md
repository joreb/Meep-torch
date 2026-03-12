# SHG measurement — what is changing (2026-08-17)

Companion to `SHG_ANOMALY_REPORT.md` (the diagnosis). This is the
change list: what we were doing before, what we do now, and which
already-reported numbers move.

## 1. Measurement method: brute-force buildup → single-pass + assembly

**Before:** measure the steady-state SHG efficiency by running the full
poled ring in nonlinear FDTD until both resonances charge. The SH
cavity's field lifetime makes that a multi-day simulation (the last
sizing was 45–67 h per amplitude), because FDTD must step through
~20000 time units of resonant charging at ~4 s per time unit.

**Now:** split the problem, the same way the linear ring was measured:

- **Simulate only the unknown:** one transit of the pump through the
  poled arc, measuring how much SH a single pass generates. No
  resonator, no charging — hundreds of time units, minutes of GPU.
- **Assemble the ring analytically** (coupled-mode theory): the pump's
  circulating buildup (×7.3 in power) and the SH's circulating buildup
  (×205) come from the already-measured, now-audited coupler and loop
  parameters (t, κ, a) of each band. Steady state is a formula
  evaluation, not a wait.
- **Validate against data we already own:** the same formula evaluated
  at the charge states of the two existing long records must reproduce
  their measured efficiencies. Both hit → the steady-state number is
  the same expression at full charge.

Justification for the split: the passed P² gate proves conversion is
second-order and undepleted at the working amplitude, which is exactly
the regime where generation and resonant enhancement factorize.

## 2. Efficiency estimator: peak FFT bin → integrated band power

The old `shg_conversion` measured each tone as its peak FFT bin, which
grows with record length for a coherent tone. All cross-record-length
eta comparisons made with it were invalid (this is what produced the
"impossible" falling efficiency that started the investigation). The
estimator becomes integrated band power normalized by the window —
record-length independent. The P² gate survives unchanged (3.772 vs
3.822; both its runs had equal length).

## 3. Corrected ring parameters (all reporting inherits these)

An audit closed a 1.8× disagreement in the SH quality factor — the
direct derivation had taken √(loop) as the round-trip amplitude when
the stored loop already is the full round trip:

| Quantity | Was quoted | Now | Why it changes |
|---|---|---|---|
| Q pump (1348 nm) | 5887–6068 (scatter) | **5877** | √ bug worth 3% here |
| Q SH (674 nm) | 48426 or 85562 (unresolved 1.8×) | **~50400** (three agreeing derivations) | √ bug worth 1.8× here |
| τ_field pump | 2529–2604 | **2522 tu** | follows Q |
| τ_field SH | 18356 (or ~10⁴, ambiguous) | **10804 tu** | follows Q |
| SH circulating enhancement | 190 or 575 (3× spread) | **205** | single-valued once a_rt fixed |
| Pump coupling | — | |t|² = 0.595, heavily overcoupled (confirmed) | unchanged |

## 4. Run plan and sizings

| Run | Old plan | New plan |
|---|---|---|
| SHG absolute | 45 h (2τ_SH charge) — **killed by user call** | single-pass (~minutes) + assembly + validation against existing records |
| Kerr ringdown | 12.4 h (sized off τ=2529, then wrongly 2.4 h off the 494 artifact) | ~5–6 h at corrected τ_field = 2522 — **on hold, awaiting user approval** |
| 3D chi2 | after run queue | unchanged: plan approved, rhs correction verified; next in queue after the above |

## 5. Labels on already-published statements

- "2τ = 75% of steady state" labeling: dead. With corrected lifetimes
  the T=5057 record sits at ~87% pump charge but only ~20% SH charge;
  no existing record is steady state.
- The falling-eta anomaly needs no exotic physics: the pump and SH
  charge on lifetimes 4.3× apart, so their ratio is not monotone in
  time. Explanation adopted, nothing to fix in the solver.
- The P² gate (second-order signature, pump exponent exactly 2.00)
  stands. The 4.5% shortfall remains attributed to slight pump
  depletion at the doubled amplitude — physical, expected, and an
  argument for never raising the working amplitude.

## 6. Process lessons now in force

- Cross-length spectral comparisons require a length-independent
  estimator; peak bins never qualify.
- Agreement between derivations that share an input is weak evidence —
  the √(loop) bug sat inside a "four-way agreement" on the pump and
  only exploded on the narrow SH line.
- Correcting a plan document means grepping the whole file for the
  wrong form (the naive rhs survived one screen below its own
  correction).
- Multi-day resonant-buildup runs are a last resort; decompose into
  transit-scale measurements + audited analytic assembly first.
