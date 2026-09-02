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
| 3D chi2 | after run queue | **cancelled by user (13:23): nonlinear stays 2D/2.5D.** The plan doc remains in docs/plans as a shelved record (rhs correction preserved there). Allowed ceiling on the vertical axis: a 2.5D correction — scale the effective chi2 by the pump²·SH vertical mode-overlap integral from mode profiles (one cheap mode solve, no 3D FDTD), factor stated. |

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

## 7. Kerr pulse-charge sizing — corrected (2026-08-17, later)

The pulse-charge Kerr variant was approved on a sidemode-suppression
figure that turned out to be wrong, and the error is worth recording
because it was caught only by simulating the pulse rather than
trusting the formula.

**What I predicted:** a Gaussian charge of 1400 tu gives −61.8 dB
suppression of the ±1 FSR sidemode.

**What the actual pulse spectrum gives: −15.3 dB.** That is ~3 % of
the pump amplitude sitting in a neighbouring azimuthal mode, in a
measurement whose entire signal is a 6.8e-5 relative frequency shift.
Frequency pulling from a 3 % sidemode is not a small perturbation on
that.

**Why the formula lied.** Two compounding mistakes: the Gaussian
Fourier exponent was off by a factor of two, and — the dominant term —
it ignored **window truncation**. The source was switched on at t=0
and off at t=T_on with the envelope still at 0.5 of its peak, and that
hard edge broadens the spectrum far more than the Gaussian itself
suppresses it. Suppression is set by σ, and a clean truncation needs
T_on ≳ 2kσ with k ≳ 3, so **T_on and suppression are coupled** — you
cannot buy suppression by shaping alone at fixed duration.

**Fix.** Subtract the pedestal so the envelope reaches exactly zero at
the source boundaries:

```
env(t) = clip( (exp(-(t-t0)²/2σ²) − exp(-k²/2)) / (1 − exp(-k²/2)), 0, None )
```

With the discontinuity removed the suppression becomes monotone in σ
(before the fix it was not: −67.3 dB at T_on=2450 but −64.6 dB at
T_on=2800, pure truncation ripple, which is the tell that the numbers
were artefacts).

**Corrected sizing**, pedestal-subtracted, ringdown 1.0 τ_field = 2522 tu:

| T_on | k | suppression | total | runtime |
|------|---|-------------|-------|---------|
| 1400 | 3.0 | −9.9 dB | 3922 tu | 4.8 h |
| 2100 | 3.0 | −36.8 dB | 4622 tu | 5.6 h |
| **2800** | **3.0** | **−59.2 dB** | **5322 tu** | **6.5 h** |

Recommended: **T_on = 2800, k = 3.0, −59.2 dB, 6.5 h.** The approved
1400 tu charge must not be used — it fails the sidemode check it was
approved on.

Also note the flat-top alternative stays rejected, and for a sharper
reason than before: its −328 dB sinc null must land exactly on the
sidemode, and T_rt is known only to a few percent (our two n_g
estimates differ by 2.6 %). A 2 % mismatch collapses that null to
−34 dB. A null you have to hit is a liability; the Gaussian degrades
gracefully.
