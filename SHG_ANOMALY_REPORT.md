# SHG run anomaly — verification report (2026-08-17)

## What we are trying to do

The device-suite deliverable set includes a **second-harmonic-generation
ring**: a periodically-poled thin-film lithium-niobate (TFLN) racetrack
(r = 50 µm, W = 1.2 µm, single bus, gap 0.25 µm) that converts a
1348 nm pump into 674 nm light — the same wavelength the SiN passive
devices (edge coupler, Y splitter, mmi_tapered) are designed for.
Material dispersion makes pump and SH walk out of phase after
L_coh = 0.86 µm, so the design relies on quasi-phase-matching: the χ⁽²⁾
sign is flipped every 1.72 µm of arc (Λ = 3.43 µm, ~183 domains).
These are the Phase-3 *nonlinear FDTD* runs that measure what the
design actually converts.

## Why the current runs matter

Two numbers are the whole deliverable:

1. **Proof the conversion is second-order** — the P² gate: doubling the
   pump amplitude must quadruple the conversion efficiency
   eta = P(2f₀)/P(f₀). This is the signature that separates real χ⁽²⁾
   physics from numerical artifacts.
2. **An honest absolute efficiency** at steady state — the headline
   number for the presentation, with the resonant buildup stated
   rather than guessed.

A third run (Kerr ringdown) measures the χ⁽³⁾ self-phase-modulation of
the same ring; its runtime and its frequency-resolution floor are both
set by the ring's photon lifetime, so it inherits any error in the Q.

## What happened

The P² gate passed, but the follow-up "2τ absolute" run produced an
efficiency *lower* than the short run, which is impossible for pure
buildup. The peer session diagnosed two bugs; this report is my
independent verification of that diagnosis against the code and the
extraction data. One of the two claims is confirmed, one is refuted.

## What ran

| Run | Record length | eta = P(2f0)/P(f0) (peak-bin) |
|---|---|---|
| Scale check (amp A) | T = 1000 | 4.5918e-5 |
| Scale check (amp 2A) | T = 1000 | 1.7552e-4 → ratio 3.822, gate PASS |
| "2τ absolute" (amp A) | T = 5057 | 1.4589e-5 — **lower than T=1000, backwards** |

## Claim 1 — estimator is record-length dependent: **CONFIRMED**

`shg_conversion()` in `examples_2/b_ring_shg.py` measures each band as the
**peak FFT bin** (`spec[m].max()`). For a coherent CW tone the peak bin
grows ~N² with record length while a less-coherent band does not, so
eta values from records of different lengths were never comparable.
The T=1000 vs T=5057 comparison was invalid from the start.

Fix (peer, in progress): integrated band power normalized by the window
— length-independent. The P² gate itself survives the estimator change
(3.772 integrated vs 3.822 peak-based) because both its runs used
identical record lengths.

## Claim 2 — ring charging time is ~494 tu, so 2τ was already steady state: **REFUTED**

The peer fitted a 1−1/e time of ~494 tu from the T=1000 envelope and
concluded the loaded Q is ~1150 (not the spectral 5887), which would
shrink the Kerr run sizing from 12.4 h to 2.4 h.

Direct computation from the committed extraction data
(`examples_2/b_ring_shg_data.npz`, pump band at f = 0.74179):

| Quantity | Value |
|---|---|
| bus transmission \|t\| | 0.7712 (\|t\|² = 0.595 — heavy overcoupling, **that part is true**) |
| loop amplitude a | 0.98155 |
| group index n_g (loop-phase slope) | 2.172 |
| round-trip time T_rt = n_g·L | 699.6 tu |
| per-round-trip energy decay (\|t\|a)² | 0.573 |
| energy lifetime τ_E = T_rt / −ln((\|t\|a)²) | **1256 tu** |
| loaded Q = ω·τ_E | **5856** |

Three independent derivations now agree: the assembled-spectrum Q
(5887), the (t, a) round-trip decay (5856, 0.5% apart), and the
original field-charging constant Q/(πf) = 2529 ≈ τ_field = 2·τ_E = 2513.
The ~494 envelope fit is the outlier and is the number that must be
rechecked, not the Q. Likely contaminations of that fit: the record
starts after `t_skip` (late-stage charging only), the mode beating the
peer itself identified, and 1−1/e of an *energy* envelope is not
τ_field (field charges as 1−e^(−t/τ_f); energy as its square).

## The falling eta still needs an explanation — but a different one

With the fixed estimator eta still falls (5.94e-5 at 0.4τ → 3.75e-5 at
2τ). The cleaner suspect: the SH resonance has its **own** photon
lifetime (Q_SH = 48426 → τ ~ 10⁴ tu). At T = 5057 the pump is ~87%
charged but the SH is nowhere near *its* steady state, and the ratio of
two differently-charging signals is not monotone. No steady-state claim
should be made from either existing record.

## Consequences

1. The "2τ = 75% of steady state" label is dead either way — but so is
   "2τ WAS steady state". Neither record supports a steady-state
   absolute claim yet.
2. Kerr run sizing stays based on τ_field ≈ 2513 tu (~12 h), **not**
   2.4 h. Do not launch until the envelope fit is reconciled.
3. The P² gate result stands: 3.772 (integrated-band), pump exponent
   exactly 2.00 — the second-order signature is solid, with a physical
   pump-depletion shortfall at 2A.
4. Every cross-record-length eta comparison in earlier notes is void;
   only same-length comparisons survive the estimator fix.

## Resolution (10:38)

The peer reproduced my Q independently (6050 vs my 5856 vs spectral
5887 — four derivations now agree) and found the exact mechanism of its
wrong ~494: the T=1000 record only covers 0.32 τ_field, and the charge
curve was normalized by the record's own endpoint — a still-climbing
curve normalized by itself looks saturated. Bonus trap found: the
extraction npz stores the loop response with the (t/|t|)² phase
correction already baked in; re-applying it doubles the phase slope and
inflates n_g (goes to the wiki bug pages).

The SH-lifetime explanation for the falling eta is adopted: at T=5057
the pump is ~87% charged but the SH (Q = 48426, τ ~ 10⁴ tu) is only at
~40% of *its* steady state — so neither existing record gives a
steady-state absolute.

**Decided plan (running now)** — after one more sizing correction, this
time mine: I quoted τ_SH ~ 10⁴ tu, which was the *energy* lifetime;
charging follows the *field* lifetime (2×), and the direct (t, a)
computation for the SH band gives τ_field = 18356 tu. A 3τ run is
therefore 67 h, not 10 h, and the cost curve has a knee (the last 15%
of SH charge costs 22 extra hours):

1. Absolute run at **T = 2.0·τ_SH = 36712 tu (~45 h)**: integrated-band
   estimator, stationary-tail measurement, labeled "eta at 74.8% SH
   charge", steady value quoted as eta/0.748 with the division stated.
   Added requirement: fit the SH envelope in the record itself with τ
   free — the fitted τ sets the final label AND arbitrates a still-open
   1.8× discrepancy in the SH Q (85562 direct vs 48426 assembled
   spectrum; the pump had four agreeing derivations, the SH does not
   yet).
2. Then Kerr ringdown at the original 12.4 h sizing (pump-resonance
   lifetimes; τ_field = 2513 was correct all along). Serialized, never
   overlapping on the GPU.
3. Then the 3D chi2 implementation (plan approved, rhs correction
   applied and numerically verified).

Phase 3 closes in ~2.4 days of serialized GPU time.

## Plan change (12:23) — 45 h run killed, faster method under investigation

USER verdict: too long. The 2τ run is killed before the descent into
multi-day territory. Now being evaluated instead: **nonlinear
single-pass + CMT assembly**, the same decomposition philosophy as the
already-validated linear half-ring method. The long run was only
measuring one unknown — the per-pass QPM conversion of the poled arc.
That is a transit-scale measurement (hundreds of tu, minutes–hours),
after which the steady-state ring absolute assembles analytically from
the measured (t, κ, a) of both bands: pump circulating enhancement
|κ_p|²/|1−t_p·a_p|², coherent two-half-arc SH generation, SH
out-coupling. The undepleted approximation is justified by the passed
P² gate. Validation gate with zero additional long runs: the same
formula, evaluated at the charge fractions of the two EXISTING records,
must reproduce both measured eta values — hit those two non-steady
points and the steady value is the same expression at charge = 1.
Kerr is also on hold until the user re-approves its timeline.
