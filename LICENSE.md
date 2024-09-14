# Licensing

meep-torch is a **mixed-license** project. Each source file declares its license
with an `SPDX-License-Identifier` header; the license of ported code follows the
license of its source. Full license texts live in `LICENSES/`.

| Component | License | Origin |
| --- | --- | --- |
| Original meep-torch code (Phase 1 engine: `meep_torch/core/*`, tests) | MIT | this project |
| `meep_torch/modes/` FD solvers and transforms (planned, Phase 2) | MIT | tidy3d `components/mode/` (© Flexcompute / Tyler Hughes, ceviche-derived) |
| Reversible-gradient engine, S-param scaffold (planned, Q2/Phase 4) | MIT | fdtdx (© 2025 Yannik Mahlau) |
| Kottke smoothing translated from `maxwell_eps.c`, plane-wave port (planned, Phase 2/Q1) | GPL-2.0-or-later | MPB (NanoComp) |
| Overlap/effective-area calculators (if ported, Phase 4) | GPL-3.0-only | femwell (© Helge Gehring) |

Every ported file carries an attribution header naming the source project, its
original license, and the exact upstream commit it was ported from.

Note on distribution: files remain individually reusable under their declared
licenses. A distribution of the combined package that includes GPL-licensed
files must comply with the corresponding GPL terms for the combined work.
Downstream users who need a pure-MIT subset can take exactly the files marked
MIT.

Proprietary reference material (tidy3d-extras binary) is never included in any
form; it is used only as an external test oracle on machines where it is
licensed.
