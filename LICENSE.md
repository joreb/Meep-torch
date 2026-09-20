# Licensing

meep-torch is a **mixed-license** project. Each source file declares its license
with an `SPDX-License-Identifier` header; the license of ported code follows the
license of its source. Full license texts live in `LICENSES/`.

| Component | License | Origin |
| --- | --- | --- |
| Original meep-torch code (Phase 1 engine `meep_torch/core/*`; Phase 2 `meep_torch/geometry/*` and `meep_torch/modes/{fd1d,effindex,eigengrad,__init__}.py`; tests) | MIT | this project |
| `meep_torch/modes/{fd2d,fd2d_tensor,transforms,derivatives}.py` FD solvers and transforms (shipped, Phase 2) | MIT | tidy3d `components/mode/` (© Flexcompute / Tyler Hughes, ceviche-derived) |
| Reversible-gradient engine, S-param scaffold (planned, Q2/Phase 4) | MIT | fdtdx (© 2025 Yannik Mahlau) |
| Kottke diagonal-projected averaging `meep_torch/geometry/smoothing.py` (shipped, Phase 2) | MIT | original from-paper implementation of Kottke, Farjadpour & Johnson, Phys. Rev. E 77, 036611 (2008); not a translation of MPB source |
| Overlap/effective-area calculators (if ported, Phase 4) | GPL-3.0-only | femwell (© Helge Gehring) |

Every ported file carries an attribution header naming the source project, its
copyright holders, and its original license.

**The tidy3d row above is a per-file carve-out, not the project's license.**
tidy3d as a whole is LGPL-2.1 ("GNU LESSER GENERAL PUBLIC LICENSE Version 2.1,
February 1999", verified against the repository's `LICENSE`). Only
`components/mode/` carries its own MIT per-file license, because that code is
ceviche-derived; that is the only part the MIT row covers. Anything else in
tidy3d -- `components/source.py` among it -- is LGPL-2.1, and copying it into an
MIT file here would be the same mistake as copying from meep (GPL-2.0).

Reading another project to learn the PHYSICS is not porting. A formula carries
no license: the paraxial Gaussian beam in `meep_torch/core/sources.py` is
written from the standard relations and is original MIT code, even though meep
and tidy3d both implement the same beam. The line is whether the file is a
transcription of theirs or an implementation of the mathematics.

Note on distribution: files remain individually reusable under their declared
licenses. A distribution of the combined package that includes GPL-licensed
files must comply with the corresponding GPL terms for the combined work.
Downstream users who need a pure-MIT subset can take exactly the files marked
MIT. The package currently ships only MIT-licensed files; the pyproject license
expression will gain GPL terms if and when GPL-derived ports (femwell/MPB,
Phase 4+) actually land in the wheel.

Proprietary reference material (tidy3d-extras binary) is never included in any
form; it is used only as an external test oracle on machines where it is
licensed.
