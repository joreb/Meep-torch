# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
from dataclasses import dataclass
from typing import Sequence

from ..core.grid import Grid

SRC_OFFSET = 3  # cells between monitor plane (port.index) and TFSF source plane
# (TM). validate_ports uses this minimum. ModeSourceTE uses a larger offset
# (weakly-guided TE mode has a long evanescent tail).


@dataclass(frozen=True)
class Port:
    axis: str
    index: int
    lo: int
    hi: int
    direction: int
    n_modes: int = 1
    #: +1 even, -1 odd, 0 unfiltered -- meep's eig_parity Y bits.
    #: The band index counts only modes of this parity, as meep counts.
    parity: int = 0


def validate_ports(ports: Sequence[Port], grid: Grid, npml: int) -> None:
    for k, p in enumerate(ports):
        if p.axis not in ("x", "y"):
            raise ValueError(f"port {k}: axis must be 'x' or 'y', got {p.axis!r}")
        if p.direction not in (+1, -1):
            raise ValueError(f"port {k}: direction must be +1 or -1, got {p.direction}")
        if p.n_modes < 1:
            raise ValueError(f"port {k}: n_modes must be >= 1, got {p.n_modes}")
        n_axis = grid.nx if p.axis == "x" else grid.ny
        n_perp = grid.ny if p.axis == "x" else grid.nx
        if not p.lo < p.hi:
            raise ValueError(f"port {k}: empty span lo={p.lo}, hi={p.hi}")
        if p.lo < 0 or p.hi > n_perp:
            raise ValueError(f"port {k}: span [{p.lo},{p.hi}) outside grid (n={n_perp})")
        if not (npml + SRC_OFFSET <= p.index < n_axis - npml - SRC_OFFSET):
            raise ValueError(
                f"port {k}: plane index={p.index} inside/adjacent to PML "
                f"(need {npml + SRC_OFFSET} <= index < {n_axis - npml - SRC_OFFSET})"
            )
        if p.lo < npml or p.hi > n_perp - npml:
            raise ValueError(f"port {k}: span [{p.lo},{p.hi}) reaches into PML (npml={npml})")