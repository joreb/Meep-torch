# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math
import os
from dataclasses import dataclass, field

import torch


def default_device() -> str:
    """Device every engine grid lands on unless told otherwise.

    ``MEEP_TORCH_DEVICE`` decides; absent it, CUDA when the machine has it.

    The GPU is worth having -- 409.9 ms/step against 1.07 ms/step on the TM
    stepper at 400x400 float64 (RTX A4000, a 382x difference) -- and as of
    2026-09-21 it is the default when present, by explicit instruction.

    The cost of that choice, which has not gone away: a script's behaviour now
    depends on which machine ran it, and any caller building its own tensors
    next to a Grid gets CPU tensors meeting CUDA state one frame later
    ("Expected all tensors to be on the same device"). Reductions also sum in
    a different order on CUDA, so absolute numbers move in the last digits and
    a golden pinned at rel=1e-12 on CPU is not the same test here.

    So: export MEEP_TORCH_DEVICE=cpu to force CPU, or pass ``device=``
    explicitly (``Simulation(device=...)``, ``Grid(device=...)``).

    Resolved per Grid construction rather than cached, so a test can set the
    environment variable and build a Grid without a module reload.
    """
    env = os.environ.get("MEEP_TORCH_DEVICE")
    if env:
        return env
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    dx: float
    courant: float = 0.5
    dtype: torch.dtype = torch.float64  # Phase 1 default: correctness-first (spec amended); float32 opt-in, validated at scale in tests/autodiff/test_float32.py
    device: str = field(default_factory=default_device)

    def __post_init__(self):
        if self.nx < 3:
            raise ValueError(f"nx must be >= 3, got {self.nx}")
        if self.ny < 3:
            raise ValueError(f"ny must be >= 3, got {self.ny}")
        if self.dx <= 0:
            raise ValueError(f"dx must be > 0, got {self.dx}")
        if not 0 < self.courant <= 1 / math.sqrt(2):
            raise ValueError(
                f"courant must be in (0, 1/sqrt(2)] for 2D stability, got {self.courant}"
            )

    @property
    def dt(self) -> float:
        return self.courant * self.dx

    def zeros(self, *shape) -> torch.Tensor:
        return torch.zeros(*shape, dtype=self.dtype, device=self.device)
