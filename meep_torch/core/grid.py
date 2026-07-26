import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    dx: float
    courant: float = 0.5
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

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
