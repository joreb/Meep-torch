# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Put hand-built test tensors on the same device the engine defaults to.

`core.grid.default_device()` returns CUDA when the machine has it (2026-09-21),
so a `Grid` lands on the GPU while a test's own
`torch.zeros(NX, NY, dtype=torch.float64)` lands on the CPU, and the two meet
one frame later as

    RuntimeError: Expected all tensors to be on the same device, but found at
    least two devices, cuda:0 and cpu!

-- 22 failures in tests/analytic/test_factory.py alone, and the same file
passes 66/66 on CPU, so it is the device and not the physics. The engine is
not at fault: torch raises here by design rather than transferring silently.

`torch.set_default_device` fixes it at the construction site instead of at
~20 call sites per file, and keeps the tests honest about which device they
are exercising: set MEEP_TORCH_DEVICE=cpu and the whole suite runs on CPU,
tensors included.
"""
import os

import pytest
import torch

from meep_torch.core.grid import default_device


@pytest.fixture(scope="session", autouse=True)
def _default_tensor_device():
    dev = default_device()
    if dev == "cpu":
        yield
        return
    torch.set_default_device(dev)
    try:
        yield
    finally:
        torch.set_default_device("cpu")


@pytest.fixture
def cpu_rng_pinned(monkeypatch):
    """CPU for gates whose hex constants are CPU-RNG values.

    torch's CPU and CUDA generators are different streams for the same seed, so
    a seeded bitwise gate starts from a different state on CUDA and its sums
    change sign, not their last bits. Both knobs move: torch's default device
    for the test's own tensors, and MEEP_TORCH_DEVICE for `Grid`, which asks
    `default_device()` and never consults torch's default.
    """
    monkeypatch.setenv("MEEP_TORCH_DEVICE", "cpu")
    prev = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(prev)
