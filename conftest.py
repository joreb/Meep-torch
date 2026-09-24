# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Process-wide test setup, for BOTH declared testpaths.

`pyproject.toml:77` declares `testpaths = ["tests", "symbolic_test"]`, and the
default-device fixture is a property of the PROCESS rather than of one
directory -- so it belongs here, at the root, not under `tests/`.

It lived in `tests/conftest.py` for two days and `symbolic_test/` never got
it. Every run in that window was `pytest tests/`, which collected two thirds
of the declared paths and was reported as "the suite". The cost: two
`symbolic_test` gates failed under the CUDA default with `Expected all tensors
to be on the same device` -- `torch.randn(...)` with no `device=` landing on
the CPU beside a CUDA `Grid` -- and nothing caught it, because nothing ran
them.
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
