# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Compatibility shim: the gdsfactory adapter now lives in
meep_torch.plugins.gdsfactory (gplugins-style plugin namespace)."""
from meep_torch.plugins.gdsfactory import component_to_eps  # noqa: F401
