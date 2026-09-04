# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep's C++ object model, reimplemented over the tensor engine.

Surface parity with the lowercase classes pymeep exposes from libmeep via
SWIG (``fields``, ``structure``, ``vec``, ``volume``, ``src_time``, ...) --
these are ports, not bindings: no FFI, no SWIG, no compiled meep. Split by
the same grouping meep uses: :mod:`core` (fields/structure/boundaries),
:mod:`geom` (vec/volume/symmetry math), :mod:`src` (sources/susceptibilities/
DFT monitors). Re-exported by name from :mod:`meep_torch.mp`.
"""
