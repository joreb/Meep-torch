# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Test setup for `tests/` lives in the ROOT conftest.py.

The default-device fixture moved there so it also covers
`symbolic_test/`, the other declared testpath, which was silently
running without it. This file is kept as the signpost rather than
deleted, because `tests/conftest.py` is where anyone will look first.
"""
