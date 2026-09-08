# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""MPB parity flags (mpb/src/maxwell/maxwell.h).

Their own module so the backend, field, raster and output mixins can name
a parity without importing ModeSolver, which imports them in turn.
``meep_torch.mpb`` re-exports the public names, so
``from meep_torch.mpb import ODD_Z`` is unaffected.
"""
NO_PARITY = 0
EVEN_Z = 1 << 0
ODD_Z = 1 << 1
EVEN_Y = 1 << 2
ODD_Y = 1 << 3

PARITY_NAMES = {NO_PARITY: "", EVEN_Z: "zeven", ODD_Z: "zodd",
                EVEN_Y: "yeven", ODD_Y: "yodd",
                EVEN_Y + EVEN_Z: "yeven, zeven",
                EVEN_Y + ODD_Z: "yeven, zodd",
                ODD_Y + EVEN_Z: "yodd, zeven",
                ODD_Y + ODD_Z: "yodd, zodd"}
