# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.mpb's MPBArray: an ndarray that remembers its lattice.

Its own module purely for file size -- ``mpb/solver.py`` re-exports it, so
``from meep_torch.mpb.solver import MPBArray`` and
``from meep_torch.mpb import MPBArray`` both still work.

The attributes ride through views and slices, which is what
``__array_finalize__`` is for: ``MPBData.convert`` reads ``lattice`` and
``kpoint`` off whatever array it is handed, including a slice of one.
"""
import numpy as np


class MPBArray(np.ndarray):
    def __new__(cls, input_array, lattice, kpoint=None, bloch_phase=False):
        obj = np.asarray(input_array).view(cls)
        obj.lattice = lattice
        obj.kpoint = kpoint
        obj.bloch_phase = bloch_phase
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.lattice = getattr(obj, "lattice", None)
        self.kpoint = getattr(obj, "kpoint", None)
        self.bloch_phase = getattr(obj, "bloch_phase", False)
