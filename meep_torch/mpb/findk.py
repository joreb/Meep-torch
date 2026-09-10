# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""MPB's ``find_k``: solve for the wavevector at a fixed frequency.

The band solvers answer ``omega(k)``. Design work usually asks the inverse
question -- what is the propagation constant of this waveguide mode at
1.55 um -- and ``find_k`` is MPB's answer: for each band in a range, root-
find ``|k|`` along a fixed direction so that the band's frequency equals a
target ``omega``.

The signature is meep's, argument for argument
(``python/mpb/solver.py``), because the examples call it positionally:

    find_k(p, omega, band_min, band_max, korig_and_kdir, tol,
           kmag_guess, kmag_min, kmag_max, *band_funcs)

``korig_and_kdir`` is either a direction on its own or a
``[korig, kdir]`` pair; ``kmag_guess`` is a scalar or one value per band.
Bands are solved from ``band_max`` DOWNWARDS, which is not an arbitrary
order: solving band n requires ``num_bands = n``, so descending lets each
solve reuse the k found for the band above as its starting guess, and it
is the order meep uses.

The root find is Newton on ``f(k) - omega`` with the group velocity as the
derivative -- ``dω/dk`` is already available in closed form from the
Hellmann-Feynman apply, so no finite difference and no second solve -- with
a bisection safeguard so a bad derivative cannot walk outside
``[kmag_min, kmag_max]``. That is the same shape as MPB's
``find_root_deriv``.
"""
from __future__ import annotations

import numpy as np

from ..mp.geometry import cartesian_to_reciprocal, reciprocal_to_cartesian
from ..mp.vec import Vector3
from ..mp.verbosity_mgr import verbosity
from .band_functions import apply_band_func_thunk

__all__ = ["FindKMixin"]


def _newton_bisect(fn, lo: float, hi: float, x0: float, tol: float,
                   max_iter: int = 50):
    """Root of ``fn(x) -> (value, derivative)`` in ``[lo, hi]``.

    Newton where the step stays inside the known bracket, bisection where
    it does not. That is the same shape as MPB's ``find_root_deriv``.

    The bracket is tracked as the best point seen on EACH SIDE of zero,
    not as "below x" and "above x". Assuming ``f`` increases with ``k``
    reads a band with negative group velocity backwards and returns the
    search bound instead of the root -- checked against a decreasing test
    function, where the assuming version returned 0.1 for a root at 1.414.
    Frequency does usually rise with |k|, which is exactly why the wrong
    version would have survived every ordinary case.
    """
    def clamp(v):
        return min(max(v, lo), hi)

    x = clamp(float(x0))
    neg = pos = None                      # best (x, f) with f < 0 / f > 0
    for _ in range(max_iter):
        f, df = fn(x)
        if abs(f) < tol:
            return x
        if f < 0:
            if neg is None or abs(f) < abs(neg[1]):
                neg = (x, f)
        else:
            if pos is None or abs(f) < abs(pos[1]):
                pos = (x, f)

        step = None
        if df and np.isfinite(df):
            step = x - f / df
        if neg is not None and pos is not None:
            a, b = sorted((neg[0], pos[0]))
            if step is None or not (a < step < b) or not np.isfinite(step):
                step = 0.5 * (a + b)
        else:
            # No bracket yet: walk in the direction the derivative says the
            # root lies, falling back to a fixed nudge when it is useless.
            if step is None or not np.isfinite(step):
                direction = -1.0 if f > 0 else 1.0
                if df and np.isfinite(df) and df < 0:
                    direction = -direction
                step = x + direction * 0.1 * max(abs(x), 1.0)
            step = clamp(step)
        if abs(step - x) < tol * max(abs(x), 1.0):
            return step
        x = step
    return x


def _band_func_at(band, band_funcs):
    """A 2-argument band function that fires only on ``band``.

    Built by a factory rather than closing over the loop variable with a
    DEFAULT argument: run_parity counts the parameters to decide whether a
    band function is a thunk, and a third parameter -- even one with a
    default -- fails that check with "Band function should take 1 or 2
    arguments".
    """
    def bfunc(ms, which_band):
        if which_band == band:
            for fn in band_funcs:
                apply_band_func_thunk(ms, fn, band, True)
    return bfunc


class FindKMixin:
    """``find_k`` on ModeSolver."""

    def find_k(self, p, omega, band_min, band_max, korig_and_kdir, tol,
               kmag_guess, kmag_min, kmag_max, *band_funcs):
        num_bands_save = self.num_bands
        kpoints_save = self.k_points
        lat = self.geometry_lattice

        if isinstance(korig_and_kdir, (list, tuple)):
            korig, kdir = korig_and_kdir[0], korig_and_kdir[1]
        else:
            korig, kdir = Vector3(), korig_and_kdir
        # Normalise the direction in CARTESIAN space and come back: a unit
        # step in the reciprocal basis is not a unit step in k unless the
        # lattice is cubic, and every oblique cell would otherwise get a
        # kmag that means something different per axis.
        kdir1 = cartesian_to_reciprocal(
            reciprocal_to_cartesian(kdir, lat).unit(), lat)

        nb = band_max - band_min + 1
        k0s = list(kmag_guess) if isinstance(kmag_guess, (list, tuple)) \
            else [kmag_guess] * nb

        def rootfun(b):
            def _f(k):
                self.num_bands = b
                self.k_points = [korig + kdir1.scale(k)]
                self.run_parity(p, False)
                v = self.compute_group_velocity_component(kdir1)
                f = float(self.freqs[-1]) - omega
                if verbosity.mpb > 0:
                    print(f"find-k {b} at {k}: {f}")
                return f, float(v[-1])
            return _f

        ks = []
        for b in range(band_max, band_min - 1, -1):
            guess = k0s[min(b - band_min, len(k0s) - 1)]
            ks.append(_newton_bisect(rootfun(b), kmag_min, kmag_max,
                                     guess, tol))

        if band_funcs:
            for b, k in zip(range(band_min, band_max + 1), reversed(ks)):
                self.num_bands = b
                self.k_points = [korig + kdir1.scale(k)]
                self.run_parity(p, False, _band_func_at(b, band_funcs))

        self.num_bands = num_bands_save
        self.k_points = kpoints_save
        ks = list(reversed(ks))
        if verbosity.mpb > 0:
            print(f"{self.parity}kvals:, {omega}, {band_min}, {band_max}",
                  end="")
            for c in korig:
                print(f", {c}", end="")
            for c in kdir1:
                print(f", {c}", end="")
            for k in ks:
                print(f", {k}", end="")
            print()
        return ks
