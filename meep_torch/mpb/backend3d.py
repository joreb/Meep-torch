# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""ModeSolver wiring for the 3D planewave backend.

Kept out of :mod:`meep_torch.mpb.backends` so neither file grows past the
size limit, and because the 3D path shares almost nothing with the 2D
dispatch: a different rasterizer, a different operator, a symmetry sector
instead of a polarization, and no harminv fallback worth reaching for.

Why 3D defaults to the planewave backend where 2D defaults to FDFD. The
Bloch-FDTD path recovers frequencies from a time series with harminv,
which has no eigenvectors and infers the band index from a sorted
detection list -- it mis-numbers wherever a mode is missed or duplicated,
and in 3D the bands are dense enough that this is the normal case rather
than the exception. It also cannot do parity sectors or fields at all. So
3D uses planewave unless the caller has explicitly set ``ms.backend``,
which is detectable because ``backend`` is a CLASS attribute on ModeSolver
and assigning it puts an entry in the instance dict.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .parity import EVEN_Y, EVEN_Z, NO_PARITY, ODD_Y, ODD_Z
from .parity3d import MirrorParity
from .fields3d import planewave_fields_3d
from .planewave3d import (ThetaOperator, kvectors_3d, lattice_grid_3d,
                          planewave_bands_3d, reciprocal_duals_3d)
from .raster3d import eps_inv_from_geometry

__all__ = ["Backend3DMixin"]

# parity flag -> (mirror axis, even?)
_PARITY_AXIS = {EVEN_Z: (2, True), ODD_Z: (2, False),
                EVEN_Y: (1, True), ODD_Y: (1, False)}


def _symmetrize_eta(eta: np.ndarray, axis: int) -> np.ndarray:
    """Force the inverse-dielectric tensor to be exactly mirror-symmetric.

    Solving inside a mirror sector is only meaningful if the STRUCTURE has
    that mirror, and "the geometry is symmetric" is not the same statement
    as "the rasterized array is symmetric". Measured on the hole slab: the
    two z-planes holding the slab surface came out asymmetric by 8.8e-2 in
    epsilon, because the surface falls between sub-samples and the strict
    inequality in the mask does not land identically on the two sides.

    That is tiny and it is fatal. It leaves [M, Theta] at 3.8e-05 instead
    of round-off, the sector stops being an invariant subspace, and the
    eigensolve then stalls at a residual floor of 2e-3 that no number of
    iterations moves -- 1000 and 3000 iterations gave the same answer.

    A tensor mirrors as ``T eta(sigma r) T`` with ``T`` the reflection, not
    as ``eta(sigma r)``: the off-diagonal components touching the mirrored
    axis flip sign. Averaging with that is a projection onto the symmetric
    part, so it is idempotent and does nothing at all to a structure that
    was already symmetric.
    """
    idx = (-np.arange(eta.shape[axis])) % eta.shape[axis]
    T = np.eye(3)
    T[axis, axis] = -1.0
    mirrored = np.einsum("ij,...jk,kl->...il", T, np.take(eta, idx, axis=axis), T)
    return 0.5 * (eta + mirrored)


class Backend3DMixin:
    """The 3D planewave path, hung off ModeSolver."""

    #: Torch device for the 3D apply. "cpu" keeps the numpy reference path,
    #: which is what every committed gate was measured against; set to
    #: "cuda" (or read from MEEP_TORCH_DEVICE, as the examples do) to move
    #: the six FFTs and the tensor contraction onto the GPU.
    device = os.environ.get("MEEP_TORCH_DEVICE", "cpu")

    def _uses_planewave_3d(self) -> bool:
        """True unless the caller explicitly asked for another backend."""
        if "backend" in vars(self):
            return self.backend == "planewave"
        return True

    def _is_cross_section(self) -> bool:
        """``size = (0, y, z)``: a waveguide cross-section, not a 2D crystal.

        Both have two nonzero lattice axes, so counting them cannot tell
        them apart. What separates them is WHICH axis is collapsed and what
        k means on it: a 2D photonic crystal collapses z and takes k
        in-plane, while a cross-section collapses x and takes k_x as the
        propagation constant beta. The second is a 3D vector problem with
        one planewave along x, not a scalar TE/TM problem.
        """
        size = self.geometry_lattice.size
        return (float(size[0]) == 0 and float(size[1]) != 0
                and float(size[2]) != 0)

    def _use_3d_path(self) -> bool:
        return (len(self._active_axes()) == 3 or self._is_cross_section()) \
            and self._uses_planewave_3d()

    def _lattice_3d(self):
        """``(a1, a2, a3)`` cartesian cell vectors, rows of ``get_lattice``."""
        lat = self.get_lattice()
        return tuple(tuple(float(x) for x in lat[i]) for i in range(3))

    def _basis_3d(self):
        """The basis vectors object centres are measured along.

        ``b_i = basis_i * basis_size_i``, NOT the unit ``basis_i``. The two
        coincide whenever ``basis_size`` is 1, which is every fixture in
        the repo and the whole conventional-cell diamond gate -- and they
        do not coincide for MPB's own fcc primitive cell, which carries
        ``basis_size = sqrt(0.5)``.

        Measured with the unit vectors instead: the two spheres land at the
        wrong separation, the cell loses its cubic symmetry, and MPB's
        near-degenerate band PAIRS come back split -- 0.38051/0.38141
        became 0.36458/0.47355, 30% off at L, while the conventional cell
        stayed at 0.48%. Nothing else in the run looks wrong; the bands are
        simply another structure's.
        """
        lat = self.geometry_lattice
        return (lat.b1, lat.b2, lat.b3)

    def _grid_3d(self):
        """``(n1, n2, n3)``, with the mirrored axis forced odd under parity.

        A zero-size lattice axis becomes ``n = 1``: MPB treats a
        ``size=(0, y, z)`` cell as 3D with a single planewave along the
        invariant axis and ``k_x`` the propagation constant, which is the
        waveguide cross-section problem and falls out of the same operator
        with no special case.

        The odd forcing is not cosmetic. An even axis has a Nyquist
        frequency whose mirror partner is not in the basis, the projector
        has to drop it, and the sector stops being an invariant subspace of
        Theta -- measured, the commutator goes from 5e-16 to 0.35. One
        extra grid line restores the symmetry exactly and costs nothing
        that matters.
        """
        gs = self._adjust_grid_size()
        size = self.geometry_lattice.size
        n = [1 if float(size[i]) == 0 else int(gs[i]) for i in range(3)]
        flag = self._parity & (EVEN_Z | ODD_Z | EVEN_Y | ODD_Y)
        if flag in _PARITY_AXIS:
            axis = _PARITY_AXIS[flag][0]
            if n[axis] > 1 and n[axis] % 2 == 0:
                n[axis] += 1
        return tuple(n)

    def _eps_inv_3d(self):
        """Cached ``(eps_inv_tensor, eps_scalar)`` for the current grid."""
        grid = self._grid_3d()
        cached = getattr(self, "_eps3d_cache", None)
        if cached is not None and cached[0] == grid:
            return cached[1], cached[2]
        a1, a2, a3 = self._lattice_3d()
        eta, eps = eps_inv_from_geometry(
            a1, a2, a3, *grid, self.geometry, self.geometry_center,
            self.default_material, mesh_size=int(self.mesh_size),
            basis=self._basis_3d())
        self._eps3d_cache = (grid, eta, eps)
        return eta, eps

    def _eps_3d_for_reporting(self):
        """``(eps_tensor_on_grid, dx)`` in the shape the rest of the class
        expects from ``_rasterize_epsilon``.

        ``get_epsilon()`` and the examples' plots read this; the operator
        reads the inverse-dielectric tensor instead. Both come out of one
        call so the two cannot drift.
        """
        import torch
        _, eps = self._eps_inv_3d()
        a1 = np.asarray(self._lattice_3d()[0], dtype=float)
        dx = float(np.linalg.norm(a1)) / self._grid_3d()[0]
        return torch.as_tensor(eps, dtype=torch.float64), dx

    def _parity_projector_3d(self, op: ThetaOperator):
        """The mirror projector for ``self._parity``, or None."""
        flag = self._parity & (EVEN_Z | ODD_Z | EVEN_Y | ODD_Y)
        if flag == NO_PARITY:
            return None
        if flag not in _PARITY_AXIS:
            raise NotImplementedError(
                f"combined parity {flag} is not supported by the 3D "
                f"planewave backend; use one mirror at a time")
        axis, even = _PARITY_AXIS[flag]
        return MirrorParity(op, axis=axis, even=even)

    def _solve_k_3d(self, k):
        """Bands at one k-point through the planewave operator.

        Stores what the field getters and the parity/velocity reporters
        need: the eigenvectors, their angular frequencies, the operator
        itself (rebuilding it from k elsewhere is how two copies drift),
        and the projector that was in force.
        """
        a1, a2, a3 = self._lattice_3d()
        grid = self._grid_3d()
        eta, _ = self._eps_inv_3d()
        B, _ = reciprocal_duals_3d(a1, a2, a3)
        kc = 2 * np.pi * (np.asarray([float(x) for x in k], dtype=float) @ B)
        flag = self._parity & (EVEN_Z | ODD_Z | EVEN_Y | ODD_Y)
        if flag in _PARITY_AXIS:
            eta = _symmetrize_eta(eta, _PARITY_AXIS[flag][0])
        # The projector is bound to THIS operator, and it is applied twice
        # per operator apply, so it has to be on the same device. Built on
        # the CPU while the solve ran on the GPU it cancelled the entire
        # speedup: 153 s against 156 s, with the apply itself 4.2x faster.
        op = ThetaOperator(eta, kvectors_3d(*grid, B, kc), grid,
                           device=self.device)
        proj = self._parity_projector_3d(op)
        freqs, vecs = planewave_bands_3d(
            eta, a1, a2, a3, kc, self.num_bands, grid_shape=grid,
            parity=proj, return_vectors=True, device=self.device,
            tol=float(self.tolerance) if getattr(self, "tolerance", None) else 1e-7)
        self._op3d = op
        self._proj3d = proj
        self._vecs3d = vecs
        self._vecs = None                      # 2D field getters must not fire
        self._omegas = np.array([2 * math.pi * float(f) for f in freqs])
        self._kc3d = kc
        self._band_pol = None
        return [float(f) for f in freqs]

    # -- mode fields -------------------------------------------------------

    def _fields3d(self, which_band, bloch_phase=True):
        """``(E, D, H)`` for a 1-based band, or None if this was not a 3D run.

        Returning None rather than raising is what lets the 2D FieldsMixin
        keep its own error message: this mixin sits ahead of it in the MRO,
        so every getter has to be able to hand the call back.
        """
        if getattr(self, "_vecs3d", None) is None:
            return None
        nb = self._vecs3d.shape[1]
        if not 1 <= which_band <= nb:
            raise ValueError(
                f"band {which_band} out of range 1..{nb} (num_bands)")
        a1, a2, a3 = self._lattice_3d()
        _, volume = reciprocal_duals_3d(a1, a2, a3)
        r = None
        if bloch_phase:
            _, _, _, r = lattice_grid_3d(*self._grid_3d(), a1, a2, a3)
        return planewave_fields_3d(
            self._op3d, self._vecs3d[:, which_band - 1],
            float(self._omegas[which_band - 1]), volume,
            envelope=not bloch_phase, r=r)

    def _get3d(self, which_band, idx, ftype, bloch_phase):
        """One 3D getter: load curfield, then hand back what was asked.

        Returning the array WITHOUT loading curfield was the bug behind
        both "no field loaded to output" (output_hfield_z) and
        "needs a D, H or B field loaded first" (output_dpwr): every writer
        and every energy function reads curfield, not the return value.
        The 2D path gets this from _finish; the 3D getters bypassed it.
        """
        f = self._fields3d(which_band, bloch_phase=False)   # envelope
        if f is None:
            return None
        return self._finish3d(f[idx], ftype, which_band, bloch_phase)

    def _finish3d(self, env, ftype, which_band, bloch_phase):
        self._set_curfield(env, ftype, which_band)
        out = env * self._bloch_phase_grid()[..., None] if bloch_phase else env
        return self._as_mpb(out, bloch_phase)

    def get_efield(self, which_band, bloch_phase=True):
        f = self._get3d(which_band, 0, "e", bloch_phase)
        return f if f is not None else super().get_efield(which_band, bloch_phase)

    def get_dfield(self, which_band, bloch_phase=True):
        f = self._get3d(which_band, 1, "d", bloch_phase)
        return f if f is not None else super().get_dfield(which_band, bloch_phase)

    def get_hfield(self, which_band, bloch_phase=True):
        f = self._get3d(which_band, 2, "h", bloch_phase)
        return f if f is not None else super().get_hfield(which_band, bloch_phase)

    def get_bfield(self, which_band, bloch_phase=True):
        # mu = 1, so B is H; separate entry point because the type letter
        # drives compute_field_energy and the output filenames.
        f = self._get3d(which_band, 2, "b", bloch_phase)
        return f if f is not None else super().get_bfield(which_band, bloch_phase)

    def get_poynting(self, which_band):
        """MPB get_poynting: ``Re(conj(E) x H)``, the time-averaged flux.

        Curfield type "v" so the output writers treat it as a vector field
        and name the file ``flux.``. No factor of 1/2: MPB does not carry
        one either, and the examples compare components rather than
        absolute powers.
        """
        f = self._fields3d(which_band, bloch_phase=False)
        if f is None:
            return super().get_poynting(which_band)
        E, _, H = f
        S = np.real(np.cross(np.conj(E), H))
        return self._finish3d(S.astype(complex), "v", which_band, False)

    # -- MPB reporters that need the eigenvector ---------------------------

    def compute_zparities(self):
        """``<h|M_z|h>`` per band, MPB's ``display_zparities`` payload."""
        return self._compute_parities(2)

    def compute_yparities(self):
        return self._compute_parities(1)

    def _compute_parities(self, axis: int):
        if getattr(self, "_vecs3d", None) is None:
            # Not a 3D planewave run: hand back to the 2D layer so its own
            # NotImplementedError stands. Raising a different exception type
            # here would quietly change the contract for every 2D caller.
            return super().compute_yparities() if axis == 1 \
                else super().compute_zparities()
        return list(MirrorParity(self._op3d, axis=axis,
                                 even=True).expectation(self._vecs3d))

    def compute_group_velocity_component(self, direction):
        """Hellmann-Feynman ``v_n = <h| dTheta/dk . dir |h> / (2 omega_n)``.

        The derivative of the operator with respect to k is available in
        closed form -- ``Theta = A^H eta A`` and only ``A = K x`` depends on
        k -- so this is one extra apply per band and needs no finite
        difference and no second solve at a nearby k. Bands with
        ``omega = 0`` report 0 rather than dividing by it.
        """
        if getattr(self, "_vecs3d", None) is None:
            return super().compute_group_velocity_component(direction)
        op = self._op3d
        # meep's semantics: `direction` is in the RECIPROCAL basis, so a
        # step of 1 along component i is a step of 2pi b_i in cartesian k.
        # find_k passes a direction normalised in CARTESIAN space, which
        # only agrees with a reciprocal-basis unit vector on a cubic cell.
        d_frac = np.asarray([float(direction.x), float(direction.y),
                             float(direction.z)]
                            if hasattr(direction, "x") else direction,
                            dtype=float)
        B, _ = reciprocal_duals_3d(*self._lattice_3d())
        d = 2 * np.pi * (d_frac @ B)
        V = self._vecs3d
        H = op.to_cartesian(V)                        # (N_G, 3, nb)
        AH = np.cross(op.K[:, :, None], H, axisa=1, axisb=1, axisc=1)
        # d/dk (K x H) = d x H, so dTheta/dk = A_d^H eta A + A^H eta A_d.
        dH = np.cross(np.broadcast_to(d, op.K.shape)[:, :, None], H,
                      axisa=1, axisb=1, axisc=1)
        eta_AH = op._eta_apply(AH)
        eta_dH = op._eta_apply(dH)
        val = (np.einsum("gim,gim->m", dH.conj(), eta_AH)
               + np.einsum("gim,gim->m", AH.conj(), eta_dH))
        # <h|dTheta/dk|h> / (2 omega) is d(omega)/d(k_frac). meep reports
        # the velocity in units of c, i.e. d(f)/d(k_frac) with f = omega/2pi
        # and k in units of 2pi/a, so the 2pi comes back out. Checked
        # against central differences of the bands: the ratio was 6.28321
        # on every band and every axis before this factor.
        out = []
        for n, w in enumerate(self._omegas):
            out.append(0.0 if w <= 0
                       else float(np.real(val[n]) / (2 * w)) / (2 * np.pi))
        return out

    def compute_group_velocities(self):
        """One ``Vector3`` per band, in the lattice's own reciprocal basis."""
        if getattr(self, "_vecs3d", None) is None:
            return super().compute_group_velocities()
        from ..mp.vec import Vector3
        cols = [self.compute_group_velocity_component(
            Vector3(*(1.0 if j == i else 0.0 for j in range(3))))
            for i in range(3)]
        return [Vector3(cols[0][n], cols[1][n], cols[2][n])
                for n in range(len(self._omegas))]
