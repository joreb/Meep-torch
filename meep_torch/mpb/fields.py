# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Per-band mode FIELDS for the 2D FDFD Bloch eigensolve.

:class:`FieldsMixin` is the half of meep's ModeSolver that needs
eigenvectors rather than eigenvalues: ``get_efield`` & co, the energy
density, MPB's canonical phase fix, and the ``output_*`` HDF5 writers.
:class:`~meep_torch.mpb.solver.ModeSolver` inherits it.

Conventions, all meep/MPB's: c = 1, mu = 1, time dependence
``exp(-i omega t)``. The eigenvector returned by the solver IS the Bloch
field F on the cell -- the wrap phase ``exp(i k L)`` lives in the
difference matrices, so the operator is the plain Laplacian and no
``exp(i k r)`` has been divided out. The periodic envelope is therefore
``F exp(-i k r)``, which is what ``bloch_phase=False`` returns and what
``_curfield`` stores; the output writers multiply the phase back in, as
meep's ``_output_vector_field`` does.

One deliberate divergence: meep's ``get_efield(b, bloch_phase=True)``
calls ``multiply_bloch_phase()`` on the STORED curfield before reading it
out, so in meep a phased fetch leaves a phased curfield and a following
``output_field_z()`` would apply the phase twice. (Its own band functions
all fetch with ``bloch_phase=False``, which is why that never bites
upstream.) Here ``_curfield`` is ALWAYS the envelope regardless of what
the caller asked for, so fetch-then-output is correct either way.

Derived fields come straight from Maxwell on the SAME difference
operators the eigensolve used, so the discrete curl identities hold to
round-off rather than to discretization order:

    TM (ODD_Z), eigenvector Ez:  B = H = curl E / (i omega)
        Hx = (Dyf Ez) / (i omega),  Hy = -(Dxf Ez) / (i omega)
    TE (EVEN_Z), eigenvector Hz:  D = i curl H / omega
        Dx = i (Dyf Hz) / omega,    Dy = -i (Dxf Hz) / omega

That is what makes ``integral eps|E|^2`` and ``integral |H|^2`` both come
out at exactly 1 from a single normalisation -- the eigen-equation forces
it. A field built by any other route would normalise to 1 too and be
wrong, so the energy check alone proves nothing; the residual against the
operator is the real gate (see tests/mp_parity/test_mpb_parity.py).

Only 2D lattices are backed. The 1D/3D harminv paths have no
eigenvectors at all and raise.
"""
from __future__ import annotations

import math

import numpy as np

from .eigensolver import te_edge_inv_eps
from .output import OutputMixin, _ENERGY_TYPES, _VECTOR_TYPES
from .parity import EVEN_Z, ODD_Z
from ..mp.vec import Vector3
from ..mp.verbosity_mgr import verbosity

class FieldsMixin(OutputMixin):
    """Mode fields and energies for a 2D ModeSolver.

    The HDF5 writers live in :mod:`meep_torch.mpb.output` and arrive
    through :class:`OutputMixin`; they are split out for file size only.
    """

    # -- state -------------------------------------------------------------
    # Set by ModeSolver.__init__ / _solve_k; declared here so the mixin is
    # readable on its own.
    _vecs = None            # (nx*ny, num_bands) complex, column-major
    _omegas = None          # (num_bands,) angular frequencies, 2*pi*f
    _kc = None              # cartesian Bloch vector (Vector3, meep units)
    _dxf = _dyf = None      # Bloch difference operators for the current k
    _band_pol = None        # per-band "tm"/"te" (2D no-parity union)
    _curfield = None        # (nx, ny, 3) complex, or (nx, ny) real energy
    _curfield_type = None
    _curfield_band = 0
    kpoint_index = 0

    # -- helpers -----------------------------------------------------------

    def _require_fields(self, name):
        if self._vecs is None:
            raise NotImplementedError(
                f"{name} needs per-band mode fields, which only the 2D FDFD "
                f"Bloch eigensolve produces. This run used the 1D/3D "
                f"Bloch-FDTD + harminv backend, which recovers frequencies "
                f"from a time series and has no eigenvectors at all. Run a "
                f"2D lattice, or use meep_torch.modes.fd1d/fd2d for guided "
                f"cross-sections")

    def _grid(self):
        eps, d = self._eps
        return eps.cpu().numpy(), float(d)

    def _band_field(self, which_band):
        """(v, omega) for a 1-based band index, v shaped (nx, ny)."""
        self._require_fields("field access")
        nb = self._vecs.shape[1]
        if not 1 <= which_band <= nb:
            raise ValueError(
                f"band {which_band} out of range 1..{nb} (num_bands)")
        eps, _ = self._grid()
        v = self._vecs[:, which_band - 1].reshape(eps.shape, order="F")
        return v, float(self._omegas[which_band - 1])

    def _lattice(self):
        """(a1, a2) of the 2D cell; falls back to the rectangular box."""
        a = getattr(self, "_a1", None)
        if a is not None:
            return self._a1, self._a2
        eps, d = self._grid()
        return (eps.shape[0] * d, 0.0), (0.0, eps.shape[1] * d)

    def _measure(self):
        """Cell area per grid point -- the quadrature weight of every
        integral here. dx*dy on a rectangular cell, |a1 x a2|/(nu nv) in
        general."""
        from .eigensolver import cell_measure
        eps, _ = self._grid()
        a1, a2 = self._lattice()
        return cell_measure(a1, a2, eps.shape[0], eps.shape[1])

    def _bloch_phase_grid(self):
        """exp(+i k.r) on the cell-centre sample points.

        In fractional coordinates this needs no basis at all: r = u a1 +
        v a2 and kc . a_i = 2 pi k_i, so kc . r = 2 pi (k1 u + k2 v). The
        k components are the ones the user passed in k_points.
        """
        eps, _ = self._grid()
        nu, nv = eps.shape
        k = self.current_k if self.current_k is not None else Vector3()
        u = (np.arange(nu) + 0.5) / nu - 0.5
        v = (np.arange(nv) + 0.5) / nv - 0.5
        return np.exp(2j * np.pi * (float(k.x) * u[:, None]
                                    + float(k.y) * v[None, :]))

    def _apply(self, op, v):
        """Difference operator on an (nx, ny) field, column-major like the
        assembled matrices."""
        eps, _ = self._grid()
        return (op @ v.reshape(-1, order="F")).reshape(eps.shape, order="F")

    def _eps_inv_apply(self, F):
        """E from D (or H from B, identity since mu = 1).

        Uses the operator's OWN edge weights for the in-plane components,
        via :func:`~meep_torch.mpb.eigensolver.te_edge_inv_eps`, so the
        energy density integrates to exactly 1 rather than to 1 plus a
        discretization error. The z component is a cell-centred scalar.
        """
        eps, _ = self._grid()
        out = np.zeros_like(F)
        if getattr(self, "backend", "fdfd") == "planewave":
            # spectral fields live at cell centres; the staggered edge
            # weights below belong to the finite-difference operator
            exx, eyy, ezz = (self._eps_diag
                             if getattr(self, "_eps_diag", None) is not None
                             else (eps, eps, eps))
            out[..., 0] = F[..., 0] / exx
            out[..., 1] = F[..., 1] / eyy
            out[..., 2] = F[..., 2] / ezz
            return out
        inv_for_ey, inv_for_ex = te_edge_inv_eps(eps, getattr(self, "_eps_te", None))
        out[..., 0] = F[..., 0] * inv_for_ex
        out[..., 1] = F[..., 1] * inv_for_ey
        out[..., 2] = F[..., 2] / eps
        return out

    def _set_curfield(self, arr, ftype, band):
        self._curfield = arr
        self._curfield_type = ftype
        self._curfield_band = band

    def _as_mpb(self, arr, bloch_phase):
        """Wrap for return, in meep's shape.

        meep pads the grid dims out to 3 and then appends the component
        axis (reference python/mpb/solver.py:415), so a 2D VECTOR field is
        (nx, ny, 1, 3), not (nx, ny, 3). Scalars keep the grid dims. The
        arrays are kept (nx, ny, 3) internally -- the extra axis would only
        get in the way of every eps broadcast -- and reshaped here, at the
        one boundary where the shape is part of the API.
        """
        from .solver import MPBArray
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], arr.shape[1], 1, arr.shape[2])
        return MPBArray(arr, self.get_lattice(), self.current_k,
                        bloch_phase=bloch_phase)

    def _finish(self, F, ftype, which_band, bloch_phase):
        """Store the envelope as curfield, hand back what the caller asked."""
        ph = self._bloch_phase_grid()[..., None]
        envelope = F * np.conj(ph)      # F exp(-i k r)
        self._set_curfield(envelope, ftype, which_band)
        return self._as_mpb(F if bloch_phase else envelope, bloch_phase)

    # -- field getters -----------------------------------------------------

    def _tm_fields(self, which_band):
        v, w = self._band_field(which_band)
        eps, _ = self._grid()
        E = np.zeros(eps.shape + (3,), dtype=np.complex128)
        E[..., 2] = v
        D = np.zeros_like(E)
        D[..., 2] = eps * v
        H = np.zeros_like(E)
        if w != 0.0:      # exact: _sorted_freqs zeroes the Gamma mode
            H[..., 0] = self._apply(self._dyf, v) / (1j * w)
            H[..., 1] = -self._apply(self._dxf, v) / (1j * w)
        return E, D, H

    def _te_fields(self, which_band):
        v, w = self._band_field(which_band)
        eps, _ = self._grid()
        H = np.zeros(eps.shape + (3,), dtype=np.complex128)
        H[..., 2] = v
        D = np.zeros_like(H)
        if w != 0.0:      # exact: _sorted_freqs zeroes the Gamma mode
            D[..., 0] = 1j * self._apply(self._dyf, v) / w
            D[..., 1] = -1j * self._apply(self._dxf, v) / w
        return self._eps_inv_apply(D), D, H

    def _planewave_band_fields(self, which_band):
        """(E, D, H) from the planewave backend's coefficients.

        planewave_fields_2d reconstructs on the SAME cell-centre grid this
        module uses (its lattice_grid), returns Bloch fields, and carries
        MPB's normalisation, so nothing here rescales. D is recovered as
        eps E at cell centres, which is exactly the relation that module
        used to build E from D -- not this module's staggered edge
        weights, which belong to the finite-difference operator.
        """
        from .planewave import planewave_fields_2d, reciprocal_k_to_cartesian
        self._require_fields("field access")
        pol = self._band_pol[which_band - 1]
        eps, _ = self._grid()
        a1, a2 = self._lattice()
        kc = reciprocal_k_to_cartesian(self._k_frac, a1, a2)
        freq = float(self._omegas[which_band - 1]) / (2 * np.pi)
        E, H = planewave_fields_2d(
            self._vecs[:, which_band - 1], eps, a1, a2, kc, freq,
            polarization=pol, eps_te=self._pw_eps_te())
        exx, eyy, ezz = (self._eps_diag if getattr(self, "_eps_diag", None)
                         is not None else (eps, eps, eps))
        D = np.empty_like(E)
        D[..., 0] = E[..., 0] * exx
        D[..., 1] = E[..., 1] * eyy
        D[..., 2] = E[..., 2] * ezz
        return E, D, H

    def _fields(self, which_band):
        if getattr(self, "backend", "fdfd") == "planewave":
            return self._planewave_band_fields(which_band)
        # Under 2D no-parity a band is TM or TE individually, so the
        # polarization is per BAND, not per run. MPB's no-parity fields are
        # the full vector field; ours are exactly TM-only or TE-only,
        # because the two polarizations decouple in a z-invariant
        # structure and each band belongs to one of them.
        pol = getattr(self, "_band_pol", None)
        te = (pol[which_band - 1] == "te" if pol
              else self._parity == EVEN_Z)
        return (self._te_fields(which_band) if te
                else self._tm_fields(which_band))

    def get_efield(self, which_band, bloch_phase=True):
        E, _, _ = self._fields(which_band)
        return self._finish(E, "e", which_band, bloch_phase)

    def get_dfield(self, which_band, bloch_phase=True):
        _, D, _ = self._fields(which_band)
        return self._finish(D, "d", which_band, bloch_phase)

    def get_hfield(self, which_band, bloch_phase=True):
        _, _, H = self._fields(which_band)
        return self._finish(H, "h", which_band, bloch_phase)

    def get_bfield(self, which_band, bloch_phase=True):
        # mu = 1 everywhere, so B and H are the same array; they are kept
        # as separate entry points because the field TYPE letter differs
        # and that is what compute_field_energy and the filenames key on.
        _, _, H = self._fields(which_band)
        return self._finish(H, "b", which_band, bloch_phase)

    def multiply_bloch_phase(self, arr=None):
        """meep multiply_bloch_phase: envelope -> full Bloch field."""
        ph = self._bloch_phase_grid()
        if arr is None:
            if self._curfield is None:
                raise RuntimeError("no current field to phase")
            self._curfield = self._curfield * (
                ph[..., None] if self._curfield.ndim == 3 else ph)
            return self._curfield
        arr = np.asarray(arr)
        return arr * (ph[..., None] if arr.ndim == 3 else ph)

    # -- energies ----------------------------------------------------------

    def compute_field_energy(self):
        """MPB compute_field_energy: replace curfield with its energy
        density and return ``[total, 6 component fractions]``.

        The density is ``sum_c Re(G_c)Re(F_c) + Im(G_c)Im(F_c)`` with
        ``G = eps^-1 F`` for a D field and ``G = F`` for H/B (mu = 1) --
        MPB fields.c compute_field_energy_internal. ``total`` is that
        summed times the cell volume over the point count, i.e. dx*dy in
        2D, and comes out at 1 for a normalised mode.
        """
        if self._curfield is None or self._curfield_type not in "dhb":
            raise RuntimeError(
                "compute_field_energy needs a D, H or B field loaded first "
                "(call get_dfield/get_hfield/get_bfield)")
        F = self._curfield
        G = self._eps_inv_apply(F) if self._curfield_type == "d" else F
        parts = []
        for c in range(3):
            parts.append(np.real(G[..., c]) * np.real(F[..., c]))
            parts.append(np.imag(G[..., c]) * np.imag(F[..., c]))
        density = sum(parts)
        total = float(density.sum())
        comp = [float(p.sum()) / (total if total else 1.0) for p in parts]
        self._set_curfield(density, self._curfield_type.upper(),
                           self._curfield_band)
        if verbosity.mpb > 0:
            fr = ", ".join(f"{comp[i] + comp[i - 1]}" for i in (1, 3, 5))
            print(f"{self._curfield_type}-energy-components:, "
                  f"{self.kpoint_index}, {self._curfield_band}, {fr}")
        return [total * self._measure()] + comp

    def _object_mask(self, objs):
        """Cells whose centre lies in one of ``objs`` (min-image periodic).

        MPB walks its objects last-to-first and breaks on the first hit.
        That precedence changes the ANSWER only for a "nothing" object
        (MATERIAL_TYPE_SELF), whose cells are skipped rather than counted;
        every other object contributes its cells either way. meep-torch
        geometry always carries a real material -- there is no
        MATERIAL_TYPE_SELF to express -- so the rule collapses to the union,
        and this is written as the union rather than as a precedence loop
        that could never take its interesting branch.

        Grid offset is a deliberate departure: MPB samples at
        ``i*d - L/2`` while this engine's epsilon lives on cell centres
        ``(i+0.5)*d - L/2`` (_rasterize_epsilon). The density being masked
        sits on the epsilon grid, so adopting MPB's offset would slide the
        mask half a cell off the very array it selects from.
        """
        from .raster import cell_centres, object_mask_2d
        eps, _ = self._grid()
        a1, a2 = self._lattice()
        X, Y = cell_centres(a1, a2, eps.shape[0], eps.shape[1])
        taken = np.zeros(eps.shape, dtype=bool)
        for obj in objs:
            taken |= object_mask_2d(obj, X, Y, a1, a2, self.geometry_center)
        return taken

    def compute_energy_in_objects(self, objs):
        """Fraction of the loaded energy density inside ``objs``."""
        if self._curfield is None or self._curfield_type not in _ENERGY_TYPES:
            raise RuntimeError(
                "compute_energy_in_objects needs an energy density loaded "
                "first (get_dfield/get_bfield then compute_field_energy)")
        if not objs:
            return 0.0
        return float(self._curfield[self._object_mask(objs)].sum()) \
            * self._measure()

    def get_dpwr(self, band):
        self.get_dfield(band, False)
        self.compute_field_energy()
        return self._as_mpb(self._curfield, False)

    def get_bpwr(self, band):
        self.get_bfield(band, False)
        self.compute_field_energy()
        return self._as_mpb(self._curfield, False)

    def get_tot_pwr(self, which_band):
        """meep get_tot_pwr: the SUM of the D and B energy densities.

        Not the average -- meep/python/mpb/solver.py does
        ``tot_pwr = epwr + hpwr`` and sets curfield type "R".
        """
        tot = np.asarray(self.get_dpwr(which_band)) + \
            np.asarray(self.get_bpwr(which_band))
        self._set_curfield(tot, "R", which_band)
        return self._as_mpb(tot, False)

    def fix_field_phase(self):
        """MPB fix_field_phase: canonical global phase and sign.

        Phase maximises the sum of squares of the real parts; the sign is
        then set from the LAST spatial entry whose |real part| is at least
        half the maximum -- not the maximum itself, which would be
        ambiguous for a field that oscillates within the cell.
        """
        if self._curfield_type is None or self._curfield is None:
            raise RuntimeError("fix_field_phase needs a vector field loaded")
        if self._curfield_type not in _VECTOR_TYPES:
            raise RuntimeError(
                f"fix_field_phase needs a vector field, not "
                f"{self._curfield_type!r}")
        f = self._curfield.reshape(-1)
        s0 = float((f.real ** 2 - f.imag ** 2).sum())
        s1 = float((2 * f.real * f.imag).sum())
        theta = 0.5 * math.atan2(-s1, s0)
        phase = complex(math.cos(theta), math.sin(theta))
        r = np.real(f * phase)
        maxabs = float(np.abs(r).max())
        sign = 1.0
        if maxabs > 0:
            hits = np.nonzero(np.abs(r) >= 0.5 * maxabs)[0]
            sign = -1.0 if r[hits[-1]] < 0 else 1.0
        rot = phase * sign
        self._curfield = self._curfield * rot
        # MPB rotates the EIGENVECTOR, not just the loaded field, so every
        # field derived afterwards carries the same canonical phase. Only
        # rotating curfield made combine_band_functions(fix_efield_phase,
        # output_efield_z) write an UNfixed field: the fix landed on a
        # curfield that the next get_efield immediately overwrote.
        if self._vecs is not None and self._curfield_band:
            self._vecs[:, self._curfield_band - 1] *= rot
        return self._curfield

