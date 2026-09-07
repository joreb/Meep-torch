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
import os
import re
import sys

import numpy as np

from .eigensolver import te_edge_inv_eps
from ..mp.geometry import Block, Cylinder, Sphere
from ..mp.vec import Vector3
from ..mp.verbosity_mgr import verbosity

# meep field-type letters (mpb/fields.c): lowercase is a vector field,
# uppercase the energy density derived from it, 'n' the dielectric function.
_VECTOR_TYPES = "dhbecv"
_ENERGY_TYPES = "DHBR"
_COMPONENTS = ("x", "y", "z")


class FieldsMixin:
    """Mode fields, energies and HDF5 output for a 2D ModeSolver."""

    # -- state -------------------------------------------------------------
    # Set by ModeSolver.__init__ / _solve_k; declared here so the mixin is
    # readable on its own.
    _vecs = None            # (nx*ny, num_bands) complex, column-major
    _omegas = None          # (num_bands,) angular frequencies, 2*pi*f
    _kc = None              # cartesian Bloch vector (Vector3, meep units)
    _dxf = _dyf = None      # Bloch difference operators for the current k
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

    def _k_rad(self):
        """Bloch vector in radians per unit length (the operator's units)."""
        k = self._kc if self._kc is not None else Vector3()
        return 2 * math.pi * float(k.x), 2 * math.pi * float(k.y)

    def _bloch_phase_grid(self):
        """exp(+i k.r) on the cell centres the epsilon grid uses."""
        eps, d = self._grid()
        nx, ny = eps.shape
        kx, ky = self._k_rad()
        x = (np.arange(nx) + 0.5) * d - nx * d / 2.0
        y = (np.arange(ny) + 0.5) * d - ny * d / 2.0
        return np.exp(1j * (kx * x[:, None] + ky * y[None, :]))

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
        from .solver import MPBArray
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
        if w != 0.0:                    # omega = 0 at Gamma: Ez const, no H
            H[..., 0] = self._apply(self._dyf, v) / (1j * w)
            H[..., 1] = -self._apply(self._dxf, v) / (1j * w)
        return E, D, H

    def _te_fields(self, which_band):
        v, w = self._band_field(which_band)
        eps, _ = self._grid()
        H = np.zeros(eps.shape + (3,), dtype=np.complex128)
        H[..., 2] = v
        D = np.zeros_like(H)
        if w != 0.0:
            D[..., 0] = 1j * self._apply(self._dyf, v) / w
            D[..., 1] = -1j * self._apply(self._dxf, v) / w
        return self._eps_inv_apply(D), D, H

    def _fields(self, which_band):
        from .solver import EVEN_Z
        return (self._te_fields(which_band) if self._parity == EVEN_Z
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
        _, d = self._grid()
        self._set_curfield(density, self._curfield_type.upper(),
                           self._curfield_band)
        if verbosity.mpb > 0:
            fr = ", ".join(f"{comp[i] + comp[i - 1]}" for i in (1, 3, 5))
            print(f"{self._curfield_type}-energy-components:, "
                  f"{self.kpoint_index}, {self._curfield_band}, {fr}")
        return [total * d * d] + comp

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
        eps, d = self._grid()
        nx, ny = eps.shape
        L = [nx * d, ny * d]
        x = (np.arange(nx) + 0.5) * d - L[0] / 2.0
        y = (np.arange(ny) + 0.5) * d - L[1] / 2.0
        X, Y = np.meshgrid(x, y, indexing="ij")
        taken = np.zeros(eps.shape, dtype=bool)
        for obj in objs:
            c = obj.center - self.geometry_center
            dp = []
            for a, (g, i) in enumerate(((X, 0), (Y, 1))):
                v = g - c[i]
                dp.append(v - L[a] * np.round(v / L[a]))
            if isinstance(obj, (Cylinder, Sphere)):
                inside = sum(v ** 2 for v in dp) < obj.radius ** 2
            elif isinstance(obj, Block):
                inside = np.ones(eps.shape, dtype=bool)
                for a, i in enumerate((0, 1)):
                    half = obj.size[i] / 2
                    if not math.isinf(half):
                        inside &= np.abs(dp[a]) < half
            else:
                raise NotImplementedError(
                    f"{type(obj).__name__} is not supported by "
                    "compute_energy_in_objects (Cylinder/Sphere/Block only)")
            taken |= inside
        return taken

    def compute_energy_in_objects(self, objs):
        """Fraction of the loaded energy density inside ``objs``."""
        if self._curfield is None or self._curfield_type not in _ENERGY_TYPES:
            raise RuntimeError(
                "compute_energy_in_objects needs an energy density loaded "
                "first (get_dfield/get_bfield then compute_field_energy)")
        if not objs:
            return 0.0
        _, d = self._grid()
        return float(self._curfield[self._object_mask(objs)].sum()) * d * d

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
        if self._curfield is None or self._curfield_type not in _VECTOR_TYPES:
            raise RuntimeError("fix_field_phase needs a vector field loaded")
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
        self._curfield = self._curfield * (phase * sign)
        return self._curfield

    # -- output ------------------------------------------------------------

    def get_parity_string(self):
        """MPB parity_string (mpb/mpb.c): a 2D run says "tm"/"te", a 3D one
        "zodd"/"zeven". This is the ``.tm``/``.te`` suffix on output files.
        """
        from .solver import EVEN_Z, ODD_Z
        two_d = len(self._active_axes()) == 2
        if self._parity == EVEN_Z:
            return "te" if two_d else "zeven"
        if self._parity == ODD_Z:
            return "tm" if two_d else "zodd"
        return self.parity or ""

    def get_filename_prefix(self):
        if self.filename_prefix:
            return f"{self.filename_prefix}-"
        _, filename = os.path.split(sys.argv[0])
        return ("" if filename in ("ipykernel_launcher.py", "__main__.py")
                else re.sub(r"\.py$", "", filename) + "-")

    def _create_fname(self, fname, prefix, parity_suffix):
        parity_str = self.get_parity_string()
        suffix = f".{parity_str}" if parity_suffix and parity_str else ""
        return prefix + fname + suffix + ".h5"

    def _description(self, kind):
        f = self.freqs[self._curfield_band - 1]
        return (f"{kind} field, kpoint {self.kpoint_index}, "
                f"band {self._curfield_band}, freq={f:.6g}")

    def output_field(self):
        self.output_field_to_file(-1, self.get_filename_prefix())

    def output_field_x(self):
        self.output_field_to_file(0, self.get_filename_prefix())

    def output_field_y(self):
        self.output_field_to_file(1, self.get_filename_prefix())

    def output_field_z(self):
        self.output_field_to_file(2, self.get_filename_prefix())

    def output_epsilon(self):
        eps, _ = self._grid()
        self._set_curfield(eps, "n", self._curfield_band)
        self.output_field_to_file(-1, self.get_filename_prefix())

    def output_field_to_file(self, component, fname_prefix):
        if self._curfield_type in _VECTOR_TYPES:
            self._output_vector_field(fname_prefix, component)
        elif self._curfield_type in ("n",) or self._curfield_type in _ENERGY_TYPES:
            self._output_scalar_field(fname_prefix)
        else:
            raise ValueError(f"Unkown field type: {self._curfield_type}")
        self._curfield = None
        self._curfield_type = None

    def _write_lattice(self, h5file):
        h5file["lattice vectors"] = self.get_lattice()

    def _output_vector_field(self, fname_prefix, component):
        import h5py
        t = self._curfield_type
        fname = f"{t}.k{self.kpoint_index:02d}.b{self._curfield_band:02d}"
        if component >= 0:
            fname += f".{_COMPONENTS[component]}"
        fname = self._create_fname(fname, fname_prefix, True)
        if verbosity.mpb > 0:
            print(f"Outputting fields to {fname}...")
        kx, ky = self._k_rad()
        field = self.multiply_bloch_phase()
        with h5py.File(fname, "w") as f:
            f["description"] = self._description(t).encode()
            f["Bloch wavevector"] = np.array(
                [kx / (2 * math.pi), ky / (2 * math.pi), 0.0])
            self._write_lattice(f)
            for idx, c in enumerate(_COMPONENTS):
                if component >= 0 and idx != component:
                    continue
                f[f"{c}.r"] = np.real(field[..., idx])
                f[f"{c}.i"] = np.imag(field[..., idx])

    def _output_scalar_field(self, fname_prefix):
        import h5py
        t = self._curfield_type
        if t == "n":
            fname, description = "epsilon", "dielectric function, epsilon"
        else:
            fname = (f"{t.lower()}pwr.k{self.kpoint_index:02d}"
                     f".b{self._curfield_band:02d}")
            description = self._description(t) + " energy density"
        fname = self._create_fname(fname, fname_prefix, t != "n")
        if verbosity.mpb > 0:
            print(f"Outputting {fname}...")
        data = np.asarray(self._curfield)
        with h5py.File(fname, "w") as f:
            f["description"] = description.encode()
            f["data"] = data
            self._write_lattice(f)
            if t == "n":
                zero = np.zeros_like(data)
                for inv, name in ((False, "epsilon"), (True, "epsilon_inverse")):
                    vals = 1.0 / data if inv else data
                    for c1 in range(3):
                        for c2 in range(c1, 3):
                            key = f"{name}.{_COMPONENTS[c1]}{_COMPONENTS[c2]}"
                            f[key] = vals if c1 == c2 else zero
