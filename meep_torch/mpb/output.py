# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""HDF5 output for mpb mode fields, in meep's filenames and datasets.

Split out of ``mpb/fields.py`` verbatim -- :class:`OutputMixin` is mixed
into :class:`~meep_torch.mpb.fields.FieldsMixin`, so these are the same
methods on the same object and every file name, dataset name and
description string is unchanged.

The parity suffix follows MPB's ``parity_string`` (mpb/mpb.c): a 2D run
writes ``.tm``/``.te``, a 3D one ``.zodd``/``.zeven``. "Bloch wavevector"
goes out in the RECIPROCAL lattice basis, which is what MPB declares
("real output_k[3]; /* kvector in reciprocal lattice basis */"), not the
cartesian vector.
"""
import os
import re
import sys

import numpy as np

from .parity import EVEN_Z, ODD_Z, PARITY_NAMES
from ..mp.verbosity_mgr import verbosity

# meep field-type letters (mpb/fields.c): lowercase is a vector field,
# uppercase the energy density derived from it, 'n' the dielectric function.
_VECTOR_TYPES = "dhbecv"
_ENERGY_TYPES = "DHBR"
_COMPONENTS = ("x", "y", "z")


class OutputMixin:
    """meep's output_* surface; needs _curfield/_curfield_type/freqs."""

    # -- output ------------------------------------------------------------

    def get_parity_string(self):
        """MPB parity_string (mpb/mpb.c): a 2D run says "tm"/"te", a 3D one
        "zodd"/"zeven". This is the ``.tm``/``.te`` suffix on output files.
        """
        # MPB's rule is nz == 1, which a 1D cell also satisfies
        two_d = len(self._active_axes()) <= 2
        if self._parity == EVEN_Z:
            return "te" if two_d else "zeven"
        if self._parity == ODD_Z:
            return "tm" if two_d else "zodd"
        # never read self.parity here: it is SET from this method
        return PARITY_NAMES.get(self._parity, "")

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

    def _description(self, kind, energy=False):
        """meep's two description strings (reference solver.py:686, :722).

        The energy one is NOT the field one with a suffix: meep writes
        "D field energy density, kpoint 1, band 25, freq=0.375", where
        appending to the field string would give "D field, kpoint 1,
        band 25, freq=0.375 energy density".
        """
        f = self.freqs[self._curfield_band - 1]
        what = f"{kind} field energy density" if energy else f"{kind} field"
        return (f"{what}, kpoint {self.kpoint_index}, "
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
        t = self._curfield_type
        if t is None:
            raise RuntimeError("no field loaded to output")
        if t in _VECTOR_TYPES:
            self._output_vector_field(fname_prefix, component)
        elif t == "n" or t in _ENERGY_TYPES:
            self._output_scalar_field(fname_prefix)
        else:
            raise ValueError(f"Unkown field type: {t}")
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
        field = self.multiply_bloch_phase()
        with h5py.File(fname, "w") as f:
            f["description"] = self._description(t).encode()
            # MPB writes k in the RECIPROCAL LATTICE BASIS, not cartesian:
            # "real output_k[3]; /* kvector in reciprocal lattice basis */"
            # (mpb/fields.c output_field_to_file). current_k already is
            # that basis -- it is what the user passed in k_points. Writing
            # the cartesian vector put 0.1 in the file where meep has 0.5
            # on a 5x5 supercell.
            f["Bloch wavevector"] = np.array(
                [float(self.current_k.x), float(self.current_k.y),
                 float(self.current_k.z)])
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
            description = self._description(t, energy=True)
        fname = self._create_fname(fname, fname_prefix, t != "n")
        if verbosity.mpb > 0:
            print(f"Outputting {fname}...")
        data = np.asarray(self._curfield)
        with h5py.File(fname, "w") as f:
            f["description"] = description.encode()
            f["data"] = data
            self._write_lattice(f)
            if t == "n":
                # one source for the tensor, shared with get_epsilon_tensor:
                # xx/yy are NOT the scalar eps when the TE operator runs on
                # a smoothed or anisotropic diagonal
                for inv, name in ((False, "epsilon"), (True, "epsilon_inverse")):
                    for c1 in range(3):
                        for c2 in range(c1, 3):
                            key = f"{name}.{_COMPONENTS[c1]}{_COMPONENTS[c2]}"
                            f[key] = np.asarray(
                                self.get_epsilon_tensor(c1, c2, inv=inv))
