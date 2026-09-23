# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.mpb-shaped ModeSolver over two band backends.

Class surface mirrors meep/python/solver.py: constructor signature, the
run_* family, solve_kpoint, freqs/all_freqs/band_range_data/gap_list
bookkeeping, retrieve_gap.

  2D lattices -> FDFD Bloch eigensolve (mpb.eigensolver), both parities:
      run_tm / run_zodd  (Ez)   and   run_te / run_zeven  (Hz)
  1D / 3D     -> Bloch-FDTD steppers (core.stepper_bloch2/3) + harminv:
      run_tm / run_zodd : 1D
      run (NO_PARITY)   : 1D (polarizations degenerate -> TM path) and
                          3D (stepper_bloch3, point-source capture)

The split is about band *numbering*, not speed. An eigensolve returns the
N lowest eigenvalues complete and ordered, so band n is eigenvalue n.
Harminv instead reports whichever resonances it detects and the band index
is that list's sort position, so a missed or duplicated mode renumbers
every band above it -- measured against the committed meep.mpb oracle,
the 2D harminv path missed band 1 entirely at small |k| and duplicated it
into band 2's slot elsewhere. The remaining harminv paths (1D, 3D) carry
that caveat until the eigensolver grows a vector 3D operator.

FDFD is 2nd-order in the grid spacing, not spectral like MPB's planewave
basis, so oracle comparisons want resolution-aware tolerances rather than
golden equality (TM ~0.3%, TE ~2.6% at resolution 32; both converge).

Per-band mode FIELDS (get_efield & co, the energy density, MPB's phase
fix, the output_* writers) come from the 2D eigensolve's eigenvectors and
live in mpb/fields.py + mpb/band_functions.py. The 1D/3D harminv paths
recover frequencies from a time series and have no eigenvectors, so the
field getters raise there (for guided strip/slab cross-sections use
meep_torch.modes.fd1d/fd2d directly).

Backend knobs beyond meep's API (fmax, t_end, courant, differentiable,
...) are plain attributes documented on the class.
"""
import inspect
import itertools
import math
import time

import numpy as np
import torch

from .array import MPBArray
from .backend3d import Backend3DMixin
from .findk import FindKMixin
from .backends import BackendMixin
from .fields import FieldsMixin
from .parity import (EVEN_Y, EVEN_Z, NO_PARITY, ODD_Y, ODD_Z, PARITY_NAMES)
from .raster import RasterMixin, _eps_of

from ..core.sources import GaussianPulse
from ..core.stepper_bloch2 import init_bloch2_state, step_bloch2_tm
from ..core.grid import to_numpy
from ..modes.harminv import harminv
from ..mp.geometry import (Block, Cylinder, Lattice, Sphere,
                           reciprocal_to_cartesian)
from ..mp.material import Medium
from ..mp.vec import Vector3
from ..mp.verbosity_mgr import verbosity
from ..geometry.smoothing import blend_te_supersampled

# meep registers an 'mpb' flag on the shared verbosity singleton and
# gates every mpb print on `verbosity.mpb > 0`, which defaults to on.
# Gating on ModeSolver.verbose instead (default False) silently
# swallowed the band-range and gap lines the upstream examples print.
if not hasattr(verbosity, "mpb"):
    verbosity.add_verbosity_var(name="mpb")

inf = math.inf

_PARITY_NAMES = PARITY_NAMES

class ModeSolver(FindKMixin, Backend3DMixin, FieldsMixin, BackendMixin, RasterMixin):
    # Bloch-FDTD backend knobs (meep-torch extensions, not meep API):
    courant = 0.4        # dt = courant * dx (2D; the 3D stepper sets its own)
    t_end = 300.0        # run length per k-point
    t_transient = 60.0   # source transient dropped before harminv
    fmin = 0.02          # harminv band floor
    fmax = None          # harminv band ceiling; None -> empty-lattice estimate
    supersample = 4      # eps coverage supersampling per axis
    # FDFD eigensolver (2D): False -> scipy eigsh, fastest, detached;
    # True -> torch path with autograd through the eigensolve, so bands
    # are differentiable w.r.t. epsilon. Same frequencies either way.
    differentiable = False
    # Which 2D band solver runs: "fdfd" (mpb.eigensolver, 2nd-order finite
    # differences, the default and what every committed golden was measured
    # with) or "planewave" (mpb.planewave, spectral). Not a meep attribute
    # -- meep has one backend -- so it is set after construction:
    # ``ms.request_backend = "planewave"``. Read the OUTCOME back from
    # ``ms.active_backend``, which is not always what was requested: 3D and
    # a size=(0, y, z) cross-section run the planewave operator whatever
    # this says. Two names because they are two different facts -- what was
    # asked for, and what solved.
    request_backend = "fdfd"
    min_Q = 100.0        # harminv mode acceptance
    min_amp = 1e-9

    def __init__(
        self,
        resolution=10,
        is_negative_epsilon_ok=False,
        eigensolver_flops=0,
        eigensolver_flags=68,
        use_simple_preconditioner=False,
        force_mu=False,
        mu_input_file="",
        epsilon_input_file="",
        mesh_size=3,
        target_freq=0.0,
        tolerance=1.0e-7,
        num_bands=1,
        k_points=None,
        ensure_periodicity=True,
        geometry=None,
        geometry_lattice=None,
        geometry_center=Vector3(0, 0, 0),
        default_material=None,
        dimensions=3,
        random_fields=False,
        filename_prefix="",
        deterministic=False,
        verbose=False,
        optimize_grid_size=True,
        eigensolver_nwork=3,
        eigensolver_block_size=-11,
    ):
        self.resolution = resolution
        self.is_negative_epsilon_ok = is_negative_epsilon_ok
        self.eigensolver_flops = eigensolver_flops
        self.eigensolver_flags = eigensolver_flags
        self.use_simple_preconditioner = use_simple_preconditioner
        self.force_mu = force_mu
        self.mu_input_file = mu_input_file
        self.epsilon_input_file = epsilon_input_file
        self.mesh_size = mesh_size
        self.target_freq = target_freq
        self.tolerance = tolerance
        self.num_bands = num_bands
        self.k_points = k_points or []
        self.ensure_periodicity = ensure_periodicity
        self.geometry = geometry or []
        self.geometry_lattice = (Lattice() if geometry_lattice is None
                                 else geometry_lattice)
        self.geometry_center = Vector3(*geometry_center)
        self.default_material = (Medium(epsilon=1) if default_material is None
                                 else default_material)
        self.dimensions = dimensions
        self.random_fields = random_fields
        self.filename_prefix = filename_prefix
        self.deterministic = deterministic
        self.verbose = verbose
        self.optimize_grid_size = optimize_grid_size
        self.eigensolver_nwork = eigensolver_nwork
        self.eigensolver_block_size = eigensolver_block_size

        self.parity = ""
        self.iterations = 0
        self.all_freqs = None
        self.freqs = []
        self.band_range_data = []
        self.gap_list = []
        self.total_run_time = 0
        self.current_k = Vector3()
        self.k_split_num = 1
        self.k_split_index = 0
        self.eigensolver_iters = []

        self._parity = NO_PARITY
        self._eps = None
        self._eps_te = None

    # -- meep-exact resolution handling -----------------------------------

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, val):
        if isinstance(val, (int, float)):
            self._resolution = [val, val, val]
        elif isinstance(val, Vector3):
            self._resolution = [val.x, val.y, val.z]
        else:
            raise TypeError(
                f"resolution must be a number or a Vector3: Got {type(val)}")
        self._eps = None

    # -- grid sizing (port of meep's _get_grid_size machinery) -------------

    def _get_grid_size(self):
        size = self.geometry_lattice.size
        return [max(math.ceil(self.resolution[i] * size[i]), 1)
                for i in range(3)]

    def next_factor2357(self, n):
        def is_factor2357(n):
            def divby(n, p):
                return divby(n // p, p) if n % p == 0 else n

            return divby(divby(divby(divby(n, 2), 3), 5), 7) == 1

        if is_factor2357(n):
            return n
        return self.next_factor2357(n + 1)

    def _optimize_grid_size(self, grid_size):
        return [self.next_factor2357(g) for g in grid_size]

    def _adjust_grid_size(self):
        grid_size = self._get_grid_size()
        if self.optimize_grid_size:
            grid_size = self._optimize_grid_size(grid_size)
        return grid_size

    def _active_axes(self):
        """Indices of periodic (nonzero-size) lattice axes."""
        size = self.geometry_lattice.size
        return [i for i in range(3) if size[i] != 0]

    # -- epsilon rasterization ---------------------------------------------

    def _check_basis_orthogonal(self):
        """Reject a non-orthogonal (oblique) lattice basis.

        _rasterize_epsilon lays the unit cell out on a cartesian grid from
        geometry_lattice.size alone, and stepper_bloch2 is rectangular-only
        ("oblique basis deferred"). Handed basis1/basis2 at 60 degrees the
        pair silently solves a *different* structure: a triangular lattice
        came back with band 1 at K off by 27.75% against the committed
        meep.mpb oracle, with nothing to flag it.

        The physics is reachable today, just not through this entry point --
        tests/examples/test_mpb_track.py builds a rectangular sqrt(3) x 1
        supercell of the triangular lattice (_tri_eps) and gets Gamma/M/K
        within 2%. Native oblique support wants the reciprocal metric
        tensor, which arrives with the planewave eigensolver.
        """
        lat = self.geometry_lattice
        b = [lat.basis1, lat.basis2, lat.basis3]
        axes = self._active_axes()
        for i in axes:
            for j in axes:
                if i >= j:
                    continue
                dot = (b[i].x * b[j].x + b[i].y * b[j].y + b[i].z * b[j].z)
                if abs(dot) > 1e-9:
                    raise NotImplementedError(
                        f"oblique lattice basis (basis{i + 1}.basis{j + 1} = "
                        f"{dot:.4f}, not orthogonal) is not supported by the "
                        "Bloch-FDTD backend, which rasterizes the cell on a "
                        "cartesian grid and would silently solve a different "
                        "structure. Build a rectangular supercell of the "
                        "lattice instead (see _tri_eps in "
                        "tests/examples/test_mpb_track.py), or wait for the "
                        "planewave eigensolver")

    # -- meep run surface --------------------------------------------------

    def init_params(self, p, reset_fields):
        self._check_parity_supported(p)
        self._parity = p
        self.parity = self.get_parity_string()
        # The 3D planewave path owns its own rasterization: it needs an
        # inverse-dielectric TENSOR on a lattice-coordinate grid, which
        # _rasterize_epsilon does not produce, and it is not subject to
        # the orthogonal-basis guard that path carries -- the operator
        # only ever sees the reciprocal duals, so an oblique cell is
        # solved directly rather than through a rectangular supercell.
        if self._use_3d_path():
            self._eps = self._eps_3d_for_reporting()
            self._eps_te = None
        else:
            self._eps = self._rasterize_epsilon()
        # meep: kpoint_index counts solves within a run and shows up in
        # every output filename (e.g. e.k01.b02.z.tm.h5)
        self.kpoint_index = 0
        self._curfield = None
        self._curfield_type = None

    def set_parity(self, p):
        self._check_parity_supported(p)
        self._parity = p
        self.parity = self.get_parity_string()

    def solve_kpoint(self, k):
        if self._eps is None:
            self.init_params(self._parity, True)
        self.freqs = self._solve_k(k)
        self.kpoint_index += 1
        # MPB's solve_kpoint ends with curfield_reset: without it a density
        # computed at k=N stays loaded and output_field would write it out
        # under k=N+1's filename
        self._curfield = None
        self._curfield_type = None

    def get_freqs(self):
        return list(self.freqs)

    def run_parity(self, p, reset_fields, *band_functions):
        self.all_freqs = np.zeros((len(self.k_points), self.num_bands))
        self.band_range_data = []

        if self.verbose:
            print(f"Computing {self.num_bands} bands "
                  f"({len(self.k_points)} k-points)")

        self.init_params(p, reset_fields)
        self.eigensolver_iters = []
        t0 = time.time()

        for i, k in enumerate(self.k_points):
            self.current_k = k
            self.solve_kpoint(k)
            # meep records one entry per k-point. Both backends here reach
            # every band in a single shot -- a direct eigensolve, or one
            # FDTD run harvested by harminv -- so the count is 1 rather
            # than MPB's iterative-eigensolver sweep count.
            self.eigensolver_iters.append(1)
            # self.freqs may be a torch tensor (differentiable path);
            # all_freqs is a plain numpy record, so detach here rather
            # than letting np.array() try to convert element-wise
            self.all_freqs[i, :] = np.array([float(f) for f in self.freqs])
            self.band_range_data = self.update_band_range_data(
                self.band_range_data,
                [f for f in self.freqs if not math.isnan(f)], k)

            for f in band_functions:
                num_args = len(inspect.signature(f).parameters)
                if num_args == 1:
                    f(self)
                elif num_args == 2:
                    band = 1
                    while band <= self.num_bands:
                        f(self, band)
                        band += 1
                else:
                    raise ValueError(
                        "Band function should take 1 or 2 arguments. "
                        "The first must be a ModeSolver instance")

        self.total_run_time += time.time() - t0

        if len(self.k_points) > 1:
            self.output_band_range_data(self.band_range_data)
            self.gap_list = self.output_gaps(self.band_range_data)
        else:
            self.gap_list = []

    def display_kpoint_data(self, name, data):
        """meep display_kpoint_data: one CSV line per k-point, prefixed by
        the parity string (e.g. ``tefreqs:, 3, 0.1, ...``)."""
        if verbosity.mpb > 0:
            print(f"{self.parity}{name}:, {self.k_split_index}", end="")
            for d in data:
                print(f", {d}", end="")
            print()

    def display_eigensolver_stats(self):
        """meep display_eigensolver_stats: iteration/flop/time summary.

        Same arithmetic and output lines as meep/python/solver.py. Both
        backends solve each k-point in one shot, so the iteration count is
        1 per k rather than an iterative sweep count, and mean flops stays
        0 unless eigensolver_flops was set (meep behaves the same way).
        """
        num_runs = len(self.eigensolver_iters)
        if num_runs <= 0:
            return

        min_iters = min(self.eigensolver_iters)
        max_iters = max(self.eigensolver_iters)
        mean_iters = np.mean(self.eigensolver_iters)
        print(f"eigensolver iterations for {num_runs} kpoints: "
              f"{min_iters}-{max_iters}, mean = {mean_iters}", end="")

        sorted_iters = sorted(self.eigensolver_iters)
        idx1 = num_runs // 2
        idx2 = ((num_runs + 1) // 2) - 1
        median_iters = 0.5 * (sorted_iters[idx1] + sorted_iters[idx2])
        print(f", median = {median_iters}")

        mean_flops = self.eigensolver_flops / (num_runs * mean_iters)
        print(f"mean flops per iteration = {mean_flops}")

        mean_time = self.total_run_time / (mean_iters * num_runs)
        print(f"mean time per iteration = {mean_time} s")

    def run(self, *band_functions):
        self.run_parity(NO_PARITY, True, *band_functions)

    def run_zeven(self, *band_functions):
        self.run_parity(EVEN_Z, True, *band_functions)

    def run_zodd(self, *band_functions):
        self.run_parity(ODD_Z, True, *band_functions)

    def run_yeven(self, *band_functions):
        self.run_parity(EVEN_Y, True, *band_functions)

    def run_yodd(self, *band_functions):
        self.run_parity(ODD_Y, True, *band_functions)

    def run_yeven_zeven(self, *band_functions):
        self.run_parity(EVEN_Y + EVEN_Z, True, *band_functions)

    def run_yeven_zodd(self, *band_functions):
        self.run_parity(EVEN_Y + ODD_Z, True, *band_functions)

    def run_yodd_zeven(self, *band_functions):
        self.run_parity(ODD_Y + EVEN_Z, True, *band_functions)

    def run_yodd_zodd(self, *band_functions):
        self.run_parity(ODD_Y + ODD_Z, True, *band_functions)

    run_te = run_zeven
    run_tm = run_zodd
    run_te_yeven = run_yeven_zeven
    run_te_yodd = run_yodd_zeven
    run_tm_yeven = run_yeven_zodd
    run_tm_yodd = run_yodd_zodd

    # -- band-range / gap bookkeeping (ported from meep/python/solver.py) --

    def update_band_range_data(self, brd, freqs, kpoint):
        def update_brd(brd, freqs, br_start):
            if not freqs:
                return br_start + brd
            br = brd[0] if brd else ((inf, -1), (-inf, -1))
            br_rest = brd[1:] if brd else []
            newmin = (freqs[0], kpoint) if freqs[0] < br[0][0] else br[0]
            newmax = (freqs[0], kpoint) if freqs[0] > br[1][0] else br[1]
            new_start = br_start + [(newmin, newmax)]
            return update_brd(br_rest, freqs[1:], new_start)

        return update_brd(brd, freqs, [])

    def output_band_range_data(self, br_data):
        if verbosity.mpb > 0:
            fmt = "Band {} range: {} at {} to {} at {}"
            for tup, band in zip(br_data, range(1, len(br_data) + 1)):
                (min_freq, min_kpoint), (max_freq, max_kpoint) = tup
                print(fmt.format(band, min_freq, min_kpoint, max_freq,
                                 max_kpoint))

    def output_gaps(self, br_data):
        def ogaps(br_cur, br_rest, i, gaps):
            if not br_rest:
                gaps = list(reversed(gaps))
                return [(gaps[i + 2], gaps[i + 1], gaps[i])
                        for i in range(0, len(gaps), 3)]
            br_rest_min_f = br_rest[0][0][0]
            br_cur_max_f = br_cur[1][0]
            if br_cur_max_f >= br_rest_min_f:
                return ogaps(br_rest[0], br_rest[1:], i + 1, gaps)
            gap_size = (200 * (br_rest_min_f - br_cur_max_f)) / (
                br_rest_min_f + br_cur_max_f)
            if verbosity.mpb > 0:
                print(f"Gap from band {i} ({br_cur_max_f}) to band "
                      f"{i + 1} ({br_rest_min_f}), {gap_size}%")
            return ogaps(br_rest[0], br_rest[1:], i + 1,
                         [gap_size, br_cur_max_f, br_rest_min_f] + gaps)

        if not br_data:
            return []
        return ogaps(br_data[0], br_data[1:], 1, [])

    def retrieve_gap(self, lower_band):
        if lower_band + 1 > len(self.band_range_data):
            raise ValueError(
                "retrieve-gap called for higher band than was calculated")
        f1 = self.band_range_data[lower_band - 1][1][0]
        f2 = self.band_range_data[lower_band][0][0]
        return (f2 - f1) / (0.005 * (f1 + f2))

    # -- lattice / epsilon output ------------------------------------------

    def get_lattice(self):
        lat = self.geometry_lattice
        s = [v if v != 0 else 1.0 for v in lat.size]
        return np.array([[float(b[i]) * si for i in range(3)]
                         for b, si in zip((lat.b1, lat.b2, lat.b3), s)])

    def get_epsilon(self):
        if self._eps is None:
            self._eps = self._rasterize_epsilon()
        eps, _ = self._eps
        return MPBArray(eps.cpu().numpy(), self.get_lattice())

    def get_epsilon_tensor(self, c1, c2, imag=0, inv=False):
        """MPB get_epsilon_tensor: one component of the epsilon tensor.

        ``c1``/``c2`` are 0/1/2 for x/y/z. The 2D operators are built from a
        DIAGONAL epsilon, so every off-diagonal component is identically
        zero and ``epsilon_offdiag`` raises at construction. ``imag`` is
        accepted for signature parity and must be 0 --
        ``with_hermitian_epsilon()`` is False here.

        The diagonal is not three copies of ``get_epsilon()``: TE runs on
        the in-plane pair, which is the Kottke-smoothed diagonal on an
        isotropic rectangular cell and the per-component coverage average
        on an anisotropic one, while TM runs on the axial component. This
        is the one accessor that shows the structure each polarization
        actually sees.
        """
        if imag:
            raise NotImplementedError(
                "imaginary epsilon: with_hermitian_epsilon() is False")
        if not (0 <= c1 <= 2 and 0 <= c2 <= 2):
            raise ValueError(f"tensor components are 0/1/2, got {c1}, {c2}")
        # host boundary: MPBArray is numpy
        ezz = to_numpy(self.get_epsilon())
        if c1 != c2:
            return MPBArray(np.zeros_like(ezz), self.get_lattice())
        te = getattr(self, "_eps_te", None)
        diag = ([to_numpy(te[0]), to_numpy(te[1]), ezz]
                if te is not None else [ezz, ezz, ezz])
        out = 1.0 / diag[c1] if inv else diag[c1]
        return MPBArray(to_numpy(out), self.get_lattice())

    def allow_negative_epsilon(self):
        self.is_negative_epsilon_ok = True
        self.target_freq = 1 / inf

