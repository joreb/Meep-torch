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

from .eigensolver import bloch_bands_2d, bloch_d_matrices
from .fields import FieldsMixin

from ..core.sources import GaussianPulse
from ..core.stepper_bloch2 import init_bloch2_state, step_bloch2_tm
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

# mpb/src/maxwell/maxwell.h parity flags
NO_PARITY = 0
EVEN_Z = 1 << 0
ODD_Z = 1 << 1
EVEN_Y = 1 << 2
ODD_Y = 1 << 3

_PARITY_NAMES = {NO_PARITY: "", EVEN_Z: "zeven", ODD_Z: "zodd",
                 EVEN_Y: "yeven", ODD_Y: "yodd",
                 EVEN_Y + EVEN_Z: "yeven, zeven",
                 EVEN_Y + ODD_Z: "yeven, zodd",
                 ODD_Y + EVEN_Z: "yodd, zeven",
                 ODD_Y + ODD_Z: "yodd, zodd"}

_FIELDS_MSG = ("{} is not ported. The 2D eigensolve does export mode "
               "fields now (mpb/fields.py: get_efield/get_dfield/"
               "get_hfield/get_bfield, compute_field_energy, "
               "compute_energy_in_objects, fix_field_phase, output_*), so "
               "this one is missing for its own reason rather than for "
               "want of eigenvectors -- see wiki/tasks/"
               "port_mpb-eigensolver.md")


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


def _eps_of(material):
    val, _ = material.resolve()
    if not isinstance(val, float):
        raise NotImplementedError(
            "dispersive/tensor materials not supported by ModeSolver")
    return val


class ModeSolver(FieldsMixin):
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

    def _rasterize_epsilon(self):
        """Supersampled coverage eps over the unit cell (min-image periodic),
        rank = number of periodic axes. First-listed object wins, like MPB."""
        axes = self._active_axes()
        if not axes:
            raise ValueError("geometry_lattice.size has no nonzero component")
        self._check_basis_orthogonal()
        gs = self._adjust_grid_size()
        size = self.geometry_lattice.size
        n = [int(gs[i]) for i in axes]
        L = [float(size[i]) for i in axes]
        d = [L[a] / n[a] for a in range(len(axes))]
        if max(d) - min(d) > 1e-9 * max(d):
            raise NotImplementedError(
                "anisotropic grid cells (resolution*size mismatch across "
                "axes) not supported by the Bloch-FDTD backend")
        ss = self.supersample
        coords = [(-L[a] / 2 + (np.arange(n[a] * ss) + 0.5) * d[a] / ss)
                  for a in range(len(axes))]
        grids = np.meshgrid(*coords, indexing="ij")
        # world point per fine cell: active axes filled, collapsed axes at 0
        P = [grids[axes.index(i)] if i in axes else np.zeros(grids[0].shape)
             for i in range(3)]

        if callable(self.default_material) and not isinstance(
                self.default_material, Medium):
            fine = np.empty(grids[0].shape)
            it = np.nditer(fine, flags=["multi_index"])
            for _ in it:
                m = it.multi_index
                med = self.default_material(
                    Vector3(P[0][m], P[1][m], P[2][m]))
                fine[m] = _eps_of(med)
        else:
            fine = np.full(grids[0].shape, _eps_of(self.default_material))

        for obj in reversed(self.geometry):
            c = obj.center - self.geometry_center
            dp = []
            for a, i in enumerate(axes):
                v = grids[a] - c[i]
                dp.append(v - L[a] * np.round(v / L[a]))  # min-image wrap
            if isinstance(obj, (Cylinder, Sphere)):
                if isinstance(obj, Cylinder) and len(axes) == 3:
                    raise NotImplementedError(
                        "3D Cylinder rasterization not supported")
                mask = sum(v ** 2 for v in dp) < obj.radius ** 2
            elif isinstance(obj, Block):
                if (obj.e1 != Vector3(1, 0, 0) or obj.e2 != Vector3(0, 1, 0)
                        or obj.e3 != Vector3(0, 0, 1)):
                    raise NotImplementedError(
                        "rotated Block not supported in ModeSolver")
                mask = np.ones(grids[0].shape, dtype=bool)
                for a, i in enumerate(axes):
                    half = obj.size[i] / 2
                    if not math.isinf(half):
                        mask &= np.abs(dp[a]) < half
            else:
                raise NotImplementedError(
                    f"{type(obj).__name__} rasterization not supported "
                    "in ModeSolver (Cylinder/Sphere/Block only)")
            fine[mask] = _eps_of(obj.material)

        shape = []
        for a in range(len(axes)):
            shape += [n[a], ss]
        blocks = fine.reshape(shape)
        eps = blocks.mean(axis=tuple(range(1, 2 * len(axes), 2)))
        self._eps_te = (blend_te_supersampled(blocks)
                        if len(axes) == 2 else None)
        return torch.as_tensor(eps, dtype=torch.float64), d[0]

    # -- band extraction backends ------------------------------------------

    def _harvest(self, series, dt, fmax):
        freqs = []
        for row in series:
            t0 = int(round(self.t_transient / dt))
            freqs += [m.freq for m in harminv(row[t0:], dt, self.fmin, fmax)
                      if m.Q > self.min_Q and m.amp > self.min_amp]
        # the same band is captured once per probe: collapse near-identical
        # modes so the [:num_bands] truncation keeps distinct bands
        out = []
        for f in sorted(freqs):
            if out and f - out[-1] <= max(1e-4, 1e-3 * f):
                continue
            out.append(f)
        return out

    def _bloch2_freqs(self, eps, kx, ky, dt, dx, fmax):
        class _G:
            pass

        g = _G()
        g.nx, g.ny = eps.shape
        g.zeros = lambda *s: torch.zeros(*s, dtype=torch.float64)
        st = init_bloch2_state(g)
        mask = torch.zeros(*eps.shape, dtype=torch.float64)
        mask[g.nx // 3 + 1, g.ny // 5 + 2] = 1.0
        pulse = GaussianPulse(fmax / 2, fmax)
        n_steps = int(round(self.t_end / dt))
        probes = [(g.nx // 3 + 1, g.ny // 5 + 2),
                  (2 * g.nx // 3, g.ny // 2 + 1),
                  (g.nx // 2 - 2, 0), (g.nx - 3, g.ny // 3)]
        series = np.empty((len(probes), n_steps), dtype=complex)
        for t in range(n_steps):
            a = pulse.amplitude((t + 0.5) * dt)
            st = step_bloch2_tm(st, eps, dt, dx, kx, ky, src_c=a, src_s=0.0,
                                src_mask=mask)
            for p, (pi, pj) in enumerate(probes):
                series[p, t] = complex(st.Ez_c[pi, pj].item(),
                                       st.Ez_s[pi, pj].item())
        return self._harvest(series, dt, fmax)

    def _bloch3_freqs(self, eps, kx, ky, kz, dx, fmax):
        from ..core.stepper_bloch3 import (build_bloch3_coeffs,
                                           init_bloch3_state, step_bloch3)

        nx, ny, nz = eps.shape
        c = build_bloch3_coeffs(eps, dx, kx, ky, n_abs_z=0, kz=kz)
        st = init_bloch3_state(nx, ny, nz)
        pulse = GaussianPulse(fmax / 2, fmax)
        src = ("Ey", nx // 3 + 1, ny // 5 + 2, nz // 2)
        n_steps = int(round(self.t_end / c.dt))
        probes = [(nx // 3 + 1, ny // 5 + 2, nz // 2),
                  (2 * nx // 3, ny // 2 + 1, nz // 3),
                  (nx // 2, 2 * ny // 3, nz // 4 + 1)]
        series = np.empty((len(probes), n_steps), dtype=complex)
        for t in range(n_steps):
            st = step_bloch3(st, c, pulse.amplitude((t + 0.5) * c.dt), src)
            for p, (pi, pj, pk) in enumerate(probes):
                series[p, t] = st.Ey[pi, pj, pk].item()
        return self._harvest(series, c.dt, fmax)

    def _estimate_fmax(self, kc, eps_avg):
        """Empty-lattice estimate: 1.2x the num_bands-th folded free-photon
        frequency in the averaged medium."""
        axes = self._active_axes()
        size = self.geometry_lattice.size
        ranges = [(np.arange(-3, 4) / size[i]) if i in axes else [0.0]
                  for i in range(3)]
        k = np.array([kc.x, kc.y, kc.z])
        f = sorted(np.linalg.norm(k + np.array(G)) / math.sqrt(eps_avg)
                   for G in itertools.product(*ranges))
        nth = f[min(self.num_bands, len(f)) - 1]
        return 1.2 * nth + 0.05

    def _check_parity_supported(self, p):
        ndims = len(self._active_axes())
        if p == ODD_Z and ndims in (1, 2):
            return
        if p == EVEN_Z and ndims == 2:
            return                        # TE via the FDFD Bloch eigensolve
        if p == NO_PARITY and ndims in (1, 3):
            return
        raise NotImplementedError(
            f"parity {_PARITY_NAMES.get(p, p)!r} in {ndims}D is not backed: "
            "available: TM/zodd (1D/2D), TE/zeven (2D, FDFD Bloch "
            "eigensolve), no-parity (1D degenerate TM; 3D via "
            "stepper_bloch3). 2D no-parity would be the union of the two "
            "polarizations and is not wired; y-parities need a mirror-"
            "reduced cell")

    def _zero_bands_at_gamma(self, kc):
        """Number of omega = 0 modes at k = 0, which harminv cannot see.

        A zero-frequency mode is constant in time, so it contributes no
        oscillation to the probe series and the harmonic inversion never
        reports it. Left uncounted, every resonance that IS found lands one
        band index too low and the whole k-row is silently misnumbered --
        band 1 at Gamma comes back carrying band 2's frequency.

        TM (Ez = const) has exactly one such mode; the committed meep.mpb
        oracles agree (tri_rods_mpb.json / honey_rods_mpb.json both give
        gamma[0] = 0.0 with bands 2/3 degenerate just above). The 3D
        no-parity path has more than one and is not validated here, so it
        raises rather than guess a band numbering.
        """
        if max(abs(kc.x), abs(kc.y), abs(kc.z)) > 1e-12:
            return 0
        ndims = len(self._active_axes())
        if self._parity == ODD_Z and ndims in (1, 2):
            return 1
        if self._parity == NO_PARITY and ndims == 1:
            return 1                      # 1D degenerate TM path
        raise NotImplementedError(
            f"k = Gamma with parity {self.parity or 'none'!r} in {ndims}D: "
            "the omega = 0 mode multiplicity is not established for this "
            "path, so bands cannot be numbered (harminv sees no zero-"
            "frequency mode). Use a non-zero k, or run TM/zodd in 1D/2D")

    def _solve_k(self, k):
        eps, dx = self._eps
        kc = reciprocal_to_cartesian(Vector3(*k), self.geometry_lattice)
        if len(self._active_axes()) == 2:
            # 2D goes through the FDFD Bloch eigensolve: the N lowest
            # eigenvalues are complete and ordered, so band n is eigenvalue
            # n and there is no band-numbering step to get wrong. The
            # harminv path below infers the index from a sorted detection
            # list and mis-assigns wherever a mode is missed or duplicated.
            if abs(kc.z) > 1e-12:
                raise NotImplementedError("2D lattice requires in-plane k")
            nx, ny = eps.shape
            kx, ky = 2 * math.pi * kc.x, 2 * math.pi * kc.y
            freqs, vecs = bloch_bands_2d(
                eps, dx, dx, kx, ky, self.num_bands,
                "te" if self._parity == EVEN_Z else "tm",
                differentiable=self.differentiable,
                eps_te=self._eps_te, return_vectors=True,
                target_freq=self.target_freq)
            # keep what the field getters need: the modes, their angular
            # frequencies, the k they belong to, and the very difference
            # operators the eigensolve used -- rebuilding those from k in
            # fields.py would risk the two drifting apart
            self._vecs = vecs
            self._omegas = np.array([2 * math.pi * float(f) for f in freqs])
            self._kc = kc
            self._dxf, self._dyf = bloch_d_matrices(nx, ny, dx, dx, kx, ky)
            return [float(f) for f in freqs]
        # harminv recovers frequencies from a time series; there is no
        # eigenvector to hand the field getters, so make that explicit
        # rather than leaving a stale set from an earlier 2D run
        self._vecs = self._omegas = None
        self._dxf = self._dyf = None
        self._kc = kc
        n_zero = self._zero_bands_at_gamma(kc)
        fmax = self.fmax
        if fmax is None:
            fmax = self._estimate_fmax(kc, float(eps.mean()))
        ndims = len(self._active_axes())
        if ndims == 1:
            if abs(kc.y) > 1e-12 or abs(kc.z) > 1e-12:
                raise NotImplementedError("1D lattice requires k along x")
            eps2 = eps[:, None].expand(eps.shape[0], 4).clone()
            got = self._bloch2_freqs(eps2, 2 * math.pi * kc.x, 0.0,
                                     self.courant * dx, dx, fmax)
        elif ndims == 2:
            if abs(kc.z) > 1e-12:
                raise NotImplementedError("2D lattice requires in-plane k")
            got = self._bloch2_freqs(eps, 2 * math.pi * kc.x,
                                     2 * math.pi * kc.y,
                                     self.courant * dx, dx, fmax)
        else:
            got = self._bloch3_freqs(eps, 2 * math.pi * kc.x,
                                     2 * math.pi * kc.y, 2 * math.pi * kc.z,
                                     dx, fmax)
        got = [0.0] * n_zero + got
        got = got[:self.num_bands]
        return got + [math.nan] * (self.num_bands - len(got))

    # -- meep run surface --------------------------------------------------

    def init_params(self, p, reset_fields):
        self._check_parity_supported(p)
        self._parity = p
        self.parity = _PARITY_NAMES.get(p, "")
        self._eps = self._rasterize_epsilon()
        # meep: kpoint_index counts solves within a run and shows up in
        # every output filename (e.g. e.k01.b02.z.tm.h5)
        self.kpoint_index = 0
        self._curfield = None
        self._curfield_type = None

    def set_parity(self, p):
        self._check_parity_supported(p)
        self._parity = p
        self.parity = _PARITY_NAMES.get(p, "")

    def solve_kpoint(self, k):
        if self._eps is None:
            self.init_params(self._parity, True)
        self.freqs = self._solve_k(k)
        self.kpoint_index += 1

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
            self.all_freqs[i, :] = np.array(self.freqs)
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

    def allow_negative_epsilon(self):
        self.is_negative_epsilon_ok = True
        self.target_freq = 1 / inf

    # -- fields ------------------------------------------------------------
    # get_efield/get_dfield/get_hfield/get_bfield, multiply_bloch_phase,
    # fix_field_phase, compute_field_energy, compute_energy_in_objects,
    # get_dpwr/get_bpwr/get_tot_pwr and the output_* writers come from
    # FieldsMixin (mpb/fields.py), backed by the 2D eigensolve's vectors.
    # What is left below genuinely has no backing yet.

    def _no_fields(self, name):
        raise NotImplementedError(_FIELDS_MSG.format(name))

    def get_charge_density(self, which_band, bloch_phase=True):
        self._no_fields("get_charge_density")

    def get_poynting(self, which_band):
        self._no_fields("get_poynting")

    def get_eigenvectors(self, first_band, num_bands):
        self._no_fields("get_eigenvectors")

    def compute_group_velocities(self):
        self._no_fields("compute_group_velocities")

    def randomize_fields(self):
        self._no_fields("randomize_fields")
