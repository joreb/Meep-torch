# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Band-solving backends behind ModeSolver, and the parity gate.

Split out of ``mpb/solver.py`` verbatim -- :class:`BackendMixin` is mixed
into :class:`~meep_torch.mpb.solver.ModeSolver`, so these are the same
methods on the same object.

Two families live here, and the split between them is about band
NUMBERING, not speed:

  2D    -> an FDFD Bloch eigensolve (``mpb.eigensolver``). The N lowest
           eigenvalues come out complete and ordered, so band n IS
           eigenvalue n.
  1D/3D -> Bloch-FDTD (``core.stepper_bloch2/3``) harvested by harminv,
           where the band index is a sort position among whatever
           resonances were detected, so a missed or duplicated mode
           renumbers every band above it.
"""
import itertools
import math

import numpy as np
import torch

from .eigensolver import bloch_bands_2d, lattice_gradients
from .parity import (EVEN_Y, EVEN_Z, NO_PARITY, ODD_Y, ODD_Z,
                     PARITY_NAMES)
from ..core.sources import GaussianPulse
from ..core.stepper_bloch2 import init_bloch2_state, step_bloch2_tm
from ..modes.harminv import harminv
from ..mp.geometry import reciprocal_to_cartesian
from ..mp.vec import Vector3

_SYM_TODO = "see wiki/tasks/port_symmetry-system.md"


class BackendMixin:
    """Band extraction; needs _eps, _parity, num_bands and the knobs."""

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
        if self._use_3d_path():
            # The planewave path solves inside a mirror sector, so the 3D
            # z-parities the FDTD backend could never do are available;
            # combined parities are not, and say so where they are asked
            # for rather than here.
            if p in (NO_PARITY, EVEN_Z, ODD_Z, EVEN_Y, ODD_Y):
                return
        if p == ODD_Z and ndims in (1, 2):
            return
        if p == EVEN_Z and ndims == 2:
            return                        # TE via the FDFD Bloch eigensolve
        if p == NO_PARITY and ndims in (1, 2, 3):
            return
        raise NotImplementedError(
            f"parity {PARITY_NAMES.get(p, p)!r} in {ndims}D is not backed: "
            "available: TM/zodd (1D/2D), TE/zeven (2D, FDFD Bloch "
            "eigensolve), no-parity (1D degenerate TM; 2D as the union "
            "of the two polarizations; 3D via stepper_bloch3). y-parities "
            "need a mirror-reduced cell")

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
        # 3D (and the size=(0,y,z) cross-section) go to the planewave
        # operator before eps is rasterized at all: that path uses its own
        # lattice-coordinate rasterizer and an inverse-dielectric TENSOR,
        # which _rasterize_epsilon does not produce.
        if self._use_3d_path():
            return self._solve_k_3d(k)
        if str(self.device) != "cpu" and not self._warned_device:
            # ModeSolver.device is read ONLY by the 3D planewave path. It is
            # a class attribute, so it exists on every solver and defaults
            # from MEEP_TORCH_DEVICE -- which the examples set globally.
            # Ignoring it silently means a 2D run looks GPU-accelerated and
            # is not, with nothing in the output to say so. Warn rather than
            # raise: a global MEEP_TORCH_DEVICE=cuda is a reasonable thing
            # to have set, and refusing would break every 2D example for a
            # setting the user did not aim at this solver.
            import warnings
            type(self)._warned_device = True
            warnings.warn(
                f"ModeSolver.device={self.device!r} is ignored for a "
                f"{len(self._active_axes())}D lattice: only the 3D planewave "
                f"backend has a device path. This run is on the CPU.",
                RuntimeWarning, stacklevel=2)
        eps, dx = self._eps
        kc = reciprocal_to_cartesian(Vector3(*k), self.geometry_lattice)
        if len(self._active_axes()) in (1, 2):
            # 2D goes through the FDFD Bloch eigensolve: the N lowest
            # eigenvalues are complete and ordered, so band n is eigenvalue
            # n and there is no band-numbering step to get wrong. The
            # harminv path below infers the index from a sorted detection
            # list and mis-assigns wherever a mode is missed or duplicated.
            if abs(kc.z) > 1e-12:
                raise NotImplementedError("2D lattice requires in-plane k")
            nx, ny = eps.shape
            a1, a2 = self._a1, self._a2
            # k as the user gave it: meep's k_points are in the RECIPROCAL
            # basis, and kc . a_i == 2 pi k_i (checked), so the fractional
            # components are the Bloch wrap phases directly and no
            # cartesian round trip is needed.
            # a 1D cell has no second reciprocal component to walk
            k1 = float(k[0])
            k2 = float(k[1]) if len(self._active_axes()) == 2 else 0.0

            if self.request_backend not in ("fdfd", "planewave"):
                raise ValueError(
                    f"backend must be 'fdfd' or 'planewave', got "
                    f"{self.request_backend!r}")

            def solve(pol):
                if self.request_backend == "fdfd":
                    return bloch_bands_2d(
                        eps, dx, dx, 0.0, 0.0, self.num_bands, pol,
                        differentiable=self.differentiable,
                        eps_te=self._eps_te, return_vectors=True,
                        target_freq=self.target_freq,
                        lattice=(a1, a2), k_frac=(k1, k2))
                from .planewave import (planewave_bands_2d,
                                        reciprocal_k_to_cartesian)
                eps_pw = (self._eps_diag[2] if pol == "tm"
                          else np.asarray(eps.cpu()))
                try:
                    return planewave_bands_2d(
                        eps_pw, a1, a2,
                        reciprocal_k_to_cartesian((k1, k2), a1, a2),
                        self.num_bands, pol,
                        eps_te=self._pw_eps_te(),
                        differentiable=self.differentiable,
                        target_freq=self.target_freq, return_vectors=True)
                except ValueError as exc:
                    if "DENSE_MAX" not in str(exc):
                        raise
                    raise NotImplementedError(
                        f"the planewave backend cannot take this case: "
                        f"{exc}\n\nSet ms.request_backend = \"fdfd\" to solve it "
                        f"with the finite-difference backend, which is "
                        f"matrix-free and has no such limit.") from exc

            if self._parity == NO_PARITY and len(self._active_axes()) == 1:
                # 1D: TE and TM are degenerate -- nothing distinguishes the
                # two in-plane directions when the structure varies along
                # one axis -- so the union would report every band TWICE.
                # meep.mpb reports each once (bragg_sine_mpb.json), and the
                # harminv path this replaced called the same case
                # "1D degenerate TM". Solve TM and label it.
                freqs, vecs = solve("tm")
                self._band_pol = ["tm"] * len(freqs)
            elif self._parity == NO_PARITY:
                # 2D no-parity is the UNION of the two polarizations. They
                # decouple completely in a z-invariant structure, so there
                # is no third operator to solve: run both and merge by
                # frequency. This is what makes a "complete" 2D gap -- a
                # gap in the union -- computable, which is the whole point
                # of run() in MPB's tutorial.
                ft, vt = solve("tm")
                fe, ve = solve("te")
                merged = ([(float(f), "tm", i) for i, f in enumerate(ft)]
                          + [(float(f), "te", i) for i, f in enumerate(fe)])
                merged.sort(key=lambda r: r[0])
                merged = merged[:self.num_bands]
                src = {"tm": (ft, vt), "te": (fe, ve)}
                freqs = [src[pol][0][i] for _, pol, i in merged]
                vecs = np.stack([src[pol][1][:, i] for _, pol, i in merged],
                                axis=1)
                self._band_pol = [pol for _, pol, _ in merged]
            else:
                pol = "te" if self._parity == EVEN_Z else "tm"
                freqs, vecs = solve(pol)
                self._band_pol = [pol] * len(freqs)
            # keep what the field getters need: the modes, their angular
            # frequencies, the k they belong to, and the very difference
            # operators the eigensolve used -- rebuilding those from k in
            # fields.py would risk the two drifting apart
            self._vecs = vecs
            self._omegas = np.array([2 * math.pi * float(f) for f in freqs])
            self._kc = kc
            self._k_frac = (k1, k2)
            gx, gy, _, _ = lattice_gradients(a1, a2, nx, ny, k1, k2)
            self._dxf, self._dyf = gx, gy
            # keep the tensor under differentiable=True: ms.freqs is the
            # only handle a caller has on the bands, so listifying it here
            # silently severs autograd for every optimiser driving epsilon
            # through the eigensolve. The scipy path is detached anyway.
            return freqs if self.differentiable else [float(f) for f in freqs]
        # harminv recovers frequencies from a time series; there is no
        # eigenvector to hand the field getters, so make that explicit
        # rather than leaving a stale set from an earlier 2D run
        self._vecs = self._omegas = None
        self._dxf = self._dyf = None
        self._band_pol = None
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

