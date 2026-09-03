# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""mp.Medium / susceptibilities / mp.PML -> meep-torch engine adapters.

Class signatures mirror meep/python/geom.py (Medium, Susceptibility family)
and meep/python/simulation.py (PML, Absorber). The engine consumes them via
the private adapters resolve() / to_pole(); meep parameters the engine cannot
honor are accepted and stored, and raise ValueError/NotImplementedError at
resolve() time with a message naming the limitation.
"""
import math
import warnings
from collections import namedtuple
from typing import Sequence

import numpy as np

from ..adjoint.problem import MaterialGrid as _EngineMaterialGrid
from ..core.materials import LorentzPole, Material
from .vec import Vector3

# side constants (meep mp.Low/High, mp.ALL); X/Y/Z live in mp/simulation.py
Low, High = 0, 1
ALL = -1

FreqRange = namedtuple("FreqRange", ["min", "max"])


def check_nonnegative(prop, val):
    if val >= 0:
        return val
    raise ValueError(f"{prop} cannot be negative. Got {val}")


def _tensor3(diag, offdiag) -> np.ndarray:
    """Real-symmetric 3x3 from meep's (diag, offdiag) Vector3 pair."""
    d, o = diag, offdiag
    return np.array([[d.x, o.x, o.y],
                     [o.x, d.y, o.z],
                     [o.y, o.z, d.z]], dtype=float)


def get_rotation_matrix(axis, theta) -> np.ndarray:
    """meep geom.get_rotation_matrix: 3x3 whose columns are the basis vectors
    rotated by theta (radians, right-hand) around axis."""
    cols = [Vector3(1, 0, 0).rotate(axis, theta),
            Vector3(0, 1, 0).rotate(axis, theta),
            Vector3(0, 0, 1).rotate(axis, theta)]
    return np.array([[c.x for c in cols],
                     [c.y for c in cols],
                     [c.z for c in cols]], dtype=float)


class Susceptibility:
    """meep geom.py Susceptibility(sigma_diag, sigma_offdiag, sigma). The
    engine supports scalar (isotropic) sigma only: an anisotropic sigma_diag
    or a nonzero sigma_offdiag raises NotImplementedError at to_pole().
    Subclasses adapt to an engine pole tuple via ``to_pole()`` (see
    coeffs._pole_recurrence)."""

    def __init__(self, sigma_diag=None, sigma_offdiag=None, sigma=None):
        sigma_diag = sigma_diag if sigma_diag is not None else Vector3()
        self.sigma_diag = Vector3(sigma, sigma, sigma) if sigma else Vector3(*sigma_diag)
        self.sigma_offdiag = (Vector3(*sigma_offdiag) if sigma_offdiag is not None
                              else Vector3())

    def transform(self, m):
        """Transform sigma to M sigma M^T / |det M| (meep Susceptibility.transform)."""
        m = np.asarray(m, dtype=float).reshape(3, 3)
        s = m @ _tensor3(self.sigma_diag, self.sigma_offdiag) @ m.T / abs(np.linalg.det(m))
        self.sigma_diag = Vector3(s[0, 0], s[1, 1], s[2, 2])
        self.sigma_offdiag = Vector3(s[0, 1], s[0, 2], s[1, 2])

    def _sigma_scalar(self) -> float:
        # private adapter: the engine takes one scalar sigma per pole
        d, o = self.sigma_diag, self.sigma_offdiag
        if o.x or o.y or o.z:
            raise NotImplementedError("anisotropic sigma_offdiag is not supported")
        if not (d.x == d.y == d.z):
            raise NotImplementedError("anisotropic sigma_diag is not supported")
        return float(d.x)

    def _sigma_matrix(self) -> np.ndarray:
        return _tensor3(self.sigma_diag, self.sigma_offdiag)


class LorentzianSusceptibility(Susceptibility):
    """meep LorentzianSusceptibility(frequency, gamma, sigma...):
    chi(f) = sigma * frequency^2 / (frequency^2 - f^2 - i*f*gamma)."""

    def __init__(self, frequency=0.0, gamma=0.0, **kwargs):
        super().__init__(**kwargs)
        self.frequency = frequency
        self.gamma = gamma

    def eval_susceptibility(self, freq):
        sigma = self._sigma_matrix()[np.newaxis, :, :]
        w2 = self.frequency * self.frequency
        if self.gamma == 0:
            return w2 / (w2 - freq * freq) * sigma
        return w2 / (w2 - freq * freq - 1j * self.gamma * freq) * sigma

    def to_pole(self) -> LorentzPole:
        return LorentzPole(fn=float(self.frequency), gamma=float(self.gamma),
                           sigma=self._sigma_scalar())


class DrudeSusceptibility(Susceptibility):
    """meep DrudeSusceptibility(frequency, gamma, sigma...):
    chi(f) = -sigma * frequency^2 / (f^2 + i*f*gamma)  (frequency = f_p).

    Maps to the engine's 4-tuple pole (fn=0, gamma, sigma, fc=f_p): the ADE
    recurrence with zero restoring frequency and coupling at the plasma
    frequency is exactly Drude (validated vs the 1D transfer matrix in
    tests/examples/test_peraxis_dispersion.py)."""

    def __init__(self, frequency=0.0, gamma=0.0, **kwargs):
        super().__init__(**kwargs)
        self.frequency = frequency
        self.gamma = gamma

    def eval_susceptibility(self, freq):
        sigma = self._sigma_matrix()[np.newaxis, :, :]
        w2 = self.frequency * self.frequency
        if self.gamma == 0:
            return -w2 / (freq * freq) * sigma
        return -w2 / (freq * (freq + 1j * self.gamma)) * sigma

    def to_pole(self) -> tuple:
        return (0.0, float(self.gamma), self._sigma_scalar(), float(self.frequency))


_GYRO_MSG = ("gyrotropic media exist only in the 1D engine stepper "
             "(meep_torch.core.stepper_1d, build_1d_coeffs(gyro=...)), which the mp "
             "Simulation shim does not reach; see wiki/tasks/port_gyrotropic-2d.md")
_NOISY_MSG = ("noisy susceptibilities are not ported; "
              "see wiki/tasks/port_noisy-susceptibilities.md")


class NoisyLorentzianSusceptibility(LorentzianSusceptibility):
    """meep NoisyLorentzianSusceptibility(noise_amp, ...): Lorentzian plus a
    Gaussian noise term in the P equation. noise_amp != 0 is unsupported."""

    def __init__(self, noise_amp=0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_amp = noise_amp

    def to_pole(self):
        if self.noise_amp:
            raise NotImplementedError(_NOISY_MSG)
        return super().to_pole()


class NoisyDrudeSusceptibility(DrudeSusceptibility):
    """meep NoisyDrudeSusceptibility(noise_amp, ...): Drude plus a Gaussian
    noise term in the P equation. noise_amp != 0 is unsupported."""

    def __init__(self, noise_amp=0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_amp = noise_amp

    def to_pole(self):
        if self.noise_amp:
            raise NotImplementedError(_NOISY_MSG)
        return super().to_pole()


class GyrotropicLorentzianSusceptibility(LorentzianSusceptibility):
    """meep GyrotropicLorentzianSusceptibility(bias, ...); zero bias degrades
    to the plain Lorentzian pole, nonzero bias is 1D-engine-only."""

    def __init__(self, bias=Vector3(), **kwargs):
        super().__init__(**kwargs)
        self.bias = bias

    def to_pole(self):
        if self.bias.x or self.bias.y or self.bias.z:
            raise NotImplementedError(_GYRO_MSG)
        return super().to_pole()


class GyrotropicDrudeSusceptibility(DrudeSusceptibility):
    """meep GyrotropicDrudeSusceptibility(bias, ...); zero bias degrades to
    the plain Drude pole, nonzero bias is 1D-engine-only."""

    def __init__(self, bias=Vector3(), **kwargs):
        super().__init__(**kwargs)
        self.bias = bias

    def to_pole(self):
        if self.bias.x or self.bias.y or self.bias.z:
            raise NotImplementedError(_GYRO_MSG)
        return super().to_pole()


class GyrotropicSaturatedSusceptibility(Susceptibility):
    """meep GyrotropicSaturatedSusceptibility(bias, frequency, gamma, alpha):
    linearized Landau-Lifshitz-Gilbert model, 1D-engine-only."""

    def __init__(self, bias=Vector3(), frequency=0.0, gamma=0.0, alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.frequency = frequency
        self.gamma = gamma
        self.bias = bias
        self.alpha = alpha

    def to_pole(self):
        raise NotImplementedError(_GYRO_MSG)


class Transition:
    """meep Transition: one atomic transition of a MultilevelAtom."""

    def __init__(self, from_level, to_level, transition_rate=0, frequency=0,
                 sigma_diag=Vector3(1, 1, 1), gamma=0, pumping_rate=0):
        self.from_level = check_nonnegative("from_level", from_level)
        self.to_level = check_nonnegative("to_level", to_level)
        self.transition_rate = transition_rate
        self.frequency = frequency
        self.sigma_diag = sigma_diag
        self.gamma = gamma
        self.pumping_rate = pumping_rate


class MultilevelAtom(Susceptibility):
    """meep MultilevelAtom(initial_populations, transitions): saturable
    gain/absorption. 1D-engine-only (build_1d_coeffs(mla=...))."""

    def __init__(self, initial_populations=None, transitions=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_populations = initial_populations or []
        self.transitions = transitions or []

    def to_pole(self):
        raise NotImplementedError(
            "MultilevelAtom exists only in the 1D engine stepper "
            "(meep_torch.core.stepper_1d, build_1d_coeffs(mla=...); see "
            "tests/examples/test_w575_multilevel.py), which the mp Simulation "
            "shim does not reach")


class _EpsMuAccessor(float):
    """Medium.epsilon / Medium.mu: a number to the shim (simulation.py and
    geometry.py paint float(medium.mu) into the mu map) and meep's accessor
    to user scripts (medium.epsilon(freq) -> 3x3 tensor)."""

    def __new__(cls, scalar, fn):
        self = super().__new__(cls, scalar)
        self._fn = fn
        return self

    def __call__(self, freq):
        return self._fn(freq)


class Medium:
    """meep geom.py Medium. Exact meep parameters, plus the meep-torch
    extensions ``eps_inf=`` (synonym for epsilon) and ``material=`` (wrap a
    dispersive engine Material); ``Medium(epsilon=<engine Material>)`` is
    also accepted for W1 compat."""

    def __init__(
        self,
        epsilon_diag=Vector3(1.0, 1.0, 1.0),
        epsilon_offdiag=Vector3(),
        mu_diag=Vector3(1.0, 1.0, 1.0),
        mu_offdiag=Vector3(),
        E_susceptibilities: Sequence[Susceptibility] = None,
        H_susceptibilities: Sequence[Susceptibility] = None,
        E_chi2_diag=Vector3(),
        E_chi3_diag=Vector3(),
        H_chi2_diag=Vector3(),
        H_chi3_diag=Vector3(),
        D_conductivity_diag=Vector3(),
        D_conductivity_offdiag=Vector3(),
        B_conductivity_diag=Vector3(),
        B_conductivity_offdiag=Vector3(),
        epsilon=None,
        index=None,
        mu=None,
        chi2=None,
        chi3=None,
        D_conductivity=None,
        B_conductivity=None,
        E_chi2=None,
        E_chi3=None,
        H_chi2=None,
        H_chi3=None,
        valid_freq_range=FreqRange(min=-math.inf, max=math.inf),
        eps_inf=None,
        material: Material = None,
    ):
        self.material = material
        if isinstance(epsilon, Material):
            self.material, epsilon = epsilon, None

        # scalar -> diag folding, exactly as meep geom.py (truthiness included)
        if epsilon:
            epsilon_diag = Vector3(epsilon, epsilon, epsilon)
        elif index:
            i2 = index * index
            epsilon_diag = Vector3(i2, i2, i2)
        elif eps_inf:
            epsilon_diag = Vector3(eps_inf, eps_inf, eps_inf)
        if mu:
            mu_diag = Vector3(mu, mu, mu)
        if D_conductivity:
            D_conductivity_diag = Vector3(D_conductivity, D_conductivity, D_conductivity)
        if B_conductivity:
            B_conductivity_diag = Vector3(B_conductivity, B_conductivity, B_conductivity)
        if E_chi2:
            E_chi2_diag = Vector3(E_chi2, E_chi2, E_chi2)
        if E_chi3:
            E_chi3_diag = Vector3(E_chi3, E_chi3, E_chi3)
        if H_chi2:
            H_chi2_diag = Vector3(H_chi2, H_chi2, H_chi2)
        if H_chi3:
            H_chi3_diag = Vector3(H_chi3, H_chi3, H_chi3)

        self.epsilon_diag = Vector3(*epsilon_diag)
        self.epsilon_offdiag = Vector3(*epsilon_offdiag)
        self.mu_diag = Vector3(*mu_diag)
        self.mu_offdiag = Vector3(*mu_offdiag)
        self.E_susceptibilities = list(E_susceptibilities or [])
        self.H_susceptibilities = list(H_susceptibilities or [])
        self.E_chi2_diag = Vector3(chi2, chi2, chi2) if chi2 else Vector3(*E_chi2_diag)
        self.E_chi3_diag = Vector3(chi3, chi3, chi3) if chi3 else Vector3(*E_chi3_diag)
        self.H_chi2_diag = Vector3(*H_chi2_diag)
        self.H_chi3_diag = Vector3(*H_chi3_diag)
        self.D_conductivity_diag = Vector3(*D_conductivity_diag)
        self.D_conductivity_offdiag = Vector3(*D_conductivity_offdiag)
        self.B_conductivity_diag = Vector3(*B_conductivity_diag)
        self.B_conductivity_offdiag = Vector3(*B_conductivity_offdiag)
        self.valid_freq_range = valid_freq_range
        # scalar mirrors the shim reads via getattr (mp/simulation.py,
        # mp/geometry.py); resolve() validates isotropy before they are used
        self.D_conductivity = float(self.D_conductivity_diag.x)
        self.B_conductivity = float(self.B_conductivity_diag.x)
        self.chi3 = float(self.E_chi3_diag.x)

    def __repr__(self):
        return "Medium()"

    # -- meep accessors ----------------------------------------------------
    @property
    def epsilon(self):
        return _EpsMuAccessor(self.epsilon_diag.x, lambda freq: self._get_epsmu(
            self.epsilon_diag, self.epsilon_offdiag, self.E_susceptibilities,
            self.D_conductivity_diag, self.D_conductivity_offdiag, freq))

    @property
    def mu(self):
        return _EpsMuAccessor(self.mu_diag.x, lambda freq: self._get_epsmu(
            self.mu_diag, self.mu_offdiag, self.H_susceptibilities,
            self.B_conductivity_diag, self.B_conductivity_offdiag, freq))

    def _get_epsmu(self, diag, offdiag, susceptibilities,
                   conductivity_diag, conductivity_offdiag, freq):
        # port of meep Medium._get_epsmu (numpy instead of meep.Matrix)
        if np.isscalar(freq):
            freqs = np.array(freq)[np.newaxis, np.newaxis, np.newaxis]
        else:
            freqs = np.squeeze(freq)[:, np.newaxis, np.newaxis]
        fr = self.valid_freq_range
        if np.min(np.squeeze(freqs)) < fr.min:
            raise ValueError(f"User specified frequency {np.min(np.squeeze(freqs))} is "
                             f"below the Medium's limit, {fr.min}.")
        if np.max(np.squeeze(freqs)) > fr.max:
            raise ValueError(f"User specified frequency {np.max(np.squeeze(freqs))} is "
                             f"above the Medium's limit, {fr.max}.")
        epsmu = _tensor3(diag, offdiag)[np.newaxis, :, :].astype(complex)
        for sus in susceptibilities:
            epsmu = epsmu + sus.eval_susceptibility(freqs)
        conductivity = _tensor3(conductivity_diag, conductivity_offdiag)[np.newaxis, :, :]
        if np.count_nonzero(conductivity) > 0:
            epsmu = (1 + 1j / (2 * np.pi * freqs) * conductivity) * epsmu
        return np.squeeze(epsmu)

    def transform(self, m):
        """Transform epsilon, mu, and all susceptibility sigmas to MxM^T/|det M|
        (meep Medium.transform); m is any 3x3 array-like."""
        m = np.asarray(m, dtype=float).reshape(3, 3)
        det = abs(np.linalg.det(m))
        eps = m @ _tensor3(self.epsilon_diag, self.epsilon_offdiag) @ m.T / det
        mu = m @ _tensor3(self.mu_diag, self.mu_offdiag) @ m.T / det
        self.epsilon_diag = Vector3(eps[0, 0], eps[1, 1], eps[2, 2])
        self.epsilon_offdiag = Vector3(eps[0, 1], eps[0, 2], eps[1, 2])
        self.mu_diag = Vector3(mu[0, 0], mu[1, 1], mu[2, 2])
        self.mu_offdiag = Vector3(mu[0, 1], mu[0, 2], mu[1, 2])
        for s in self.E_susceptibilities:
            s.transform(m)
        for s in self.H_susceptibilities:
            s.transform(m)

    def rotate(self, axis, theta):
        self.transform(get_rotation_matrix(axis, theta))

    # -- private engine adapter --------------------------------------------
    def _engine_check(self) -> float:
        """Validate that stored meep params the engine cannot honor are at
        their defaults; returns the scalar chi3."""
        def iso(v, what):
            if not (v.x == v.y == v.z):
                raise ValueError(
                    f"anisotropic {what} is not supported through the mp shim; the "
                    f"core engine takes an in-plane tensor directly "
                    f"(meep_torch.core.coeffs.build_te_coeffs(eps_tensor=...))")
            return float(v.x)

        def zero(v, what):
            if v.x or v.y or v.z:
                raise ValueError(f"nonzero {what} is not supported by the engine")

        iso(self.epsilon_diag, "epsilon_diag")
        zero(self.epsilon_offdiag, "epsilon_offdiag")
        iso(self.mu_diag, "mu_diag")
        zero(self.mu_offdiag, "mu_offdiag")
        if self.H_susceptibilities:
            raise ValueError("H_susceptibilities (magnetic dispersion) are not "
                             "supported by the engine")
        zero(self.E_chi2_diag, "chi2/E_chi2 (engine nonlinearity is Kerr chi3 only)")
        zero(self.H_chi2_diag, "H_chi2")
        zero(self.H_chi3_diag, "H_chi3")
        iso(self.D_conductivity_diag, "D_conductivity_diag")
        zero(self.D_conductivity_offdiag, "D_conductivity_offdiag")
        iso(self.B_conductivity_diag, "B_conductivity_diag")
        zero(self.B_conductivity_offdiag, "B_conductivity_offdiag")
        return iso(self.E_chi3_diag, "E_chi3_diag")

    def resolve(self) -> tuple:
        """Return (eps_value_or_Material, chi3) for the engine painters."""
        chi3 = self._engine_check()
        if self.material is not None:
            return self.material, chi3
        base = float(self.epsilon_diag.x)
        poles = tuple(s.to_pole() for s in self.E_susceptibilities)
        if poles:
            fr = self.valid_freq_range
            return Material(eps_inf=base, poles=poles,
                            valid_fmin=max(0.0, float(fr.min)),
                            valid_fmax=float(fr.max)), chi3
        return base, chi3


def DefaultPMLProfile(u):
    return u * u


class _PMLMeta(type):
    # mp/simulation.py dispatches `isinstance(bl, PML)` before its Absorber
    # elif; meep's Absorber(PML) subclassing would send absorbers down the
    # CPML branch. An Absorber is deliberately NOT a PML *instance* here
    # (issubclass stays True for meep parity).
    # ponytail: drop this metaclass once the shim checks Absorber first.
    def __instancecheck__(cls, obj):
        if cls is PML and isinstance(obj, Absorber):
            return False
        return super().__instancecheck__(obj)


class PML(metaclass=_PMLMeta):
    """meep mp.PML boundary layer. The engine applies CPML with a fixed
    profile on every cell edge: only the meep defaults (direction=ALL,
    side=ALL, R_asymptotic=1e-15, mean_stretch=1, quadratic profile) are
    accepted; anything else raises ValueError naming the limitation."""

    def __init__(self, thickness=None, direction=ALL, side=ALL,
                 R_asymptotic=1e-15, mean_stretch=1.0,
                 pml_profile=DefaultPMLProfile):
        if thickness is None:
            raise ValueError("PML thickness must be specified.")
        if R_asymptotic <= 0:
            raise ValueError(f"PML.R_asymptotic must be positive. Got {R_asymptotic}")
        if mean_stretch < 1.0:
            raise ValueError(f"PML.mean_stretch must be >= 1. Got {mean_stretch}")
        self.thickness = float(thickness)
        self.direction = direction
        self.side = side
        self.R_asymptotic = R_asymptotic
        self.mean_stretch = mean_stretch
        self.pml_profile = pml_profile
        name = type(self).__name__
        if direction != ALL or side != ALL:
            raise ValueError(
                f"{name} with direction/side is not supported: the engine "
                f"applies boundary layers on all cell edges (direction=ALL, side=ALL)")
        if mean_stretch != 1.0:
            raise ValueError(f"{name}.mean_stretch != 1 is not supported by the engine")
        if pml_profile is not DefaultPMLProfile:
            raise ValueError(f"custom {name}.pml_profile is not supported: the "
                             f"engine profile is fixed (see meep_torch.core.pml)")
        if R_asymptotic != 1e-15:
            raise ValueError(f"{name}.R_asymptotic is fixed in the engine "
                             f"(see meep_torch.core.pml); only the default 1e-15 is accepted")


class Absorber(PML):
    """meep mp.Absorber: drop-in for PML using a matched sigma_d = sigma_b
    quadratic conductivity ramp instead of coordinate stretching. Same
    signature as PML; applied on all cell edges."""


class MaterialGrid(_EngineMaterialGrid):
    """meep MaterialGrid(grid_size, medium1, medium2, weights, grid_type,
    do_averaging, beta, eta, damping).

    Engine mapping: medium1/medium2 resolve (Medium.resolve) to the scalar
    eps_min/eps_max endpoints of the engine design grid
    (meep_torch.adjoint.problem.MaterialGrid), so this object plugs straight
    into adjoint DesignRegion; beta/eta become the default tanh projection of
    adjoint OptimizationProblem when it is built over this grid (beta=0, the
    meep default, means no projection -> the engine default beta=1.0).
    ``weights`` follow meep: flat float64 in [0,1], zeros by default; use
    update_weights to replace them (or pass them as OptimizationProblem
    rho0). Stored only: do_averaging (design-grid subpixel smoothing),
    grid_type (overlapping-grid merge -- grids never overlap here), damping
    (meep's binarization penalty). Dispersive endpoints raise: the engine
    interpolates one scalar eps."""

    def __init__(self, grid_size, medium1, medium2, weights=None,
                 grid_type="U_DEFAULT", do_averaging=False, beta=0, eta=0.5,
                 damping=0):
        gs = Vector3(*grid_size)
        # meep: a ~0 component means "not varying along that axis" -> 1
        gs = Vector3(*[1 if abs(v) < 1e-9 else int(v) for v in (gs.x, gs.y, gs.z)])
        self.grid_size = gs
        self.num_params = int(gs.x * gs.y * gs.z)
        self.medium1 = medium1
        self.medium2 = medium2

        def scalar_eps(m):
            if isinstance(m, Medium):
                v, _ = m.resolve()
                if not isinstance(v, float):
                    raise NotImplementedError(
                        "dispersive MaterialGrid endpoints are not supported: "
                        "the engine interpolates a scalar eps (adjoint tier)")
                return v
            return float(m)

        shape = ((int(gs.x), int(gs.y)) if gs.z == 1
                 else (int(gs.x), int(gs.y), int(gs.z)))
        super().__init__(shape, eps_min=scalar_eps(medium1),
                         eps_max=scalar_eps(medium2))
        self.do_averaging = do_averaging
        self.beta = beta
        self.eta = eta
        self.damping = damping
        if weights is None:
            self.weights = np.zeros((self.num_params,))
        elif np.asarray(weights).size != self.num_params:
            raise ValueError(
                f"weights of shape {np.asarray(weights).size} do not match "
                f"user specified grid dimension: {self.grid_size}")
        else:
            self.weights = self.check_weights(
                np.asarray(weights).flatten().astype(np.float64))
        grid_type_dict = {"U_MIN": 0, "U_PROD": 1, "U_MEAN": 2, "U_DEFAULT": 3}
        if grid_type not in grid_type_dict:
            raise ValueError(
                f"Invalid grid_type: {grid_type}. Must be either U_MIN, "
                f"U_PROD, U_MEAN, or U_DEFAULT")
        self.grid_type = grid_type_dict[grid_type]
        self.swigobj = None

    def check_weights(self, w):
        """meep check_weights: clip to [0,1] with a warning when outside."""
        w = np.asarray(w)
        if np.amin(w) >= 0.0 and np.amax(w) <= 1.0:
            return w
        warnings.warn(
            "The weights parameter of MaterialGrid must be in the range [0,1].")
        return np.clip(w, 0.0, 1.0)

    def update_weights(self, x):
        """meep update_weights: replace the design weights in place."""
        x = np.asarray(x)
        if x.size != self.num_params:
            raise ValueError(
                f"weights of shape {self.weights.size} do not match user "
                f"specified grid dimension: {self.grid_size}")
        self.weights[:] = self.check_weights(x).flatten().astype(np.float64)
