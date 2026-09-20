# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""mp source objects -> meep-torch LineSource / ModeSource (via Simulation).

W1 ships basic semantics: GaussianSource -> GaussianPulse; ContinuousSource ->
CWRamp (faithful width/slowness/cutoff lands in W2). Component selects
polarization (Ez/Ex(1D) -> TM, Hz -> TE). EigenModeSource wraps ports/ModeSource.
"""
import cmath
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .coords import GridSpec, cell_index, cell_index_y, span_index_range, to_corner
from .vec import Vector3

# Component sentinels (subset meep exposes that the 34 examples use).
Ez = "Ez"
Ex = "Ex"
Ey = "Ey"
Hz = "Hz"
Hy = "Hy"
Hx = "Hx"
Dx = "Dx"
Dy = "Dy"
Dz = "Dz"
Bx = "Bx"
By = "By"
Bz = "Bz"
Dielectric = "Dielectric"
ALL_COMPONENTS = "ALL_COMPONENTS"

# meep enum sentinels; parity bits follow meep_torch.mpb.solver (MPB values)
AUTOMATIC = -1
NO_PARITY, EVEN_Z, ODD_Z, EVEN_Y, ODD_Y = 0, 1, 2, 4, 8

_EZ_COMPONENTS = {Ez, Ex}      # Ex used in 1D -> TM Ez in meep-torch
_HZ_COMPONENTS = {Hz, Hy, Ey}  # Ey drives the 2D TE (Hz) polarization


def pol_for(component: str) -> str:
    if component in _EZ_COMPONENTS:
        return "tm"
    if component in _HZ_COMPONENTS:
        return "te"
    raise ValueError(f"unsupported component {component!r}")


class _Shifted:
    """Delayed time profile: amplitude 0 before ``start_time``, then the
    wrapped profile shifted so its t=0 lands at ``start_time``. Other
    attributes (fcen, t_end, ...) forward to the wrapped profile."""

    def __init__(self, inner, start_time: float):
        self._inner = inner
        self.start_time = float(start_time)

    def amplitude(self, t: float) -> float:
        if t < self.start_time:
            return 0.0
        return self._inner.amplitude(t - self.start_time)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Quadrature:
    """The 90-degree-shifted carrier of a real cos-phase profile (env*sin
    instead of env*cos), for injecting the imaginary part of a complex
    spatial amplitude into the real-field engine: Re[a * env * e^{-iwt}] =
    Re(a)*env*cos(wt) + Im(a)*env*sin(wt)."""
    # ponytail: quarter-period carrier delay also shifts the envelope by
    # 1/(4 fcen) -- negligible for narrowband pulses (envelope width 1/fwidth
    # >> period); evaluate env and carrier separately if broadband complex
    # sources ever matter

    def __init__(self, inner, fcen: float):
        self._inner = inner
        self._delay = 1.0 / (4.0 * float(fcen))

    def amplitude(self, t: float) -> float:
        return self._inner.amplitude(t - self._delay)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Windowed:
    """meep custom_src_time semantics: src_func evaluated at t (NOT shifted)
    inside [start_time, end_time], 0 outside."""

    def __init__(self, inner, start_time: float, end_time: float):
        self._inner = inner
        self.start_time = float(start_time)
        self.end_time = float(end_time)

    def amplitude(self, t: float) -> float:
        if self.start_time <= t <= self.end_time:
            return self._inner.amplitude(t)
        return 0.0

    def __getattr__(self, name):
        return getattr(self._inner, name)


class SourceTime:
    """meep source.py SourceTime: ``is_integrated`` only (stored; True raises
    NotImplementedError at profile() -- the engine injects soft current
    sources, there is no integrated d(current)/dt path). Subclasses supply
    ``_base_profile()`` and start/end-time gating via profile()."""

    def __init__(self, is_integrated=False):
        self.is_integrated = is_integrated

    def _checked_base_profile(self):
        if self.is_integrated:
            raise NotImplementedError(
                "is_integrated=True is not supported: the engine injects soft "
                "current sources only (no integrated d(current)/dt path)")
        return self._base_profile()


class GaussianSource(SourceTime):
    """meep GaussianSource(frequency, width, fwidth, start_time, cutoff,
    is_integrated, wavelength) -> engine GaussianPulse (real cos-modulated
    Gaussian, peak at start_time + cutoff*width). All params supported."""

    def __init__(self, frequency=None, width=0, fwidth=float("inf"),
                 start_time=0, cutoff=5.0, is_integrated=False, wavelength=None):
        if frequency is None and wavelength is None:
            raise ValueError("GaussianSource requires either frequency or wavelength")
        super().__init__(is_integrated=is_integrated)
        self.frequency = 1.0 / wavelength if wavelength else float(frequency)
        self.width = max(width, 1.0 / fwidth)
        self.start_time = start_time
        self.cutoff = cutoff

    @property
    def _fw(self):
        # private adapter: effective fwidth for the engine / EigenModeSource
        return 1.0 / float(self.width)

    def _base_profile(self):
        from ..core.sources import GaussianPulse
        if self.width == 0:
            raise ValueError("GaussianSource needs width or fwidth")
        return GaussianPulse(self.frequency, self._fw, cutoff=self.cutoff)

    def profile(self):
        p = self._checked_base_profile()
        return _Shifted(p, self.start_time) if self.start_time else p

    def fourier_transform(self, freq: float) -> complex:
        """meep's analytic FT (exp(+i*omega*t) convention) of the complex
        profile exp(-i*omega0*(t - t_peak)) * exp(-(t - t_peak)^2 / (2 w^2)):
        w*sqrt(2*pi) * exp(i*omega*t_peak) * exp(-w^2 (omega - omega0)^2 / 2).
        The engine's real cos profile transforms as 0.5*(F(f) + conj(F(-f)))."""
        w = float(self.width)
        t_peak = float(self.start_time) + self.cutoff * w
        omega = 2.0 * math.pi * freq
        omega0 = 2.0 * math.pi * self.frequency
        return (w * math.sqrt(2.0 * math.pi) * cmath.exp(1j * omega * t_peak)
                * math.exp(-0.5 * (w * (omega - omega0)) ** 2))


class ContinuousSource(SourceTime):
    """meep ContinuousSource(frequency, start_time, end_time, width, fwidth,
    cutoff, slowness, wavelength, is_integrated) -> engine ContinuousSource
    (sin carrier, 1-cos turn-on over slowness*width; width=0 falls back to a
    few periods). Not supported: end_time (no smooth turn-off path; raises on
    non-default)."""

    def __init__(self, frequency=None, start_time=0, end_time=1.0e20, width=0,
                 fwidth=float("inf"), cutoff=3.0, slowness=3.0, wavelength=None,
                 is_integrated=False):
        if frequency is None and wavelength is None:
            raise ValueError("ContinuousSource requires either frequency or wavelength")
        super().__init__(is_integrated=is_integrated)
        self.frequency = 1.0 / wavelength if wavelength else float(frequency)
        self.start_time = start_time
        self.end_time = end_time
        self.width = max(width, 1.0 / fwidth)
        self.cutoff = cutoff
        self.slowness = slowness
        self._fcen = self.frequency    # private adapter (EigenModeSource path)

    def _base_profile(self):
        from ..core.sources import ContinuousSource as _CS
        if self.end_time < 1.0e20:
            raise NotImplementedError(
                "ContinuousSource end_time is not supported (engine has no "
                "smooth turn-off); use a GaussianSource pulse instead")
        return _CS(self.frequency, width=(self.width or None),
                   slowness=self.slowness, cutoff=self.cutoff)

    def profile(self):
        p = self._checked_base_profile()
        return _Shifted(p, self.start_time) if self.start_time else p


class CustomSource(SourceTime):
    """meep CustomSource(src_func, start_time, end_time, is_integrated,
    center_frequency, fwidth): src_func(t) inside [start_time, end_time], 0
    outside (NOT shifted). center_frequency/fwidth are stored only (meep uses
    them for DFT decimation, which the engine does not do)."""

    def __init__(self, src_func, start_time=-1.0e20, end_time=1.0e20,
                 is_integrated=False, center_frequency=0, fwidth=0):
        super().__init__(is_integrated=is_integrated)
        self.src_func = src_func
        self.start_time = start_time
        self.end_time = end_time
        self.center_frequency = center_frequency
        self.fwidth = fwidth

    def profile(self):
        p = self._checked_base_profile()
        if self.start_time > -1.0e20 or self.end_time < 1.0e20:
            return _Windowed(p, self.start_time, self.end_time)
        return p

    def _base_profile(self):
        from ..core.sources import CustomSource as _CS
        return _CS(self.src_func)


@dataclass
class Source:
    """meep Source: src (time profile) + component over a center/size (or
    volume) region. amp_data / amp_func_file (gridded amplitude from HDF5)
    have no engine path and raise when passed at construction."""

    src: object
    component: str = Ez
    center: Vector3 = None
    volume: object = None       # mp.Volume; supplies center/size when given
    size: Vector3 = None
    amplitude: float = 1.0
    amp_func: object = None     # callable(Vector3) -> spatial weight on the source line
    amp_func_file: str = ""
    amp_data: object = None

    def __post_init__(self):
        if self.volume is not None:
            self.center = self.volume.center
            self.size = self.volume.size
        if self.amp_data is not None:
            raise NotImplementedError(
                "Source amp_data (gridded amplitude array) is not supported; "
                "use amp_func")
        if self.amp_func_file:
            raise NotImplementedError(
                "Source amp_func_file (HDF5 amplitude) is not supported; "
                "use amp_func")

    def add_source(self, sim):
        """meep Source.add_source(sim): register with the Simulation."""
        sim._sources.append(self)


def _beam_amp(src_obj):
    """meep's own beam amplitude for this source, as a complex amp_func.

    Uses the ported gaussianbeam::get_fields rather than a paraxial formula,
    so the transverse plane comes from beam_kdir instead of being hard-coded.
    The component taken is the one the 2D engine drives, since it injects a
    single soft source rather than meep's six-component equivalent-current
    sheet -- that limitation is unchanged; only the amplitude is now meep's.
    """
    from ..core.gaussianbeam import beam_amp_func
    idx = {"Ex": 0, "Ey": 1, "Ez": 2}.get(str(src_obj.component), 2)
    freq = getattr(src_obj.src, "frequency", None) or getattr(
        src_obj.src, "fcen", 1.0)
    return beam_amp_func(src_obj.beam_E0, src_obj.beam_kdir, src_obj.beam_w0,
                         src_obj.beam_x0, freq, idx)


def _beam_envelope(beam_E0, beam_w0, beam_kdir=None, beam_x0=None):
    """Real Gaussian waist envelope exp(-rho^2/w0^2) about the BEAM AXIS.

    rho is the distance from the beam axis, which is meep's own definition --
    `sources.cpp:624-641` computes

        xrel = x - x0                 relative to the beam focus
        zhat = kdir / |kdir|          unit propagation direction
        rho  = |zhat x xrel|          transverse distance from the axis

    and the field goes as exp(-rho^2/w(z)^2) (source.py:695).

    THE PREVIOUS VERSION USED exp(-(p.y^2 + p.z^2)/w0^2), which takes y-z as
    the transverse plane and is therefore correct ONLY for a beam propagating
    along x. gaussian-beam.py sets beam_kdir = Vector3(0, 1, 0) -- its own
    comment reads "(0: +y axis)" -- so its transverse plane is x-z and the
    envelope was evaluated along the propagation direction instead of across
    it. A cross product has no preferred axis and is right for any kdir.

    ``p`` arrives relative to the source centre (meep's amp_func contract), and
    ``beam_x0`` is the focus in that same frame, so xrel = p - beam_x0.
    """
    if beam_w0 is None:
        raise ValueError("GaussianBeamSource requires beam_w0")
    w0 = float(beam_w0)
    e0 = max(abs(beam_E0.x), abs(beam_E0.y), abs(beam_E0.z)) or 1.0
    k = beam_kdir or Vector3(0.0, 0.0, 1.0)
    kn = math.sqrt(k.x * k.x + k.y * k.y + k.z * k.z) or 1.0
    zx, zy, zz = k.x / kn, k.y / kn, k.z / kn
    x0 = beam_x0 or Vector3()

    def envelope(p):
        rx, ry, rz = p.x - x0.x, p.y - x0.y, p.z - x0.z
        cx = zy * rz - zz * ry
        cy = zz * rx - zx * rz
        cz = zx * ry - zy * rx
        rho2 = cx * cx + cy * cy + cz * cz
        return e0 * math.exp(-rho2 / (w0 ** 2))

    return envelope


class GaussianBeam3DSource(Source):
    """meep GaussianBeam3DSource: exact 3D Gaussian beam via equivalent
    currents. Accepted at construction; the 2D engine has no complex
    equivalent-current injection path, so Simulation.run raises on the exact
    class (use GaussianBeam2DSource / GaussianBeamSource, which launch the
    2D envelope approximation)."""

    def __init__(self, src, center=None, volume=None, component=ALL_COMPONENTS,
                 beam_x0=None, beam_kdir=None, beam_w0=None, beam_E0=None,
                 **kwargs):
        if volume is not None:
            raise NotImplementedError(
                "GaussianBeamSource volume= is not supported; pass center/size")
        if component is ALL_COMPONENTS:
            component = Ez           # 2D engine needs one polarization (TM default)
        super().__init__(src=src, component=component, center=center, **kwargs)
        self.beam_x0 = beam_x0 if beam_x0 is not None else Vector3()
        self.beam_kdir = beam_kdir if beam_kdir is not None else Vector3()
        self.beam_w0 = beam_w0
        self.beam_E0 = beam_E0 if beam_E0 is not None else Vector3()


class GaussianBeam2DSource(GaussianBeam3DSource):
    """meep GaussianBeam2DSource -- 2D amplitude-only approximation.

    Ships as a line Source whose amp_func is the real Gaussian envelope
    exp(-(y^2 + z^2) / beam_w0^2) around the source center, scaled by
    |beam_E0|'s largest component. The engine injects a real static spatial
    mask times a real scalar time profile, so the focusing/tilt phase cannot
    be applied: ``beam_x0`` and ``beam_kdir`` are stored but do NOT curve or
    steer the beam. A user-supplied amp_func overrides the envelope. The
    genuine meep 2D beam math (green2d / get_r_rhat / incoming_mask) is
    ported below for callers that evaluate the fields directly."""

    def __init__(self, src, center=None, volume=None, component=ALL_COMPONENTS,
                 beam_x0=None, beam_kdir=None, beam_w0=None, beam_E0=None,
                 **kwargs):
        super().__init__(src, center=center, volume=volume, component=component,
                         beam_x0=beam_x0, beam_kdir=beam_kdir, beam_w0=beam_w0,
                         beam_E0=beam_E0, **kwargs)
        if self.amp_func is None:
            self.amp_func = _beam_amp(self)

    @staticmethod
    def get_r_rhat(X, X0):
        """(r, rhat): distance and unit vector from the 2-vector X0 to every
        point of the (2, nx, ny) grid X."""
        rhat = X - np.asarray(X0)[:, None, None]
        r = np.sqrt(np.sum(rhat * rhat, axis=0))
        return r, rhat / r[np.newaxis, :, :]

    def green2d(self, X, freq, eps, mu, X0, kdir, hankel, beam_E0):
        """meep's green2d: 6-component EH of a 2D dipole pair (electric p and
        the magnetic dipole that cancels the backward wave) at frequency
        ``freq``; ``kdir`` is the complex direction cos+i*sin; ``hankel`` is
        scipy's hankel1/hankel2."""
        EH = np.zeros([6] + list(X.shape)[1:], dtype=complex)
        r, rhat = self.get_r_rhat(X, X0)
        omega = 2 * np.pi * freq
        k = omega * np.sqrt(eps * mu)
        ik = 1j * k
        kr = k * r
        Z = np.sqrt(mu / eps)
        H0_kr, H1_kr, H2_kr = hankel(0, kr), hankel(1, kr), hankel(2, kr)
        ikH1 = 0.25 * ik * H1_kr
        p = np.zeros(3, dtype=complex)
        p[0], p[1], p[2] = beam_E0.x, beam_E0.y, beam_E0.z
        pdotrhat = p[0] * rhat[0] + p[1] * rhat[1]
        rhatcrossp = rhat[0] * p[1] - rhat[1] * p[0]
        eHx = -rhat[1] * ikH1 * p[2]
        eHy = rhat[0] * ikH1 * p[2]
        eHz = -rhatcrossp * ikH1
        eEx = -(rhat[0] * (pdotrhat / r * 0.25 * Z)) * H1_kr + \
            (rhat[1] * (rhatcrossp * omega * mu * 0.125)) * (H0_kr - H2_kr)
        eEy = -(rhat[1] * (pdotrhat / r * 0.25 * Z)) * H1_kr - \
            (rhat[0] * (rhatcrossp * omega * mu * 0.125)) * (H0_kr - H2_kr)
        eEz = (-0.25 * omega * mu) * H0_kr * p[2]
        p = -np.array([-p[2] * np.imag(kdir), p[2] * np.real(kdir),
                       p[0] * np.imag(kdir) - p[1] * np.real(kdir)])
        pdotrhat = p[0] * rhat[0] + p[1] * rhat[1]
        rhatcrossp = rhat[0] * p[1] - rhat[1] * p[0]
        hEx = rhat[1] * ikH1 * p[2]
        hEy = -rhat[0] * ikH1 * p[2]
        hEz = rhatcrossp * ikH1
        hHx = -(rhat[0] * (pdotrhat / r * 0.25 / Z)) * H1_kr + \
            (rhat[1] * (rhatcrossp * omega * eps * 0.125)) * (H0_kr - H2_kr)
        hHy = -(rhat[1] * (pdotrhat / r * 0.25 / Z)) * H1_kr - \
            (rhat[0] * (rhatcrossp * omega * eps * 0.125)) * (H0_kr - H2_kr)
        hHz = (-0.25 * omega * eps) * H0_kr * p[2]
        EH[:] = np.array([eEx, eEy, eEz, eHx, eHy, eHz]) + \
            np.array([hEx, hEy, hEz, hHx, hHy, hHz])
        return EH

    @staticmethod
    def incoming_mask(x0, y0, kre, kim, X):
        """(incoming, outgoing, waist) boolean masks splitting the plane by
        the waist line through (x0, y0) normal to the beam direction
        (kre, kim) -- points downstream of the waist are 'incoming' for the
        equivalent-current construction."""
        s = (X[0] - x0) * kre + (X[1] - y0) * kim
        waist = np.abs(s) <= 1e-12
        return (s > 0) & ~waist, (s < 0) & ~waist, waist

    def get_fields(self, sim):
        raise NotImplementedError(
            "GaussianBeam2DSource.get_fields: the complex equivalent-current "
            "fields cannot be reconstructed from the engine's real injected "
            "envelope; evaluate green2d directly instead")


class GaussianBeamSource(GaussianBeam3DSource):
    """meep GaussianBeamSource (rebased on GaussianBeam3DSource per meep);
    behaves as the 2D envelope approximation here, like GaussianBeam2DSource."""

    def __init__(self, src, center=None, volume=None, component=ALL_COMPONENTS,
                 beam_x0=None, beam_kdir=None, beam_w0=None, beam_E0=None,
                 **kwargs):
        super().__init__(src, center=center, volume=volume, component=component,
                         beam_x0=beam_x0, beam_kdir=beam_kdir, beam_w0=beam_w0,
                         beam_E0=beam_E0, **kwargs)
        if self.amp_func is None:
            self.amp_func = _beam_amp(self)


class IndexedSource(Source):
    """meep IndexedSource(src, srcdata, amp_arr): one amplitude per grid point
    of the srcdata region (here: an object carrying center/size, e.g.
    mp.Volume). The 2D engine applies amp_arr along the source line; the
    point count is validated at run time and complex amplitudes raise (the
    engine injects a real static mask)."""

    def __init__(self, src, srcdata, amp_arr, needs_boundary_fix=False):
        super().__init__(src=src, component=getattr(srcdata, "component", Ez),
                         center=srcdata.center, size=srcdata.size)
        self.srcdata = srcdata
        self.amp_arr = np.asarray(amp_arr)
        self.needs_boundary_fix = needs_boundary_fix


class DiffractedPlanewave:
    """meep DiffractedPlanewave(g, axis, s, p): a diffraction order spec
    (structural only -- carried in place of an eig_band; the engine has no
    diffraction-order decomposition, so consumers raise)."""

    def __init__(self, g=None, axis=None, s=None, p=None):
        self.g = g
        self.axis = axis
        self.s = complex(s if s is not None else 0)
        self.p = complex(p if p is not None else 0)


class EigenModeSource(Source):
    """meep EigenModeSource: a Source whose spatial profile is a waveguide
    eigenmode. The engine's S-parameter path (compute_smatrix via
    meep_torch.ports) consumes direction/eig_band; the eig_* knobs with no
    engine path (eig_kpoint, eig_match_freq=False, eig_vol, transverse
    parity) raise."""

    def __init__(self, src, center=None, volume=None, eig_lattice_size=None,
                 eig_lattice_center=None, eig_vol=None, component=ALL_COMPONENTS,
                 direction=AUTOMATIC, eig_band=1, eig_kpoint=None,
                 eig_match_freq=True, eig_parity=NO_PARITY, eig_resolution=0,
                 eig_tolerance=1e-12, **kwargs):
        if isinstance(eig_band, DiffractedPlanewave):
            raise NotImplementedError(
                "EigenModeSource with a DiffractedPlanewave band is not "
                "supported (no diffraction-order engine path)")
        if eig_band < 1:
            raise ValueError(
                f"eig_band is 1-based like meep (1 = fundamental mode), got "
                f"{eig_band}")
        if eig_kpoint is not None:
            raise NotImplementedError(
                "EigenModeSource eig_kpoint is not supported (modes are "
                "solved along the port normal only)")
        if eig_match_freq is not True:
            raise NotImplementedError(
                "EigenModeSource eig_match_freq=False (fixed-k mode) is not "
                "supported")
        if eig_vol is not None:
            raise NotImplementedError(
                "EigenModeSource eig_vol is not supported; the mode is solved "
                "on the source line")
        if eig_parity & (EVEN_Y | ODD_Y):
            raise NotImplementedError(
                "EigenModeSource transverse (y) parity filtering is not "
                "supported; use EVEN_Z/ODD_Z to select TE/TM")
        want = Hz if eig_parity & EVEN_Z else (Ez if eig_parity & ODD_Z else None)
        if component is ALL_COMPONENTS:
            component = want or Ez       # meep ALL_COMPONENTS -> TM default here
        elif want is not None and component != want:
            raise ValueError(
                f"component={component} conflicts with eig_parity (implies {want})")
        super().__init__(src=src, component=component, center=center,
                         volume=volume, **kwargs)
        self.direction = direction
        self.eig_band = eig_band
        self.eig_kpoint = eig_kpoint
        self.eig_match_freq = eig_match_freq
        self.eig_parity = eig_parity
        self.eig_resolution = eig_resolution
        self.eig_tolerance = eig_tolerance
        self.eig_lattice_size = eig_lattice_size if eig_lattice_size is not None else self.size
        self.eig_lattice_center = (eig_lattice_center if eig_lattice_center is not None
                                   else self.center)

    def eig_power(self, freq):
        """Total power of the source at ``freq`` (meep eig_power)."""
        return abs(self.amplitude * self.src.fourier_transform(freq)) ** 2


def source_axis(size: Vector3 | None) -> str:
    """Which axis a source segment EXTENDS along: "y", "x", or "point".

    meep places no constraint on a source's orientation -- a beam travelling
    along +y needs a source line extended along x, and gaussian-beam.py is
    exactly that: `size=mp.Vector3(s)`, 14um in x and zero in y.

    _point_or_span used to read only `size.y`, so an x-extended source fell
    into its "no y extent" branch and became a SINGLE CELL. The 14um line
    collapsed to a point, and with it the line measure dx/dx^2 became the
    point measure 1/dx^2 -- a factor of 1/dx too much amplitude on top of
    losing the extent. Nothing raised; the run just produced the wrong field.
    """
    if size is None:
        return "point"
    ey = math.isinf(size.y) or size.y != 0
    ex = math.isinf(size.x) or size.x != 0
    if ey and ex:
        raise NotImplementedError(
            "a source extended in BOTH x and y is an area source, not a line; "
            "pass a zero extent on one axis")
    return "y" if ey else ("x" if ex else "point")


def _span_x(center: Vector3, size: Vector3 | None, gs: GridSpec):
    """(jy, i0, i1) for a source line extended along x at a fixed y."""
    c = center or Vector3()
    jy = max(0, min(gs.ny - 1, cell_index_y(c.y, gs)))
    if math.isinf(size.x):
        return jy, 0, gs.nx
    x0 = c.x - size.x / 2.0 + gs.sx / 2.0
    x1 = c.x + size.x / 2.0 + gs.sx / 2.0
    i0, i1 = span_index_range(max(0.0, x0), min(gs.sx, x1), gs, "x")
    return jy, i0, i1


def _point_or_span(center: Vector3, size: Vector3 | None, gs: GridSpec):
    """Return (ix, j0, j1) for a source line normal to x (Ez/Ex/Hy 2D).

    If size is None or its y-extent is 0 -> single-cell point at center;
    else a y-span [j0, j1). x is the propagation/normal axis here.
    """
    c = center or Vector3()
    ix = cell_index(c.x, gs)
    if size is None or size.y == 0:
        jy = max(0, min(gs.ny, cell_index_y(c.y, gs)))
        return ix, jy, jy + 1
    if math.isinf(size.y):
        return ix, 0, gs.ny          # full-y plane source (1D / infinite-height line)
    y0 = c.y - size.y / 2.0 + gs.sy / 2.0
    y1 = c.y + size.y / 2.0 + gs.sy / 2.0
    j0, j1 = span_index_range(max(0.0, y0), min(gs.sy, y1), gs, "y")
    return ix, j0, j1


def build_line_source(spec: Source, gs: GridSpec, pol: str):
    axis = source_axis(spec.size)
    if axis == "x":
        # Extended along x: a horizontal segment at fixed y. Reported with its
        # own tag so Simulation builds a row mask rather than a column one.
        jy, i0, i1 = _span_x(spec.center, spec.size, gs)
        return ("xline", pol, jy, i0, i1, spec.src.profile(), spec.amplitude,
                spec.amp_func, spec.center, gs)
    ix, j0, j1 = _point_or_span(spec.center, spec.size, gs)
    # LineSource is constructed on the Grid by Simulation (needs grid); return spec
    return ("line", pol, ix, j0, j1, spec.src.profile(), spec.amplitude, spec.amp_func,
            spec.center, gs)


def build_mode_source(spec: EigenModeSource, gs: GridSpec):
    c = spec.center or Vector3()
    # propagation axis = axis of smallest size component (meep convention)
    s = spec.size or Vector3(0, 1, 0)
    if abs(s.x) <= abs(s.y):
        axis, idx, lo, hi = "x", cell_index(c.x, gs), *_y_span(c, s, gs)
    else:
        axis, idx = "y", max(0, min(gs.ny, cell_index_y(c.y, gs)))
        x0 = c.x - s.x / 2 + gs.sx / 2
        x1 = c.x + s.x / 2 + gs.sx / 2
        lo, hi = span_index_range(x0, x1, gs, "x")
    return ("mode", axis, idx, lo, hi, spec.direction, spec.eig_band,
            spec.src.profile(), spec.amplitude)


def _y_span(c: Vector3, s: Vector3, gs: GridSpec):
    y0 = c.y - s.y / 2 + gs.sy / 2
    y1 = c.y + s.y / 2 + gs.sy / 2
    lo, hi = span_index_range(max(0.0, y0), min(gs.sy, y1), gs, "y")
    return lo, hi