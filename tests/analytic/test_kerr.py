import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import CWRamp, LineSource
from meep_torch.core.stepper import init_tm_state

F0 = 0.25          # fundamental; THG at 0.75 still resolved at dx=0.02


def probe_spectrum_at(amp: float) -> tuple[float, float]:
    g = Grid(nx=1000, ny=8, dx=0.02)
    m = build_tm_coeffs(g, eps_inf=2.25, chi3=0.02, npml=40, y_pmc=True)

    class Scaled:
        def __init__(self, base, a):
            self.base, self.a = base, a

        def amplitude(self, t):
            return self.a * self.base.amplitude(t)

    src = LineSource(g, ix=100, j0=0, j1=8, profile=Scaled(CWRamp(F0, 40.0), amp))
    probe = PointProbe(700, 4)
    s = init_tm_state(g, y_pmc=True)
    _, (series,) = run_tm(s, m, 8000, source=src, monitors=[probe])
    steady = series[4000:]                      # discard ramp transient
    spec = torch.fft.rfft(steady).abs()
    freqs = torch.fft.rfftfreq(len(steady), g.dt)
    def peak(f):
        i = (freqs - f).abs().argmin()
        return spec[i - 2 : i + 3].max().item()
    return peak(F0), peak(3 * F0)


def test_thg_scales_as_cube():
    _, thg1 = probe_spectrum_at(0.02)
    _, thg2 = probe_spectrum_at(0.04)
    ratio = thg2 / thg1
    assert 8 * 0.7 < ratio < 8 * 1.3            # 2x input -> 8x THG (+-30%)


def test_thg_absent_without_chi3():
    g = Grid(nx=1000, ny=8, dx=0.02)
    m = build_tm_coeffs(g, eps_inf=2.25, chi3=0.0, npml=40, y_pmc=True)
    src = LineSource(g, ix=100, j0=0, j1=8, profile=CWRamp(F0, 40.0))
    probe = PointProbe(700, 4)
    s = init_tm_state(g, y_pmc=True)
    _, (series,) = run_tm(s, m, 8000, source=src, monitors=[probe])
    steady = series[4000:]
    spec = torch.fft.rfft(steady).abs()
    freqs = torch.fft.rfftfreq(len(steady), g.dt)
    i0 = (freqs - F0).abs().argmin()
    i3 = (freqs - 3 * F0).abs().argmin()
    assert spec[i3 - 2 : i3 + 3].max() < 1e-6 * spec[i0 - 2 : i0 + 3].max()
