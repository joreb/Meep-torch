import pytest

from meep_torch.core.materials import Si3N4, SiO2, cSi

F_1550 = 1 / 1.55


def test_si3n4_at_1550():
    # Luke fit is excellent here: published n(1.55) = 1.9963
    assert cSi is not Si3N4
    assert Si3N4.n(F_1550).real == pytest.approx(1.9963, abs=0.002)


def test_sio2_at_1550():
    # Horiba broadband fit: expect ~1.458 from the model (textbook 1.444; fit bias known)
    assert SiO2.n(F_1550).real == pytest.approx(1.4577, abs=0.003)
    assert SiO2.n(F_1550).real == pytest.approx(1.444, abs=0.02)   # sanity vs textbook


def test_csi_at_1550_extrapolated():
    # Deinega fit valid 0.4-1.0 um; extrapolates to ~3.497 at 1.55 (textbook 3.4757)
    n = cSi.n(F_1550).real
    assert n == pytest.approx(3.497, abs=0.01)
    assert n == pytest.approx(3.4757, abs=0.05)                    # sanity vs textbook
    assert F_1550 < cSi.valid_fmin  # document that 1.55um is outside fit range


def test_all_lossless_in_telecom():
    for mat in (SiO2, Si3N4):
        assert abs(mat.n(F_1550).imag) < 1e-3
