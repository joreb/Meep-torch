import torch

from meep_torch.core.pml import cpml_profiles


def test_interior_is_identity():
    b, a, k = cpml_profiles(n=100, npml=10, offset=0.0, dt=0.01, dx=0.02)
    # interior: sigma=0, alpha=0 -> b=1, a=0, kappa=1
    assert torch.all(b[20:80] == 1.0)
    assert torch.all(a[20:80] == 0.0)
    assert torch.all(k == 1.0)


def test_pml_ranges():
    b, a, _ = cpml_profiles(n=100, npml=10, offset=0.0, dt=0.01, dx=0.02)
    assert torch.all(b[:10] < 1.0) and torch.all(b[:10] > 0.0)
    assert torch.all(a[:5] < 0.0)          # a = sigma*(b-1)/... < 0 where sigma>0
    assert torch.allclose(b[:10], b[-10:].flip(0))  # symmetric ends


def test_npml_zero_degenerates():
    b, a, k = cpml_profiles(n=50, npml=0, offset=0.5, dt=0.01, dx=0.02)
    assert torch.all(b == 1.0) and torch.all(a == 0.0) and torch.all(k == 1.0)
