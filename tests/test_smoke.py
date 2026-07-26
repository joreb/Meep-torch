import meep_torch


def test_import():
    assert meep_torch.__version__ == "0.1.0"


def test_torch_available():
    import torch

    assert torch.tensor([1.0]).sum().item() == 1.0
