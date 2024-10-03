import pytest
from meep_torch.core.grid import Grid
from meep_torch.ports.port import Port, validate_ports


def _grid():
    return Grid(nx=100, ny=60, dx=0.1)


def test_valid_port_passes():
    p = Port(axis="x", index=20, lo=15, hi=45, direction=+1)
    validate_ports([p], _grid(), npml=10)


@pytest.mark.parametrize("kw,msg", [
    (dict(axis="z"), "axis"),
    (dict(direction=0), "direction"),
    (dict(n_modes=0), "n_modes"),
    (dict(lo=45, hi=15), "span"),
    (dict(index=11), "PML"),          # 11 < npml+3 = 13
    (dict(index=95), "PML"),          # 95 >= nx-npml-3 = 87
    (dict(lo=5), "PML"),              # span dips into PML
])
def test_invalid_ports_raise(kw, msg):
    base = dict(axis="x", index=20, lo=15, hi=45, direction=+1, n_modes=1)
    base.update(kw)
    with pytest.raises(ValueError, match=msg):
        validate_ports([Port(**base)], _grid(), npml=10)


def test_y_normal_axis_lengths():
    # y-normal: index checked against ny, span against nx
    p = Port(axis="y", index=20, lo=15, hi=85, direction=-1)
    validate_ports([p], _grid(), npml=10)
    bad = Port(axis="y", index=55, lo=15, hi=85, direction=-1)  # 55 >= ny-13
    with pytest.raises(ValueError, match="PML"):
        validate_ports([bad], _grid(), npml=10)