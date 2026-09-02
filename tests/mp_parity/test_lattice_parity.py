# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Matrix/Lattice/Mesh parity gate against real pymeep (the oracle env).

The same COMPUTE snippet runs through our mp shim and through pymeep in a
subprocess (~/miniforge3/envs/meep-oracle); every produced value must match
to 1e-12. Covers Matrix algebra (det/inverse/transpose/H/conj, mm/mv/scalar,
diag+offdiag hermitian construction), get_rotation_matrix, a triangular
Lattice (basis/b1..b3/metric, all six lattice/cartesian/reciprocal
conversions on both Vector3 and Matrix operands, rotate_lattice/
rotate_reciprocal), and Mesh centroid/recentering.
"""
import json
import os
import subprocess

import pytest

NAMES = [
    "Vector3", "Matrix", "Lattice", "Mesh", "get_rotation_matrix",
    "lattice_to_cartesian", "cartesian_to_lattice",
    "reciprocal_to_cartesian", "cartesian_to_reciprocal",
    "lattice_to_reciprocal", "reciprocal_to_lattice",
]

ORACLE = os.path.expanduser("~/miniforge3/envs/meep-oracle/bin/python")

# Every op below runs identically through both namespaces; numbers are
# serialized as [re, im] pairs so complex and real compare uniformly.
COMPUTE = r'''
import math


def num(c):
    c = complex(c)
    return [c.real, c.imag]


def vec(v):
    return [num(v[i]) for i in range(3)]


def mat(m):
    return [vec(m.row(i)) for i in range(3)]


def compute(ns):
    Vector3, Matrix, Lattice, Mesh = (
        ns["Vector3"], ns["Matrix"], ns["Lattice"], ns["Mesh"])

    out = {}
    A = Matrix(Vector3(1.0, 4.0, 7.0), Vector3(2.0, 5.0, 8.0),
               Vector3(3.0, 6.0, 10.0))
    B = Matrix(Vector3(2.0, 0.0, 1.0), Vector3(-1.0, 3.0, 0.0),
               Vector3(0.5, 2.0, -2.0))
    v = Vector3(0.3, -0.4, 0.5)

    out["A"] = mat(A)
    out["det"] = num(A.determinant())
    out["inv"] = mat(A.inverse())
    out["transpose"] = mat(A.transpose())
    out["mm"] = mat(A * B)
    out["mv"] = vec(A * v)
    out["scale_l"] = mat(2.5 * A)
    out["scale_r"] = mat(A * 2.5)
    out["div"] = mat(A / 2.0)
    out["add"] = mat(A + B)
    out["sub"] = mat(A - B)
    out["getitem_row"] = vec(A[1])
    out["repr"] = repr(A)

    C = Matrix(Vector3(1.0, 2j, 3.0), Vector3(0.0, 1 + 1j, 2.0),
               Vector3(4.0, 5.0, 6 - 2j))
    out["conj"] = mat(C.conj())
    out["getH"] = mat(C.getH())
    out["H"] = mat(C.H)
    D = Matrix(diag=Vector3(1.0, 2.0, 3.0), offdiag=Vector3(1j, 2.0, 3 - 1j))
    out["diag_offdiag"] = mat(D)

    grm = ns["get_rotation_matrix"]
    out["rot"] = mat(grm(Vector3(1.0, 1.0, 1.0), 0.7))
    out["rot_v"] = vec(grm(Vector3(1.0, 2.0, 3.0), -1.2) * v)

    lat = Lattice(size=Vector3(1.0, 7.0, 3.0),
                  basis_size=Vector3(1.0, 1.0, 1.0),
                  basis1=Vector3(math.sqrt(3) / 2, 0.5),
                  basis2=Vector3(math.sqrt(3) / 2, -0.5))
    out["basis"] = mat(lat.basis)
    out["b1"] = vec(lat.b1)
    out["b2"] = vec(lat.b2)
    out["b3"] = vec(lat.b3)
    out["metric"] = mat(lat.metric)
    for tag, fn in [("l2c", "lattice_to_cartesian"),
                    ("c2l", "cartesian_to_lattice"),
                    ("r2c", "reciprocal_to_cartesian"),
                    ("c2r", "cartesian_to_reciprocal"),
                    ("l2r", "lattice_to_reciprocal"),
                    ("r2l", "reciprocal_to_lattice")]:
        out[tag + "_v"] = vec(ns[fn](v, lat))
        out[tag + "_m"] = mat(ns[fn](B, lat))
    out["roundtrip"] = vec(ns["cartesian_to_lattice"](
        ns["lattice_to_cartesian"](v, lat), lat))
    out["rot_lat"] = vec(v.rotate_lattice(Vector3(0.0, 0.0, 1.0), 0.9, lat))
    out["rot_rec"] = vec(v.rotate_reciprocal(Vector3(0.0, 0.0, 1.0), 0.9, lat))

    verts = [Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0),
             Vector3(0.0, 1.0, 0.0), Vector3(0.0, 0.0, 1.0)]
    tris = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    m1 = Mesh(verts, tris)
    m2 = Mesh(verts, tris, center=Vector3(1.0, 2.0, 3.0))
    out["mesh_center"] = vec(m1.center)
    out["mesh_verts"] = [vec(p) for p in m2.vertices]
    out["mesh_center2"] = vec(m2.center)
    out["mesh_tris"] = [list(t) for t in m2.triangles]
    return out
'''

pytestmark = pytest.mark.skipif(
    not os.path.exists(ORACLE), reason="meep-oracle env not available")


@pytest.fixture(scope="module")
def oracle():
    script = COMPUTE + (
        "\nimport json\nimport meep as mp\n"
        f"ns = {{n: getattr(mp, n) for n in {NAMES!r}}}\n"
        "print('ORACLE_JSON=' + json.dumps(compute(ns)))\n")
    r = subprocess.run([ORACLE, "-c", script], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    line = [ln for ln in r.stdout.splitlines()
            if ln.startswith("ORACLE_JSON=")][-1]
    return json.loads(line[len("ORACLE_JSON="):])


@pytest.fixture(scope="module")
def ours():
    from meep_torch.mp import geometry, vec
    ns = {n: getattr(vec, n, None) or getattr(geometry, n) for n in NAMES}
    env = {}
    exec(COMPUTE, env)
    return env["compute"](ns)


def _diff(a, b, path=""):
    """Collect mismatches between two nested JSON-ish structures."""
    bad = []
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} != {len(b)}"]
        for i, (x, y) in enumerate(zip(a, b)):
            bad += _diff(x, y, f"{path}[{i}]")
    elif isinstance(a, str) or isinstance(b, str):
        if a != b:
            bad.append(f"{path}: {a!r} != {b!r}")
    else:
        if abs(a - b) > 1e-12:
            bad.append(f"{path}: {a} != {b} (|d|={abs(a - b):.3e})")
    return bad


def test_key_sets_match(ours, oracle):
    assert set(ours) == set(oracle)


@pytest.mark.parametrize("key", [
    "A", "det", "inv", "transpose", "mm", "mv", "scale_l", "scale_r", "div",
    "add", "sub", "getitem_row", "repr", "conj", "getH", "H", "diag_offdiag",
    "rot", "rot_v", "basis", "b1", "b2", "b3", "metric",
    "l2c_v", "l2c_m", "c2l_v", "c2l_m", "r2c_v", "r2c_m",
    "c2r_v", "c2r_m", "l2r_v", "l2r_m", "r2l_v", "r2l_m",
    "roundtrip", "rot_lat", "rot_rec",
    "mesh_center", "mesh_center2", "mesh_verts", "mesh_tris",
])
def test_parity(key, ours, oracle):
    bad = _diff(ours[key], oracle[key], key)
    assert not bad, "\n".join(bad)
