# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Flexcompute
# Copyright (c) 2019 Tyler Hughes
# Derived from the ceviche package (https://github.com/fancompute/ceviche).
# Adaptations (c) 2026 meep-torch contributors, MIT.
import numpy as np
import scipy.sparse as sp

from meep_torch.modes.derivatives import (
    C_0,
    create_d_matrices,
    create_s_matrices,
    make_dxb,
    make_dxf,
    make_dyb,
    make_dyf,
)


def _uniform_dls(Nx, Ny, dl):
    dlf = (np.full(Nx, dl), np.full(Ny, dl))
    dlb = (np.full(Nx, dl), np.full(Ny, dl))
    return (dlf, dlb)


def _isotropic_tensor(val, N):
    t = np.ones((3, 3, N), dtype=np.complex128)
    for i in range(3):
        t[i, i, :] = val
    return t


def test_d_matrices_forward_difference_stencil():
    # Forward x derivative on (4,3): diags([-1,1],[0,1]) with PEC (dxf[0,0]=0), /dl.
    Nx, Ny, dl = 4, 3, 0.1
    dxf = make_dxf(np.full(Nx, dl), (Nx, Ny), pmc=False)
    assert dxf.shape == (Nx * Ny, Nx * Ny)
    # Linear ramp in x: f[i,j] = i*dl -> forward diff = dl, /dl = 1 interior; 0 at PEC row 0.
    f = np.zeros(Nx * Ny)
    for j in range(Ny):
        for i in range(Nx):
            f[i * Ny + j] = i * dl
    out = dxf @ f
    out_2d = out.reshape(Nx, Ny)
    # interior x rows (i=1..Nx-2): forward diff = 1.0; last row (i=Nx-1): f[Nx]-f[Nx-1] out of
    # domain -> -f[Nx-1]/dl = -(Nx-1); row 0 (PEC): super-only = f[1]-f[0] = dl -> /dl = 1.
    assert np.allclose(out_2d[0, :], 1.0)
    assert np.allclose(out_2d[1, :], 1.0)
    assert np.allclose(out_2d[2, :], 1.0)
    assert np.allclose(out_2d[3, :], -(Nx - 1))  # boundary wrap: -f[last]/dl
    # Constant field: strictly-interior rows (1..Nx-2) -> 0; row 0 (PEC ghost=0) -> f[1]/dl;
    # last row (out-of-domain wrap) -> -f[last]/dl.
    const_out = (dxf @ np.ones(Nx * Ny)).reshape(Nx, Ny)
    assert np.allclose(const_out[1 : Nx - 1, :], 0.0)
    assert np.allclose(const_out[0, :], 1.0 / dl)
    assert np.allclose(const_out[Nx - 1, :], -1.0 / dl)


def test_d_matrices_backward_difference_and_pmc():
    # Backward derivative under PMC sets dxb[0,0]=2 (Neumann-like); PEC sets 0.
    Nx, Ny, dl = 4, 2, 0.1
    dxb_pec = make_dxb(np.full(Nx, dl), (Nx, Ny), pmc=False).tocsr()
    dxb_pmc = make_dxb(np.full(Nx, dl), (Nx, Ny), pmc=True).tocsr()
    # PMC bumps the (0,0) entry; PEC zeros it. kron yields a BSR matrix -> convert to csr
    # for element indexing.
    assert dxb_pec[0, 0] == 0.0
    assert dxb_pmc[0, 0] == 2.0 / dl


def test_d_matrices_y_derivative_shape_and_kron():
    Nx, Ny, dl = 3, 5, 0.1
    dyf = make_dyf(np.full(Ny, dl), (Nx, Ny), pmc=False)
    dyb = make_dyb(np.full(Ny, dl), (Nx, Ny), pmc=False)
    assert dyf.shape == (Nx * Ny, Nx * Ny)
    assert dyb.shape == (Nx * Ny, Nx * Ny)
    # y-derivative of a y-ramp should be 1 in interior y rows.
    f = np.zeros(Nx * Ny)
    for i in range(Nx):
        for j in range(Ny):
            f[i * Ny + j] = j * dl
    out = (dyf @ f).reshape(Nx, Ny)
    assert np.allclose(out[:, 0], 1.0)
    assert np.allclose(out[:, 1:-1], 1.0)


def test_create_d_matrices_returns_four():
    Nx, Ny, dl = 4, 4, 0.1
    dls = _uniform_dls(Nx, Ny, dl)
    dxf, dxb, dyf, dyb = create_d_matrices((Nx, Ny), dls, dmin_pmc=(False, False))
    for m in (dxf, dxb, dyf, dyb):
        assert sp.issparse(m)
        assert m.shape == (Nx * Ny, Nx * Ny)


def test_s_matrices_identity_without_pml():
    Nx, Ny, dl = 4, 3, 0.1
    N = Nx * Ny
    dls = _uniform_dls(Nx, Ny, dl)
    eps = _isotropic_tensor(2.25, N)
    mu = _isotropic_tensor(1.0, N)
    omega = 2 * np.pi * C_0 / 1.0  # lambda = 1 um
    sx_f, sx_b, sy_f, sy_b = create_s_matrices(
        omega, (Nx, Ny), npml=(0, 0), dls=dls,
        eps_tensor=eps, mu_tensor=mu, dmin_pml=(True, True),
    )
    eye = sp.eye(N, format="csr")
    for s in (sx_f, sx_b, sy_f, sy_b):
        assert s.shape == (N, N)
        assert np.allclose((s - eye).data, 0.0)


def test_s_matrices_complex_inside_pml():
    Nx, Ny, dl = 6, 5, 0.1
    N = Nx * Ny
    dls = _uniform_dls(Nx, Ny, dl)
    eps = _isotropic_tensor(2.25, N)
    mu = _isotropic_tensor(1.0, N)
    omega = 2 * np.pi * C_0 / 1.0
    sx_f, sx_b, sy_f, sy_b = create_s_matrices(
        omega, (Nx, Ny), npml=(2, 2), dls=dls,
        eps_tensor=eps, mu_tensor=mu, dmin_pml=(True, True),
    )
    # Inside PML the s-factors are complex (kappa + 1j*sigma/...) so 1/s has nonzero imag part.
    for s in (sx_f, sx_b, sy_f, sy_b):
        d = s.data
        assert np.any(d.imag != 0.0), "PML s-matrix must carry imaginary absorption"
    # And it must differ from identity (|s| != 1 somewhere).
    eye = sp.eye(N, format="csr")
    assert not np.allclose((sx_f - eye).data, 0.0)


def test_s_factors_dmin_pml_false_keeps_lower_side_real():
    # dmin_pml=False disables PML on the 'bottom' (low-index) side; the high-index side still
    # absorbs. The low-index side stays 1 (real), high side goes complex.
    Nx, Ny, dl = 6, 5, 0.1
    N = Nx * Ny
    dls = _uniform_dls(Nx, Ny, dl)
    eps = _isotropic_tensor(2.25, N)
    mu = _isotropic_tensor(1.0, N)
    omega = 2 * np.pi * C_0 / 1.0
    sx_f, _, _, _ = create_s_matrices(
        omega, (Nx, Ny), npml=(2, 2), dls=dls,
        eps_tensor=eps, mu_tensor=mu, dmin_pml=(False, False),
    )
    # Inspect the diagonal of the S-matrix reshaped to (Nx, Ny). Bottom x rows (low-index
    # side) keep PML disabled -> real; top x rows still absorb -> complex.
    diag = np.asarray(sx_f.diagonal()).reshape(Nx, Ny)
    assert np.allclose(diag[0, :].imag, 0.0)  # bottom x side: PML disabled -> real
    assert np.any(diag[-1, :].imag != 0.0)    # top x side: still absorbing -> complex
