# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""MPB's planewave band eigensolver for 2D lattices (z-invariant, in-plane k).

The FDFD sibling :mod:`meep_torch.mpb.eigensolver` discretizes the same
omega(k) problem with 2nd-order finite differences on a Yee grid. This one
uses MPB's own basis -- one planewave per FFT grid index, no cutoff -- so
the operator error is spectral and what is left is entirely the epsilon
averaging. Measured against the committed meep.mpb oracles at resolution
32 the square-lattice rods come out at 1.4e-3 relative (TM), where the
FDFD path measures 3.4e-3, and the triangular lattice -- solved on its own
primitive cell -- at 2.7e-4. TE is a different story and the ceiling is
the epsilon model, not the basis; see ``tests/mp_parity/test_planewave.py``.

The other thing the planewave basis buys is an arbitrary lattice: the
operator only ever sees the reciprocal duals, so a triangular or oblique
cell is solved directly instead of through a rectangular supercell whose
bands come out folded and have to be matched back.

Conventions (MPB's, except where noted):

* ``eps`` is (nu, nv) on a uniform grid in LATTICE (fractional) coordinates,
  cell i at ``u = (i + 0.5)/nu - 1/2``. MPB itself samples at ``u = i/nu``;
  a half-cell offset is a rigid translation, which is a diagonal unitary
  similarity on the operator and therefore invisible in the spectrum.
* Planewaves ``G = 2pi (m b1 + n b2)`` with ``b_i . a_j = delta_ij`` and
  (m, n) running the numpy FFT frequency ordering of (nu, nv). Coefficient
  vectors are flattened C-order over that (m, n) grid, so a coefficient
  column reshapes to (nu, nv) and feeds ``np.fft.ifft2`` directly.
* ``kc`` is CARTESIAN, radians per unit length: ``2*pi`` times meep's
  cartesian k. From MPB's reciprocal-lattice k use
  ``kc = 2*pi*(k1*b1 + k2*b2)`` (:func:`reciprocal_k_to_cartesian`).
* Frequencies are meep units (a = 1, c = 1, f = omega/2pi).

TM (Ez) follows MPB's transverse-H formulation: the in-plane H amplitude is
``H_G = h_G (zhat x K_G)/|K_G|`` and the operator is ``|K_G| eta(G-G')
|K_G'|`` with the SCALAR ``eta = 1/<eps>`` -- Ez runs parallel to every
in-plane interface, so the correct effective inverse is the reciprocal of
the cell-averaged epsilon, MPB's "mean epsilon" case (maxwell_eps.c).

TE (Hz) is ``-div(eta grad Hz) = omega^2 Hz``, but written the way the
tensor demands: ``curl H = -(zhat x grad Hz)``, so

    A_{GG'} = (zhat x K_G) . eta(G-G') . (zhat x K_G')

which for a diagonal Kottke eta is ``Kx eta_yy Kx' + Ky eta_xx Ky'``. The
crossing is the same one :func:`~meep_torch.mpb.eigensolver.te_edge_inv_eps`
carries and for the same reason: d/dx Hz is set by E_y. Contracting the
tensor with K instead of (zhat x K) is numerically identical for scalar eta
and wrong for a smoothed one.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import torch

__all__ = ["planewave_bands_2d", "planewave_fields_2d", "lattice_grid",
           "reciprocal_duals", "reciprocal_k_to_cartesian"]

# Dense eigh ceiling: N = nu*nv unknowns cost N^2 complex128, so 1024 cells
# is 16 MB and 4096 is 268 MB. Above this, TM at target_freq = 0 goes
# matrix-free: measured on a 5x5 rod supercell at resolution 16 (N = 6400,
# 8 bands) that is 220 s dense against ~0.01 s, same answer. Everything
# else raises rather than quietly allocating a gigabyte; raise this
# constant to override that.
DENSE_MAX = 4096


# --- lattice bookkeeping --------------------------------------------------

def _vec2(v) -> np.ndarray:
    """(x, y) of a Vector3, a 2-tuple or a 3-tuple, as float64."""
    a = np.asarray([v.x, v.y] if hasattr(v, "x") else v, dtype=float).ravel()
    return a[:2]


def reciprocal_duals(a1, a2):
    """``(b1, b2)`` with ``b_i . a_j = delta_ij`` (no 2pi), plus the cell area.

    Returns ``(B, area)`` where B's ROWS are b1, b2 -- B is the inverse of
    the column matrix of a1, a2, which is exactly the dual condition.
    """
    A = np.column_stack([_vec2(a1), _vec2(a2)])
    return np.linalg.inv(A), abs(np.linalg.det(A))


def reciprocal_k_to_cartesian(k, a1, a2) -> np.ndarray:
    """MPB k (reciprocal-lattice fractional) -> cartesian radians/length.

    MPB's ``k_points`` are in the reciprocal basis, so ``Vector3(-1/3, 1/3)``
    on the triangular lattice is the K point. This is what
    :func:`planewave_bands_2d` wants for ``kc``.
    """
    B, _ = reciprocal_duals(a1, a2)
    return 2 * np.pi * (_vec2(k) @ B)


def lattice_grid(nu: int, nv: int, a1, a2):
    """``(u, v, r)``: fractional coordinates and cartesian positions.

    ``u`` is (nu,), ``v`` is (nv,), ``r`` is (nu, nv, 2) with
    ``r[i, j] = u[i]*a1 + v[j]*a2``. The module's grid convention lives
    here so callers rasterizing epsilon cannot drift from the solver.
    """
    u = (np.arange(nu) + 0.5) / nu - 0.5
    v = (np.arange(nv) + 0.5) / nv - 0.5
    r = (u[:, None, None] * _vec2(a1) + v[None, :, None] * _vec2(a2))
    return u, v, r


def _kvectors(nu: int, nv: int, B: np.ndarray, kc) -> np.ndarray:
    """(N, 2) cartesian ``K_G = kc + G``, C-order over the FFT (m, n) grid."""
    m = np.rint(np.fft.fftfreq(nu) * nu)
    n = np.rint(np.fft.fftfreq(nv) * nv)
    G = m[:, None, None] * B[0] + n[None, :, None] * B[1]
    return 2 * np.pi * G.reshape(-1, 2) + _vec2(kc)[None, :]


def _operator_parts(eps, eps_te, polarization: str, K: np.ndarray):
    """``(etas, kcols)``: real-space inverse-eps grids and the K columns each
    one is contracted with, so ``A = sum_j diag(kcol_j) Eta_j diag(kcol_j)``.

    TM contracts the scalar ``1/eps`` with ``|K_G|``. TE contracts
    ``(eta_xx, eta_yy)`` with ``(zhat x K)_x, (zhat x K)_y = (-Ky, Kx)`` --
    see the module docstring for why it is (zhat x K) and not K.
    """
    if polarization == "tm":
        return (1.0 / eps,), (np.linalg.norm(K, axis=1),)
    exx, eyy = (eps, eps) if eps_te is None else eps_te
    return (1.0 / exx, 1.0 / eyy), (-K[:, 1], K[:, 0])


# --- dense assembly -------------------------------------------------------

def _shift_index(nu: int, nv: int):
    """Broadcast index pair picking ``eta_hat(G - G')`` out of ``fft2(eta)``.

    Shaped (nu,1,nu,1) and (1,nv,1,nv) so the fancy-index result is
    (nu, nv, nu, nv) = row (m,n) by column (m',n'), reshaping straight to
    (N, N) in the same C-order the coefficient vectors use. Keeping the
    index arrays tiny matters: the materialized N x N difference arrays
    would be as large as the operator itself.
    """
    mi, ni = np.arange(nu), np.arange(nv)
    return ((mi[:, None] - mi[None, :]) % nu)[:, None, :, None], \
           ((ni[:, None] - ni[None, :]) % nv)[None, :, None, :]


def _assemble(etas, kcols, nu: int, nv: int):
    """Dense Hermitian operator, numpy or torch depending on ``etas``.

    ``Eta = F diag(eta) F^H`` with F the unitary 2D DFT is the circulant
    convolution by eta-hat, i.e. ``Eta[a, b] = eta_hat(G_a - G_b)`` with
    ``eta_hat = fft2(eta)/N``. One code path serves both backends because
    the fancy indexing and the fft agree between numpy and torch.
    """
    n_cells = nu * nv
    dm, dn = _shift_index(nu, nv)
    tmode = isinstance(etas[0], torch.Tensor)
    if tmode:
        dm, dn = torch.as_tensor(dm), torch.as_tensor(dn)
    out = None
    for eta, kk in zip(etas, kcols):
        if tmode:
            ehat = torch.fft.fft2(eta) / n_cells
            kk = torch.as_tensor(kk, dtype=torch.complex128)
        else:
            ehat = np.fft.fft2(eta) / n_cells
            kk = kk.astype(complex)
        term = kk[:, None] * ehat[dm, dn].reshape(n_cells, n_cells) * kk[None, :]
        out = term if out is None else out + term
    return out


# --- eigen bookkeeping ----------------------------------------------------

def _pick(vals: np.ndarray, num_bands: int, target_freq: float) -> np.ndarray:
    """Indices of the wanted eigenvalues, ascending in frequency.

    ``target_freq`` 0 means the lowest bands; nonzero is MPB's targeted
    solver -- the ``num_bands`` modes nearest ``(2pi f)^2``, which finds a
    defect state in a gap without computing every band below it.
    """
    if target_freq > 1e-12:
        sel = np.argsort(np.abs(vals - (2 * np.pi * target_freq) ** 2))[:num_bands]
        return sel[np.argsort(vals[sel])]
    return np.argsort(vals)[:num_bands]


def _freqs(vals) -> np.ndarray:
    """omega^2 -> f = omega/2pi, with the omega = 0 band reported exactly 0.

    At Gamma the G = 0 row of both operators is identically zero (|K| = 0
    for TM, zhat x K = 0 for TE), so one eigenvalue is a numerical zero of
    order 1e-13. MPB reports band 1 = 0.0 there; snapping the window
    [0, 1e-12*max) to zero reproduces that exactly rather than to 7 digits.

    The window is one-sided on purpose. Snapping everything BELOW the
    threshold would swallow a genuinely negative eigenvalue -- which is not
    round-off but an indefinite operator, i.e. a non-positive epsilon
    somewhere -- and report it as another omega = 0 band at a generic k
    (measured: eps[0,0] = -4 turned a raw -61.5 into a reported 0.0). A
    negative eigenvalue has no real frequency, so say so instead.
    """
    vals = np.asarray(vals, dtype=float)
    hi = max(float(vals.max()) if vals.size else 0.0, 1e-300)
    if vals.size and vals.min() < -1e-12 * hi:
        raise ValueError(
            f"operator is indefinite (eigenvalue {vals.min():.4g}): the "
            f"planewave formulation needs eps > 0 everywhere, and omega = "
            f"sqrt(lambda) has no real value here")
    return np.sqrt(np.where(vals < 1e-12 * hi, 0.0, vals)) / (2 * np.pi)


def _normalize(vecs: np.ndarray, area: float) -> np.ndarray:
    """MPB's normalisation, expressed in the planewave coefficients.

    ``integral |H|^2 = area * ||h||^2`` exactly (Parseval, and the TM
    polarization vector is a unit vector), so ``||h|| = 1/sqrt(area)`` is
    the whole of it -- no field reconstruction needed. The companion
    identity ``integral Re(E* . D) = ||h||^2 * area`` then holds to
    round-off, which is what ties the operator, the reconstruction and this
    scale factor together (test_field_energy_identity).
    """
    nrm = np.linalg.norm(vecs, axis=0) * np.sqrt(area)
    return vecs / np.where(nrm == 0, 1.0, nrm)[None, :]


# --- matrix-free path (TM only) -------------------------------------------

def _solve_sparse_tm(eps: np.ndarray, kn: np.ndarray, nu: int, nv: int,
                     num_bands: int):
    """Lowest ``num_bands`` of the TM operator without ever forming it.

    The TM operator factorizes: ``A = Kn Eta Kn`` with ``Kn = diag(|K_G|)``
    and ``Eta = F diag(1/eps) F^H``, and eps > 0 makes Eta positive
    definite, so on the subspace where |K_G| > 0

        A^-1 = Kn^-1 F diag(eps) F^H Kn^-1

    exactly -- ``x -> fft2(eps * ifft2(x/|K|))/|K|`` -- and eigsh(which=
    "LM") on THAT converges in a second where eigsh(which="SA") on A never
    would: the spectrum of A runs from 0 to ~(pi*nu)^2 and its lowest bands
    have no relative separation, while its inverse's largest eigenvalues
    are exactly the well-separated ones.

    null(A) = null(Kn) exactly (Eta is PD), so exactly one direction dies
    when kc is a reciprocal lattice vector (Gamma) and none otherwise --
    K_G = kc + G vanishes for at most one G because the G are distinct.
    That direction is A's omega = 0 band and is put back by hand.

    Deflating it is where the easy version is WRONG, and silently: A on the
    live block is ``Kn_L Eta_LL Kn_L``, whose inverse needs
    ``(Eta_LL)^-1``, and ``(Eta^-1)_LL`` is not that. Just zeroing the dead
    row of Kn^-1 solves the wrong problem and returns a plausible extra
    band (measured: a spurious 0.4217 below the true 0.5606 at Gamma). The
    fix is the Schur complement, which for a single dead index is rank one:

        (Eta_LL)^-1 = M_LL - w_L w_L^H / w_d ,  w = M e_d , M = Eta^-1

    one extra transform up front and one inner product per apply.

    ponytail: TM only. TE's operator is ``Kn P^H Eta P Kn`` with P the
    transverse projection, and the projected Eta has no cheap inverse; the
    upgrade is MPB's own preconditioned block CG. Plain scipy LOBPCG with
    the diagonal 1/(|K|^2 <eta>) preconditioner was tried and rejected --
    with a random start block it returned answers two orders of magnitude
    wrong WITHOUT failing, and even seeded it would not converge for TE at
    Gamma. TE above DENSE_MAX raises rather than guesses.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh
    n_cells = nu * nv
    live = kn > 0
    dead = np.flatnonzero(~live)
    n_dead = dead.size
    inv_kn = np.where(live, 1.0 / np.where(live, kn, 1.0), 0.0)

    def apply_m(X):
        """M = Eta^-1 = F diag(eps) F^H, i.e. fft2(eps * ifft2(x))."""
        t = np.fft.ifft2(X.reshape(nu, nv, -1), axes=(0, 1))
        return np.fft.fft2(eps[:, :, None] * t, axes=(0, 1)).reshape(n_cells, -1)

    if n_dead:
        e_d = np.zeros((n_cells, 1), dtype=complex)
        e_d[dead[0]] = 1.0
        w = apply_m(e_d)[:, 0]
        w_dd = float(np.real(w[dead[0]]))

    def mm(X):
        Z = inv_kn[:, None] * np.asarray(X, dtype=complex).reshape(n_cells, -1)
        T = apply_m(Z)
        if n_dead:
            T = T - np.outer(w, (w.conj() @ Z) / w_dd)
        return inv_kn[:, None] * T

    k = num_bands - n_dead
    if k > 0:
        op = LinearOperator((n_cells, n_cells), matvec=mm, matmat=mm,
                            dtype=complex)
        mu, phi = eigsh(op, k=k, which="LM")
        order = np.argsort(-np.real(mu))          # largest 1/lam == lowest lam
        vals, vecs = 1.0 / np.real(mu)[order], phi[:, order]
    else:
        vals, vecs = np.zeros(0), np.zeros((n_cells, 0), dtype=complex)
    if n_dead:
        null_vecs = np.zeros((n_cells, n_dead), dtype=complex)
        null_vecs[dead, np.arange(n_dead)] = 1.0
        vals = np.concatenate([np.zeros(n_dead), vals])
        vecs = np.concatenate([null_vecs, vecs], axis=1)
    return vals[:num_bands], vecs[:, :num_bands]


# --- public API -----------------------------------------------------------

def planewave_bands_2d(eps, a1, a2, kc, num_bands: int,
                       polarization: str = "tm", eps_te=None,
                       differentiable: bool = False, target_freq: float = 0.0,
                       return_vectors: bool = False):
    """Lowest ``num_bands`` frequencies at cartesian Bloch vector ``kc``.

    ``eps``: (nu, nv) cell-averaged relative permittivity on the lattice
    grid of :func:`lattice_grid` -- u along ``a1``, v along ``a2``.
    ``a1``/``a2``: cartesian lattice vectors (Vector3 or length-2/3).
    ``kc``: cartesian Bloch vector in radians per unit length, i.e. 2pi
    times meep's cartesian k. See :func:`reciprocal_k_to_cartesian` to get
    there from MPB's reciprocal-basis ``k_points``.
    ``polarization``: ``"tm"`` (Ez) or ``"te"`` (Hz).
    ``eps_te``: optional ``(eps_xx, eps_yy)`` Kottke diagonal for the TE
    operator, from
    :func:`~meep_torch.geometry.smoothing.blend_te_supersampled`. The
    diagonal projection is only meaningful on a grid whose axes are the
    interface-normal axes, so this is for RECTANGULAR cells; leave it None
    on an oblique lattice. Ignored for TM.
    ``differentiable``: torch path, dense, autograd through the eigensolve
    down to ``eps``; returns a tensor instead of an array. It clamps the
    omega = 0 band at zero rather than snapping it there, so at Gamma band
    1 comes back as round-off (~1e-8) instead of exactly 0.0 -- snapping
    would sever a gradient, and the band has none worth keeping anyway.
    ``target_freq``: 0 for the lowest bands, else the bands nearest it.
    ``return_vectors``: also return ``(N, num_bands)`` complex planewave
    coefficients, C-order over the FFT (m, n) grid, normalised so the cell
    integral of |H|^2 is 1, ready for :func:`planewave_fields_2d`. Always
    detached, on both paths.

    Returns frequencies ascending, exactly ``num_bands`` of them -- band n
    is eigenvalue n, no detection step that can renumber.
    """
    if polarization not in ("tm", "te"):
        raise ValueError(f"polarization must be 'tm' or 'te', got {polarization!r}")
    eps_np = (eps.detach().cpu().numpy() if isinstance(eps, torch.Tensor)
              else np.asarray(eps, dtype=float))
    if eps_np.ndim != 2:
        raise ValueError(f"eps must be a 2D (nu, nv) grid, got {eps_np.shape}")
    if not eps_np.min() > 0:
        raise ValueError(
            f"eps must be positive everywhere (min {eps_np.min():.4g}): the "
            f"operator is built from 1/eps and goes indefinite otherwise")
    nu, nv = eps_np.shape
    n_cells = nu * nv
    if not 1 <= num_bands <= n_cells:
        raise ValueError(f"num_bands must be in 1..{n_cells}, got {num_bands}")
    # Before either backend: the torch path and the targeted solver are
    # dense-only, and a 128x128 cell would ask torch for 4.3 GB without
    # this. Only TM at target_freq = 0 has somewhere else to go.
    if n_cells > DENSE_MAX and not (polarization == "tm" and not differentiable
                                    and target_freq <= 1e-12):
        raise ValueError(
            f"{nu}x{nv} = {n_cells} planewaves needs the dense operator "
            f"({n_cells ** 2 * 16 / 2 ** 20:.0f} MB) -- only TM at "
            f"target_freq = 0 on the numpy path has a matrix-free path. "
            f"Lower the resolution or raise "
            f"meep_torch.mpb.planewave.DENSE_MAX (currently {DENSE_MAX}) "
            f"knowingly.")
    B, area = reciprocal_duals(a1, a2)
    K = _kvectors(nu, nv, B, kc)

    if differentiable:
        eps_t = eps if isinstance(eps, torch.Tensor) else torch.as_tensor(eps_np)
        te_t = (None if eps_te is None else
                tuple(torch.as_tensor(np.asarray(e, dtype=float))
                      if not isinstance(e, torch.Tensor) else e for e in eps_te))
        etas, kcols = _operator_parts(eps_t, te_t, polarization, K)
        A = _assemble(etas, kcols, nu, nv)
        A = 0.5 * (A + A.conj().T)
        vals, vecs = torch.linalg.eigh(A)
        sel = _pick(vals.detach().cpu().numpy(), num_bands, target_freq)
        idx = torch.as_tensor(sel.copy())
        freqs = torch.sqrt(torch.clamp(vals[idx], min=0.0)) / (2 * np.pi)
        if not return_vectors:
            return freqs
        return freqs, _normalize(vecs[:, idx].detach().cpu().numpy(), area)

    etas, kcols = _operator_parts(eps_np, eps_te, polarization, K)
    if n_cells > DENSE_MAX:
        vals, vecs = _solve_sparse_tm(eps_np, kcols[0], nu, nv, num_bands)
    else:
        A = _assemble(etas, kcols, nu, nv)
        A = 0.5 * (A + A.conj().T)
        if target_freq > 1e-12:
            vals, vecs = sla.eigh(A)
        else:
            # subset_by_index, not a full eigh: the whole spectrum costs
            # ~4x more and every caller here wants the bottom handful.
            vals, vecs = sla.eigh(A, subset_by_index=[0, num_bands - 1])
        sel = _pick(np.real(vals), num_bands, target_freq)
        vals, vecs = np.real(vals)[sel], vecs[:, sel]
    freqs = _freqs(vals)
    if not return_vectors:
        return freqs
    return freqs, _normalize(np.asarray(vecs), area)


def planewave_fields_2d(coeffs, eps, a1, a2, kc, freq: float,
                        polarization: str = "tm", eps_te=None,
                        envelope: bool = False):
    """Reconstruct ``(E, H)`` on the real-space grid from one band's coeffs.

    ``coeffs``: one column of :func:`planewave_bands_2d`'s ``return_vectors``
    output, (N,) or (nu, nv). ``freq``: that band's frequency (meep units).

    Returns two (nu, nv, 3) complex arrays. By default these are the BLOCH
    fields, carrying ``exp(i kc . r)`` with r from :func:`lattice_grid`;
    ``envelope=True`` returns the periodic part instead. The two differ by
    exactly that factor, which is the cheapest way to check the phase
    convention (test_bloch_phase_factor).

    Signs come from exp(-i omega t), c = mu = 1: ``D = i curl H / omega``
    and ``E = eta D`` pointwise in real space, ``B = H``. An omega = 0 band
    (Gamma, TM band 1) comes back with E and H both zero: its coefficient
    sits on the G where |K_G| = 0, which has no polarization vector, and
    the static uniform Ez it stands for carries no H for the formulation to
    hold on to. That is the honest answer, not a special case.

    Normalisation is inherited from the coefficients: with MPB's scaling
    the cell integral of |H|^2 is 1, and Re(E* . D) integrates to 1 too.
    """
    if polarization not in ("tm", "te"):
        raise ValueError(f"polarization must be 'tm' or 'te', got {polarization!r}")
    eps_np = np.asarray(eps, dtype=float)
    nu, nv = eps_np.shape
    h = np.asarray(coeffs, dtype=complex).reshape(nu, nv)
    B, _ = reciprocal_duals(a1, a2)
    K = _kvectors(nu, nv, B, kc).reshape(nu, nv, 2)
    omega = 2 * np.pi * float(freq)

    def to_real(c):
        """coefficients -> sum_G c_G exp(i G . r) on the DFT's own grid,
        r = (i/nu) a1 + (j/nv) a2 -- not lattice_grid's half-cell-offset r.

        The two differ by a rigid shift, and it cancels: the eigenvector
        already carries the matching gauge phase (the operator is the same
        one up to that diagonal unitary), so the reconstructed fields land
        on lattice_grid's cells with only a global phase to show for it.
        The energy identity closing to 1e-13 is the check on that.
        """
        return np.fft.ifft2(c) * (nu * nv)

    E = np.zeros((nu, nv, 3), dtype=complex)
    H = np.zeros((nu, nv, 3), dtype=complex)
    if polarization == "tm":
        # H_G = h_G (zhat x K_G)/|K_G|; the G with |K| = 0 (Gamma) carries no
        # transverse direction, and the eigenvector's weight there is zero
        # for every omega > 0 band because that row of A vanishes.
        kn = np.linalg.norm(K, axis=2)
        safe = np.where(kn == 0, 1.0, kn)
        ehat = np.stack([-K[..., 1], K[..., 0]], axis=-1) / safe[..., None]
        ehat[kn == 0] = 0.0
        H[..., 0] = to_real(h * ehat[..., 0])
        H[..., 1] = to_real(h * ehat[..., 1])
        if omega > 0:
            # (curl H)_z = i |K_G| h_G  ->  D_z = i(curl H)_z/omega
            E[..., 2] = to_real(-kn * h) / omega / eps_np
    else:
        H[..., 2] = to_real(h)
        if omega > 0:
            # D_G = (zhat x K_G) h_G / omega
            d = np.stack([to_real(-K[..., 1] * h), to_real(K[..., 0] * h)],
                         axis=-1) / omega
            exx, eyy = (eps_np, eps_np) if eps_te is None else eps_te
            E[..., 0] = d[..., 0] / np.asarray(exx, dtype=float)
            E[..., 1] = d[..., 1] / np.asarray(eyy, dtype=float)
    if envelope:
        return E, H
    _, _, r = lattice_grid(nu, nv, a1, a2)
    phase = np.exp(1j * (r @ _vec2(kc)))[..., None]
    return E * phase, H * phase
