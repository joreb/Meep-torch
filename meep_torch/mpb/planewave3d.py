# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""MPB's planewave band eigensolver for 3D lattices, matrix-free.

The 2D sibling :mod:`meep_torch.mpb.planewave` assembles the operator
densely, which is affordable because a 2D cell has ``nu*nv`` planewaves.
A 3D cell has ``n1*n2*n3`` of them and two transverse components each --
16^3 is 8192 unknowns, so the dense operator is 1 GB and the assembly, not
the eigensolve, is the wall. MPB's own answer is to never form it: the
operator is a pair of cross products with an FFT round trip in between,
and every step of that is O(N log N).

Formulation (MPB maxwell_op.c). Expand the magnetic field as

    H(r) = sum_G H_G exp(i K_G . r),   K_G = kc + G,

which is transverse (div B = 0) exactly when ``K_G . H_G = 0``, so each G
carries only TWO degrees of freedom. Storing three and constraining them
would put a large null space in the spectrum and hand the eigensolver
spurious zero modes to find. Instead each G gets an orthonormal frame
``(t1, t2)`` perpendicular to ``K_G`` and the unknown is the coefficient
pair in that frame -- MPB's ``k_data`` m/n vectors.

With ``curl -> i K_G x`` the eigenproblem ``curl(eta curl H) = omega^2 H``
becomes ``Theta h = omega^2 h`` with

    Theta = A^H eta A,   A H = K_G x H

(``A`` is the real antisymmetric cross-product matrix, so ``A^H = -A`` and
``Theta = -A eta A``), Hermitian and positive semidefinite by construction
for any Hermitian positive-definite ``eta``. Applying it is:

    frame -> cartesian, cross with K_G, inverse FFT to real space,
    multiply by the effective inverse-dielectric TENSOR per cell,
    FFT back, cross with -K_G, project onto the frame.

Conventions match the 2D module: ``eps_inv`` lives on a uniform grid in
LATTICE (fractional) coordinates, ``kc`` is CARTESIAN in radians per unit
length (2pi times meep's cartesian k), frequencies are meep units
(``f = omega/2pi``), and coefficient arrays are C-order over the numpy FFT
frequency grid so a column reshapes to ``(n1, n2, n3, 2)`` directly.

The dense assembly here is NOT a fallback path -- it is the reference the
matrix-free apply is checked against, and it is why
:func:`theta_matvec_residual` exists. An operator bug and an eigensolver
bug produce the same wrong eigenvalues; only comparing ``Theta v`` against
``A @ v`` for the same ``v`` tells them apart.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

__all__ = ["reciprocal_duals_3d", "lattice_grid_3d", "kvectors_3d",
           "transverse_frame", "ThetaOperator", "assemble_dense_3d",
           "theta_matvec_residual", "planewave_bands_3d"]


# --- lattice bookkeeping --------------------------------------------------

def _vec3(v) -> np.ndarray:
    """(x, y, z) of a Vector3, a 3-tuple or an array, as float64."""
    a = np.asarray([v.x, v.y, v.z] if hasattr(v, "x") else v, dtype=float)
    return a.ravel()[:3]


def reciprocal_duals_3d(a1, a2, a3):
    """``(B, volume)`` with B's ROWS the duals ``b_i . a_j = delta_ij``.

    No 2pi: the factor is applied once in :func:`kvectors_3d`, the same
    split the 2D module uses. B is the inverse of the column matrix of the
    lattice vectors, which is exactly the dual condition, so this is one
    ``inv`` rather than the three cross products the textbook writes -- and
    it stays correct for a basis whose handedness is negative, where the
    cross-product form silently flips sign.
    """
    A = np.column_stack([_vec3(a1), _vec3(a2), _vec3(a3)])
    det = float(np.linalg.det(A))
    if abs(det) < 1e-12:
        raise ValueError(
            f"lattice vectors are degenerate (|det| = {abs(det):.3g}): "
            f"a1, a2, a3 must span 3D")
    return np.linalg.inv(A), abs(det)


def lattice_grid_3d(n1: int, n2: int, n3: int, a1, a2, a3):
    """``(u, v, w, r)``: fractional coordinates and cartesian positions.

    Cell ``(i, j, k)`` sits at ``u = (i + 0.5)/n1 - 1/2`` and so on, the
    half-cell-offset convention of the 2D module. MPB samples at ``i/n``;
    the offset is a rigid translation, a diagonal unitary similarity on the
    operator, and therefore invisible in the spectrum.
    """
    u = (np.arange(n1) + 0.5) / n1 - 0.5
    v = (np.arange(n2) + 0.5) / n2 - 0.5
    w = (np.arange(n3) + 0.5) / n3 - 0.5
    r = (u[:, None, None, None] * _vec3(a1)
         + v[None, :, None, None] * _vec3(a2)
         + w[None, None, :, None] * _vec3(a3))
    return u, v, w, r


def kvectors_3d(n1: int, n2: int, n3: int, B: np.ndarray, kc) -> np.ndarray:
    """``(N_G, 3)`` cartesian ``K_G = kc + G``, C-order over the FFT grid.

    ``G = 2pi (m b1 + n b2 + p b3)`` with (m, n, p) in numpy's FFT
    frequency ordering, so index ``g`` of the returned array and index
    ``g`` of a flattened ``fftn`` output are the same planewave.
    """
    m = np.rint(np.fft.fftfreq(n1) * n1)
    n = np.rint(np.fft.fftfreq(n2) * n2)
    p = np.rint(np.fft.fftfreq(n3) * n3)
    G = (m[:, None, None, None] * B[0]
         + n[None, :, None, None] * B[1]
         + p[None, None, :, None] * B[2])
    return 2 * np.pi * G.reshape(-1, 3) + _vec3(kc)[None, :]


def transverse_frame(K: np.ndarray):
    """``(t1, t2)``, each (N_G, 3), orthonormal and perpendicular to each K.

    MPB's ``k_data``. The frame is built from a reference axis that is not
    parallel to K -- zhat, swapped for xhat where K is too close to it --
    because ``K x zhat`` collapses for a K along z and the normalisation
    would divide by zero on exactly the high-symmetry points every band
    diagram visits.

    For ``K = 0`` (the G = 0 row at Gamma) any frame will do: the operator
    row is ``K x (.)`` and vanishes identically whatever the frame is, so
    there is nothing to special-case here. The two zero eigenvalues it
    contributes are physical -- a uniform H with no curl -- and the
    empty-lattice gate counts them.
    """
    K = np.asarray(K, dtype=float)
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (K.shape[0], 1))
    kn = np.linalg.norm(K, axis=1)
    safe_kn = np.where(kn == 0, 1.0, kn)
    khat = K / safe_kn[:, None]
    # |khat . zhat| near 1 means zhat is a bad reference; xhat cannot also
    # be parallel to it, so one swap is enough.
    near_z = np.abs(khat @ np.array([0.0, 0.0, 1.0])) > 0.9
    ref[near_z] = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(khat, ref)
    n1 = np.linalg.norm(t1, axis=1)
    # K = 0 rows: khat is zero, so t1 is zero. Plant a valid frame anyway
    # rather than leaving NaN to propagate through an inner product.
    dead = n1 < 1e-12
    t1[dead] = np.array([1.0, 0.0, 0.0])
    n1 = np.where(dead, 1.0, n1)
    t1 = t1 / n1[:, None]
    t2 = np.cross(khat, t1)
    t2[dead] = np.array([0.0, 1.0, 0.0])
    return t1, t2


# --- effective inverse dielectric ----------------------------------------

def _as_eps_inv_tensor(eps_inv, shape):
    """Accept a scalar grid or a full tensor grid; return (n1, n2, n3, 3, 3).

    A scalar grid is ``eta * I``. Passing epsilon rather than its inverse
    is the easy mistake here and it is not detectable from the values, so
    the argument is named for what it is and never inverted internally.
    """
    a = np.asarray(eps_inv)
    if a.shape == shape:
        out = np.zeros(shape + (3, 3), dtype=float)
        idx = np.arange(3)
        out[..., idx, idx] = a[..., None]
        return out
    if a.shape == shape + (3, 3):
        return np.asarray(a, dtype=float)
    raise ValueError(
        f"eps_inv must be {shape} (scalar per cell) or {shape + (3, 3)} "
        f"(tensor per cell), got {a.shape}")


# --- the operator ---------------------------------------------------------

class ThetaOperator:
    """``Theta = A^H eta A`` applied matrix-free, ``A H = K_G x H``.

    Holds the frame and the real-space inverse-dielectric tensor; the
    apply is three FFTs' worth of work per column and allocates nothing
    that scales as N^2. ``shape`` is (2 N_G, 2 N_G) so it drops straight
    into ``scipy.sparse.linalg.LinearOperator``.
    """

    def __init__(self, eps_inv, K: np.ndarray, grid_shape, device="cpu"):
        self.device = str(device)
        self.grid = tuple(int(s) for s in grid_shape)
        self.n_g = int(np.prod(self.grid))
        self.n = 2 * self.n_g
        self.K = np.asarray(K, dtype=float)
        self.t1, self.t2 = transverse_frame(self.K)
        self.eta = _as_eps_inv_tensor(eps_inv, self.grid)
        self.shape = (self.n, self.n)
        self.dtype = np.dtype(complex)
        # Diagonal Fourier preconditioner, MPB maxwell_pre.c: the operator
        # is |K|^2 times a mean inverse-dielectric, so this is its symbol.
        eta_mean = float(np.mean(np.trace(self.eta, axis1=-2, axis2=-1)) / 3.0)
        kk = np.sum(self.K ** 2, axis=1) * max(eta_mean, 1e-300)
        self._precond_diag = np.where(kk > 0, 1.0 / np.where(kk > 0, kk, 1.0), 0.0)
        self._t = None
        if self.device != "cpu":
            self._t = _TorchArrays(self)

    # -- pieces, exposed because the tests check them individually --------

    def to_cartesian(self, h):
        """(2 N_G, m) frame coefficients -> (N_G, 3, m) cartesian H."""
        h = np.asarray(h, dtype=complex).reshape(self.n_g, 2, -1)
        return (self.t1[:, :, None] * h[:, 0][:, None, :]
                + self.t2[:, :, None] * h[:, 1][:, None, :])

    def to_frame(self, v):
        """(N_G, 3, m) cartesian -> (2 N_G, m) frame coefficients."""
        c1 = np.einsum("gi,gim->gm", self.t1, v)
        c2 = np.einsum("gi,gim->gm", self.t2, v)
        return np.stack([c1, c2], axis=1).reshape(self.n, -1)

    def _cross_k(self, v):
        """``K_G x v`` for (N_G, 3, m)."""
        return np.cross(self.K[:, :, None], v, axisa=1, axisb=1, axisc=1)

    def _eta_apply(self, v):
        """Real-space multiply by the inverse-dielectric tensor.

        The FFT normalisations cancel exactly -- coefficients to real space
        is ``N * ifftn`` and back is ``fftn / N`` -- so neither factor is
        written, which also keeps the apply free of a size-dependent
        constant that would drift between here and the dense assembly.
        """
        g = self.grid
        m = v.shape[-1]
        f = np.fft.ifftn(v.reshape(g + (3, m)), axes=(0, 1, 2))
        f = np.einsum("xyzij,xyzjm->xyzim", self.eta, f)
        f = np.fft.fftn(f, axes=(0, 1, 2))
        return f.reshape(self.n_g, 3, m)

    # -- the apply --------------------------------------------------------

    def matmat(self, X):
        """``Theta X``. numpy in, numpy out, whichever device runs it.

        The CPU path stays numpy on purpose: every gate in the suite was
        measured against it, and torch-on-CPU would differ in the last bits
        of the FFT. The GPU path is a separate implementation of the same
        five steps, held to the numpy one at 1e-12 by
        ``test_cuda_matches_cpu_bands``.
        """
        X = np.asarray(X, dtype=complex)
        squeeze = X.ndim == 1
        if squeeze:
            X = X[:, None]
        if self._t is not None:
            out = self._t.matmat(X)
        else:
            d = self._cross_k(self.to_cartesian(X))
            e = self._eta_apply(d)
            out = self.to_frame(-self._cross_k(e))
        return out[:, 0] if squeeze else out

    matvec = matmat

    def precond(self, X):
        """``M^-1 X`` with M diagonal in the planewave basis."""
        X = np.asarray(X, dtype=complex)
        d = np.repeat(self._precond_diag, 2)
        return d[:, None] * X if X.ndim == 2 else d * X


class _TorchArrays:
    """The same operator on a torch device, for the 3D apply only.

    Why the apply and not the whole eigensolve: scipy's LOBPCG is the
    driver and it is cheap -- the cost is the matvec, which is six 3D FFTs
    per column plus a tensor contraction. Moving just that costs two host
    transfers of the block per apply and keeps the convergence behaviour,
    the retry and every gate exactly as measured.

    ``torch.lobpcg`` would avoid the transfers but does not accept complex
    Hermitian input, so the whole-in-torch version would need its own
    block iteration; that is a bigger change than the transfers are worth
    until they show up in a profile.
    """

    def __init__(self, op):
        import torch
        self.torch = torch
        self.dev = torch.device(op.device)
        self.grid = op.grid
        self.n_g, self.n = op.n_g, op.n
        cd = torch.complex128
        self.K = torch.as_tensor(op.K, device=self.dev, dtype=torch.float64)
        self.t1 = torch.as_tensor(op.t1, device=self.dev, dtype=cd)
        self.t2 = torch.as_tensor(op.t2, device=self.dev, dtype=cd)
        self.eta = torch.as_tensor(op.eta, device=self.dev, dtype=cd)
        self.Kc = self.K.to(cd)

    def matmat(self, X):
        torch = self.torch
        x = torch.as_tensor(np.ascontiguousarray(X), device=self.dev,
                            dtype=torch.complex128)
        m = x.shape[1]
        h = x.reshape(self.n_g, 2, m)
        v = (self.t1[:, :, None] * h[:, 0][:, None, :]
             + self.t2[:, :, None] * h[:, 1][:, None, :])
        v = torch.linalg.cross(self.Kc[:, :, None].expand_as(v), v, dim=1)
        f = torch.fft.ifftn(v.reshape(self.grid + (3, m)), dim=(0, 1, 2))
        f = torch.einsum("xyzij,xyzjm->xyzim", self.eta, f)
        f = torch.fft.fftn(f, dim=(0, 1, 2)).reshape(self.n_g, 3, m)
        f = -torch.linalg.cross(self.Kc[:, :, None].expand_as(f), f, dim=1)
        c1 = torch.einsum("gi,gim->gm", self.t1, f)
        c2 = torch.einsum("gi,gim->gm", self.t2, f)
        out = torch.stack([c1, c2], dim=1).reshape(self.n, m)
        return out.cpu().numpy()

    def mirror(self, X, axis: int, signs, keep):
        """The parity mirror, on device.

        Left on the host it would dominate a GPU parity run: the projector
        is applied twice per operator apply, and each application is two
        more 3D FFTs of the same block.
        """
        torch = self.torch
        x = torch.as_tensor(np.ascontiguousarray(X), device=self.dev,
                            dtype=torch.complex128)
        keep_t = torch.as_tensor(keep, device=self.dev,
                                 dtype=torch.complex128)[:, None]
        x = x * keep_t
        m = x.shape[1]
        h = x.reshape(self.n_g, 2, m)
        v = (self.t1[:, :, None] * h[:, 0][:, None, :]
             + self.t2[:, :, None] * h[:, 1][:, None, :])
        f = torch.fft.ifftn(v.reshape(self.grid + (3, m)), dim=(0, 1, 2))
        n_ax = self.grid[axis]
        idx = torch.as_tensor((-np.arange(n_ax)) % n_ax, device=self.dev)
        f = torch.index_select(f, axis, idx)
        sg = torch.as_tensor(signs, device=self.dev,
                             dtype=torch.complex128)
        f = f * sg[None, None, None, :, None]
        f = torch.fft.fftn(f, dim=(0, 1, 2)).reshape(self.n_g, 3, m)
        c1 = torch.einsum("gi,gim->gm", self.t1, f)
        c2 = torch.einsum("gi,gim->gm", self.t2, f)
        out = torch.stack([c1, c2], dim=1).reshape(self.n, m) * keep_t
        return out.cpu().numpy()


def assemble_dense_3d(eps_inv, K: np.ndarray, grid_shape) -> np.ndarray:
    """The explicit ``(2 N_G, 2 N_G)`` operator -- reference, not a path.

    Built from the convolution form directly,

        Theta[(G,a),(G',b)] = conj(K_G x t_a(G)) . etahat(G - G') . (K_G' x t_b(G'))

    with ``etahat = fftn(eta)/N``, so it shares no code with
    :meth:`ThetaOperator.matmat` beyond the frame. That independence is the
    whole point: it is what makes the agreement between the two evidence
    rather than a tautology.

    Cost is ``(2 N_G)^2`` complex128, so 8^3 is 16 MB and 16^3 is 1 GB.
    Callers are expected to keep this to small grids.
    """
    op = ThetaOperator(eps_inv, K, grid_shape)
    g, n_g = op.grid, op.n_g
    if op.n > 4096:
        raise ValueError(
            f"assemble_dense_3d refuses {op.n} x {op.n} "
            f"({op.n ** 2 * 16 / 2 ** 20:.0f} MB); it exists to check the "
            f"matrix-free apply on a small grid, not to solve real cells")
    # etahat[dG] for every index difference, as (n1, n2, n3, 3, 3).
    etahat = np.fft.fftn(op.eta, axes=(0, 1, 2)) / n_g
    # Index difference table over the flattened G ordering.
    idx = [np.arange(s) for s in g]
    diffs = []
    for axis, s in enumerate(g):
        d = (idx[axis][:, None] - idx[axis][None, :]) % s
        diffs.append(d)
    # (N_G, N_G) flat index into the etahat grid.
    strides = (g[1] * g[2], g[2], 1)
    flat = np.zeros((n_g, n_g), dtype=np.int64)
    ii = np.unravel_index(np.arange(n_g), g)
    for axis in range(3):
        flat += diffs[axis][ii[axis][:, None], ii[axis][None, :]] * strides[axis]
    eh = etahat.reshape(n_g, 3, 3)[flat]              # (N_G, N_G, 3, 3)

    # A t_a(G) = K_G x t_a(G), the two columns per G.
    at = np.stack([np.cross(op.K, op.t1), np.cross(op.K, op.t2)], axis=1)
    at = at.reshape(op.n, 3)                          # row (G, a)
    theta = np.einsum("ri,rcij,cj->rc", at.conj(),
                      eh.repeat(2, axis=0).repeat(2, axis=1), at)
    return 0.5 * (theta + theta.conj().T)


def _eig_residuals(op: "ThetaOperator", vals, vecs, scale=None) -> np.ndarray:
    """Relative ``||Theta v - lambda v|| / lambda`` per column.

    The only honest convergence statement. scipy's LOBPCG reports its own
    accuracies through a UserWarning and then silently postprocesses, so a
    run that warns can still be right and a run that does not warn is not
    thereby verified -- measured here, one column sat at 2.8e+01 at the
    iteration cap and came back at 1e-6 after postprocessing. Recomputing
    the residual against the operator costs one apply per band and does not
    depend on believing any of that.
    """
    vals = np.asarray(vals, dtype=float)
    R = op.matmat(vecs) - vecs * vals.astype(complex)[None, :]
    # Normalised by the SPECTRAL scale, not by each band's own eigenvalue.
    # At Gamma the K_G = 0 row makes two eigenvalues exactly zero, and a
    # per-band denominator turns their round-off residual into 1e+11 and
    # fails a converged solve. The zero bands are physical and their
    # eigenvectors are as accurate as any other.
    # The scale comes from the whole solved BLOCK, not from the bands being
    # returned. At Gamma on an empty lattice every requested band can be one
    # of the two omega = 0 modes, and a scale taken from those alone is zero
    # -- the normalisation then divides round-off by 1e-300 and fails a
    # perfectly converged solve (measured 4.2e+04). Falling back to the
    # operator's own magnitude keeps the criterion meaningful there.
    if scale is None:
        scale = float(np.max(np.abs(vals))) if vals.size else 0.0
    if not scale > 0:
        scale = float(np.max(np.sum(op.K ** 2, axis=1))) or 1.0
    return np.linalg.norm(R, axis=0) / scale


def theta_matvec_residual(eps_inv, K, grid_shape, n_probe: int = 4,
                          seed: int = 0):
    """Largest relative ``||Theta v - A v||`` over random probes.

    The gate that separates an operator bug from an eigensolver bug. Run it
    before ever comparing eigenvalues: both bugs move the spectrum, only
    this one moves the matvec.
    """
    op = ThetaOperator(eps_inv, K, grid_shape)
    A = assemble_dense_3d(eps_inv, K, grid_shape)
    rng = np.random.default_rng(seed)
    V = (rng.standard_normal((op.n, n_probe))
         + 1j * rng.standard_normal((op.n, n_probe)))
    got, want = op.matmat(V), A @ V
    den = np.linalg.norm(want, axis=0)
    den = np.where(den == 0, 1.0, den)
    return float(np.max(np.linalg.norm(got - want, axis=0) / den))


# --- public API -----------------------------------------------------------

def _freqs_3d(vals, scale=None) -> np.ndarray:
    """omega^2 -> f = omega/2pi, sharing the 2D module's zero/negative rule.

    An eigenvalue below ``-1e-12 * max`` is not round-off: the operator is
    Hermitian PSD for any positive-definite eta, so a negative value means
    the inverse-dielectric tensor was not positive definite, i.e. a
    non-positive epsilon somewhere. Saying so beats reporting a spurious
    ``omega = 0`` band, which is what clamping would do.
    """
    vals = np.asarray(vals, dtype=float)
    # Same scaling argument as the residual gate: the threshold that
    # separates "round-off zero" from "indefinite operator" has to come
    # from the operator, not from the bands being returned. At Gamma with
    # num_bands = 1 the only returned band IS the zero mode, and a
    # self-scaled threshold turned its -7.1e-15 into a hard error.
    if scale is None:
        scale = float(vals.max()) if vals.size else 0.0
    hi = max(float(scale), 1e-300)
    if vals.size and vals.min() < -1e-12 * hi:
        raise ValueError(
            f"operator is indefinite (eigenvalue {vals.min():.4g}): the "
            f"effective inverse-dielectric tensor must be positive definite, "
            f"which needs eps > 0 in every cell")
    # The zero window is RELATIVE 1e-9, not 1e-12. The omega = 0 modes at
    # Gamma are structurally exact -- the K_G = 0 row is identically null --
    # but the eigensolver only resolves them to whatever tolerance it ran
    # at, and MPB's eigenvalue-change stopping rule leaves them further from
    # zero than a residual rule did: band 2 came back at 2.6e-06 instead of
    # 0.0. A genuine band inside this window would mean f/f_max ~ 3e-5,
    # which is not a band. The one-sided test still lets a NEGATIVE
    # eigenvalue through to the indefinite check above.
    return np.sqrt(np.where(vals < 1e-9 * hi, 0.0, vals)) / (2 * np.pi)


def planewave_bands_3d(eps_inv, a1, a2, a3, kc, num_bands: int,
                       grid_shape=None, tol: float = 1e-7,
                       maxiter: int = 500, extra: int = 6,
                       seed: int = 0, return_vectors: bool = False,
                       parity=None, residual_warn: float = 1e-3,
                       device: str = "cpu"):
    """Lowest ``num_bands`` frequencies at cartesian Bloch vector ``kc``.

    ``eps_inv``: ``(n1, n2, n3)`` scalar or ``(n1, n2, n3, 3, 3)`` tensor
    INVERSE permittivity, from
    :func:`~meep_torch.mpb.raster3d.eps_inv_from_geometry`.
    ``a1``/``a2``/``a3``: cartesian lattice vectors.
    ``kc``: cartesian Bloch vector, radians per unit length.
    ``parity``: optional projector callable applied inside the operator and
    to the start block, from :mod:`meep_torch.mpb.parity3d`.
    Convergence is MPB's rule: the relative change in the EIGENVALUES
    between the last two iterations, against ``tol`` (``ms.tolerance``).
    It is read off lobpcg's own lambda history rather than from its
    residual estimate, and it does not raise -- MPB does not either. A
    marginal miss at a degenerate level is a slightly less converged
    number, not a failed calculation, and raising on one turned
    ``mpb_data_analysis`` into a traceback over a residual of 1.259e-05
    against a 1e-05 threshold.

    ``residual_warn``: the recomputed ``||Theta v - lambda v||`` is still
    measured against the operator and WARNS above this, because scipy's
    LOBPCG reports non-convergence through a warning it then postprocesses
    away and the eigenvalue history alone cannot see a stalled sector --
    the hole-slab symmetry bug sat at 2e-3 while looking converged.

    The block carries ``extra`` spare columns. Degenerate bands are the
    reason: diamond at X has a four-fold level, and a block exactly the
    size of the wanted set has no room to separate a degenerate cluster
    from the next one up.
    """
    from scipy.sparse.linalg import LinearOperator, lobpcg
    import warnings

    a = np.asarray(eps_inv)
    if grid_shape is None:
        grid_shape = a.shape[:3] if a.ndim >= 3 else a.shape
    B, _ = reciprocal_duals_3d(a1, a2, a3)
    K = kvectors_3d(*grid_shape, B, kc)
    op = ThetaOperator(eps_inv, K, grid_shape, device=device)
    if not 1 <= num_bands <= op.n:
        raise ValueError(f"num_bands must be in 1..{op.n}, got {num_bands}")

    apply_a, apply_m = op.matmat, op.precond
    if parity is not None:
        # P Theta P alone is WRONG for a smallest-eigenvalue solver: it
        # sends the entire discarded sector to zero, so the bottom of the
        # spectrum is a null space of dimension ~N/2 and lobpcg returns
        # that instead of the bands (measured: residual 0.89, and the
        # "frequencies" are round-off). Lifting the complement above the
        # spectrum with a shift makes the wanted sector the true bottom.
        # The shift is an upper bound on Theta's spectrum: |K|^2 times the
        # largest eigenvalue of eta, doubled.
        eta_hi = float(np.max(np.linalg.eigvalsh(op.eta.reshape(-1, 3, 3))))
        shift = 2.0 * float(np.max(np.sum(op.K ** 2, axis=1))) * max(eta_hi, 1e-300)

        def apply_a(X, _p=parity, _op=op, _s=shift):
            PX = _p(X)
            return _p(_op.matmat(PX)) + _s * (X - PX)

        def apply_m(X, _p=parity, _op=op, _s=shift):
            PX = _p(X)
            return _p(_op.precond(PX)) + (1.0 / _s) * (X - PX)

    block = min(num_bands + max(extra, 0), op.n)
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((op.n, block))
         + 1j * rng.standard_normal((op.n, block)))
    if parity is not None:
        X = parity(X)
    A = LinearOperator(op.shape, matvec=apply_a, matmat=apply_a, dtype=complex)
    M = LinearOperator(op.shape, matvec=apply_m, matmat=apply_m, dtype=complex)
    with warnings.catch_warnings():
        # scipy's own accuracy warning is superseded by the two checks
        # below, both measured against the operator rather than against
        # the solver's internal estimate.
        warnings.simplefilter("ignore", UserWarning)
        vals, vecs, hist = lobpcg(A, X, M=M, tol=tol, maxiter=maxiter,
                                  largest=False, retLambdaHistory=True)
    n_iter = max(len(hist) - 1, 0)
    eig_change = float("inf")
    if len(hist) >= 2:
        # lobpcg's history can carry a complex dtype with zero
        # imaginary part; take the real part rather than let numpy
        # warn about discarding it on every k-point.
        last = np.real(np.asarray(hist[-1])).ravel()
        prev = np.real(np.asarray(hist[-2])).ravel()
        m = min(last.size, prev.size, num_bands)
        scale = max(float(np.max(np.abs(last[:m]))), 1e-300)
        eig_change = float(np.max(np.abs(last[:m] - prev[:m])) / scale)
    order = np.argsort(np.real(vals))
    vals, vecs = np.real(vals)[order], vecs[:, order]

    block_scale = float(np.max(np.abs(vals))) if vals.size else 0.0
    res = _eig_residuals(op, vals[:num_bands], vecs[:, :num_bands],
                         scale=block_scale)
    worst = float(np.max(res, initial=0.0))

    if worst > residual_warn:
        # scipy's LOBPCG stops on its OWN residual estimate and then
        # postprocesses, so it can declare success early and be badly
        # wrong. Measured on the hole slab at 9 bands, resolution 16: it
        # returned after 117 iterations of a 500 budget with the true
        # residual at 7.1e-01 -- not a marginal miss, a wrong answer that
        # looks converged, and the eigenvalue-change criterion agreed with
        # it (1.7e-02). Retry once, seeded with the vectors it returned and
        # at a tighter tolerance, before saying anything: that took the
        # same case to 3.0e-09. A warning the caller cannot act on is
        # worse than a second attempt, and returning the 0.71 silently
        # would be worse than either.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            v2, w2, hist2 = lobpcg(A, vecs, M=M, tol=tol * 1e-3,
                                   maxiter=maxiter, largest=False,
                                   retLambdaHistory=True)
        o2 = np.argsort(np.real(v2))
        v2, w2 = np.real(v2)[o2], w2[:, o2]
        s2 = float(np.max(np.abs(v2))) if v2.size else 0.0
        r2 = _eig_residuals(op, v2[:num_bands], w2[:, :num_bands], scale=s2)
        if float(np.max(r2, initial=0.0)) < worst:
            vals, vecs, res = v2, w2, r2
            block_scale, worst = s2, float(np.max(r2, initial=0.0))
            n_iter += max(len(hist2) - 1, 0)
            if len(hist2) >= 2:
                a_, b_ = (np.real(np.asarray(hist2[-1])).ravel(),
                          np.real(np.asarray(hist2[-2])).ravel())
                m_ = min(a_.size, b_.size, num_bands)
                sc_ = max(float(np.max(np.abs(a_[:m_]))), 1e-300)
                eig_change = float(np.max(np.abs(a_[:m_] - b_[:m_])) / sc_)

    if worst > residual_warn:
        warnings.warn(
            f"planewave 3D eigensolve is weakly converged: worst band "
            f"residual ||Theta v - lambda v|| = {worst:.3e} (> "
            f"{residual_warn:g}) after {n_iter} iterations, relative "
            f"eigenvalue change {eig_change:.3e}. Raise maxiter (currently "
            f"{maxiter}), raise extra (currently {extra}; degenerate levels "
            f"need room), or tighten ms.tolerance.", RuntimeWarning,
            stacklevel=2)
    planewave_bands_3d.last_stats = {
        "iterations": n_iter, "eig_change": eig_change,
        "residual": worst, "converged": eig_change <= tol}

    freqs = _freqs_3d(vals[:num_bands], scale=block_scale)
    if not return_vectors:
        return freqs
    return freqs, vecs[:, :num_bands]
