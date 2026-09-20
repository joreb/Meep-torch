"""
A module for fitting the complex refractive index profile over a broad
bandwidth to a sum of Lorentzian polarizability terms using gradient-based
optimization via NLopt (nlopt.readthedocs.io). The fitting parameters are
then used to define a `Medium` object.
"""
# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/eps_fit_lorentzian.py by scripts/port_upstream_example.py.
# The body below is upstream's, unchanged but for the meep import and the generated ModeSolver lines, so that a difference between
# this file and its twin is a difference between the ENGINES. See that
# script's docstring for why nothing here rewrites a call site.
#
# The literal spellings below (`matplotlib.use("Agg")`, `OUT = ...`,
# `OUT.mkdir`, `DEVICE`) are the ones
# examples/example_template.py greps for. An earlier version used private
# `_mpl` / `_OUT` names, which satisfied the RULES and failed the CHECKER on
# 216 counts -- two dialects for one convention, and the checker was right to
# complain: a rule nothing can verify is a comment.
import os
import pathlib

import matplotlib
import numpy as _np

matplotlib.use("Agg")                # rule 1: headless, before any pyplot import
for _old, _new in (("trapz", "trapezoid"), ("complex_", "complex128"),
                   ("float_", "float64"), ("alltrue", "all"),
                   ("sometrue", "any"), ("product", "prod"),
                   ("cumproduct", "cumprod"), ("round_", "round")):
    if not hasattr(_np, _old) and hasattr(_np, _new):
        setattr(_np, _old, getattr(_np, _new))   # upstream predates numpy 2.0
OUT = pathlib.Path(__file__).parent / "output/eps_fit_lorentzian/"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda"
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/


def _dump_plot_data(_fig, _path):
    """Write the NUMBERS behind a figure next to it, as <name>.plotdata.json.

    Several tutorials compute a result and only plot it -- binary_grating_
    phasemap, gaussian-beam and refl-quartz print nothing numeric and write no
    data file, so a stdout-and-files comparison has nothing to say about them
    at all. The numbers are right there in the axes.

    This lives in the generated preamble and not in the body, which is the
    whole point: it is emitted identically into both twins, so the
    byte-comparability that makes a twin difference mean anything survives. A
    record() call in the body would not.
    """
    import json as _json
    out, n = {}, 0
    for ai, ax in enumerate(_fig.get_axes()):
        for li, ln in enumerate(ax.get_lines()):
            xs, ys = ln.get_xdata(), ln.get_ydata()
            if len(xs):
                out[f"ax{ai}.line{li}.x"] = [float(v) for v in xs]
                out[f"ax{ai}.line{li}.y"] = [float(v) for v in ys]
                n += 1
        for ii, im in enumerate(ax.get_images()):
            a = im.get_array()
            if a is not None and a.size:
                import numpy as _np
                a = _np.asarray(a, dtype=float)
                out[f"ax{ai}.image{ii}.shape"] = list(a.shape)
                out[f"ax{ai}.image{ii}.stats"] = [float(a.min()), float(a.max()),
                                                  float(a.mean()),
                                                  float((a ** 2).mean() ** 0.5)]
                n += 1
    if n:
        # PREFIXED WITH THE SCRIPT STEM, because the figure's name comes from
        # upstream's body and is therefore IDENTICAL on both twins. Without the
        # prefix both write Ez_angle0.plotdata.json into one directory and the
        # second run overwrites the first -- the same defect that made the two
        # mpb twins clobber each other's h5 output.
        _p = pathlib.Path(str(_path))
        _p = _p.with_name("eps_fit_lorentzian_mp-" + _p.stem)
        _p.with_suffix(".plotdata.json").write_text(
            _json.dumps(out, indent=1, sort_keys=True))


import matplotlib.figure as _mplfig   # submodule, not an attribute of
                                     # matplotlib until imported
_savefig = _mplfig.Figure.savefig


def _savefig_and_dump(self, fname, *a, **k):
    r = _savefig(self, fname, *a, **k)
    try:
        _dump_plot_data(self, fname)
    except Exception:
        pass          # a figure we cannot read must not kill the tutorial
    return r


_mplfig.Figure.savefig = _savefig_and_dump
# --- end preamble ----------------------------------------------------------

from typing import Tuple

import matplotlib
import meep_torch.mp as mp
import nlopt
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt


def lorentzfunc(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Returns the complex ε profile given a set of Lorentzian parameters p
    (σ_0, ω_0, γ_0, σ_1, ω_1, γ_1, ...) for a set of frequencies x.
    """
    N = len(p) // 3
    y = np.zeros(len(x))
    for n in range(N):
        A_n = p[3 * n + 0]
        x_n = p[3 * n + 1]
        g_n = p[3 * n + 2]
        y = y + A_n / (np.square(x_n) - np.square(x) - 1j * x * g_n)
    return y


def lorentzerr(p: np.ndarray, x: np.ndarray, y: np.ndarray, grad: np.ndarray) -> float:
    """
    Returns the error (or residual or loss) as the L2 norm
    of the difference of ε(p,x) and y over a set of frequencies x as
    well as the gradient of this error with respect to each Lorentzian
    polarizability parameter in p and saving the result in grad.
    """
    N = len(p) // 3
    yp = lorentzfunc(p, x)
    val = np.sum(np.square(abs(y - yp)))
    for n in range(N):
        A_n = p[3 * n + 0]
        x_n = p[3 * n + 1]
        g_n = p[3 * n + 2]
        d = 1 / (np.square(x_n) - np.square(x) - 1j * x * g_n)
        if grad.size > 0:
            grad[3 * n + 0] = 2 * np.real(np.dot(np.conj(yp - y), d))
            grad[3 * n + 1] = (
                -4 * x_n * A_n * np.real(np.dot(np.conj(yp - y), np.square(d)))
            )
            grad[3 * n + 2] = (
                -2 * A_n * np.imag(np.dot(np.conj(yp - y), x * np.square(d)))
            )
    return val


def lorentzfit(
    p0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    alg=nlopt.LD_LBFGS,
    tol: float = 1e-25,
    maxeval: float = 10000,
) -> Tuple[np.ndarray, float]:
    """
    Returns the optimal Lorentzian polarizability parameters and error
    which minimize the error in ε(p0,x) relative to y for an initial
    set of Lorentzian polarizability parameters p0 over a set of
    frequencies x using the NLopt algorithm alg for a relative
    tolerance tol and a maximum number of iterations maxeval.
    """
    opt = nlopt.opt(alg, len(p0))
    opt.set_ftol_rel(tol)
    opt.set_maxeval(maxeval)
    opt.set_lower_bounds(np.zeros(len(p0)))
    opt.set_upper_bounds(float("inf") * np.ones(len(p0)))
    opt.set_min_objective(lambda p, grad: lorentzerr(p, x, y, grad))
    local_opt = nlopt.opt(nlopt.LD_LBFGS, len(p0))
    local_opt.set_ftol_rel(1e-10)
    local_opt.set_xtol_rel(1e-8)
    opt.set_local_optimizer(local_opt)
    popt = opt.optimize(p0)
    minf = opt.last_optimum_value()
    return popt, minf


if __name__ == "__main__":
    # Import the complex refractive index profile from a CSV file.
    # The file format is three comma-separated columns:
    #     wavelength (nm), real(n), imag(n).
    mydata = np.genfromtxt("mymaterial.csv", delimiter=",")
    n = mydata[:, 1] + 1j * mydata[:, 2]

    # Fitting parameter: the instantaneous (infinite frequency) dielectric.
    # Should be > 1.0 for stability and chosen such that
    # np.amin(np.real(eps)) is ~1.0. eps is defined below.
    eps_inf = 1.1

    eps = np.square(n) - eps_inf

    # Fit only the data in the wavelength range of [wl_min, wl_max].
    wl = mydata[:, 0]
    wl_min = 399  # minimum wavelength (units of nm)
    wl_max = 701  # maximum wavelength (units of nm)
    start_idx = np.where(wl > wl_min)
    idx_start = start_idx[0][0]
    end_idx = np.where(wl < wl_max)
    idx_end = end_idx[0][-1] + 1

    # The fitting function is ε(f) where f is the frequency, rather than ε(λ).
    # Note: an equally spaced grid of wavelengths results in the larger
    #       wavelengths having a finer frequency grid than smaller ones.
    #       This feature may impact the accuracy of the fit.
    freqs = 1000 / wl  # units of 1/μm
    freqs_reduced = freqs[idx_start:idx_end]
    wl_reduced = wl[idx_start:idx_end]
    eps_reduced = eps[idx_start:idx_end]

    # Fitting parameter: number of Lorentzian terms to use in the fit
    num_lorentzians = 2

    # Number of times to repeat local optimization with random initial values.
    num_repeat = 30

    ps = np.zeros((num_repeat, 3 * num_lorentzians))
    mins = np.zeros(num_repeat)
    for m in range(num_repeat):
        # Initial values for the Lorentzian polarizability terms. Each term
        # consists of three parameters (σ, ω, γ) and is chosen randomly.
        # Note: for the case of no absorption, γ should be set to zero.
        p_rand = [10 ** (np.random.random()) for _ in range(3 * num_lorentzians)]
        ps[m, :], mins[m] = lorentzfit(
            p_rand, freqs_reduced, eps_reduced, nlopt.LD_MMA, 1e-25, 50000
        )
        ps_str = "( " + ", ".join(f"{prm:.4f}" for prm in ps[m, :]) + " )"
        print(f"iteration:, {m:3d}, ps_str, {mins[m]:.6f}")

    # Find the best performing set of parameters.
    idx_opt = np.where(np.min(mins) == mins)[0][0]
    popt_str = "( " + ", ".join(f"{prm:.4f}" for prm in ps[idx_opt]) + " )"
    print(f"optimal:, {popt_str}, {mins[idx_opt]:.6f}")

    # Define a `Medium` class object using the optimal fitting parameters.
    E_susceptibilities = []

    for n in range(num_lorentzians):
        mymaterial_freq = ps[idx_opt][3 * n + 1]
        mymaterial_gamma = ps[idx_opt][3 * n + 2]

        if mymaterial_freq == 0:
            mymaterial_sigma = ps[idx_opt][3 * n + 0]
            E_susceptibilities.append(
                mp.DrudeSusceptibility(
                    frequency=1.0, gamma=mymaterial_gamma, sigma=mymaterial_sigma
                )
            )
        else:
            mymaterial_sigma = ps[idx_opt][3 * n + 0] / mymaterial_freq**2
            E_susceptibilities.append(
                mp.LorentzianSusceptibility(
                    frequency=mymaterial_freq,
                    gamma=mymaterial_gamma,
                    sigma=mymaterial_sigma,
                )
            )

    mymaterial = mp.Medium(epsilon=eps_inf, E_susceptibilities=E_susceptibilities)

    # Plot the fit and the actual data for comparison.
    mymaterial_eps = [mymaterial.epsilon(f)[0][0] for f in freqs_reduced]

    fig, ax = plt.subplots(ncols=2)

    ax[0].plot(wl_reduced, np.real(eps_reduced) + eps_inf, "bo-", label="actual")
    ax[0].plot(wl_reduced, np.real(mymaterial_eps), "ro-", label="fit")
    ax[0].set_xlabel("wavelength (nm)")
    ax[0].set_ylabel(r"real($\epsilon$)")
    ax[0].legend()

    ax[1].plot(wl_reduced, np.imag(eps_reduced), "bo-", label="actual")
    ax[1].plot(wl_reduced, np.imag(mymaterial_eps), "ro-", label="fit")
    ax[1].set_xlabel("wavelength (nm)")
    ax[1].set_ylabel(r"imag($\epsilon$)")
    ax[1].legend()

    fig.suptitle(
        f"Comparison of Actual Material Data and Fit\n"
        f"using Drude-Lorentzian Susceptibility"
    )

    fig.subplots_adjust(wspace=0.3)
    fig.savefig("eps_fit_sample.png", dpi=150, bbox_inches="tight")
