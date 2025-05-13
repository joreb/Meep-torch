# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Analytic 2D Mie scattering (infinite dielectric cylinder, TM: E parallel
to the axis; Bohren & Huffman case I). Reference for the scattering ports.

  b_n = [m J_n(mx) J_n'(x) - J_n(x) J_n'(mx)]
      / [m J_n(mx) H_n'(x) - H_n(x) J_n'(mx)],  x = k a, m = n_cyl
  Q_sca = (2/x) (|b_0|^2 + 2 sum_n |b_n|^2)
  T(theta) = b_0 + 2 sum_n b_n cos(n theta);  dsigma/dtheta ~ |T|^2
"""
import numpy as np
from scipy.special import h1vp, hankel1, jv, jvp


def mie2d_bn(m: float, x: float, n_max: int = 12) -> np.ndarray:
    n = np.arange(0, n_max + 1)
    num = m * jv(n, m * x) * jvp(n, x) - jv(n, x) * jvp(n, m * x)
    den = m * jv(n, m * x) * h1vp(n, x) - hankel1(n, x) * jvp(n, m * x)
    return num / den


def q_sca(m: float, x: float, n_max: int = 12) -> float:
    b = mie2d_bn(m, x, n_max)
    return float((2.0 / x) * (np.abs(b[0]) ** 2 + 2 * np.sum(np.abs(b[1:]) ** 2)))


def dsigma_dtheta(m: float, x: float, thetas, n_max: int = 12) -> np.ndarray:
    b = mie2d_bn(m, x, n_max)
    th = np.asarray(thetas)
    T = b[0] + 2 * sum(b[n] * np.cos(n * th) for n in range(1, n_max + 1))
    return np.abs(T) ** 2
