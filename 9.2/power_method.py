from __future__ import annotations
from dataclasses import dataclass
from typing import List

from matrix_utils import (
    EPS,
    dot,
    matrix_vector_multiply,
    normalize,
    residual_vector,
    vector_norm_2,
)


@dataclass
class PowerMethodResult:
    eigenvalue: float
    eigenvector: List[float]
    iterations: int
    residual_norm: float


def power_method(
    a: List[List[float]],
    x0: List[float],
    eps: float = 1e-8,
    max_iter: int = 1000,
) -> PowerMethodResult:

    x = normalize(x0)
    lam_prev = None

    for k in range(1, max_iter + 1):

        y = matrix_vector_multiply(a, x)
        x_new = normalize(y)
        lam = dot(y, x)

        r = residual_vector(a, x_new, lam)
        res_norm = vector_norm_2(r)

        if lam_prev is not None and abs(lam - lam_prev) < eps:
            return PowerMethodResult(lam, x_new, k, res_norm)

        if res_norm < eps:
            return PowerMethodResult(lam, x_new, k, res_norm)

        x = x_new
        lam_prev = lam

    return PowerMethodResult(lam, x, max_iter, res_norm)
