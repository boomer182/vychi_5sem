from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import List

from matrix_utils import copy_matrix, identity_matrix, matrix_multiply


@dataclass
class QRResult:
    q: List[List[float]]
    r: List[List[float]]
    operations: int
    method_name: str


def qr_householder(a: List[List[float]]) -> QRResult:
    n = len(a)
    r = copy_matrix(a)
    q = identity_matrix(n)
    ops = 0

    for k in range(n - 1):
        x = [r[i][k] for i in range(k, n)]
        norm_x = sqrt(sum(value * value for value in x))
        ops += 2 * len(x)

        if norm_x < 1e-15:
            continue

        sign = 1.0 if x[0] >= 0 else -1.0
        v = x[:]
        v[0] += sign * norm_x
        v_norm = sqrt(sum(value * value for value in v))
        ops += 2 * len(v)

        if v_norm < 1e-15:
            continue

        w = [value / v_norm for value in v]
        ops += len(v)

        h_small = identity_matrix(n - k)
        for i in range(n - k):
            for j in range(n - k):
                h_small[i][j] -= 2.0 * w[i] * w[j]
                ops += 3

        h = identity_matrix(n)
        for i in range(k, n):
            for j in range(k, n):
                h[i][j] = h_small[i - k][j - k]

        r = matrix_multiply(h, r)
        q = matrix_multiply(q, h)

        m = n
        ops += 2 * m * m * m

    return QRResult(
        q=q,
        r=r,
        operations=ops,
        method_name="Householder reflections",
    )
