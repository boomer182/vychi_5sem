from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import List

from matrix_utils import copy_matrix, identity_matrix, transpose


@dataclass
class QRResult:
    q: List[List[float]]
    r: List[List[float]]
    operations: int
    method_name: str


def qr_givens(a: List[List[float]]) -> QRResult:
    n = len(a)
    r = copy_matrix(a)
    q_left = identity_matrix(n)
    ops = 0

    for j in range(n):
        for i in range(j + 1, n):
            if abs(r[i][j]) < 1e-15:
                continue

            denom = hypot(r[j][j], r[i][j])
            c = r[j][j] / denom
            s = r[i][j] / denom
            ops += 5

            for k in range(j, n):
                temp1 = c * r[j][k] + s * r[i][k]
                temp2 = -s * r[j][k] + c * r[i][k]
                r[j][k] = temp1
                r[i][k] = temp2
                ops += 6

            for k in range(n):
                temp1 = c * q_left[j][k] + s * q_left[i][k]
                temp2 = -s * q_left[j][k] + c * q_left[i][k]
                q_left[j][k] = temp1
                q_left[i][k] = temp2
                ops += 6

    q = transpose(q_left)

    return QRResult(
        q=q,
        r=r,
        operations=ops,
        method_name="Givens rotations",
    )
