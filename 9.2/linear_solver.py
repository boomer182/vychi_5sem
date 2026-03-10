from __future__ import annotations
from typing import List
from matrix_utils import EPS, copy_matrix


def solve_gauss_with_partial_pivot(a: List[List[float]], b: List[float]) -> List[float]:

    n = len(a)
    m = copy_matrix(a)
    rhs = b[:]

    for k in range(n):

        pivot_row = max(range(k, n), key=lambda i: abs(m[i][k]))

        if abs(m[pivot_row][k]) < EPS:
            raise ValueError("Матрица вырождена или почти вырождена.")

        if pivot_row != k:
            m[k], m[pivot_row] = m[pivot_row], m[k]
            rhs[k], rhs[pivot_row] = rhs[pivot_row], rhs[k]

        pivot = m[k][k]

        for i in range(k + 1, n):

            factor = m[i][k] / pivot
            m[i][k] = 0.0

            for j in range(k + 1, n):
                m[i][j] -= factor * m[k][j]

            rhs[i] -= factor * rhs[k]

    x = [0.0] * n

    for i in range(n - 1, -1, -1):

        s = 0.0

        for j in range(i + 1, n):
            s += m[i][j] * x[j]

        if abs(m[i][i]) < EPS:
            raise ValueError("Ошибка обратного хода метода Гаусса.")

        x[i] = (rhs[i] - s) / m[i][i]

    return x
