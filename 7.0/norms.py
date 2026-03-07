from __future__ import annotations

from math import sqrt

from matrix import Matrix
from vector import Vector


def vector_norm_1(v: Vector) -> float:
    return v.norm1()


def vector_norm_2(v: Vector) -> float:
    return v.norm2()


def vector_norm_inf(v: Vector) -> float:
    return v.norm_inf()


def matrix_norm_1(a: Matrix) -> float:
    return max(
        sum(abs(a[i][j]) for i in range(a.rows))
        for j in range(a.cols)
    )


def matrix_norm_inf(a: Matrix) -> float:
    return max(
        sum(abs(x) for x in row)
        for row in a.data
    )


def frobenius_norm(a: Matrix) -> float:
    return sqrt(sum(x * x for row in a.data for x in row))


def matrix_norm_2(
    a: Matrix,
    tol: float = 1e-12,
    max_iter: int = 10_000
) -> float:
    """
    Приближённое вычисление спектральной нормы:
    ||A||_2 = sqrt(lambda_max(A^T A))

    Используется степенной метод для матрицы A^T A.
    """
    if a.rows == 0 or a.cols == 0:
        raise ValueError("Matrix must be non-empty")

    ata = a.transpose() * a
    x = Vector([1.0] * ata.cols)
    x_norm = x.norm2()
    x = x / x_norm

    prev_lambda = 0.0

    for _ in range(max_iter):
        y = ata * x
        y_norm = y.norm2()

        if y_norm == 0:
            return 0.0

        x = y / y_norm
        current_lambda = x.dot(ata * x)

        if abs(current_lambda - prev_lambda) < tol:
            return sqrt(current_lambda)

        prev_lambda = current_lambda

    return sqrt(prev_lambda)


def matrix_norm(a: Matrix, p: int | str) -> float:
    if p == 1:
        return matrix_norm_1(a)
    if p == 2:
        return matrix_norm_2(a)
    if p == "inf":
        return matrix_norm_inf(a)
    raise ValueError("Supported norms: 1, 2, 'inf'")


def vector_norm(v: Vector, p: int | str) -> float:
    if p == 1:
        return vector_norm_1(v)
    if p == 2:
        return vector_norm_2(v)
    if p == "inf":
        return vector_norm_inf(v)
    raise ValueError("Supported norms: 1, 2, 'inf'")
