from __future__ import annotations

from matrix import Matrix


def inverse_matrix(a: Matrix, eps: float = 1e-12) -> Matrix:
    """
    Нахождение обратной матрицы методом Гаусса-Жордана
    через расширенную матрицу (A | E).
    """
    if not a.is_square():
        raise ValueError("Only square matrices can be inverted")

    n = a.rows
    aug = [
        a[i].copy() + Matrix.identity(n)[i].copy()
        for i in range(n)
    ]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < eps:
            raise ValueError("Matrix is singular or nearly singular")

        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    inverse_part = [row[n:] for row in aug]
    return Matrix(inverse_part)
