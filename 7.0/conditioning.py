from __future__ import annotations

from inverse import inverse_matrix
from matrix import Matrix
from norms import matrix_norm


def condition_number(a: Matrix, p: int | str = 2) -> float:
    inv_a = inverse_matrix(a)
    return matrix_norm(a, p) * matrix_norm(inv_a, p)
