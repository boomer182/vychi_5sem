from __future__ import annotations

from typing import List
from matrix_utils import matrix_vector_multiply


def case_near_diagonal() -> tuple[List[List[float]], List[float], List[float], str]:
    a = [
        [10.0, 0.5, 0.2],
        [0.3, 9.0, 0.4],
        [0.2, 0.1, 8.0],
    ]
    x_exact = [1.0, 2.0, -1.0]
    b = matrix_vector_multiply(a, x_exact)
    description = (
        "Матрица близка к диагональной. "
        "Ожидается быстрая сходимость и у метода простой итерации, и у метода Зейделя."
    )
    return a, b, x_exact, description


def case_near_lower_triangular() -> tuple[List[List[float]], List[float], List[float], str]:
    a = [
        [6.0, 0.4, 0.1],
        [2.5, 7.0, 0.3],
        [1.5, 2.0, 8.0],
    ]
    x_exact = [1.0, -2.0, 3.0]
    b = matrix_vector_multiply(a, x_exact)
    description = (
        "Матрица близка к нижней треугольной. "
        "Для метода Зейделя это особенно благоприятный случай, так как новые значения сразу используются."
    )
    return a, b, x_exact, description


def case_spd() -> tuple[List[List[float]], List[float], List[float], str]:
    a = [
        [6.0, 1.0, 1.0],
        [1.0, 7.0, 1.0],
        [1.0, 1.0, 8.0],
    ]
    x_exact = [2.0, -1.0, 1.0]
    b = matrix_vector_multiply(a, x_exact)
    description = (
        "Симметричная положительно определённая матрица. "
        "По теории метод Зейделя должен сходиться при любом начальном приближении."
    )
    return a, b, x_exact, description


def initial_guess_zero(n: int) -> List[float]:
    return [0.0] * n
