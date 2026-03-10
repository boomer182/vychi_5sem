from __future__ import annotations

from typing import List
from matrix_utils import matrix_vector_multiply


def case_good_3x3() -> tuple[List[List[float]], List[float], List[float], str]:
    a = [
        [10.0, 1.0, 1.0],
        [2.0, 10.0, 1.0],
        [2.0, 2.0, 10.0],
    ]
    x_exact = [1.0, 2.0, 3.0]
    b = matrix_vector_multiply(a, x_exact)
    description = "Хорошо обусловленная диагонально преобладающая система, для которой метод должен сходиться быстро."
    return a, b, x_exact, description


def case_slow_3x3() -> tuple[List[List[float]], List[float], List[float], str]:
    a = [
        [5.0, 2.0, 1.0],
        [2.0, 5.0, 2.0],
        [1.0, 2.0, 5.0],
    ]
    x_exact = [1.0, -1.0, 2.0]
    b = matrix_vector_multiply(a, x_exact)
    description = "Система, для которой метод сходится, но заметно медленнее, так как норма матрицы B ближе к единице."
    return a, b, x_exact, description


def case_bad_3x3() -> tuple[List[List[float]], List[float], List[float], str]:
    a = [
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 2.0],
        [3.0, 2.0, 1.0],
    ]
    x_exact = [1.0, 1.0, 1.0]
    b = matrix_vector_multiply(a, x_exact)
    description = "Система, для которой простое приведение к виду x = Bx + c приводит к плохой матрице B; сходимость может отсутствовать."
    return a, b, x_exact, description


def initial_guess_zero(n: int) -> List[float]:
    return [0.0] * n


def initial_guess_ones(n: int) -> List[float]:
    return [1.0] * n


def initial_guess_rhs(b: List[float]) -> List[float]:
    return b[:]
