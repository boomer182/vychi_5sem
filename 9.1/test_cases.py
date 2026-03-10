from __future__ import annotations

from typing import List


def case_symmetric_clear_gap() -> tuple[List[List[float]], List[float], float, str]:
    a = [
        [5.0, 1.0, 0.0],
        [1.0, 3.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    x0 = [1.0, 1.0, 1.0]
    lambda_expected = 5.414213562373095
    description = (
        "Симметричная матрица с хорошо отделённым максимальным по модулю "
        "собственным числом. Оба варианта степенного метода должны работать устойчиво."
    )
    return a, x0, lambda_expected, description


def case_large_dominant() -> tuple[List[List[float]], List[float], float, str]:
    a = [
        [20.0, 2.0, 0.0],
        [2.0, 3.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    x0 = [1.0, 1.0, 1.0]
    lambda_expected = 20.233687939614086
    description = (
        "Матрица с большим доминирующим собственным числом. "
        "На таком примере хорошо видно, зачем полезна нормировка: "
        "в базовом методе длина вектора быстро растёт."
    )
    return a, x0, lambda_expected, description


def case_small_dominant() -> tuple[List[List[float]], List[float], float, str]:
    a = [
        [0.5, 0.1, 0.0],
        [0.1, 0.2, 0.0],
        [0.0, 0.0, 0.05],
    ]
    x0 = [1.0, 1.0, 1.0]
    lambda_expected = 0.5302775637731994
    description = (
        "Матрица, у которой доминирующее собственное число по модулю меньше единицы. "
        "В базовом методе длина вектора быстро уменьшается, а нормированный вариант работает устойчивее."
    )
    return a, x0, lambda_expected, description
