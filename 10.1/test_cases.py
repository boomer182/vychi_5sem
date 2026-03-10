from __future__ import annotations

from typing import List, Tuple


def case_tall_4x3() -> Tuple[List[List[float]], str]:
    a = [
        [3.0, 1.0, 0.0],
        [1.0, 4.0, 1.0],
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
    ]
    description = "Прямоугольная высокая матрица 4x3."
    return a, description


def case_wide_3x5() -> Tuple[List[List[float]], str]:
    a = [
        [2.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 3.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 2.0, 1.0, 1.0],
    ]
    description = "Прямоугольная широкая матрица 3x5."
    return a, description


def case_square_4x4() -> Tuple[List[List[float]], str]:
    a = [
        [4.0, 1.0, 0.0, 1.0],
        [1.0, 3.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 1.0],
        [1.0, 0.0, 1.0, 2.0],
    ]
    description = "Квадратная матрица 4x4."
    return a, description
