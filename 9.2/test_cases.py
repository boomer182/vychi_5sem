from __future__ import annotations
from typing import List


def case_symmetric_3x3() -> tuple[List[List[float]], List[float], str]:

    a = [
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 0.0],
        [1.0, 0.0, 2.0],
    ]

    x0 = [1.0, 1.0, 1.0]

    description = (
        "Симметричная матрица 3x3. "
        "Хороший пример для метода обратных итераций и метода с отношением Рэлея."
    )

    return a, x0, description


def case_symmetric_4x4() -> tuple[List[List[float]], List[float], str]:

    a = [
        [6.0, 2.0, 0.0, 0.0],
        [2.0, 5.0, 1.0, 0.0],
        [0.0, 1.0, 4.0, 1.0],
        [0.0, 0.0, 1.0, 3.0],
    ]

    x0 = [1.0, 1.0, 1.0, 1.0]

    description = (
        "Симметричная матрица 4x4 с хорошо разделёнными собственными числами."
    )

    return a, x0, description
