from __future__ import annotations

from typing import List

from matrix_utils import (
    determinant_gauss,
    format_matrix,
    format_vector,
    is_symmetric,
    leading_principal_minor,
    matrix_vector_multiply,
)


def spd_example() -> tuple[List[List[float]], List[float], List[float], str]:
    """
    Симметричная положительно определённая матрица 5x5.
    Построена как A = B^T B + I.
    """
    a = [
        [10.0, 2.0, 1.0, 0.0, 1.0],
        [2.0, 9.0, 2.0, 1.0, 0.0],
        [1.0, 2.0, 8.0, 2.0, 1.0],
        [0.0, 1.0, 2.0, 7.0, 2.0],
        [1.0, 0.0, 1.0, 2.0, 6.0],
    ]
    x_exact = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = matrix_vector_multiply(a, x_exact)
    theory = (
        "Матрица симметрична и положительно определена, поэтому разложение "
        "A = L L^T существует, и метод Холецкого применим."
    )
    return a, b, x_exact, theory


def non_symmetric_example() -> tuple[List[List[float]], List[float], str]:
    a = [
        [4.0, 1.0, 0.0, 2.0, 1.0],
        [0.0, 3.0, 1.0, 0.0, 2.0],
        [1.0, 2.0, 5.0, 1.0, 0.0],
        [2.0, 0.0, 1.0, 6.0, 1.0],
        [0.0, 1.0, 0.0, 2.0, 4.0],
    ]
    b = [1.0, 2.0, 3.0, 4.0, 5.0]
    theory = (
        "Матрица несимметрична, а метод Холецкого требует симметричную "
        "положительно определённую матрицу. Поэтому метод неприменим."
    )
    return a, b, theory


def symmetric_not_positive_example() -> tuple[List[List[float]], List[float], str]:
    a = [
        [1.0, 2.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 2.0],
    ]
    b = [1.0, 1.0, 1.0, 1.0, 1.0]
    theory = (
        "Матрица симметрична, но не является положительно определённой. "
        "Например, её первый главный минор второго порядка отрицателен, "
        "поэтому разложение Холецкого невозможно."
    )
    return a, b, theory


def principal_minors_report(a: List[List[float]]) -> List[str]:
    report = []
    n = len(a)

    for k in range(1, n + 1):
        minor = leading_principal_minor(a, k)
        det = determinant_gauss(minor)
        report.append(f"det(A_{k}) = {det:.12g}")

    return report


def matrix_short_report(a: List[List[float]]) -> str:
    lines = []
    lines.append(f"Размер: {len(a)} x {len(a)}")
    lines.append(f"Симметрична: {'да' if is_symmetric(a) else 'нет'}")
    lines.extend(principal_minors_report(a))
    return "\n".join(lines)
