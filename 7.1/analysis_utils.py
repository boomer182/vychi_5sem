from __future__ import annotations

from typing import List


EPS = 1e-12


def matrix_norm_inf(a: List[List[float]]) -> float:
    return max(sum(abs(x) for x in row) for row in a)


def vector_norm_inf(v: List[float]) -> float:
    return max(abs(x) for x in v) if v else 0.0


def is_square(a: List[List[float]]) -> bool:
    if not a:
        return False
    n = len(a)
    return all(len(row) == n for row in a)


def has_zero_row(a: List[List[float]]) -> bool:
    return any(all(abs(x) < EPS for x in row) for row in a)


def diagonal_dominance_info(a: List[List[float]]) -> str:
    strict = True
    weak = True

    for i, row in enumerate(a):
        diag = abs(row[i])
        off_diag_sum = sum(abs(row[j]) for j in range(len(row)) if j != i)

        if diag < off_diag_sum:
            weak = False
            strict = False
            break
        if diag <= off_diag_sum:
            strict = False

    if strict:
        return "матрица обладает строгим диагональным преобладанием"
    if weak:
        return "матрица обладает нестрогим диагональным преобладанием"
    return "диагональное преобладание отсутствует"


def preliminary_assessment(a: List[List[float]], b: List[float]) -> List[str]:
    comments = []

    if not a:
        comments.append("Матрица пуста.")
        return comments

    if not is_square(a):
        comments.append("Матрица коэффициентов не квадратная: система не подходит для стандартного метода Гаусса.")
        return comments

    n = len(a)
    if len(b) != n:
        comments.append("Размер вектора правой части не совпадает с размером матрицы.")
        return comments

    comments.append(f"Размер системы: {n} x {n}.")
    comments.append(f"Норма матрицы ||A||_inf = {matrix_norm_inf(a):.6g}.")
    comments.append(f"Норма правой части ||b||_inf = {vector_norm_inf(b):.6g}.")
    comments.append(f"Предварительная оценка: {diagonal_dominance_info(a)}.")

    if has_zero_row(a):
        comments.append("В матрице есть нулевая строка, система может быть вырожденной или несовместной.")
    else:
        comments.append("Нулевых строк в матрице нет.")

    comments.append(
        "Для численной устойчивости рекомендуется использовать выбор главного элемента."
    )

    return comments


def matrix_vector_product(a: List[List[float]], x: List[float]) -> List[float]:
    n = len(a)
    result = []
    for i in range(n):
        s = 0.0
        for j in range(len(a[i])):
            s += a[i][j] * x[j]
        result.append(s)
    return result


def vector_subtract(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def relative_error_inf(exact: List[float], approx: List[float]) -> float:
    numerator = vector_norm_inf(vector_subtract(approx, exact))
    denominator = vector_norm_inf(exact)
    if denominator < EPS:
        return numerator
    return numerator / denominator
