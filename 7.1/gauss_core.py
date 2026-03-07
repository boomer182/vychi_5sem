from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


EPS = 1e-12


@dataclass
class GaussResult:
    solution: List[float]
    determinant: float
    operations_forward: int
    operations_backward: int
    operations_total: int
    theoretical_operations: float
    triangular_matrix: List[List[float]]
    transformed_rhs: List[float]
    row_permutation: List[int]
    pivot_strategy: str


def copy_matrix(a: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in a]


def copy_vector(v: List[float]) -> List[float]:
    return v[:]


def format_vector(v: List[float], digits: int = 10) -> str:
    return "[" + ", ".join(f"{x:.{digits}g}" for x in v) + "]"


def format_matrix(a: List[List[float]], digits: int = 10) -> str:
    lines = []
    for row in a:
        lines.append(" ".join(f"{x:12.{digits}g}" for x in row))
    return "\n".join(lines)


def theoretical_gauss_operations(n: int) -> float:
    return (2.0 / 3.0) * n**3 + (3.0 / 2.0) * n**2 - (7.0 / 6.0) * n


def argmax_abs_in_column(
    a: List[List[float]],
    start_row: int,
    col: int
) -> int:
    best_row = start_row
    best_value = abs(a[start_row][col])
    for i in range(start_row + 1, len(a)):
        current = abs(a[i][col])
        if current > best_value:
            best_value = current
            best_row = i
    return best_row


def argmax_abs_in_row(
    a: List[List[float]],
    row: int,
    start_col: int
) -> int:
    best_col = start_col
    best_value = abs(a[row][start_col])
    for j in range(start_col + 1, len(a[row])):
        current = abs(a[row][j])
        if current > best_value:
            best_value = current
            best_col = j
    return best_col


def argmax_abs_in_submatrix(
    a: List[List[float]],
    start: int
) -> tuple[int, int]:
    best_row = start
    best_col = start
    best_value = abs(a[start][start])

    n = len(a)
    for i in range(start, n):
        for j in range(start, n):
            current = abs(a[i][j])
            if current > best_value:
                best_value = current
                best_row = i
                best_col = j

    return best_row, best_col


def swap_rows(a: List[List[float]], b: List[float], i: int, j: int) -> None:
    if i != j:
        a[i], a[j] = a[j], a[i]
        b[i], b[j] = b[j], b[i]


def swap_columns(
    a: List[List[float]],
    col_order: List[int],
    i: int,
    j: int
) -> None:
    if i != j:
        for row in a:
            row[i], row[j] = row[j], row[i]
        col_order[i], col_order[j] = col_order[j], col_order[i]


def restore_solution_order(
    solution_permuted: List[float],
    col_order: List[int]
) -> List[float]:
    n = len(solution_permuted)
    solution = [0.0] * n
    for new_pos, old_pos in enumerate(col_order):
        solution[old_pos] = solution_permuted[new_pos]
    return solution


def solve_gauss(
    a_input: List[List[float]],
    b_input: List[float],
    pivot_strategy: str = "column"
) -> GaussResult:
    """
    pivot_strategy:
        'column' - выбор главного элемента по столбцу
        'row'    - выбор главного элемента по строке
        'full'   - выбор главного элемента по столбцу и строке
        'none'   - без выбора главного элемента
    """
    a = copy_matrix(a_input)
    b = copy_vector(b_input)

    n = len(a)
    if n == 0:
        raise ValueError("Матрица не должна быть пустой")

    for row in a:
        if len(row) != n:
            raise ValueError("Матрица коэффициентов должна быть квадратной")

    if len(b) != n:
        raise ValueError("Размер правой части не совпадает с размером матрицы")

    col_order = list(range(n))
    determinant_sign = 1
    operations_forward = 0

    for k in range(n):
        pivot_row = k
        pivot_col = k

        if pivot_strategy == "column":
            pivot_row = argmax_abs_in_column(a, k, k)
        elif pivot_strategy == "row":
            pivot_col = argmax_abs_in_row(a, k, k)
        elif pivot_strategy == "full":
            pivot_row, pivot_col = argmax_abs_in_submatrix(a, k)
        elif pivot_strategy == "none":
            pass
        else:
            raise ValueError(
                "Неизвестная стратегия выбора главного элемента. "
                "Используйте: 'none', 'column', 'row', 'full'."
            )

        if pivot_strategy in ("column", "full"):
            if pivot_row != k:
                swap_rows(a, b, k, pivot_row)
                determinant_sign *= -1

        if pivot_strategy in ("row", "full"):
            if pivot_col != k:
                swap_columns(a, col_order, k, pivot_col)
                determinant_sign *= -1

        pivot = a[k][k]
        if abs(pivot) < EPS:
            raise ValueError(
                "Система не может быть надежно решена методом Гаусса: "
                "получен нулевой или слишком малый ведущий элемент."
            )

        for i in range(k + 1, n):
            mu = a[i][k] / pivot
            operations_forward += 1  # деление

            a[i][k] = 0.0
            for j in range(k + 1, n):
                a[i][j] -= mu * a[k][j]
                operations_forward += 2  # умножение + вычитание

            b[i] -= mu * b[k]
            operations_forward += 2  # умножение + вычитание

    determinant = determinant_sign
    for i in range(n):
        determinant *= a[i][i]

    x_permuted = [0.0] * n
    operations_backward = 0

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += a[i][j] * x_permuted[j]
            operations_backward += 2  # умножение + сложение

        numerator = b[i] - s
        operations_backward += 1  # вычитание

        if abs(a[i][i]) < EPS:
            raise ValueError("Обратный ход невозможен: нулевой диагональный элемент.")

        x_permuted[i] = numerator / a[i][i]
        operations_backward += 1  # деление

    solution = restore_solution_order(x_permuted, col_order)
    total_operations = operations_forward + operations_backward

    return GaussResult(
        solution=solution,
        determinant=determinant,
        operations_forward=operations_forward,
        operations_backward=operations_backward,
        operations_total=total_operations,
        theoretical_operations=theoretical_gauss_operations(n),
        triangular_matrix=a,
        transformed_rhs=b,
        row_permutation=col_order,
        pivot_strategy=pivot_strategy,
    )
