from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from matrix_utils import (
    EPS,
    matrix_norm_1,
    matrix_norm_inf,
    power_method_spectral_radius,
    residual,
    vector_norm_inf,
    vector_subtract,
    zeros,
)


@dataclass
class SeidelStep:
    iteration: int
    x_prev: List[float]
    x_curr: List[float]
    diff_inf: float
    residual_inf: float
    aposterior_estimate: Optional[float]


@dataclass
class SeidelResult:
    solution: List[float]
    steps: List[SeidelStep]
    iterations: int
    operations: int
    stop_reason: str
    norm_b_inf: float
    norm_b1: float
    norm_b1_lower: float
    norm_b2_upper: float
    q_estimate_inf: Optional[float]
    q_estimate_1: Optional[float]
    spectral_radius_equivalent: float


def build_iteration_form(a: List[List[float]], b: List[float]) -> tuple[List[List[float]], List[float]]:
    n = len(a)
    b_matrix = zeros(n, n)
    c = [0.0] * n

    for i in range(n):
        if abs(a[i][i]) < EPS:
            raise ValueError("На диагонали матрицы A есть нулевой элемент.")
        c[i] = b[i] / a[i][i]

        for j in range(n):
            if i == j:
                b_matrix[i][j] = 0.0
            else:
                b_matrix[i][j] = -a[i][j] / a[i][i]

    return b_matrix, c


def split_b_matrix(b_matrix: List[List[float]]) -> tuple[List[List[float]], List[List[float]]]:
    n = len(b_matrix)
    b1 = zeros(n, n)
    b2 = zeros(n, n)

    for i in range(n):
        for j in range(n):
            if i > j:
                b1[i][j] = b_matrix[i][j]
            elif i < j:
                b2[i][j] = b_matrix[i][j]

    return b1, b2


def one_seidel_iteration(a: List[List[float]], b: List[float], x_prev: List[float]) -> tuple[List[float], int]:
    n = len(a)
    x_curr = x_prev[:]
    ops = 0

    for i in range(n):
        s1 = 0.0
        for j in range(i):
            s1 += a[i][j] * x_curr[j]
            ops += 2

        s2 = 0.0
        for j in range(i + 1, n):
            s2 += a[i][j] * x_prev[j]
            ops += 2

        x_curr[i] = (b[i] - s1 - s2) / a[i][i]
        ops += 3

    return x_curr, ops


def invert_lower_unit(l: List[List[float]]) -> List[List[float]]:
    n = len(l)
    inv = zeros(n, n)

    for j in range(n):
        inv[j][j] = 1.0
        for i in range(j + 1, n):
            s = 0.0
            for k in range(j, i):
                s += l[i][k] * inv[k][j]
            inv[i][j] = -s

    return inv


def equivalent_iteration_matrix(b1: List[List[float]], b2: List[List[float]]) -> List[List[float]]:
    n = len(b1)
    e_minus_b1 = zeros(n, n)

    for i in range(n):
        for j in range(n):
            e_minus_b1[i][j] = -b1[i][j]
        e_minus_b1[i][i] += 1.0

    inv = invert_lower_unit(e_minus_b1)

    from matrix_utils import matrix_multiply
    return matrix_multiply(inv, b2)


def seidel_method(
    a: List[List[float]],
    b: List[float],
    x0: List[float],
    eps: float,
    max_iter: int = 10_000,
) -> SeidelResult:
    b_matrix, c = build_iteration_form(a, b)
    b1, b2 = split_b_matrix(b_matrix)

    norm_b_inf = matrix_norm_inf(b_matrix)
    norm_b1 = matrix_norm_1(b_matrix)
    norm_b1_lower = matrix_norm_inf(b1)
    norm_b2_upper = matrix_norm_inf(b2)

    q_estimate_inf = None
    if norm_b_inf < 1.0:
        q_estimate_inf = norm_b2_upper / (1.0 - norm_b1_lower)

    from matrix_utils import matrix_norm_1 as mat_norm1
    q_estimate_1 = None
    lower1 = mat_norm1(b1)
    upper1 = mat_norm1(b2)
    if lower1 < 1.0:
        q_estimate_1 = upper1 / (1.0 - lower1)

    beq = equivalent_iteration_matrix(b1, b2)
    spectral_radius_equivalent = power_method_spectral_radius(beq)

    x_prev = x0[:]
    steps: List[SeidelStep] = []
    operations = 0

    for k in range(1, max_iter + 1):
        x_curr, ops = one_seidel_iteration(a, b, x_prev)
        operations += ops

        diff = vector_subtract(x_curr, x_prev)
        diff_inf = vector_norm_inf(diff)

        res = residual(a, x_curr, b)
        res_inf = vector_norm_inf(res)

        aposterior = None
        if norm_b_inf < 1.0 and norm_b2_upper > EPS:
            aposterior = norm_b2_upper / (1.0 - norm_b_inf) * diff_inf

        steps.append(
            SeidelStep(
                iteration=k,
                x_prev=x_prev[:],
                x_curr=x_curr[:],
                diff_inf=diff_inf,
                residual_inf=res_inf,
                aposterior_estimate=aposterior,
            )
        )

        if aposterior is not None and aposterior < eps:
            return SeidelResult(
                solution=x_curr,
                steps=steps,
                iterations=k,
                operations=operations,
                stop_reason="Апостериорная оценка метода Зейделя стала меньше eps.",
                norm_b_inf=norm_b_inf,
                norm_b1=norm_b1,
                norm_b1_lower=norm_b1_lower,
                norm_b2_upper=norm_b2_upper,
                q_estimate_inf=q_estimate_inf,
                q_estimate_1=q_estimate_1,
                spectral_radius_equivalent=spectral_radius_equivalent,
            )

        x_prev = x_curr

    return SeidelResult(
        solution=x_prev,
        steps=steps,
        iterations=max_iter,
        operations=operations,
        stop_reason="Достигнуто максимальное число итераций.",
        norm_b_inf=norm_b_inf,
        norm_b1=norm_b1,
        norm_b1_lower=norm_b1_lower,
        norm_b2_upper=norm_b2_upper,
        q_estimate_inf=q_estimate_inf,
        q_estimate_1=q_estimate_1,
        spectral_radius_equivalent=spectral_radius_equivalent,
    )
