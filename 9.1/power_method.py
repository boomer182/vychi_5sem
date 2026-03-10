from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from matrix_utils import (
    EPS,
    angle_between_vectors,
    dot,
    matrix_vector_multiply,
    normalize,
    rayleigh_quotient,
    residual_vector,
    vector_norm_2,
)


@dataclass
class PowerStep:
    iteration: int
    eigenvalue: float
    eigenvector: List[float]
    residual_norm: float
    aposterior_estimate: float
    difference_lambda: Optional[float]


@dataclass
class PowerResult:
    eigenvalue: float
    eigenvector: List[float]
    steps: List[PowerStep]
    iterations: int
    stop_reason: str
    operations: int
    method_name: str


def power_method_basic(
    a: List[List[float]],
    x0: List[float],
    eps: float,
    max_iter: int = 10_000,
) -> PowerResult:
    x_prev = x0[:]
    steps: List[PowerStep] = []
    operations = 0
    prev_lambda = None

    for k in range(1, max_iter + 1):
        x_curr = matrix_vector_multiply(a, x_prev)
        n = len(x_prev)
        operations += 2 * n * n

        denominator = dot(x_prev, x_prev)
        if abs(denominator) < EPS:
            raise ValueError("Получен почти нулевой вектор в базовом степенном методе.")

        lam = dot(x_curr, x_prev) / denominator
        operations += 2 * n + 1

        x_for_report = normalize(x_curr)
        operations += n

        r = residual_vector(a, x_for_report, lam)
        res_norm = vector_norm_2(r)
        operations += 3 * n

        aposterior = res_norm / max(vector_norm_2(x_for_report), EPS)
        operations += n

        diff_lambda = None if prev_lambda is None else abs(lam - prev_lambda)

        steps.append(
            PowerStep(
                iteration=k,
                eigenvalue=lam,
                eigenvector=x_for_report,
                residual_norm=res_norm,
                aposterior_estimate=aposterior,
                difference_lambda=diff_lambda,
            )
        )

        if aposterior < eps:
            return PowerResult(
                eigenvalue=lam,
                eigenvector=x_for_report,
                steps=steps,
                iterations=k,
                stop_reason="Апостериорная оценка по невязке стала меньше eps.",
                operations=operations,
                method_name="Basic power method",
            )

        x_prev = x_curr
        prev_lambda = lam

    return PowerResult(
        eigenvalue=steps[-1].eigenvalue,
        eigenvector=steps[-1].eigenvector,
        steps=steps,
        iterations=max_iter,
        stop_reason="Достигнуто максимальное число итераций.",
        operations=operations,
        method_name="Basic power method",
    )


def power_method_normalized(
    a: List[List[float]],
    x0: List[float],
    eps: float,
    max_iter: int = 10_000,
) -> PowerResult:
    x_prev = normalize(x0)
    steps: List[PowerStep] = []
    operations = len(x0)
    prev_lambda = None

    for k in range(1, max_iter + 1):
        y = matrix_vector_multiply(a, x_prev)
        n = len(x_prev)
        operations += 2 * n * n

        lam = dot(y, x_prev)
        operations += 2 * n - 1

        x_curr = normalize(y)
        operations += n

        r = residual_vector(a, x_curr, lam)
        res_norm = vector_norm_2(r)
        operations += 3 * n

        aposterior = res_norm / max(vector_norm_2(x_curr), EPS)
        operations += n

        diff_lambda = None if prev_lambda is None else abs(lam - prev_lambda)

        steps.append(
            PowerStep(
                iteration=k,
                eigenvalue=lam,
                eigenvector=x_curr[:],
                residual_norm=res_norm,
                aposterior_estimate=aposterior,
                difference_lambda=diff_lambda,
            )
        )

        if aposterior < eps:
            return PowerResult(
                eigenvalue=lam,
                eigenvector=x_curr,
                steps=steps,
                iterations=k,
                stop_reason="Апостериорная оценка по невязке стала меньше eps.",
                operations=operations,
                method_name="Normalized power method",
            )

        x_prev = x_curr
        prev_lambda = lam

    return PowerResult(
        eigenvalue=steps[-1].eigenvalue,
        eigenvector=steps[-1].eigenvector,
        steps=steps,
        iterations=max_iter,
        stop_reason="Достигнуто максимальное число итераций.",
        operations=operations,
        method_name="Normalized power method",
    )
