from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from linear_solver import solve_gauss_with_partial_pivot
from matrix_utils import (
    normalize,
    rayleigh_quotient,
    residual_vector,
    scalar_matrix_subtract,
    vector_norm_2,
)


@dataclass
class IterationStep:
    iteration: int
    eigenvalue: float
    eigenvector: List[float]
    residual_norm: float
    delta_lambda: Optional[float]


@dataclass
class InverseIterationResult:
    eigenvalue: float
    eigenvector: List[float]
    iterations: int
    residual_norm: float
    steps: List[IterationStep]
    method_name: str
    stop_reason: str


def inverse_iteration(
    a: List[List[float]],
    lambda_star: float,
    x0: List[float],
    eps: float = 1e-8,
    max_iter: int = 100,
) -> InverseIterationResult:
    x = normalize(x0)
    steps: List[IterationStep] = []
    prev_lambda = None

    shifted = scalar_matrix_subtract(a, lambda_star)

    for k in range(1, max_iter + 1):
        y = solve_gauss_with_partial_pivot(shifted, x)
        x = normalize(y)

        lam = rayleigh_quotient(a, x)
        r = residual_vector(a, x, lam)
        res_norm = vector_norm_2(r)

        delta_lambda = None if prev_lambda is None else abs(lam - prev_lambda)

        steps.append(
            IterationStep(
                iteration=k,
                eigenvalue=lam,
                eigenvector=x[:],
                residual_norm=res_norm,
                delta_lambda=delta_lambda,
            )
        )

        if res_norm < eps:
            return InverseIterationResult(
                eigenvalue=lam,
                eigenvector=x,
                iterations=k,
                residual_norm=res_norm,
                steps=steps,
                method_name="Метод обратных итераций",
                stop_reason="Невязка стала меньше заданной точности.",
            )

        prev_lambda = lam

    return InverseIterationResult(
        eigenvalue=lam,
        eigenvector=x,
        iterations=max_iter,
        residual_norm=res_norm,
        steps=steps,
        method_name="Метод обратных итераций",
        stop_reason="Достигнуто максимальное число итераций.",
    )


def inverse_iteration_rayleigh(
    a: List[List[float]],
    x0: List[float],
    eps: float = 1e-10,
    max_iter: int = 50,
) -> InverseIterationResult:
    x = normalize(x0)
    steps: List[IterationStep] = []
    prev_lambda = None

    for k in range(1, max_iter + 1):
        lam = rayleigh_quotient(a, x)
        shifted = scalar_matrix_subtract(a, lam)

        try:
            y = solve_gauss_with_partial_pivot(shifted, x)
        except ValueError:
            r = residual_vector(a, x, lam)
            res_norm = vector_norm_2(r)
            delta_lambda = None if prev_lambda is None else 0.0

            steps.append(
                IterationStep(
                    iteration=k,
                    eigenvalue=lam,
                    eigenvector=x[:],
                    residual_norm=res_norm,
                    delta_lambda=delta_lambda,
                )
            )

            return InverseIterationResult(
                eigenvalue=lam,
                eigenvector=x,
                iterations=k,
                residual_norm=res_norm,
                steps=steps,
                method_name="Метод обратных итераций с отношением Рэлея",
                stop_reason=(
                    "Матрица A - lambda I стала почти вырожденной; "
                    "это соответствует приближению к собственному числу."
                ),
            )

        x = normalize(y)
        lam_new = rayleigh_quotient(a, x)

        r = residual_vector(a, x, lam_new)
        res_norm = vector_norm_2(r)

        delta_lambda = None if prev_lambda is None else abs(lam_new - prev_lambda)

        steps.append(
            IterationStep(
                iteration=k,
                eigenvalue=lam_new,
                eigenvector=x[:],
                residual_norm=res_norm,
                delta_lambda=delta_lambda,
            )
        )

        if res_norm < eps:
            return InverseIterationResult(
                eigenvalue=lam_new,
                eigenvector=x,
                iterations=k,
                residual_norm=res_norm,
                steps=steps,
                method_name="Метод обратных итераций с отношением Рэлея",
                stop_reason="Невязка стала меньше заданной точности.",
            )

        prev_lambda = lam_new

    return InverseIterationResult(
        eigenvalue=lam_new,
        eigenvector=x,
        iterations=max_iter,
        residual_norm=res_norm,
        steps=steps,
        method_name="Метод обратных итераций с отношением Рэлея",
        stop_reason="Достигнуто максимальное число итераций.",
    )
