from __future__ import annotations

from dataclasses import dataclass
from typing import List

from matrix_utils import (
    add_scalar_to_diagonal,
    copy_matrix,
    diagonal_entries,
    off_diagonal_frobenius_norm,
    subtract_scalar_from_diagonal,
)
from qr_decomposition import qr_householder


@dataclass
class QRStep:
    iteration: int
    matrix_snapshot: List[List[float]]
    diagonal: List[float]
    offdiag_norm: float
    shift: float


@dataclass
class QRAlgorithmResult:
    eigenvalues: List[float]
    final_matrix: List[List[float]]
    iterations: int
    steps: List[QRStep]
    operations: int
    method_name: str
    stop_reason: str


def qr_algorithm(
    a: List[List[float]],
    eps: float = 1e-10,
    max_iter: int = 500,
) -> QRAlgorithmResult:
    current = copy_matrix(a)
    steps: List[QRStep] = []
    operations = 0

    for k in range(1, max_iter + 1):
        qr = qr_householder(current)
        current = [
            [sum(qr.r[i][t] * qr.q[t][j] for t in range(len(a))) for j in range(len(a))]
            for i in range(len(a))
        ]
        operations += qr.operations + 2 * (len(a) ** 3)

        offdiag = off_diagonal_frobenius_norm(current)
        steps.append(
            QRStep(
                iteration=k,
                matrix_snapshot=copy_matrix(current),
                diagonal=diagonal_entries(current),
                offdiag_norm=offdiag,
                shift=0.0,
            )
        )

        if offdiag < eps:
            return QRAlgorithmResult(
                eigenvalues=diagonal_entries(current),
                final_matrix=current,
                iterations=k,
                steps=steps,
                operations=operations,
                method_name="Обычный QR-алгоритм",
                stop_reason="Внедиагональная норма стала меньше заданной точности.",
            )

    return QRAlgorithmResult(
        eigenvalues=diagonal_entries(current),
        final_matrix=current,
        iterations=max_iter,
        steps=steps,
        operations=operations,
        method_name="Обычный QR-алгоритм",
        stop_reason="Достигнуто максимальное число итераций.",
    )


def qr_algorithm_shifted(
    a: List[List[float]],
    eps: float = 1e-10,
    max_iter: int = 500,
) -> QRAlgorithmResult:
    current = copy_matrix(a)
    steps: List[QRStep] = []
    operations = 0
    n = len(a)

    for k in range(1, max_iter + 1):
        mu = current[n - 1][n - 1]

        shifted = subtract_scalar_from_diagonal(current, mu)
        qr = qr_householder(shifted)

        rq = [
            [sum(qr.r[i][t] * qr.q[t][j] for t in range(n)) for j in range(n)]
            for i in range(n)
        ]
        current = add_scalar_to_diagonal(rq, mu)
        operations += qr.operations + 2 * (n ** 3)

        offdiag = off_diagonal_frobenius_norm(current)
        steps.append(
            QRStep(
                iteration=k,
                matrix_snapshot=copy_matrix(current),
                diagonal=diagonal_entries(current),
                offdiag_norm=offdiag,
                shift=mu,
            )
        )

        if offdiag < eps:
            return QRAlgorithmResult(
                eigenvalues=diagonal_entries(current),
                final_matrix=current,
                iterations=k,
                steps=steps,
                operations=operations,
                method_name="QR-алгоритм со сдвигом",
                stop_reason="Внедиагональная норма стала меньше заданной точности.",
            )

    return QRAlgorithmResult(
        eigenvalues=diagonal_entries(current),
        final_matrix=current,
        iterations=max_iter,
        steps=steps,
        operations=operations,
        method_name="QR-алгоритм со сдвигом",
        stop_reason="Достигнуто максимальное число итераций.",
    )
