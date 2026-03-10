from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import List, Tuple

from matrix_utils import (
    EPS,
    copy_matrix,
    diag_rect,
    format_vector,
    identity,
    matmul,
    offdiag_fro_tridiagonal_like,
    transpose,
    zeros,
)


@dataclass
class BidiagonalQRStep:
    iteration: int
    shift: float
    offdiag_norm: float
    diag_snapshot: List[float]


@dataclass
class BidiagonalQRResult:
    sigma: List[float]
    u_tilde: List[List[float]]
    v_tilde: List[List[float]]
    t_final: List[List[float]]
    steps: List[BidiagonalQRStep]
    stop_reason: str


def bidiagonal_to_btb(b: List[List[float]]) -> List[List[float]]:
    bt = transpose(b)
    return matmul(bt, b)


def wilkinson_shift_2x2(a: float, b: float, d: float) -> float:
    delta = (a - d) / 2.0
    sign = 1.0 if delta >= 0 else -1.0
    return d - sign * b * b / (abs(delta) + sqrt(delta * delta + b * b))


def givens_qr_symmetric_tridiagonal(t: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    n = len(t)
    r = copy_matrix(t)
    q = identity(n)

    for j in range(n - 1):
        a = r[j][j]
        b = r[j + 1][j]
        if abs(b) < EPS:
            continue

        rho = sqrt(a * a + b * b)
        c = a / rho
        s = b / rho

        for k in range(j, n):
            x = r[j][k]
            y = r[j + 1][k]
            r[j][k] = c * x + s * y
            r[j + 1][k] = -s * x + c * y

        for k in range(n):
            x = q[k][j]
            y = q[k][j + 1]
            q[k][j] = c * x + s * y
            q[k][j + 1] = -s * x + c * y

    return q, r


def jacobi_eigen_symmetric(a: List[List[float]], eps: float = 1e-12, max_iter: int = 5000) -> Tuple[List[float], List[List[float]]]:
    n = len(a)
    d = copy_matrix(a)
    v = identity(n)

    for _ in range(max_iter):
        p = 0
        q = 1
        best = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(d[i][j]) > best:
                    best = abs(d[i][j])
                    p, q = i, j

        if best < eps:
            return [d[i][i] for i in range(n)], v

        if abs(d[p][p] - d[q][q]) < EPS:
            angle_c = 1.0 / sqrt(2.0)
            angle_s = 1.0 / sqrt(2.0)
        else:
            tau = (d[q][q] - d[p][p]) / (2.0 * d[p][q])
            t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + sqrt(1.0 + tau * tau))
            angle_c = 1.0 / sqrt(1.0 + t * t)
            angle_s = t * angle_c

        for k in range(n):
            if k != p and k != q:
                dkp = d[k][p]
                dkq = d[k][q]
                d[k][p] = angle_c * dkp - angle_s * dkq
                d[p][k] = d[k][p]
                d[k][q] = angle_s * dkp + angle_c * dkq
                d[q][k] = d[k][q]

        dpp = d[p][p]
        dqq = d[q][q]
        dpq = d[p][q]
        d[p][p] = angle_c * angle_c * dpp - 2.0 * angle_c * angle_s * dpq + angle_s * angle_s * dqq
        d[q][q] = angle_s * angle_s * dpp + 2.0 * angle_c * angle_s * dpq + angle_c * angle_c * dqq
        d[p][q] = 0.0
        d[q][p] = 0.0

        for k in range(n):
            vkp = v[k][p]
            vkq = v[k][q]
            v[k][p] = angle_c * vkp - angle_s * vkq
            v[k][q] = angle_s * vkp + angle_c * vkq

    return [d[i][i] for i in range(n)], v


def bidiagonal_qr_with_shifts(
    b: List[List[float]],
    eps: float = 1e-12,
    max_iter: int = 1000,
) -> BidiagonalQRResult:
    t = bidiagonal_to_btb(b)
    n = len(t)
    v_tilde = identity(n)
    steps: List[BidiagonalQRStep] = []

    for k in range(1, max_iter + 1):
        offdiag = offdiag_fro_tridiagonal_like(t)
        diag_now = [t[i][i] for i in range(n)]
        if offdiag < eps:
            eigvals, eigvecs = jacobi_eigen_symmetric(t, eps=eps)
            sigma = [sqrt(max(val, 0.0)) for val in eigvals]
            return BidiagonalQRResult(
                sigma=sigma,
                u_tilde=[],
                v_tilde=eigvecs,
                t_final=t,
                steps=steps,
                stop_reason="QR-итерации для B^T B сошлись.",
            )

        a = t[n - 2][n - 2]
        b_last = t[n - 2][n - 1]
        d = t[n - 1][n - 1]
        mu = wilkinson_shift_2x2(a, b_last, d)

        shifted = copy_matrix(t)
        for i in range(n):
            shifted[i][i] -= mu

        q, r = givens_qr_symmetric_tridiagonal(shifted)
        t = matmul(r, q)
        for i in range(n):
            t[i][i] += mu

        v_tilde = matmul(v_tilde, q)

        steps.append(
            BidiagonalQRStep(
                iteration=k,
                shift=mu,
                offdiag_norm=offdiag,
                diag_snapshot=diag_now,
            )
        )

    eigvals, eigvecs = jacobi_eigen_symmetric(t, eps=eps)
    sigma = [sqrt(max(val, 0.0)) for val in eigvals]
    return BidiagonalQRResult(
        sigma=sigma,
        u_tilde=[],
        v_tilde=eigvecs,
        t_final=t,
        steps=steps,
        stop_reason="Достигнуто максимальное число QR-итераций.",
    )
