from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import List

from bidiagonal_qr import bidiagonal_qr_with_shifts, jacobi_eigen_symmetric
from golub_kahan import GolubKahanResult, golub_kahan_bidiagonalize
from matrix_utils import (
    EPS,
    columns_of_matrix,
    diag_rect,
    format_vector,
    gram_schmidt_columns,
    mat_sub,
    matmul,
    matrix_from_columns,
    matvec,
    norm2,
    norm_fro,
    normalize,
    sort_svd_desc,
    thin_diag,
    transpose,
    zeros,
)


@dataclass
class SVDResult:
    u: List[List[float]]
    sigma: List[float]
    v: List[List[float]]
    b: List[List[float]]
    golub_kahan: GolubKahanResult
    bidiag_qr_steps: list
    reconstruction_error_fro: float
    singular_value_error_inf: float
    reference_sigma: List[float]
    left_residuals: List[float]
    right_residuals: List[float]


def build_left_vectors_from_sigma(
    b: List[List[float]],
    v_tilde: List[List[float]],
    sigma: List[float],
    m: int,
    p: int,
) -> List[List[float]]:
    v_cols = columns_of_matrix(v_tilde)
    u_cols = []

    for i in range(p):
        vi = v_cols[i]
        bvi = matvec(b, vi)
        if sigma[i] > 1e-10:
            ui = [x / sigma[i] for x in bvi]
            ui = normalize(ui)
        else:
            ui = [0.0] * m
        u_cols.append(ui)

    return matrix_from_columns(u_cols)


def symmetric_reference_singular_values(a: List[List[float]]) -> List[float]:
    ata = matmul(transpose(a), a)
    eigvals, _ = jacobi_eigen_symmetric(ata, eps=1e-13, max_iter=8000)
    sigma = [sqrt(max(x, 0.0)) for x in eigvals]
    sigma.sort(reverse=True)
    return sigma


def svd_via_golub_kahan_and_bidiag_qr(a: List[List[float]], eps: float = 1e-12) -> SVDResult:
    m = len(a)
    n = len(a[0])
    p = min(m, n)

    gk = golub_kahan_bidiagonalize(a)
    bidiag_qr = bidiagonal_qr_with_shifts(gk.b, eps=eps, max_iter=1500)

    sigma_full = bidiag_qr.sigma
    v_tilde_full = bidiag_qr.v_tilde

    sigma = sigma_full[:p]
    v_tilde = [row[:p] for row in v_tilde_full]

    u_tilde = build_left_vectors_from_sigma(gk.b, v_tilde, sigma, m, p)

    u = matmul(gk.u_left, u_tilde)
    v = matmul(gk.v_right, v_tilde)

    sigma, u, v = sort_svd_desc(sigma, u, v)

    a_rec = matmul(matmul(u, thin_diag(sigma)), transpose(v))
    rec_err = norm_fro(mat_sub(a, a_rec))

    ref_sigma = symmetric_reference_singular_values(a)[:p]
    sv_err = max(abs(sigma[i] - ref_sigma[i]) for i in range(p)) if p > 0 else 0.0

    u_cols = columns_of_matrix(u)
    v_cols = columns_of_matrix(v)
    left_residuals = []
    right_residuals = []

    at = transpose(a)
    for i in range(p):
        av = matvec(a, v_cols[i])
        su = [sigma[i] * x for x in u_cols[i]]
        left_res = norm2([av[j] - su[j] for j in range(len(av))])

        atu = matvec(at, u_cols[i])
        sv = [sigma[i] * x for x in v_cols[i]]
        right_res = norm2([atu[j] - sv[j] for j in range(len(atu))])

        left_residuals.append(left_res)
        right_residuals.append(right_res)

    return SVDResult(
        u=u,
        sigma=sigma,
        v=v,
        b=gk.b,
        golub_kahan=gk,
        bidiag_qr_steps=bidiag_qr.steps,
        reconstruction_error_fro=rec_err,
        singular_value_error_inf=sv_err,
        reference_sigma=ref_sigma,
        left_residuals=left_residuals,
        right_residuals=right_residuals,
    )
