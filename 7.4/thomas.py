from dataclasses import dataclass
from typing import List


@dataclass
class ThomasResult:
    solution: List[float]
    alpha: List[float]
    beta: List[float]
    operations: int


def check_applicability(a, b, c):
    n = len(b)

    for k in range(n):
        left = abs(b[k])
        right = abs(a[k]) + abs(c[k])
        if left < right:
            return False

    return True


def thomas_algorithm(a, b, c, d):
    n = len(b)

    alpha = [0.0] * n
    beta = [0.0] * n
    gamma = [0.0] * n

    ops = 0

    gamma[0] = b[0]
    alpha[0] = -c[0] / gamma[0]
    beta[0] = d[0] / gamma[0]
    ops += 2

    for i in range(1, n):
        gamma[i] = b[i] + a[i] * alpha[i - 1]
        ops += 2

        if i < n - 1:
            alpha[i] = -c[i] / gamma[i]
            ops += 1

        beta[i] = (d[i] - a[i] * beta[i - 1]) / gamma[i]
        ops += 2

    x = [0.0] * n
    x[n - 1] = beta[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
        ops += 2

    return ThomasResult(x, alpha, beta, ops)
