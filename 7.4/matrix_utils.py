from typing import List


def format_vector(v: List[float], digits: int = 6) -> str:
    return "[" + ", ".join(f"{x:.{digits}g}" for x in v) + "]"


def vector_subtract(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def vector_norm_inf(v: List[float]) -> float:
    return max(abs(x) for x in v)


def tridiagonal_matrix_vector(a, b, c, x):
    n = len(b)
    result = [0.0] * n

    for i in range(n):
        result[i] = b[i] * x[i]
        if i > 0:
            result[i] += a[i] * x[i - 1]
        if i < n - 1:
            result[i] += c[i] * x[i + 1]

    return result
