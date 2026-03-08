from typing import List
from math import sin


def build_problem(n: int):

    h = 1 / n

    a = [0.0] * (n + 1)
    b = [0.0] * (n + 1)
    c = [0.0] * (n + 1)
    d = [0.0] * (n + 1)

    for i in range(1, n):
        a[i] = -1 / h**2
        b[i] = 1 + 2 / h**2
        c[i] = -1 / h**2
        d[i] = sin(i * h)

    b[0] = 1
    b[n] = 1

    d[0] = 0
    d[n] = 0

    return a, b, c, d
