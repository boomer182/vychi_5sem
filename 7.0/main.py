from matrix import Matrix
from vector import Vector
from norms import (
    vector_norm_1,
    vector_norm_2,
    vector_norm_inf,
    matrix_norm_1,
    matrix_norm_2,
    matrix_norm_inf,
    frobenius_norm,
)
from inverse import inverse_matrix
from conditioning import condition_number


def main() -> None:
    v = Vector([1, -2, 3])
    a = Matrix([
        [4, 2, 1],
        [0, 5, -1],
        [2, 1, 3],
    ])

    print("Vector:", v)
    print("||v||_1   =", vector_norm_1(v))
    print("||v||_2   =", vector_norm_2(v))
    print("||v||_inf =", vector_norm_inf(v))
    print()

    print("Matrix A:")
    for row in a.data:
        print(row)
    print()

    print("||A||_1   =", matrix_norm_1(a))
    print("||A||_2   =", matrix_norm_2(a))
    print("||A||_inf =", matrix_norm_inf(a))
    print("||A||_F   =", frobenius_norm(a))
    print()

    inv_a = inverse_matrix(a)
    print("A^{-1}:")
    for row in inv_a.data:
        print(row)
    print()

    print("cond_1(A)   =", condition_number(a, 1))
    print("cond_2(A)   =", condition_number(a, 2))
    print("cond_inf(A) =", condition_number(a, "inf"))
    print()

    x = Vector([1, 2, 3])
    ax = a * x
    print("A * x =", ax)


if __name__ == "__main__":
    main()
