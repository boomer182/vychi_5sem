from __future__ import annotations

from matrix_utils import (
    format_matrix,
    format_vector,
    matrix_vector_multiply,
    vector_norm_inf,
    vector_subtract,
)
from cholesky import cholesky_solve
from models import (
    matrix_short_report,
    non_symmetric_example,
    spd_example,
    symmetric_not_positive_example,
)


def print_solution_case(title: str, a, b, x_exact=None, theory_text="") -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)

    print("\nМатрица A:")
    print(format_matrix(a, 8))

    print("\nВектор b:")
    print(format_vector(b, 8))

    print("\nТеоретическая проверка:")
    print(matrix_short_report(a))

    print("\nОбоснование:")
    print(theory_text)

    try:
        result = cholesky_solve(a, b)

        print("\nМатрица L:")
        print(format_matrix(result.l, 8))

        print("\nПроверка разложения ||A - L L^T||_inf:")
        print(f"{result.factorization_residual:.12g}")

        print("\nЧисленное решение x:")
        print(format_vector(result.solution, 10))

        if x_exact is not None:
            error = vector_norm_inf(vector_subtract(result.solution, x_exact))
            print("\nТочное решение x_exact:")
            print(format_vector(x_exact, 10))
            print(f"||x_num - x_exact||_inf = {error:.12g}")

            residual = vector_subtract(matrix_vector_multiply(a, result.solution), b)
            print(f"||A x - b||_inf = {vector_norm_inf(residual):.12g}")

        print("\nЧисло операций:")
        print(f"Разложение, факт        : {result.decomposition_operations}")
        print(f"Решение, факт           : {result.solve_operations}")
        print(f"Всего, факт             : {result.total_operations}")
        print(f"Разложение, теория      : {result.theoretical_decomposition_ops:.6f}")
        print(f"Решение, теория         : {result.theoretical_solve_ops:.6f}")
        print(f"Всего, теория           : {result.theoretical_total_ops:.6f}")

        n = len(a)
        gauss_estimate = (2.0 / 3.0) * n**3 + 2.0 * n**2
        print(f"\nОценка для метода Гаусса: {gauss_estimate:.6f}")
        print(
            "Вывод: для больших n метод Холецкого примерно вдвое дешевле "
            "обычного метода Гаусса для симметричных положительно "
            "определённых матриц."
        )

    except ValueError as exc:
        print("\nМетод Холецкого не применим.")
        print("Причина:", exc)


def main() -> None:
    print("Практикум 7.3. Метод Холецкого\n")

    a1, b1, x1, theory1 = spd_example()
    print_solution_case(
        "Пример 1: матрица, для которой метод Холецкого применим",
        a1,
        b1,
        x_exact=x1,
        theory_text=theory1,
    )

    a2, b2, theory2 = non_symmetric_example()
    print()
    print_solution_case(
        "Пример 2: матрица, для которой метод Холецкого неприменим (несимметричность)",
        a2,
        b2,
        theory_text=theory2,
    )

    a3, b3, theory3 = symmetric_not_positive_example()
    print()
    print_solution_case(
        "Пример 3: матрица, для которой метод Холецкого неприменим "
        "(симметрична, но не положительно определена)",
        a3,
        b3,
        theory_text=theory3,
    )


if __name__ == "__main__":
    main()
