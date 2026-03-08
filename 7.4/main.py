from matrix_utils import (
    format_vector,
    tridiagonal_matrix_vector,
    vector_norm_inf,
    vector_subtract,
)

from thomas import (
    check_applicability,
    thomas_algorithm,
)

from model_problem import build_problem


def main():

    print("Практикум 7.4 — метод прогонки\n")

    n = int(input("Введите размер системы n: "))

    a, b, c, d = build_problem(n)

    print("\nПроверка условий применимости метода")

    if check_applicability(a, b, c):
        print("Условия выполнены")
    else:
        print("Условия НЕ выполнены")

    result = thomas_algorithm(a, b, c, d)

    print("\nКоэффициенты α:")
    print(format_vector(result.alpha))

    print("\nКоэффициенты β:")
    print(format_vector(result.beta))

    print("\nРешение x:")
    print(format_vector(result.solution, 10))

    Ax = tridiagonal_matrix_vector(a, b, c, result.solution)

    residual = vector_subtract(Ax, d)

    print("\n||Ax - d||_inf =", vector_norm_inf(residual))

    print("\nЧисло операций (примерно):", result.operations)

    print("\nОценка:")
    print("Метод прогонки ≈ 8n операций")
    print("Метод Гаусса ≈ (2/3)n^3")


if __name__ == "__main__":
    main()
