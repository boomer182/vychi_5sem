from __future__ import annotations

from matrix_utils import (
    angle_between_vectors,
    format_matrix,
    format_vector,
    gershgorin_circles,
)
from power_method import power_method_basic, power_method_normalized
from test_cases import (
    case_large_dominant,
    case_small_dominant,
    case_symmetric_clear_gap,
)


def choose_case():
    print("Выберите тестовую матрицу:")
    print("1 - симметричная матрица с хорошим разделением спектра")
    print("2 - доминирующее собственное число больше 1")
    print("3 - доминирующее собственное число меньше 1")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            return case_symmetric_clear_gap()
        if choice == "2":
            return case_large_dominant()
        if choice == "3":
            return case_small_dominant()
        print("Нужно выбрать 1, 2 или 3.")


def choose_eps():
    print("\nВыберите точность:")
    print("1 - 1e-3")
    print("2 - 1e-6")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            return 1e-3
        if choice == "2":
            return 1e-6
        print("Нужно выбрать 1 или 2.")


def print_gershgorin(a):
    print("\nКруги Гершгорина:")
    circles = gershgorin_circles(a)
    for i, (center, radius) in enumerate(circles, start=1):
        print(f"S{i}: center = {center:.10g}, radius = {radius:.10g}")


def print_steps(title, result, lambda_expected):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    for step in result.steps:
        print("-" * 78)
        print(f"Итерация {step.iteration}")
        print(f"lambda^(k)          = {step.eigenvalue:.12g}")
        if step.difference_lambda is None:
            print("delta lambda        = ---")
        else:
            print(f"delta lambda        = {step.difference_lambda:.12g}")
        print(f"||r^(k)||_2         = {step.residual_norm:.12g}")
        print(f"апостериорная оценка = {step.aposterior_estimate:.12g}")
        print("x^(k) =", format_vector(step.eigenvector, 10))

    print("\nИтог:")
    print(f"Метод               = {result.method_name}")
    print(f"Число итераций      = {result.iterations}")
    print(f"Причина остановки   = {result.stop_reason}")
    print(f"Число операций      = {result.operations}")
    print(f"Найденное lambda    = {result.eigenvalue:.12g}")
    print(f"Ожидаемое lambda    = {lambda_expected:.12g}")
    print(f"|error lambda|      = {abs(result.eigenvalue - lambda_expected):.12g}")
    print(f"Найденный вектор    = {format_vector(result.eigenvector, 10)}")
    print(f"||x||_2             = {sum(x*x for x in result.eigenvector) ** 0.5:.12g}")


def main():
    print("Практикум 9.1 — степенной метод\n")

    a, x0, lambda_expected, description = choose_case()
    eps = choose_eps()

    print("\nМатрица A:")
    print(format_matrix(a, 8))

    print("\nНачальное приближение x^(0):")
    print(format_vector(x0, 10))

    print("\nОписание примера:")
    print(description)

    print_gershgorin(a)

    basic_result = power_method_basic(a, x0, eps, max_iter=1000)
    normalized_result = power_method_normalized(a, x0, eps, max_iter=1000)

    print_steps("Базовый степенной метод", basic_result, lambda_expected)
    print_steps("Степенной метод с нормировкой", normalized_result, lambda_expected)

    print("\n" + "=" * 78)
    print("СРАВНЕНИЕ")
    print("=" * 78)
    print(f"Точность eps                  = {eps}")
    print(f"Итерации, базовый метод       = {basic_result.iterations}")
    print(f"Итерации, нормированный метод = {normalized_result.iterations}")
    print(f"Операции, базовый метод       = {basic_result.operations}")
    print(f"Операции, нормированный метод = {normalized_result.operations}")

    print("\nВывод:")
    print(
        "Нормированный вариант степенного метода лучше защищён от переполнения "
        "и исчезновения порядка, потому что вектор на каждом шаге приводится к норме 1."
    )


if __name__ == "__main__":
    main()
