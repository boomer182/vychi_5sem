from __future__ import annotations

from inverse_iteration import inverse_iteration, inverse_iteration_rayleigh
from matrix_utils import format_matrix, format_vector
from power_method import power_method
from test_cases import case_symmetric_3x3, case_symmetric_4x4


def choose_case():
    print("Выберите тестовую матрицу:")
    print("1 - симметричная 3x3")
    print("2 - симметричная 4x4")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            return case_symmetric_3x3()
        if choice == "2":
            return case_symmetric_4x4()
        print("Нужно выбрать 1 или 2.")


def choose_eps():
    print("\nВыберите точность:")
    print("1 - 1e-6")
    print("2 - 1e-10")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            return 1e-6
        if choice == "2":
            return 1e-10
        print("Нужно выбрать 1 или 2.")


def choose_start_vector(n: int):
    print("\nВыберите начальное приближение для метода Рэлея:")
    print("1 - вектор из единиц")
    print("2 - стандартный базисный вектор e1")
    print("3 - ввести свой вектор")

    while True:
        choice = input("Ваш выбор: ").strip()

        if choice == "1":
            return [1.0] * n, "вектор из единиц"

        if choice == "2":
            x0 = [0.0] * n
            x0[0] = 1.0
            return x0, "базисный вектор e1"

        if choice == "3":
            while True:
                raw = input(f"Введите {n} чисел через пробел: ").strip().split()
                if len(raw) != n:
                    print(f"Нужно ввести ровно {n} чисел.")
                    continue
                try:
                    return [float(x) for x in raw], "пользовательский вектор"
                except ValueError:
                    print("Ошибка: вводите только числа.")

        print("Нужно выбрать 1, 2 или 3.")


def print_steps(result):
    print("\n" + "=" * 72)
    print(result.method_name)
    print("=" * 72)

    for step in result.steps:
        print(f"Итерация {step.iteration}")
        print(f"  lambda       = {step.eigenvalue:.12g}")
        if step.delta_lambda is None:
            print("  delta lambda = ---")
        else:
            print(f"  delta lambda = {step.delta_lambda:.12g}")
        print(f"  residual     = {step.residual_norm:.12g}")
        print(f"  x            = {format_vector(step.eigenvector, 10)}")
        print("-" * 72)

    print("Итог:")
    print(f"  собственное число  = {result.eigenvalue:.12g}")
    print(f"  собственный вектор = {format_vector(result.eigenvector, 10)}")
    print(f"  ||e||_2            = {sum(x * x for x in result.eigenvector) ** 0.5:.12g}")
    print(f"  ||r||_2            = {result.residual_norm:.12g}")
    print(f"  итераций           = {result.iterations}")
    print(f"  причина остановки  = {result.stop_reason}")


def main():
    print("Практикум 9.2 — метод обратных итераций\n")

    a, x0_power, description = choose_case()
    eps = choose_eps()

    print("\nМатрица A:")
    print(format_matrix(a, 8))

    print("\nОписание примера:")
    print(description)

    print("\nНачальное приближение для степенного метода:")
    print(format_vector(x0_power, 10))

    print("\nШаг 1. Находим предварительное собственное число степенным методом.")
    power_result = power_method(a, x0_power, eps=1e-8, max_iter=1000)

    print("\nРезультат степенного метода:")
    print(f"  lambda_max ~= {power_result.eigenvalue:.12g}")
    print(f"  вектор      = {format_vector(power_result.eigenvector, 10)}")
    print(f"  ||r||_2     = {power_result.residual_norm:.12g}")
    print(f"  итераций    = {power_result.iterations}")

    print("\nШаг 2. Используем найденное lambda_max в методе обратных итераций.")
    inv_result = inverse_iteration(
        a,
        lambda_star=power_result.eigenvalue,
        x0=x0_power,
        eps=eps,
        max_iter=30,
    )

    x0_rayleigh, rayleigh_name = choose_start_vector(len(a))

    print("\nШаг 3. Метод обратных итераций с отношением Рэлея.")
    print("Начальное приближение:")
    print(rayleigh_name, "=", format_vector(x0_rayleigh, 10))

    rayleigh_result = inverse_iteration_rayleigh(
        a,
        x0=x0_rayleigh,
        eps=eps,
        max_iter=30,
    )

    print_steps(inv_result)
    print_steps(rayleigh_result)

    print("\n" + "=" * 72)
    print("СРАВНЕНИЕ МЕТОДОВ")
    print("=" * 72)
    print("Обычный метод обратных итераций:")
    print(f"  lambda = {inv_result.eigenvalue:.12g}")
    print(f"  ||r||_2 = {inv_result.residual_norm:.12g}")
    print(f"  итераций = {inv_result.iterations}")

    print("\nМетод с отношением Рэлея:")
    print(f"  lambda = {rayleigh_result.eigenvalue:.12g}")
    print(f"  ||r||_2 = {rayleigh_result.residual_norm:.12g}")
    print(f"  итераций = {rayleigh_result.iterations}")

    print("\nВывод:")
    print(
        "Обычный метод обратных итераций использует заранее выбранное приближение "
        "к собственному числу. Метод с отношением Рэлея дополнительно уточняет "
        "собственное число на каждом шаге и обычно сходится быстрее."
    )


if __name__ == "__main__":
    main()
