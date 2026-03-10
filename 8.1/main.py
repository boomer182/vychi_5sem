from __future__ import annotations

from matrix_utils import (
    format_matrix,
    format_vector,
    random_vector,
    vector_norm_inf,
    vector_subtract,
)
from iteration_method import build_iteration_form, residual, simple_iteration
from test_cases import (
    case_bad_3x3,
    case_good_3x3,
    case_slow_3x3,
    initial_guess_ones,
    initial_guess_rhs,
    initial_guess_zero,
)


def choose_case():
    print("Выберите тестовую систему:")
    print("1 - быстро сходящаяся")
    print("2 - медленно сходящаяся")
    print("3 - проблемная")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            return case_good_3x3()
        if choice == "2":
            return case_slow_3x3()
        if choice == "3":
            return case_bad_3x3()
        print("Нужно выбрать 1, 2 или 3.")


def choose_initial_guess(b):
    n = len(b)
    print("\nВыберите начальное приближение:")
    print("1 - вектор из нулей")
    print("2 - вектор из единиц")
    print("3 - правая часть b")
    print("4 - случайный вектор")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            return initial_guess_zero(n), "нулевой вектор"
        if choice == "2":
            return initial_guess_ones(n), "вектор из единиц"
        if choice == "3":
            return initial_guess_rhs(b), "правая часть b"
        if choice == "4":
            return random_vector(n, -1.0, 1.0), "случайный вектор"
        print("Нужно выбрать 1, 2, 3 или 4.")


def choose_eps():
    print("\nВыберите требуемую точность:")
    print("1 - 1e-3")
    print("2 - 1e-6")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            return 1e-3
        if choice == "2":
            return 1e-6
        print("Нужно выбрать 1 или 2.")


def choose_stop_mode():
    print("\nВыберите критерий остановки:")
    print("1 - апостериорная оценка")
    print("2 - простая проверка ||x^(m) - x^(m-1)|| < eps")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            return "aposterior", "апостериорная оценка"
        if choice == "2":
            return "difference", "простая разность соседних приближений"
        print("Нужно выбрать 1 или 2.")


def print_iteration_table(result):
    print("\nПромежуточные итерации:")
    for step in result.steps:
        print("-" * 70)
        print(f"Итерация {step.iteration}")
        print("x^(m-1) =", format_vector(step.x_prev, 10))
        print("x^(m)   =", format_vector(step.x_curr, 10))
        print(f"||x^(m) - x^(m-1)||_inf = {step.diff_inf:.12g}")
        if step.aposterior_estimate is not None:
            print(f"Апостериорная оценка    = {step.aposterior_estimate:.12g}")
        else:
            print("Апостериорная оценка    = не вычисляется, так как ||B|| >= 1")
        print(f"||A x^(m) - b||_inf     = {step.residual_inf:.12g}")


def main():
    print("Практикум 8.1 — метод простой итерации\n")

    a, b, x_exact, description = choose_case()

    print("\nМатрица A:")
    print(format_matrix(a, 8))

    print("\nВектор b:")
    print(format_vector(b, 10))

    print("\nТочное решение:")
    print(format_vector(x_exact, 10))

    print("\nОписание примера:")
    print(description)

    b_matrix, c = build_iteration_form(a, b)

    print("\nМатрица B в представлении x = Bx + c:")
    print(format_matrix(b_matrix, 8))

    print("\nВектор c:")
    print(format_vector(c, 10))

    x0, x0_name = choose_initial_guess(b)
    print("\nНачальное приближение:")
    print(x0_name, "=", format_vector(x0, 10))

    eps = choose_eps()
    stop_mode, stop_name = choose_stop_mode()

    result = simple_iteration(a, b, x0, eps, stop_mode=stop_mode, max_iter=1000)

    print("\nХарактеристики матрицы B:")
    print(f"||B||_inf = {result.norm_b_inf:.12g}")
    print(f"||B||_1   = {result.norm_b1:.12g}")
    print(f"rho(B)    = {result.spectral_radius:.12g}")

    if result.norm_b_inf < 1.0:
        print("По достаточному условию сходимость гарантирована в норме ||·||_inf.")
    else:
        print("Достаточное условие ||B||_inf < 1 не выполнено.")

    if result.spectral_radius < 1.0:
        print("По спектральному радиусу метод должен сходиться.")
    else:
        print("По спектральному радиусу сходимость не гарантируется.")

    print_iteration_table(result)

    final_residual = residual(a, result.solution, b)
    actual_error = vector_subtract(result.solution, x_exact)

    print("\n" + "=" * 70)
    print("ИТОГ")
    print("=" * 70)
    print("Критерий остановки:", stop_name)
    print("Требуемая точность:", eps)
    print("Причина остановки :", result.stop_reason)
    print("Число итераций    :", result.iterations)
    print("Число операций    :", result.operations)
    print("\nПриближённое решение:")
    print(format_vector(result.solution, 12))

    print("\nФактическая погрешность:")
    print(format_vector(actual_error, 12))
    print(f"||x_num - x_exact||_inf = {vector_norm_inf(actual_error):.12g}")

    print("\nНевязка:")
    print(format_vector(final_residual, 12))
    print(f"||A x - b||_inf = {vector_norm_inf(final_residual):.12g}")

    print("\nКраткий вывод:")
    if stop_mode == "difference":
        print("Простой критерий по разности соседних приближений удобен, но может завершить процесс слишком рано.")
    else:
        print("Апостериорный критерий более обоснован, если ||B|| < 1.")

    print("Оценка трудоёмкости одной итерации: порядка 2 n^2 операций.")
    print("Общая трудоёмкость: порядка 2 n^2 * m, где m — число итераций.")


if __name__ == "__main__":
    main()
