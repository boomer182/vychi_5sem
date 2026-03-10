from __future__ import annotations

from matrix_utils import (
    format_matrix,
    format_vector,
    is_positive_definite_by_minors,
    is_symmetric,
    relative_diagonal_dominance_report,
    residual,
    vector_norm_inf,
    vector_subtract,
)
from simple_iteration import build_iteration_form, simple_iteration
from seidel import build_iteration_form as build_iteration_form_seidel
from seidel import seidel_method
from test_cases import (
    case_near_diagonal,
    case_near_lower_triangular,
    case_spd,
    initial_guess_zero,
)


def choose_case():
    print("Выберите тестовую систему:")
    print("1 - матрица, близкая к диагональной")
    print("2 - матрица, близкая к нижней треугольной")
    print("3 - симметричная положительно определённая матрица")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            return case_near_diagonal()
        if choice == "2":
            return case_near_lower_triangular()
        if choice == "3":
            return case_spd()
        print("Нужно выбрать 1, 2 или 3.")


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


def print_matrix_assessment(a):
    print("\nОценка матрицы:")
    print(f"Симметрична: {'да' if is_symmetric(a) else 'нет'}")
    print(
        f"Положительно определена: "
        f"{'да' if is_symmetric(a) and is_positive_definite_by_minors(a) else 'нет'}"
    )
    print("Диагональное преобладание по строкам:")
    for line in relative_diagonal_dominance_report(a):
        print(" ", line)


def print_simple_iteration_short(simple_result):
    print("\nМетод простой итерации:")
    print(f"  ||B||_inf = {simple_result.norm_b_inf:.12g}")
    print(f"  ||B||_1   = {simple_result.norm_b1:.12g}")
    print(f"  rho(B)    = {simple_result.spectral_radius:.12g}")
    print(f"  Итераций  = {simple_result.iterations}")
    print(f"  Операций  = {simple_result.operations}")
    print(f"  Причина остановки: {simple_result.stop_reason}")


def print_seidel_iteration_table(result):
    print("\nПромежуточные результаты метода Зейделя:")
    for step in result.steps:
        print("-" * 72)
        print(f"Итерация {step.iteration}")
        print("x^(m-1) =", format_vector(step.x_prev, 10))
        print("x^(m)   =", format_vector(step.x_curr, 10))
        print(f"||x^(m)-x^(m-1)||_inf = {step.diff_inf:.12g}")
        print(f"||A x^(m)-b||_inf     = {step.residual_inf:.12g}")
        if step.aposterior_estimate is not None:
            print(f"Апостериорная оценка  = {step.aposterior_estimate:.12g}")
        else:
            print("Апостериорная оценка  = не вычисляется по выбранной оценке")


def print_final_comparison(a, b, x_exact, seidel_result, simple_result):
    seidel_error = vector_subtract(seidel_result.solution, x_exact)
    simple_error = vector_subtract(simple_result.solution, x_exact)

    seidel_residual = residual(a, seidel_result.solution, b)
    simple_residual = residual(a, simple_result.solution, b)

    print("\n" + "=" * 72)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("=" * 72)

    print("\nМетод Зейделя:")
    print("Решение x =", format_vector(seidel_result.solution, 12))
    print(f"||x - x_exact||_inf = {vector_norm_inf(seidel_error):.12g}")
    print(f"||Ax - b||_inf      = {vector_norm_inf(seidel_residual):.12g}")
    print(f"Итераций            = {seidel_result.iterations}")
    print(f"Операций            = {seidel_result.operations}")

    print("\nМетод простой итерации:")
    print("Решение x =", format_vector(simple_result.solution, 12))
    print(f"||x - x_exact||_inf = {vector_norm_inf(simple_error):.12g}")
    print(f"||Ax - b||_inf      = {vector_norm_inf(simple_residual):.12g}")
    print(f"Итераций            = {simple_result.iterations}")
    print(f"Операций            = {simple_result.operations}")

    print("\nВывод:")
    if seidel_result.iterations < simple_result.iterations:
        print("Метод Зейделя сошёлся быстрее метода простой итерации.")
    elif seidel_result.iterations > simple_result.iterations:
        print("В данном примере метод простой итерации потребовал меньше итераций.")
    else:
        print("В данном примере оба метода потребовали одинаковое число итераций.")

    print("У метода Зейделя на каждом шаге используются уже обновлённые компоненты вектора,")
    print("поэтому для матриц, близких к нижним треугольным, он обычно выигрывает особенно заметно.")


def main():
    print("Практикум 8.2 — метод Зейделя\n")

    a, b, x_exact, description = choose_case()
    eps = choose_eps()

    print("\nМатрица A:")
    print(format_matrix(a, 8))

    print("\nВектор b:")
    print(format_vector(b, 10))

    print("\nТочное решение x_exact:")
    print(format_vector(x_exact, 10))

    print("\nОписание примера:")
    print(description)

    print_matrix_assessment(a)

    b_matrix, c = build_iteration_form(a, b)
    print("\nПредставление для простой итерации x = Bx + c:")
    print("Матрица B:")
    print(format_matrix(b_matrix, 8))
    print("\nВектор c:")
    print(format_vector(c, 10))

    x0 = initial_guess_zero(len(a))
    print("\nНачальное приближение:")
    print(format_vector(x0, 10))

    seidel_result = seidel_method(a, b, x0, eps, max_iter=10_000)
    simple_result = simple_iteration(a, b, x0, eps, max_iter=10_000)

    print("\nХарактеристики метода Зейделя:")
    print(f"||B||_inf             = {seidel_result.norm_b_inf:.12g}")
    print(f"||B||_1               = {seidel_result.norm_b1:.12g}")
    print(f"||B1||_inf            = {seidel_result.norm_b1_lower:.12g}")
    print(f"||B2||_inf            = {seidel_result.norm_b2_upper:.12g}")
    if seidel_result.q_estimate_inf is not None:
        print(f"q_inf estimate        = {seidel_result.q_estimate_inf:.12g}")
    else:
        print("q_inf estimate        = не вычисляется")
    if seidel_result.q_estimate_1 is not None:
        print(f"q_1 estimate          = {seidel_result.q_estimate_1:.12g}")
    else:
        print("q_1 estimate          = не вычисляется")
    print(f"rho(B_tilde)          = {seidel_result.spectral_radius_equivalent:.12g}")
    print(f"Причина остановки     = {seidel_result.stop_reason}")

    print_seidel_iteration_table(seidel_result)
    print_simple_iteration_short(simple_result)
    print_final_comparison(a, b, x_exact, seidel_result, simple_result)


if __name__ == "__main__":
    main()
