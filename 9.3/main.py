from __future__ import annotations

from matrix_utils import (
    format_matrix,
    format_vector,
    is_positive_definite_by_minors,
    is_symmetric,
    off_diagonal_frobenius_norm,
    sort_descending,
    symmetry_error_inf,
)
from qr_algorithm import qr_algorithm, qr_algorithm_shifted
from test_cases import case_spd_3x3, case_spd_4x4, case_spd_5x5


def choose_case():
    print("Выберите тестовую матрицу:")
    print("1 - SPD матрица 3x3")
    print("2 - SPD матрица 4x4")
    print("3 - SPD матрица 5x5")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            return case_spd_3x3()
        if choice == "2":
            return case_spd_4x4()
        if choice == "3":
            return case_spd_5x5()
        print("Нужно выбрать 1, 2 или 3.")


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


def print_matrix_assessment(a):
    print("\nПроверка входной матрицы:")
    print(f"Симметрична: {'да' if is_symmetric(a) else 'нет'}")
    print(f"Положительно определена: {'да' if is_positive_definite_by_minors(a) else 'нет'}")
    print(f"Ошибка симметрии ||A-A^T||_inf = {symmetry_error_inf(a):.12g}")
    print(f"Норма внедиагональной части   = {off_diagonal_frobenius_norm(a):.12g}")


def print_steps_short(result):
    print("\n" + "=" * 78)
    print(result.method_name)
    print("=" * 78)

    for step in result.steps:
        print(f"Итерация {step.iteration}")
        print(f"  сдвиг                = {step.shift:.12g}")
        print(f"  диагональ            = {format_vector(step.diagonal, 10)}")
        print(f"  ||A_offdiag||_F      = {step.offdiag_norm:.12g}")
        print("-" * 78)

    print("Итог:")
    print(f"  причина остановки    = {result.stop_reason}")
    print(f"  число итераций       = {result.iterations}")
    print(f"  число операций       = {result.operations}")
    print(f"  найденные λ          = {format_vector(sort_descending(result.eigenvalues), 12)}")
    print("\nФинальная матрица:")
    print(format_matrix(result.final_matrix, 10))


def main():
    print("Практикум 9.3 — QR-алгоритм для полной проблемы собственных значений\n")

    a, description = choose_case()
    eps = choose_eps()

    print("\nИсходная матрица A:")
    print(format_matrix(a, 8))

    print("\nОписание примера:")
    print(description)

    print_matrix_assessment(a)

    print("\nЗапускается обычный QR-алгоритм...")
    basic_result = qr_algorithm(a, eps=eps, max_iter=300)

    print("\nЗапускается ускоренный QR-алгоритм со сдвигом...")
    shifted_result = qr_algorithm_shifted(a, eps=eps, max_iter=300)

    print_steps_short(basic_result)
    print_steps_short(shifted_result)

    print("\n" + "=" * 78)
    print("СРАВНЕНИЕ")
    print("=" * 78)
    print(f"Точность eps                     = {eps}")
    print(f"Итерации, обычный QR             = {basic_result.iterations}")
    print(f"Итерации, QR со сдвигом          = {shifted_result.iterations}")
    print(f"Операции, обычный QR             = {basic_result.operations}")
    print(f"Операции, QR со сдвигом          = {shifted_result.operations}")

    print("\nСобственные числа:")
    print("Обычный QR:     ", format_vector(sort_descending(basic_result.eigenvalues), 12))
    print("QR со сдвигом:  ", format_vector(sort_descending(shifted_result.eigenvalues), 12))

    print("\nВывод:")
    print(
        "Обычный QR-алгоритм сходится к почти диагональной матрице, "
        "а вариант со сдвигом обычно требует заметно меньше итераций. "
        "Для симметричных положительно определённых матриц оба метода "
        "должны давать вещественные собственные значения, расположенные на диагонали "
        "предельной матрицы."
    )


if __name__ == "__main__":
    main()
