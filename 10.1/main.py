from __future__ import annotations

from matrix_utils import format_matrix, format_vector
from svd_pipeline import svd_via_golub_kahan_and_bidiag_qr
from test_cases import case_square_4x4, case_tall_4x3, case_wide_3x5


def choose_case():
    print("Выберите тестовую матрицу:")
    print("1 - высокая матрица 4x3")
    print("2 - широкая матрица 3x5")
    print("3 - квадратная матрица 4x4")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            return case_tall_4x3()
        if choice == "2":
            return case_wide_3x5()
        if choice == "3":
            return case_square_4x4()
        print("Нужно выбрать 1, 2 или 3.")


def choose_eps():
    print("\nВыберите точность QR-итераций:")
    print("1 - 1e-8")
    print("2 - 1e-12")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            return 1e-8
        if choice == "2":
            return 1e-12
        print("Нужно выбрать 1 или 2.")


def print_golub_kahan_report(result):
    print("\n" + "=" * 78)
    print("ПРОМЕЖУТОЧНЫЙ РЕЗУЛЬТАТ АЛГОРИТМА ГОЛУБА — КАХАНА")
    print("=" * 78)
    for text in result.steps_text:
        print(text)
        print("-" * 78)

    print("Бидиагональная матрица B:")
    print(format_matrix(result.b, 10))
    print("\nГлавная диагональ alpha:")
    print(format_vector(result.alpha, 12))
    print("\nНаддиагональ beta:")
    print(format_vector(result.beta, 12))


def print_bidiag_qr_report(steps):
    print("\n" + "=" * 78)
    print("QR-ИТЕРАЦИИ СО СДВИГАМИ ДЛЯ БИДИАГОНАЛЬНОЙ МАТРИЦЫ")
    print("=" * 78)

    for step in steps:
        print(f"Итерация {step.iteration}")
        print(f"  сдвиг Уилкинсона     = {step.shift:.12g}")
        print(f"  диагональ B^T B      = {format_vector(step.diag_snapshot, 12)}")
        print(f"  ||offdiag(B^T B)||_F = {step.offdiag_norm:.12g}")
        print("-" * 78)


def print_final_report(result):
    print("\n" + "=" * 78)
    print("ИТОГОВОЕ СИНГУЛЯРНОЕ РАЗЛОЖЕНИЕ")
    print("=" * 78)

    print("Матрица U (тонкое SVD):")
    print(format_matrix(result.u, 10))

    print("\nСингулярные числа:")
    print(format_vector(result.sigma, 12))

    print("\nМатрица V (тонкое SVD):")
    print(format_matrix(result.v, 10))

    print("\nСправочные сингулярные числа:")
    print(format_vector(result.reference_sigma, 12))

    print("\nПогрешности:")
    print(f"||A - U Σ V^T||_F               = {result.reconstruction_error_fro:.12g}")
    print(f"max |σ_i - σ_i(ref)|            = {result.singular_value_error_inf:.12g}")

    print("\nПроверка сингулярных троек:")
    for i in range(len(result.sigma)):
        print(
            f"i = {i + 1}: "
            f"||A v_i - σ_i u_i||_2 = {result.left_residuals[i]:.12g}, "
            f"||A^T u_i - σ_i v_i||_2 = {result.right_residuals[i]:.12g}"
        )


def main():
    print("Практикум 10.1 — сингулярное разложение матрицы\n")

    a, description = choose_case()
    eps = choose_eps()

    print("\nИсходная матрица A:")
    print(format_matrix(a, 8))

    print("\nОписание примера:")
    print(description)
    print(f"Размер матрицы: {len(a)} x {len(a[0])}")

    result = svd_via_golub_kahan_and_bidiag_qr(a, eps=eps)

    print_golub_kahan_report(result.golub_kahan)
    print_bidiag_qr_report(result.bidiag_qr_steps)
    print_final_report(result)


if __name__ == "__main__":
    main()
