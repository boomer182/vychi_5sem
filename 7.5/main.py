from __future__ import annotations

from matrix_utils import (
    format_matrix,
    format_vector,
    matrix_vector_multiply,
)
from qr_givens import qr_givens
from qr_householder import qr_householder
from solver import solve_via_qr


def input_int(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value <= 0:
                print("Введите положительное целое число.")
                continue
            return value
        except ValueError:
            print("Ошибка: нужно ввести целое число.")


def input_float_list(prompt: str, expected_len: int):
    while True:
        try:
            parts = input(prompt).strip().split()
            if len(parts) != expected_len:
                print(f"Нужно ввести ровно {expected_len} чисел.")
                continue
            return [float(x) for x in parts]
        except ValueError:
            print("Ошибка: вводите только числа.")


def input_matrix():
    n = input_int("Введите размер квадратной матрицы n: ")
    a = []
    print("Введите матрицу A построчно:")
    for i in range(n):
        a.append(input_float_list(f"Строка {i + 1}: ", n))
    return a


def input_vector(n: int, name: str):
    return input_float_list(f"Введите вектор {name} из {n} чисел: ", n)


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_report(method_result, solve_report, theoretical_ops: float) -> None:
    print_section(f"Метод: {method_result.method_name}")

    print("Матрица Q:")
    print(format_matrix(method_result.q, 8))

    print("\nМатрица R:")
    print(format_matrix(method_result.r, 8))

    print("\nНайденное решение x:")
    print(format_vector(solve_report.x, 10))

    if solve_report.solution_error is not None:
        print("\nФактическая погрешность решения x - x_exact:")
        print(format_vector(solve_report.solution_error, 10))
        print(f"||x - x_exact||_inf = {solve_report.solution_error_inf:.12g}")
        print(f"||x - x_exact||_2   = {solve_report.solution_error_2:.12g}")
        print(
            f"Относительная погрешность (inf) = "
            f"{solve_report.relative_solution_error_inf:.12g}"
        )

    print("\nНевязка Ax - b:")
    print(format_vector(solve_report.residual, 10))
    print(f"||Ax - b||_inf = {solve_report.residual_inf:.12g}")
    print(f"||Ax - b||_2   = {solve_report.residual_2:.12g}")
    print(f"Относительная невязка (inf) = {solve_report.relative_residual_inf:.12g}")

    print("\nПроверка разложения и ортогональности:")
    print(f"||A - QR||_inf        = {solve_report.decomposition_error_inf:.12g}")
    print(f"||Q^T Q - I||_inf     = {solve_report.orthogonality_error_inf:.12g}")

    print("\nОценка числа операций:")
    print(f"Фактически посчитано        : {method_result.operations}")
    print(f"Теоретическая оценка        : {theoretical_ops:.6f}")


def main() -> None:
    print("Практикум 7.5 — QR-разложение")
    print("1) метод вращений Гивенса")
    print("2) метод отражений Хаусхолдера")

    a = input_matrix()
    n = len(a)

    print("\nВыберите способ задания правой части:")
    print("1 - ввести b вручную")
    print("2 - ввести точное решение x_exact и вычислить b = A x_exact")

    mode = input_int("Ваш выбор: ")
    while mode not in (1, 2):
        print("Нужно выбрать 1 или 2.")
        mode = input_int("Ваш выбор: ")

    x_exact = None
    if mode == 1:
        b = input_vector(n, "b")
    else:
        x_exact = input_vector(n, "x_exact")
        b = matrix_vector_multiply(a, x_exact)

    print_section("Исходные данные")
    print("Матрица A:")
    print(format_matrix(a, 8))
    print("\nВектор b:")
    print(format_vector(b, 10))
    if x_exact is not None:
        print("\nТочное решение x_exact:")
        print(format_vector(x_exact, 10))

    givens_result = qr_givens(a)
    givens_report = solve_via_qr(a, b, givens_result.q, givens_result.r, x_exact)

    householder_result = qr_householder(a)
    householder_report = solve_via_qr(
        a,
        b,
        householder_result.q,
        householder_result.r,
        x_exact,
    )

    print_report(givens_result, givens_report, 2.0 * (n ** 3))
    print_report(householder_result, householder_report, (4.0 / 3.0) * (n ** 3))

    print_section("Сравнение методов")
    print(f"Вращения Гивенса:      ~ 2 n^3   = {2.0 * (n ** 3):.6f}")
    print(f"Отражения Хаусхолдера: ~ 4/3 n^3 = {(4.0 / 3.0) * (n ** 3):.6f}")
    print(f"Метод Гаусса:          ~ 2/3 n^3 = {(2.0 / 3.0) * (n ** 3):.6f}")
    print(f"LU-разложение:         ~ 2/3 n^3 = {(2.0 / 3.0) * (n ** 3):.6f}")
    print(f"Холецкий:              ~ 1/3 n^3 = {(1.0 / 3.0) * (n ** 3):.6f}")

    print("\nВывод:")
    print(
        "QR через отражения обычно быстрее QR через вращения, "
        "а оба метода дают ортогональное разложение, удобное "
        "для устойчивого решения СЛАУ."
    )


if __name__ == "__main__":
    main()
