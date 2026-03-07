from __future__ import annotations

from typing import List

from matrix_utils import format_matrix, format_vector
from lu import lu_decomposition, solve_with_lu, check_lu
from lup import lup_decomposition, solve_with_lup, check_lup
from experiments import (
    repeated_lu_experiment,
    repeated_lup_experiment,
    compare_operation_counts,
)


def input_int(prompt: str, min_value: int | None = None) -> int:
    while True:
        try:
            value = int(input(prompt))
            if min_value is not None and value < min_value:
                print(f"Введите число не меньше {min_value}.")
                continue
            return value
        except ValueError:
            print("Ошибка: нужно ввести целое число.")


def input_float_list(prompt: str, expected_len: int) -> List[float]:
    while True:
        try:
            parts = input(prompt).strip().split()
            if len(parts) != expected_len:
                print(f"Нужно ввести ровно {expected_len} чисел.")
                continue
            return [float(x) for x in parts]
        except ValueError:
            print("Ошибка: вводите только числа.")


def input_matrix() -> List[List[float]]:
    n = input_int("Введите размер квадратной матрицы n: ", 1)
    a = []
    print("Введите матрицу построчно:")
    for i in range(n):
        row = input_float_list(f"Строка {i + 1}: ", n)
        a.append(row)
    return a


def input_rhs(n: int) -> List[float]:
    print("Введите правую часть:")
    return input_float_list("b: ", n)


def print_lu_solution(a: List[List[float]], b: List[float]) -> None:
    print("\n=== LU-разложение ===")
    try:
        l, u, dec_ops = lu_decomposition(a)
        x, solve_ops = solve_with_lu(l, u, b)
        residual = check_lu(a, l, u)

        print("Матрица L:")
        print(format_matrix(l, 8))
        print("\nМатрица U:")
        print(format_matrix(u, 8))
        print("\nПроверка ||A - LU||_inf:")
        print(f"{residual:.12g}")
        print("\nРешение x:")
        print(format_vector(x, 10))

        det = 1.0
        for i in range(len(u)):
            det *= u[i][i]

        print(f"\ndet(A) = {det:.12g}")
        print(f"Операции разложения: {dec_ops}")
        print(f"Операции решения Ly=b и Ux=y: {solve_ops}")
        print(f"Всего: {dec_ops + solve_ops}")
    except ValueError as exc:
        print("LU-разложение не выполнено:")
        print(exc)


def print_lup_solution(a: List[List[float]], b: List[float]) -> None:
    print("\n=== LUP-разложение ===")
    try:
        p, l, u, dec_ops = lup_decomposition(a)
        x, solve_ops = solve_with_lup(p, l, u, b)
        residual = check_lup(a, p, l, u)

        print("Матрица P:")
        print(format_matrix(p, 8))
        print("\nМатрица L:")
        print(format_matrix(l, 8))
        print("\nМатрица U:")
        print(format_matrix(u, 8))
        print("\nПроверка ||PA - LU||_inf:")
        print(f"{residual:.12g}")
        print("\nРешение x:")
        print(format_vector(x, 10))

        perm_sign = 1
        used_cols = []
        for i in range(len(p)):
            for j in range(len(p)):
                if abs(p[i][j] - 1.0) < 1e-12:
                    used_cols.append(j)
                    break

        inversions = 0
        for i in range(len(used_cols)):
            for j in range(i + 1, len(used_cols)):
                if used_cols[i] > used_cols[j]:
                    inversions += 1
        perm_sign = -1 if inversions % 2 else 1

        det = perm_sign
        for i in range(len(u)):
            det *= u[i][i]

        print(f"\ndet(A) = {det:.12g}")
        print(f"Операции разложения: {dec_ops}")
        print(f"Операции решения Pb, Ly=Pb и Ux=y: {solve_ops}")
        print(f"Всего: {dec_ops + solve_ops}")
    except ValueError as exc:
        print("LUP-разложение не выполнено:")
        print(exc)


def ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("да", "д", "yes", "y"):
            return True
        if ans in ("нет", "н", "no", "n"):
            return False
        print("Введите да или нет.")


def run_repeated_experiments(a: List[List[float]]) -> None:
    count = input_int("\nСколько правых частей сгенерировать? ", 1)

    print("\n=== Повторные решения через LU ===")
    try:
        lu_results = repeated_lu_experiment(a, count)
        for idx, item in enumerate(lu_results, start=1):
            print("-" * 60)
            print(f"Тест {idx}")
            print("x_exact =", format_vector(item["x_exact"], 8))
            print("b       =", format_vector(item["b"], 8))
            print("x_num   =", format_vector(item["x_num"], 8))
            print(f"||x_num - x_exact||_inf = {item['error_inf']:.12g}")

        cmp_lu = compare_operation_counts(
            len(a),
            count,
            lu_results[0]["decomposition_operations"],
            lu_results[0]["solve_operations"],
        )

        print("\nСравнение числа операций для LU:")
        print(f"Метод Гаусса каждый раз: {cmp_lu['gauss_each_time_estimate']:.6f}")
        print(f"Одно LU + много решений : {cmp_lu['reuse_decomposition_estimate']:.6f}")
        print(f"Выгода                  : {cmp_lu['gain']:.6f}")
    except ValueError as exc:
        print("Эксперимент LU невозможен:")
        print(exc)

    print("\n=== Повторные решения через LUP ===")
    try:
        lup_results = repeated_lup_experiment(a, count)
        for idx, item in enumerate(lup_results, start=1):
            print("-" * 60)
            print(f"Тест {idx}")
            print("x_exact =", format_vector(item["x_exact"], 8))
            print("b       =", format_vector(item["b"], 8))
            print("x_num   =", format_vector(item["x_num"], 8))
            print(f"||x_num - x_exact||_inf = {item['error_inf']:.12g}")

        cmp_lup = compare_operation_counts(
            len(a),
            count,
            lup_results[0]["decomposition_operations"],
            lup_results[0]["solve_operations"],
        )

        print("\nСравнение числа операций для LUP:")
        print(f"Метод Гаусса каждый раз: {cmp_lup['gauss_each_time_estimate']:.6f}")
        print(f"Одно LUP + много решений: {cmp_lup['reuse_decomposition_estimate']:.6f}")
        print(f"Выгода                  : {cmp_lup['gain']:.6f}")
    except ValueError as exc:
        print("Эксперимент LUP невозможен:")
        print(exc)


def main() -> None:
    print("=== Практикум 7.2: LU и LUP разложения ===")
    a = input_matrix()
    b = input_rhs(len(a))

    print_lu_solution(a, b)
    print_lup_solution(a, b)

    if ask_yes_no("\nЗапустить серию решений с разными правыми частями? (да/нет): "):
        run_repeated_experiments(a)


if __name__ == "__main__":
    main()
