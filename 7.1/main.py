from __future__ import annotations

from typing import List

from analysis_utils import preliminary_assessment
from gauss_core import format_matrix, format_vector, solve_gauss
from hilbert_experiment import run_hilbert_experiment


def input_int(prompt: str, min_value: int | None = None) -> int:
    while True:
        try:
            value = int(input(prompt))
            if min_value is not None and value < min_value:
                print(f"Введите целое число не меньше {min_value}.")
                continue
            return value
        except ValueError:
            print("Ошибка: нужно ввести целое число.")


def input_float_list(prompt: str, expected_len: int) -> List[float]:
    while True:
        try:
            parts = input(prompt).strip().split()
            if len(parts) != expected_len:
                print(f"Ошибка: нужно ввести ровно {expected_len} чисел.")
                continue
            return [float(x) for x in parts]
        except ValueError:
            print("Ошибка: вводите только числа.")


def input_matrix_and_rhs() -> tuple[List[List[float]], List[float]]:
    n = input_int("Введите размер системы n: ", min_value=1)
    print("\nВведите матрицу A построчно.")
    a = []
    for i in range(n):
        row = input_float_list(f"Строка {i + 1}: ", n)
        a.append(row)

    print("\nВведите вектор правой части b.")
    b = input_float_list("b: ", n)

    return a, b


def choose_pivot_strategy() -> str:
    print("\nВыбор главного элемента:")
    print("1 - без выбора главного элемента")
    print("2 - по столбцу")
    print("3 - по строке")
    print("4 - по столбцу и строке")

    mapping = {
        1: "none",
        2: "column",
        3: "row",
        4: "full",
    }

    while True:
        choice = input_int("Ваш выбор: ")
        if choice in mapping:
            return mapping[choice]
        print("Нужно выбрать 1, 2, 3 или 4.")


def print_preliminary_assessment(a: List[List[float]], b: List[float]) -> None:
    print("\nПредварительная оценка системы:")
    for line in preliminary_assessment(a, b):
        print("-", line)


def print_gauss_result(result) -> None:
    print("\nРезультат решения:")
    print("Стратегия выбора главного элемента:", result.pivot_strategy)
    print("Решение x =", format_vector(result.solution, digits=12))
    print(f"Определитель det(A) = {result.determinant:.12g}")

    print("\nВерхнетреугольная матрица после прямого хода:")
    print(format_matrix(result.triangular_matrix, digits=10))

    print("\nПреобразованная правая часть:")
    print(format_vector(result.transformed_rhs, digits=10))

    print("\nЧисло арифметических операций:")
    print(f"Прямой ход     : {result.operations_forward}")
    print(f"Обратный ход   : {result.operations_backward}")
    print(f"Всего          : {result.operations_total}")
    print(f"По теории      : {result.theoretical_operations:.6f}")
    print(
        "Отклонение     : "
        f"{result.operations_total - result.theoretical_operations:.6f}"
    )


def ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("y", "yes", "д", "да"):
            return True
        if ans in ("n", "no", "н", "нет"):
            return False
        print("Введите да/нет.")


def print_hilbert_experiment() -> None:
    print("\nЭксперимент с матрицей Гильберта")
    print("Будем решать Ax = b, где точное решение x = (1, ..., 1)^T.")

    count = input_int("Сколько значений n проверить? ", min_value=1)
    n_values = []
    for i in range(count):
        n_values.append(input_int(f"n[{i + 1}] = ", min_value=1))

    pivot_strategy = choose_pivot_strategy()
    results = run_hilbert_experiment(n_values, pivot_strategy=pivot_strategy)

    print("\nРезультаты эксперимента:")
    for item in results:
        print("-" * 60)
        print(f"n = {item['n']}")
        if item["success"]:
            print("Решение найдено.")
            print(f"det(A) = {item['determinant']:.12g}")
            print(f"||x_exact - x_num||_inf / ||x_exact||_inf = {item['error_inf']:.12g}")
            print(f"Операций (факт) = {item['operations_total']}")
            print(f"Операций (теория) = {item['theoretical_operations']:.6f}")
            print("Найденное решение:")
            print(format_vector(item["solution"], digits=10))
        else:
            print("Решение не найдено.")
            print("Причина:", item["message"])


def main() -> None:
    print("=== Метод Гаусса для решения СЛАУ ===")

    a, b = input_matrix_and_rhs()
    print_preliminary_assessment(a, b)

    pivot_strategy = choose_pivot_strategy()

    try:
        result = solve_gauss(a, b, pivot_strategy=pivot_strategy)
        print_gauss_result(result)
    except ValueError as exc:
        print("\nОшибка при решении системы:")
        print(exc)

    if ask_yes_no("\nЗапустить эксперимент с матрицей Гильберта? (да/нет): "):
        print_hilbert_experiment()


if __name__ == "__main__":
    main()
