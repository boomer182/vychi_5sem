from matrix import Matrix
from vector import Vector
from norms import (
    vector_norm_1,
    vector_norm_2,
    vector_norm_inf,
    matrix_norm_1,
    matrix_norm_2,
    matrix_norm_inf,
    frobenius_norm,
)
from inverse import inverse_matrix
from conditioning import condition_number


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


def input_vector() -> Vector:
    n = input_int("Введите размер вектора: ")

    while True:
        try:
            raw = input(f"Введите {n} координат вектора через пробел: ").strip().split()
            if len(raw) != n:
                print(f"Ошибка: нужно ввести ровно {n} чисел.")
                continue
            values = [float(x) for x in raw]
            return Vector(values)
        except ValueError:
            print("Ошибка: координаты должны быть числами.")


def input_matrix() -> Matrix:
    rows = input_int("Введите количество строк матрицы: ")
    cols = input_int("Введите количество столбцов матрицы: ")

    matrix_data = []
    print("Введите строки матрицы. Каждая строка вводится через пробел.")

    for i in range(rows):
        while True:
            try:
                raw = input(f"Строка {i + 1}: ").strip().split()
                if len(raw) != cols:
                    print(f"Ошибка: в строке должно быть ровно {cols} чисел.")
                    continue
                row = [float(x) for x in raw]
                matrix_data.append(row)
                break
            except ValueError:
                print("Ошибка: элементы матрицы должны быть числами.")

    return Matrix(matrix_data)


def print_matrix(title: str, matrix: Matrix) -> None:
    print(title)
    for row in matrix.data:
        print(" ".join(f"{x:10.6f}" for x in row))
    print()


def main() -> None:
    print("=== Ввод вектора ===")
    v = input_vector()
    print()

    print("=== Ввод матрицы ===")
    a = input_matrix()
    print()

    print("Введённый вектор:")
    print(v)
    print(f"||v||_1   = {vector_norm_1(v)}")
    print(f"||v||_2   = {vector_norm_2(v)}")
    print(f"||v||_inf = {vector_norm_inf(v)}")
    print()

    print_matrix("Введённая матрица:", a)

    print(f"||A||_1   = {matrix_norm_1(a)}")
    print(f"||A||_2   = {matrix_norm_2(a)}")
    print(f"||A||_inf = {matrix_norm_inf(a)}")
    print(f"||A||_F   = {frobenius_norm(a)}")
    print()

    if a.is_square():
        try:
            inv_a = inverse_matrix(a)
            print_matrix("Обратная матрица A^{-1}:", inv_a)

            print(f"cond_1(A)   = {condition_number(a, 1)}")
            print(f"cond_2(A)   = {condition_number(a, 2)}")
            print(f"cond_inf(A) = {condition_number(a, 'inf')}")
            print()
        except ValueError as error:
            print(f"Не удалось найти обратную матрицу: {error}")
            print("Число обусловленности для этой матрицы не вычисляется.")
            print()
    else:
        print("Матрица не квадратная, поэтому обратная матрица и число обусловленности не вычисляются.")
        print()

    if a.cols == len(v):
        av = a * v
        print("Произведение A * v:")
        print(av)
    else:
        print("Нельзя вычислить A * v: число столбцов матрицы должно совпадать с размером вектора.")


if __name__ == "__main__":
    main()
