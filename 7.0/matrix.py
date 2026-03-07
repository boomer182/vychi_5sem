from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List

from vector import Vector


@dataclass
class Matrix:
    data: List[List[float]]

    def __init__(self, data: Iterable[Iterable[float]]) -> None:
        rows = [[float(x) for x in row] for row in data]
        if not rows:
            raise ValueError("Matrix cannot be empty")

        row_length = len(rows[0])
        if row_length == 0:
            raise ValueError("Matrix rows cannot be empty")

        for row in rows:
            if len(row) != row_length:
                raise ValueError("All rows must have the same length")

        self.data = rows

    @property
    def rows(self) -> int:
        return len(self.data)

    @property
    def cols(self) -> int:
        return len(self.data[0])

    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    def __getitem__(self, index: int) -> List[float]:
        return self.data[index]

    def __setitem__(self, index: int, value: List[float]) -> None:
        if len(value) != self.cols:
            raise ValueError("Invalid row length")
        self.data[index] = [float(x) for x in value]

    def __iter__(self) -> Iterator[List[float]]:
        return iter(self.data)

    def __add__(self, other: Matrix) -> Matrix:
        self._check_same_shape(other)
        return Matrix(
            [a + b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self.data, other.data)
        )

    def __sub__(self, other: Matrix) -> Matrix:
        self._check_same_shape(other)
        return Matrix(
            [a - b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self.data, other.data)
        )

    def __mul__(self, other: float | Vector | Matrix) -> Matrix | Vector:
        if isinstance(other, (int, float)):
            return Matrix(
                [float(other) * value for value in row]
                for row in self.data
            )

        if isinstance(other, Vector):
            if self.cols != len(other):
                raise ValueError("Matrix and vector sizes are incompatible")
            result = []
            for row in self.data:
                result.append(sum(a * b for a, b in zip(row, other)))
            return Vector(result)

        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Matrix sizes are incompatible")
            result = []
            other_t = other.transpose()
            for row in self.data:
                result_row = []
                for col in other_t.data:
                    result_row.append(sum(a * b for a, b in zip(row, col)))
                result.append(result_row)
            return Matrix(result)

        raise TypeError("Unsupported operand type for multiplication")

    def __rmul__(self, scalar: float) -> Matrix:
        if not isinstance(scalar, (int, float)):
            raise TypeError("Left operand must be a scalar")
        return self.__mul__(float(scalar))

    def transpose(self) -> Matrix:
        return Matrix(zip(*self.data))

    def is_square(self) -> bool:
        return self.rows == self.cols

    def copy(self) -> Matrix:
        return Matrix(row.copy() for row in self.data)

    @staticmethod
    def identity(n: int) -> Matrix:
        if n <= 0:
            raise ValueError("Size must be positive")
        return Matrix(
            [1.0 if i == j else 0.0 for j in range(n)]
            for i in range(n)
        )

    def row(self, index: int) -> Vector:
        return Vector(self.data[index])

    def col(self, index: int) -> Vector:
        return Vector(self.data[i][index] for i in range(self.rows))

    def _check_same_shape(self, other: Matrix) -> None:
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same shape")

    def __repr__(self) -> str:
        return f"Matrix({self.data})"
