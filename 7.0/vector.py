from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Iterator, List


@dataclass
class Vector:
    data: List[float]

    def __init__(self, data: Iterable[float]) -> None:
        self.data = [float(x) for x in data]
        if not self.data:
            raise ValueError("Vector cannot be empty")

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[float]:
        return iter(self.data)

    def __getitem__(self, index: int) -> float:
        return self.data[index]

    def __setitem__(self, index: int, value: float) -> None:
        self.data[index] = float(value)

    def __add__(self, other: Vector) -> Vector:
        self._check_same_size(other)
        return Vector(a + b for a, b in zip(self.data, other.data))

    def __sub__(self, other: Vector) -> Vector:
        self._check_same_size(other)
        return Vector(a - b for a, b in zip(self.data, other.data))

    def __mul__(self, scalar: float) -> Vector:
        return Vector(scalar * x for x in self.data)

    def __rmul__(self, scalar: float) -> Vector:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector:
        if scalar == 0:
            raise ZeroDivisionError("Division by zero")
        return Vector(x / scalar for x in self.data)

    def dot(self, other: Vector) -> float:
        self._check_same_size(other)
        return sum(a * b for a, b in zip(self.data, other.data))

    def norm1(self) -> float:
        return sum(abs(x) for x in self.data)

    def norm2(self) -> float:
        return sqrt(sum(x * x for x in self.data))

    def norm_inf(self) -> float:
        return max(abs(x) for x in self.data)

    def copy(self) -> Vector:
        return Vector(self.data)

    def to_list(self) -> List[float]:
        return self.data.copy()

    def _check_same_size(self, other: Vector) -> None:
        if len(self) != len(other):
            raise ValueError("Vectors must have the same size")

    def __repr__(self) -> str:
        return f"Vector({self.data})"
