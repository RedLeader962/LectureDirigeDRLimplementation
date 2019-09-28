# coding=utf-8
from collections import namedtuple
from typing import NamedTuple

Point = namedtuple('Point', ['x', 'y'])

Point3D = namedtuple('Point3D', Point._fields + ('z',))


class PointWithStr(namedtuple('Point', Point._fields)):
    __slots__ = ()
    # The subclass shown above sets __slots__ to an empty tuple.
    # This helps keep memory requirements low by preventing the creation of instance dictionaries.
    # source: https://docs.python.org/3.7/library/collections.html#collections.namedtuple

    @property
    def hypot(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __str__(self):
        return 'Point: x=%6.3f  y=%6.3f  hypot=%6.3f' % (self.x, self.y, self.hypot)


class Employee(NamedTuple):
    """Represents an employee."""
    name: str
    id: int = 3

    def __repr__(self) -> str:
        return f'<Employee {self.name}, id={self.id}>'


