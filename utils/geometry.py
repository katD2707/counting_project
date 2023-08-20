import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum
from color import Color, ColorPalette

@dataclass
class Point:
    x: float
    y: float

    def as_xy_int_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

    def as_xy_float_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


def get_polygon_center(polygon: np.ndarray) -> Point:
    """
    Calculate the center of a polygon.

    This function takes in a polygon as a 2-dimensional numpy ndarray and
    returns the center of the polygon as a Point object.
    The center is calculated as the mean of the polygon's vertices along each axis,
    and is rounded down to the nearest integer.

    Parameters:
        polygon (np.ndarray): A 2-dimensional numpy ndarray representing the
            vertices of the polygon.

    Returns:
        Point: The center of the polygon, represented as a
            Point object with x and y attributes.
    """
    center = np.mean(polygon, axis=0).astype(int)
    return Point(x=center[0], y=center[1])


class Position(Enum):
    CENTER = "CENTER"
    BOTTOM_CENTER = "BOTTOM_CENTER"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    def pad(self, padding):
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding,
        )