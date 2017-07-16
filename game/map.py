import numpy as np

from tools.point import Point


class Map:
    def __init__(self, size):
        self.size = size
        self.coord = Point(size[0], size[1])

    def render(self, point):
        map = np.zeros(self.size)
        map[point.x, point.y] = 1
        return map

    def is_inside(self, point):
        return point.x >= 0 and point.x < self.coord.x and point.y >= 0 and point.y < self.coord.y