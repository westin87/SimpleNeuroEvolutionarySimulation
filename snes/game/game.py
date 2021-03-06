from numpy.random import choice
from matplotlib import pyplot as plt

from snes.game.map import Map
from snes.tools.point import Point
from snes.trainer.task import Task


class Game(Task):
    @property
    def number_of_inputs(self):
        return 4

    @property
    def number_of_outputs(self):
        return 2

    @classmethod
    def create(cls):
        return cls()

    def __init__(self):
        pass

    def start(self, draw=False):
        self.map = Map((10, 10))
        self.player = Point(4, 4)
        self.movement = Point(0, 0)
        self.ticks = 0
        self.direction = Point(0, 1)
        self._player_is_alive = True
        self._draw = draw

        if self._draw:
            plt.figure()
            self._im = plt.imshow(self.get_map())

    def tick(self):
        self.ticks += 1

        self.player += self.movement
        self.movement = Point(0, 0)

        if self.ticks % 25:
            self.direction = Point(choice([-1, 1]), choice([-1, 1]))

        self.player += self.direction

        if self.ticks > 10000 or self._not_on_map(self.player):
            self._player_is_alive = False

        if self._draw:
            self._im.set_data(self.get_map())
            plt.pause(0.001)

    def set_input(self, inputs):
        if inputs[0] > 0.5:
            self.movement += Point(-1, 0)

        if inputs[1] > 0.5:
            self.movement += Point(1, 0)

        if inputs[2] > 0.5:
            self.movement += Point(0, -1)

        if inputs[3] > 0.5:
            self.movement += Point(0, 1)

    def get_output(self):
        return self.player.x / 10, self.player.y / 10

    def get_score(self):
        return self.ticks

    def get_map(self):
        return self.map.render(self.player)

    def is_running(self):
        return self._player_is_alive

    def _not_on_map(self, player):
        return not self.map.is_inside(player)
