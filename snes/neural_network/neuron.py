from abc import ABCMeta, abstractmethod

from snes.tools.point import Point


class Neuron(metaclass=ABCMeta):
    def __init__(self):
        self.position = Point(0, 0)

    @abstractmethod
    def action_potential(self):
        pass

    @abstractmethod
    def add_connection_to(self, neuron):
        pass

    @abstractmethod
    def is_connected_downstream(self, neuron):
        pass

    @abstractmethod
    def is_connected_upstream(self, neuron):
        pass




