from abc import ABCMeta, abstractmethod


class Neuron(metaclass=ABCMeta):
    @abstractmethod
    def action_potential(self):
        pass

    @abstractmethod
    def add_connection_to(self, neuron):
        pass


