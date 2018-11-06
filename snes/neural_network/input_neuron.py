from snes.neural_network.axon import Axon
from snes.neural_network.neuron import Neuron
from snes.tools.point import Point


class InputNeuron(Neuron):
    def __init__(self):
        self.value = 0
        self.outgoing_axons = list()
        self.position = Point(0, 0)

    def action_potential(self, thought=None):
        return self.value

    def add_connection_to(self, neuron: Neuron):
        if isinstance(neuron, InputNeuron):
            raise ValueError("Can not connect a input neuron to another input neuron.")

        return Axon.create_connection(self, neuron)