import numpy as np

from neural_network.axon import Axon
from neural_network.neuron import Neuron
from tools.math_helpers import sigmoid
from tools.point import Point


class OutputNeuron(Neuron):
    def __init__(self):
        self.incoming_axons = list()
        self.position = Point(0, 0)
        self._action_potential = None
        self._last_thought = None

    def action_potential(self, thought=None):
        if thought == self._last_thought:
            return self._action_potential

        incoming_potential = np.sum([axon.action_potential(thought) for axon in self.incoming_axons])
        self._action_potential = sigmoid(incoming_potential)
        return self._action_potential

    def add_connection_to(self, neuron: Neuron):
        if isinstance(neuron, OutputNeuron):
            raise ValueError("Can not connect a output neuron to another output neuron.")

        return Axon.create_connection(neuron, self)