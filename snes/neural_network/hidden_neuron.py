import numpy as np

from snes.neural_network.axon import Axon
from snes.neural_network.input_neuron import InputNeuron
from snes.neural_network.neuron import Neuron
from snes.tools.math_helpers import sigmoid
from snes.tools.point import Point


class HiddenNeuron(Neuron):
    def __init__(self):
        self.incoming_axons = list()
        self.outgoing_axons = list()
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
        if isinstance(neuron, InputNeuron):
            return Axon.create_connection(neuron, self)
        else:
            return Axon.create_connection(self, neuron)

    def is_connected_up_stream(self, neuron):
        hidden_neurons = [axon.incoming_neuron for axon in self.incoming_axons if isinstance(axon.incoming_neuron, HiddenNeuron)]
        if neuron is self:
            return True
        elif not hidden_neurons:
            return False
        else:
            output = list()
            for hidden_neuron in hidden_neurons:
                output.append(hidden_neuron.is_connected_up_stream(neuron))

            return np.any(output)