import numpy as np
from numpy.random import randn

from snes.neural_network.axon import Axon
from snes.neural_network.input_neuron import InputNeuron
from snes.neural_network.neuron import Neuron
from snes.neural_network.output_neuron import OutputNeuron
from snes.tools.math_helpers import sigmoid


class HiddenNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.incoming_axons = list()
        self.outgoing_axons = list()
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
        elif isinstance(neuron, OutputNeuron):
            return Axon.create_connection(self, neuron)
        else:
            if randn() > 0.5:
                return Axon.create_connection(neuron, self)
            else:
                return Axon.create_connection(self, neuron)

    def is_connected(self, neuron):
        return self.is_connected_upstream(neuron) or self.is_connected_downstream(neuron)

    def is_connected_upstream(self, neuron):
        if neuron is self:
            return True

        upstream_neurons = [axon.incoming_neuron for axon in self.incoming_axons]

        output = list()
        for connected_neuron in upstream_neurons:
            output.append(connected_neuron.is_connected_upstream(neuron))

        return np.any(output)

    def is_connected_downstream(self, neuron):
        if neuron is self:
            return True

        downstream_neurons = [axon.outgoing_neuron for axon in self.outgoing_axons]

        output = list()
        for connected_neuron in downstream_neurons:
            output.append(connected_neuron.is_connected_downstream(neuron))

        return np.any(output)

    def is_used(self):
        return bool(self.incoming_axons and self.outgoing_axons)
