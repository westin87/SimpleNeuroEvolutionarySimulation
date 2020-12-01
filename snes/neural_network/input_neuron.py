from snes.neural_network.axon import Axon
from snes.neural_network.neuron import Neuron


class InputNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.value = 0
        self.outgoing_axons = list()

    def action_potential(self, thought=None):
        return self.value

    def add_connection_to(self, neuron: Neuron):
        if isinstance(neuron, InputNeuron):
            raise ValueError("Can not connect a input neuron to another input neuron.")

        return Axon.create_connection(self, neuron)

    def is_connected_upstream(self, neuron):
        return False

    def is_connected_downstream(self, neuron):
        connected_neurons = [axon.outgoing_neuron for axon in self.outgoing_axons]
        return neuron in connected_neurons
