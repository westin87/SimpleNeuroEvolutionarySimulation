from numpy.random import randn


class Axon:
    def __init__(self, input_neuron, output_neuron, weight):
        self.weight = weight

        self.incoming_neuron = input_neuron
        self.outgoing_neuron = output_neuron

    @classmethod
    def create_connection(cls, input_neuron, output_neuron, weight=None):
        if weight is None:
            weight = randn()

        axon = cls(input_neuron, output_neuron, weight)
        axon.incoming_neuron.outgoing_axons.append(axon)
        axon.outgoing_neuron.incoming_axons.append(axon)
        return axon

    def remove(self):
        self.incoming_neuron.outgoing_axons.remove(self)
        self.outgoing_neuron.incoming_axons.remove(self)

    def action_potential(self, thought=None):
        return self.weight * self.incoming_neuron.action_potential(thought)
