from copy import deepcopy
from datetime import datetime
from enum import Enum
from pathlib import Path
from random import randrange
from typing import List

import pickle
from matplotlib import pyplot as plt
from numpy.random import randn, choice

from neural_network.axon import Axon
from neural_network.neuron import Neuron
from neural_network.output_neuron import OutputNeuron
from neural_network.hidden_neuron import HiddenNeuron
from neural_network.input_neuron import InputNeuron
from tools.point import Point


class Brain:
    def __init__(self, number_of_inputs, number_of_outputs):
        self._axons = list()
        self._thought = 0;
        self.input_neurons = [InputNeuron() for _ in range(number_of_inputs)]
        self.hidden_neurons = [HiddenNeuron()]
        self.output_neurons = [OutputNeuron() for _ in range(number_of_outputs)]
        self.number_of_neurons = number_of_inputs + number_of_outputs + 1
        self.number_of_axons = 0

    def think(self, input_values):
        self._thought += 1
        for input_neuron, value in zip(self.input_neurons, input_values):
            input_neuron.value = value

        output_values = list()
        for output_neuron in self.output_neurons:
            output_values.append(output_neuron.action_potential(self._thought))

        return output_values

    def mutate(self):
        mutation = self._get_random_mutation()

        if mutation == Mutation.NewNeuron:
            number_of_neurons = choice([1, 2, 3, 4], p=[0.9, 0.05, 0.04, 0.01])
            self._add_neurons(number_of_neurons)
        elif mutation == Mutation.NewConnection:
            number_of_axons = choice([1, 2, 3, 4, 5], p=[0.8, 0.1, 0.05, 0.04, 0.01])
            self._add_axons(number_of_axons)
        elif mutation == Mutation.IncreasedAxonInfluence:
            self._increase_influence_of_random_axon()
        elif mutation == Mutation.ChangeAxonActivity:
            self._change_influence_of_random_neuron()

    def get_all_neurons(self) -> List[Neuron]:
        return self.input_neurons + self.hidden_neurons + self.output_neurons

    def clone(self):
        return deepcopy(self)

    def plot(self):
        for i, n in enumerate(self.input_neurons):
            n.position += Point(0, 5*i)

        for i, n in enumerate(self.hidden_neurons):
            n.position += Point(randrange(1, 9), 3*i)

        for i, n in enumerate(self.output_neurons):
            n.position += Point(10, 5*i)

        x = [n.position.x for n in self.get_all_neurons()]
        y = [n.position.y for n in self.get_all_neurons()]

        plt.figure()
        plt.plot(x, y, 'o')
        for axon in self._axons:
            x = [axon.incoming_neuron.position.x,
                 axon.outgoing_neuron.position.x]
            y = [axon.incoming_neuron.position.y,
                 axon.outgoing_neuron.position.y]
            plt.plot(x, y)

        plt.show()

    def save(self):
        path = Path(f"brain_{datetime.now().isoformat(timespec='seconds')}.bin")
        with path.open('wb') as fo:
            pickle.dump(self, fo)

        return path

    @staticmethod
    def load(path):
        path = Path(path)
        with path.open('rb') as fo:
            return pickle.load(fo)

    def _add_neuron(self):
        self.number_of_neurons += 1
        neuron = HiddenNeuron()
        self.hidden_neurons.append(neuron)

    def _add_neurons(self, number_of_neurons):
        for _ in range(number_of_neurons):
            self._add_neuron()

    def _add_axon(self):
        self.number_of_axons += 1

        neuron = self._get_any_neuron()

        if isinstance(neuron, InputNeuron):
            random_neuron = self._get_any_non_input_neuron()

        elif isinstance(neuron, HiddenNeuron):
            random_neuron = self._get_any_neuron()
            while neuron.is_connected_up_stream(random_neuron):
                random_neuron = self._get_any_neuron()

        elif isinstance(neuron, OutputNeuron):
            random_neuron = self._get_any_non_output_neuron()
        else:
            raise ValueError(f"Unknown neuron type trying to connect: "
                             f"'{type(neuron)}'")

        self._axons.append(neuron.add_connection_to(random_neuron))

        return random_neuron

    def _add_axons(self, number_of_axons):
        for _ in range(number_of_axons):
            self._add_axon()

    def _increase_influence_of_random_axon(self):
        axon = self._get_any_axon()
        if axon:
            axon.weight *= 1.05

    def _change_influence_of_random_neuron(self):
        axon = self._get_any_axon()
        if axon:
            axon.weight = randn()

    def _get_any_axon(self) -> Axon:
        any_neuron = self._get_any_non_input_neuron()
        if any_neuron.incoming_axons:
            return choice(any_neuron.incoming_axons)
        return None

    def _get_any_neuron(self) -> Neuron:
        return choice(self.get_all_neurons())

    def _get_any_non_input_neuron(self) -> Neuron:
        return choice(self.hidden_neurons + self.output_neurons)

    def _get_any_non_output_neuron(self) -> Neuron:
        return choice(self.input_neurons + self.hidden_neurons)

    def _get_random_mutation(self):
        mutation_list = [
            Mutation.NewNeuron,
            Mutation.NewConnection,
            Mutation.IncreasedAxonInfluence,
            Mutation.ChangeAxonActivity]

        return choice(mutation_list, p=[0.05, 0.1, 0.65, 0.2])


class Mutation(Enum):
    NewNeuron = 1
    NewConnection = 2
    IncreasedAxonInfluence = 3
    ChangeAxonActivity = 4
