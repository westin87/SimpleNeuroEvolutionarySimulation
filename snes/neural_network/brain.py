import sys
from copy import deepcopy
from datetime import datetime
from enum import Enum
from pathlib import Path
from random import randrange
from typing import List

import pickle
from matplotlib import pyplot as plt
from numpy.random import randn, choice

from snes.neural_network.axon import Axon
from snes.neural_network.input_neuron import InputNeuron
from snes.neural_network.hidden_neuron import HiddenNeuron
from snes.neural_network.neuron import Neuron
from snes.neural_network.output_neuron import OutputNeuron
from snes.tools.point import Point
from snes.trainer.configuration import Configuration


class Brain:
    def __init__(self, number_of_inputs, number_of_outputs):
        self._axons = list()
        self._thought = 0
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
        for _ in range(choice(Configuration.number_of_mutations_per_iteration)):
            mutation = self._get_random_mutation()

            if mutation == Mutation.NewNeuron:
                number_of_neurons = choice(
                    Configuration.number_of_new_neurons_in_mutation,
                    p=Configuration.probability_for_new_neurons)
                self._add_neurons(number_of_neurons)

            elif mutation == Mutation.NewAxon:
                number_of_axons = choice(
                    Configuration.number_of_new_axons_in_mutation,
                    p=Configuration.probability_for_new_axons)
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

        all_neurons = self.get_all_neurons()
        x = [n.position.x for n in all_neurons]
        y = [n.position.y for n in all_neurons]

        plt.figure()
        plt.plot(x, y, 'o')

        for axon in self._axons:
            x = [axon.incoming_neuron.position.x,
                 axon.outgoing_neuron.position.x]
            y = [axon.incoming_neuron.position.y,
                 axon.outgoing_neuron.position.y]
            plt.plot(x, y)
            x = axon.incoming_neuron.position.x + (axon.outgoing_neuron.position.x-axon.incoming_neuron.position.x)/2
            y = axon.incoming_neuron.position.y + (axon.outgoing_neuron.position.y-axon.incoming_neuron.position.y)/2
            plt.text(x, y, f"{axon.weight:.3f}")

    def save(self):
        self._remove_unused_neurons()
        path = Path(f"brain_{datetime.now().isoformat(timespec='seconds')}.bin")
        with path.open('wb') as fo:
            pickle.dump(self, fo)

        return path

    @staticmethod
    def load(path):
        path = Path(path)
        with path.open('rb') as fo:
            return pickle.load(fo)

    def _add_neurons(self, number_of_neurons):
        for _ in range(number_of_neurons):
            self._add_neuron()

    def _add_neuron(self):
        self.number_of_neurons += 1
        neuron = HiddenNeuron()
        self.hidden_neurons.append(neuron)

    def _add_axons(self, number_of_axons):
        for _ in range(number_of_axons):
            self._add_axon()

    def _add_axon(self):
        neuron = self._get_any_neuron()

        if isinstance(neuron, InputNeuron):
            random_neuron = self._get_any_non_input_neuron()
            emergency_break = 0
            while neuron.is_connected_downstream(random_neuron):
                random_neuron = self._get_any_non_input_neuron()
                emergency_break += 1
                if emergency_break > 1000:
                    return
        elif isinstance(neuron, HiddenNeuron):
            random_neuron = self._get_any_neuron()
            emergency_break = 0
            while neuron.is_connected(random_neuron):
                random_neuron = self._get_any_neuron()
                emergency_break += 1
                if emergency_break > 1000:
                    return
        elif isinstance(neuron, OutputNeuron):
            random_neuron = self._get_any_non_output_neuron()
            emergency_break = 0
            while neuron.is_connected_upstream(random_neuron):
                random_neuron = self._get_any_non_output_neuron()
                emergency_break += 1
                if emergency_break > 1000:
                    return
        else:
            return

        self.number_of_axons += 1
        self._axons.append(neuron.add_connection_to(random_neuron))

    def _add_axon_tmp(self):
        neuron = self._get_any_hidden_neuron()
        random_neuron = self._get_any_neuron()

        emergency_break = 0

        while (Axon(neuron, random_neuron, 0) in self._axons) or neuron.is_connected(random_neuron):
            random_neuron = self._get_any_neuron()
            emergency_break += 1
            if emergency_break > 1000:
                return

        self.number_of_axons += 1
        self._axons.append(neuron.add_connection_to(random_neuron))

    def _increase_influence_of_random_axon(self):
        axon = self._get_any_axon()
        if axon:
            axon.weight *= Configuration.axon_change_factor

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

    def _get_any_hidden_neuron(self) -> HiddenNeuron:
        return choice(self.hidden_neurons)

    def _get_any_non_input_neuron(self) -> Neuron:
        return choice(self.hidden_neurons + self.output_neurons)

    def _get_any_non_output_neuron(self) -> Neuron:
        return choice(self.input_neurons + self.hidden_neurons)

    def _remove_unused_neurons(self):
        print(f"Removing {len([n for n in self.hidden_neurons if not n.is_used()])} unused neurons")
        self.hidden_neurons = [n for n in self.hidden_neurons if n.is_used()]

    def _get_random_mutation(self):
        mutation_list = [
            Mutation.NewNeuron,
            Mutation.NewAxon,
            Mutation.IncreasedAxonInfluence,
            Mutation.ChangeAxonActivity]

        return choice(mutation_list, p=Configuration.probability_for_mutation_type)


class Mutation(Enum):
    NewNeuron = 1
    NewAxon = 2
    IncreasedAxonInfluence = 3
    ChangeAxonActivity = 4
