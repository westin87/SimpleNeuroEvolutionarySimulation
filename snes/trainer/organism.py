from snes.neural_network.brain import Brain
from snes.tools.math_helpers import relative_difference_of
from snes.trainer.task import Task


class Organism:
    def __init__(self, task: Task, brain: Brain):
        self.age = 0
        self._task: Task = task.create()
        self._brain = brain
        self._historical_fitness = list()

    def evaluate(self):
        self.age += 1
        self._task.start()

        while self._task.is_running():
            output = self._task.get_output()
            inputs = self._brain.think(output)
            self._task.set_input(inputs)
            self._task.tick()

        self._update_fitness()

    def mutate(self) -> bool:
        old_brain = self._brain.clone()
        self._brain.mutate()

        return self._is_new_species(self._brain, old_brain)

    @property
    def fitness(self):
        if self._historical_fitness:
            return self._historical_fitness[-1]
        else:
            return 0

    def clone(self):
        return Organism(self._task.create(), self._brain.clone())

    def _update_fitness(self):
        fitness = self._task.get_score()
        self._historical_fitness.append(fitness)

    @staticmethod
    def _is_new_species(brain, old_brain) -> bool:
        to_large_diff_in_neurons = relative_difference_of(brain.number_of_neurons, old_brain.number_of_neurons) > 0.2
        to_large_diff_in_axons = relative_difference_of(brain.number_of_axons, old_brain.number_of_axons) > 0.6
        return to_large_diff_in_neurons or to_large_diff_in_axons

    def plot_brain(self):
        self._brain.plot()

    def save(self):
        self._brain.save()

    @classmethod
    def load(cls, path, task):
        brain = Brain.load(path)
        return Organism(task, brain)

    def __str__(self):
        return f"F: {self.fitness}, N: {self._brain.number_of_neurons}, A: {self._brain.number_of_axons}"
