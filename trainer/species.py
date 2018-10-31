import numpy as np

from tools.math_helpers import relative_difference_of
from trainer.configuration import Configuration


class Species:
    def __init__(self, organism):
        self.age = 0
        self._historical_fitness = list()
        self._goal_number_of_organisms = 40
        self._organisms = [organism.clone() for _ in range(self._goal_number_of_organisms)]

    @property
    def fitness(self):
        return np.max([organism.fitness for organism in self._organisms])

    def evolve(self):
        self.age += 1

        for organism in self._organisms:
            organism.evaluate()
            self._save_fitness()

        self._sort_by_fitness(self._organisms)
        self._organisms = self._remove_poorest_organisms(self._organisms)

        self._organisms += self._create_new_organisms_from(self._organisms[0])

        new_species = list()
        for organism in self._organisms[3:]:
            organism_has_become_a_new_species = organism.mutate()

            if organism_has_become_a_new_species:
                new_species.append(Species(organism))

        self._organisms = self._remove_organisms(new_species)

        return new_species

    def has_stopped_evolving(self):
        evolution_limit = Configuration.minimum_life_of_species
        if self.age <= evolution_limit:
            return False

        max_fitness = np.max(self._historical_fitness[-evolution_limit//2:])
        min_fitness = np.min(self._historical_fitness[-evolution_limit:-evolution_limit//2])

        return relative_difference_of(max_fitness, min_fitness) < Configuration.minimum_improvement_of_species

    def get_best_organism(self):
        return self._organisms[0]

    def _remove_poorest_organisms(self, organisms):
        return organisms[:(len(organisms) // 2)]

    def _sort_by_fitness(self, organisms):
        organisms.sort(key=lambda o: o.fitness, reverse=True)

    def _remove_organisms(self, new_species):
        return [organism for organism in self._organisms if not organism in new_species]

    def _save_fitness(self):
        self._historical_fitness.append(self.fitness)

    def _create_new_organisms_from(self, organism):
        number_new_organisms_to_create = self._goal_number_of_organisms - len(self._organisms)
        return [organism.clone() for _ in range(number_new_organisms_to_create)]