import numpy as np

from snes.trainer.configuration import Configuration
from snes.trainer.organism import Organism


class Species:
    def __init__(self, organism: Organism):
        self.age = 0
        self._goal_number_of_organisms = Configuration.number_organisms_per_specie
        self._organisms = [organism.clone() for _ in range(self._goal_number_of_organisms)]

    @property
    def fitness(self):
        return np.max([organism.fitness for organism in self._organisms])

    def evolve(self):
        self.age += 1

        for organism in self._organisms:
            organism.evaluate()

        self._remove_poorest_organisms()

        self._organisms += self._create_new_organisms_from(self._organisms[0])

        new_species = list()
        for organism in self._organisms[3:]:
            organism_has_become_a_new_species = organism.mutate()

            if organism_has_become_a_new_species:
                self._organisms.remove(organism)
                new_species.append(Species(organism))

        if Configuration.run_on_multiple_cores:
            return [self] + new_species
        else:
            return new_species

    def get_best_organism(self):
        return self._organisms[0]

    def _remove_poorest_organisms(self):
        self._organisms.sort(key=lambda o: o.fitness, reverse=True)
        self._organisms = self._organisms[:(3 * len(self._organisms) // 4)]

    def _create_new_organisms_from(self, organism):
        number_new_organisms_to_create = self._goal_number_of_organisms - len(self._organisms)
        return [organism.clone() for _ in range(number_new_organisms_to_create)]
