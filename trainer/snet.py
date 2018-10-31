from neural_network.brain import Brain
from trainer.configuration import Configuration
from trainer.organism import Organism
from trainer.species import Species
from trainer.task import Task


class SNET:
    def __init__(self, task: Task):
        self.generation = 0

        initial_brain = Brain(task.number_of_outputs, task.number_of_inputs)
        self._speciess = [Species(Organism(task, initial_brain))]

    def train(self) -> Organism:
        while self.generation < Configuration.max_number_of_generations:
            self.generation += 1
            print("=" * 10 + f"Generation: {self.generation}" + "=" * 10)

            new_speciess = list()
            for species in self._speciess:
                new_speciess += species.evolve()
            self._speciess += new_speciess

            self._kill_speciess_that_have_stoped_evolving()

            if self._all_speciess_has_been_extinct():
                print("All species's has become extinct. Ether increase probability "
                      "for new species's or lower limit for when to kill a species.")
                return None

            best_organism = self._get_best_organism()
            print(f"Current best organism: {best_organism}")
            print(f"Number of species in next iteration: {len(self._speciess)}")
            if best_organism.fitness > Configuration.success_fitness:
                break

        return best_organism

    def _kill_speciess_that_have_stoped_evolving(self):
        speciess_to_delete = list()
        for species in self._speciess:
            if species.has_stopped_evolving():
                speciess_to_delete.append(species)

        self._remove_speciess(speciess_to_delete)

    def _remove_speciess(self, speciess):
        self._speciess = [species for species in self._speciess if not species in speciess]

    def _get_best_organism(self):
        self._speciess.sort(key=lambda s: s.fitness, reverse=True)
        return self._speciess[0].get_best_organism()

    def _all_speciess_has_been_extinct(self):
        return not self._speciess
