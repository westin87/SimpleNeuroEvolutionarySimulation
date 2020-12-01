import os


class Configuration:
    max_number_of_generations = 1000  # This is to stop thing for going on forever and ever...
    success_fitness = 5000  # If an organism achieves this fitness the training end and the organism id returned, this has to be adapted to fit the game score.

    number_organisms_per_specie = 60
    number_of_species = 20

    number_of_mutations_per_iteration = [1, 2, 4, 8, 16]
    probability_for_mutation_type = [0.01, 0.08, 0.71, 0.2]  # [NewNeuron, NewAxon, IncreasedAxonInfluence, ChangeAxonActivity]

    number_of_new_neurons_in_mutation = [1, 2, 3, 4]  # How many new neurons should be added.
    probability_for_new_neurons = [0.9, 0.05, 0.04, 0.01]  # Probability for the above (must be same length).

    number_of_new_axons_in_mutation = [1, 2, 3, 4, 5]  # How many new axons should be added.
    probability_for_new_axons = [0.8, 0.1, 0.05, 0.04, 0.01]  # Probability for the above (must be same length).

    axon_change_factor = 1.05  # Factor of change for IncreasedAxonInfluence

    run_on_multiple_cores = True
    number_of_cores = os.cpu_count()
