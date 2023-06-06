# Libraries
# =======================
import os
from GA_HyperSolver import genetic_algorithm as ga
from knapsack_HyperSolver import read_instance
from problem import ProblemCharacteristics


def train(num_tournament, num_parents_to_select, individuals_to_exchange, number_islands,
          population_size, generations, crossover_probability, mutation_probability, migration_probability,
          features, heuristics, number_rules, training_split, training_set, lb, ub, nm, pc, nc):
    # Paths to the problem instance and to the solution folder
    root = os.getcwd()
    experiment = os.path.join('ga', training_set, training_split)
    folder_instance = os.path.join(root, 'Instances KP', experiment)
    folder_solution = os.path.join(root, 'experiments', experiment)
    folder_name = f"{population_size}-{generations}-{number_islands}"
    folder_path = os.path.join(folder_solution, folder_name)

    os.makedirs(folder_path, exist_ok=True)

    num_experiment = len(os.listdir(folder_path))
    experiment_folder_path = os.path.join(folder_path, str(num_experiment))
    os.makedirs(experiment_folder_path, exist_ok=True)

    file_paths = [os.path.join(folder_instance, file) for file in os.listdir(folder_instance)]

    parameters_filepath = os.path.join(experiment_folder_path, 'General Parameters.txt')
    with open(parameters_filepath, 'a') as parameters:
        parameters.write(f'Population per island: {population_size}\n'
                         f'Generations: {generations}\n'
                         f'Number of islands: {number_islands}\n'
                         f'Crossover probabilities: {crossover_probability}\n'
                         f'Mutation probabilities: {mutation_probability}\n'
                         f'Migration probabilities: {migration_probability}\n'
                         f'Number of tournaments: {num_tournament}\n'
                         f'Number of individuals to exchange: {individuals_to_exchange}\n'
                         f'Number of parents for crossover: {num_parents_to_select}\n'
                         f'Number of rules: {number_rules}\n'
                         f'Heuristics: {heuristics}\n'
                         f'Features: {features}\n'
                         f'Training split: {training_split}\n'
                         f'Training Set: {training_set}\n'
                         '\n### Polynomial mutation parameters ###\n'
                         f'nm: {100}\n'
                         f'pm: {"1/n"}\n'
                         f'lb: {lb}\n'
                         f'ub: {ub}\n'
                         '\n### Simulated binary crossover parameters ###\n'
                         f'pc: {1}\n'
                         f'nc: {30}\n'
                         f'lb: {lb}\n'
                         f'ub: {ub}\n')

    problem_pool = [ProblemCharacteristics(read_instance(file)) for file in file_paths]

    ga(num_tournament, num_parents_to_select, individuals_to_exchange,
       number_islands, population_size, generations, crossover_probability, mutation_probability,
       migration_probability, features, heuristics, number_rules, problem_pool, experiment_folder_path, lb, ub, nm, pc,
       nc)
