# Libraries
# =======================
import os
from GA_HyperSolver import genetic_algorithm as ga
from knapsack_HyperSolver import read_instance
from problem import ProblemCharacteristics


def train(num_tournament, num_parents_to_select, individuals_to_exchange, number_islands, run_times,
          population_size, generations, crossover_probability, mutation_probability, migration_probability,
          features, heuristics, number_rules, training_split):
    # Variables
    problem_pool = []
    file_path = []
    sub_folders = []

    # Paths to the problem instance and to the solution folder
    experiment = 'ga\\Test set A\\' + training_split + '\\Training\\'
    folder_instance = os.getcwd() + '\\Instances KP\\' + experiment
    folder_solution = os.getcwd() + '\\experiments\\' + experiment
    folder_name = str(population_size) + '-' + str(generations) + '-' + str(number_islands) + '-' + str(run_times)
    folder_path = os.path.join(folder_solution, folder_name)

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    os.chdir(folder_path)
    for folder in os.listdir():
        sub_folders.append(f"folder")

    num_experiment = len(sub_folders)
    if not os.path.isdir(str(num_experiment)):
        os.mkdir(str(num_experiment))
    folder_path = os.path.join(folder_path, str(num_experiment))

    os.chdir(folder_instance)
    # Iterate over all the files in the directory
    for file in os.listdir():
        # Create the filepath of particular file
        file_path.append(f"{folder_instance}\\{file}")

    parameters = open(folder_path + '\\' + 'General Parameters.txt', 'a')
    parameters.write('Population per island: {}\n'
                     'Generations: {}\n'
                     'Number of islands: {}\n'
                     'Crossover probabilities: {}\n'
                     'Mutation probabilities: {}\n'
                     'Migration probabilities: {}\n'
                     'Run times: {}\n'
                     'Number of tournaments: {}\n'
                     'Number of individuals to exchange: {}\n'
                     'Number of parents for crossover: {}\n'
                     'Number of rules: {}\n'
                     'Training split: {}'
                     .format(population_size, generations, number_islands, crossover_probability, mutation_probability,
                             migration_probability, run_times, num_tournament, individuals_to_exchange,
                             num_parents_to_select, number_rules, training_split))
    parameters.close()

    for file in file_path:
        problem_pool.append(ProblemCharacteristics(read_instance(file)))

    ga(num_tournament, num_parents_to_select, individuals_to_exchange,
       number_islands, population_size, generations, crossover_probability, mutation_probability,
       migration_probability, features, heuristics, number_rules, problem_pool, folder_path, run_times)
