# Libraries
# =======================
import os
from HyperSolver import GA_HyperSolver as GA
from items_HyperSolver import Items
from HyperHeuristic import HyperHeuristic

# Variables for the GA
num_tournament = 7
num_parents_to_select = 2
individuals_to_exchange = 2
number_islands = 4
run_times = 30
population_size = 10
generations = 1000
# [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
crossover_probability = [0.9, 0.9, 0.9, 0.9]
# [0.01, 0.05, 0.07, 0.08, 0.10, 0.11, 0.12, 0.15, 0.18, 0.20]
mutation_probability = [0.01, 0.05, 0.07, 0.08]
# [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]
migration_probability = [0.4]

# Variables for the Hyper Heuristics
features = ['WEIGHT', 'ITEMS_IN_KNAPSACK', 'ITEMS_OUT_KNAPSACK', 'TOTAL_WEIGHT_LEFT', 'TOTAL_VALUE_LEFT']
heuristics = ['MAXP', 'MAXPW', 'MINW', 'MARK']
number_rules = 8

# Paths to the problem instance and to the solution folder
experiment = 'ga\\Test set (reduced)\\Training Set\\'
folder_instance = os.getcwd() + '\\Instances KP\\' + experiment
folder_solution = os.getcwd() + '\\experiments\\' + experiment
folder_name = str(population_size) + '-' + str(generations) + '-' + str(number_islands) + '-' + str(run_times)
folder_path = os.path.join(folder_solution, folder_name)

if not os.path.isdir(folder_path):
    os.makedirs(folder_path)

os.chdir(folder_path)
sub_folders = []
for folder in os.listdir():
    sub_folders.append(f"folder")

num_experiment = len(sub_folders)
if not os.path.isdir(str(num_experiment)):
    os.mkdir(str(num_experiment))
folder_path = os.path.join(folder_path, str(num_experiment))

os.chdir(folder_instance)
file_path = []
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
                 'Number of parents for crossover: {}'
                 .format(population_size, generations, number_islands, crossover_probability, mutation_probability,
                         migration_probability, run_times, num_tournament, individuals_to_exchange,
                         num_parents_to_select))
parameters.close()

if __name__ == '__main__':
    GA.geneticAlgorithm(num_tournament, num_parents_to_select, individuals_to_exchange,
                        number_islands, population_size, generations, crossover_probability, mutation_probability,
                        migration_probability, features, heuristics, number_rules, file_path, folder_path, run_times)
