# Libraries
# =======================
import os
from DirectSolver import GA as ga
from items import Items
from knapsack import Knapsack

# Variables
num_tournament = 5
num_parents_to_select = 2
individuals_to_exchange = 10
number_islands = 4
run_times = 30
population_size = [2000, 500]
generations = 100
# [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
crossover_probability = [0.9, 0.9, 0.9, 0.9]
# [0.01, 0.05, 0.07, 0.08, 0.10, 0.11, 0.12, 0.15, 0.18, 0.20]
mutation_probability = [0.01, 0.05, 0.07, 0.08]
migration_probability = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]

# Paths to the problem instance and to the solution folder
experiment = 'ga\\Test set (reduced)\\'
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
                         migration_probability, run_times, num_tournament, individuals_to_exchange, num_parents_to_select))
parameters.close()

if __name__ == '__main__':
    table = open(folder_path + '\\' + 'table.txt', 'a')
    for kp in range(len(file_path)):
        # Creates a list of lists to save the different populations from the islands
        population = [[]] * number_islands
        # List with the items of the problem
        list_items = []
        # Reading files with the instance problem
        instance = open(file_path[kp], 'r')
        problemCharacteristics = instance.readline().rstrip("\n")
        problemCharacteristics = problemCharacteristics.split(", ")

        # This information needs to be taken from the .txt files
        # First element in the first row indicates the number of items
        # second element of the first row indicates the backpack capacity
        # from the second row and forth, the first element represent the profit
        # the second element represent the weight
        backpack_capacity = int(problemCharacteristics[0])  # Number of items in the problem
        max_weight = float(problemCharacteristics[1])       # Maximum weight for the backpack to carry

        # Creation of item's characteristics with the information from the .txt file
        for idx in range(backpack_capacity):
            instanceItem = instance.readline().rstrip("\n")
            instanceItem = instanceItem.split(", ")
            list_items.append(Items(idx, float(instanceItem[1]), int(instanceItem[0])))
        instance.close()

        # Opening text file to save the data of each run
        problem = file_path[kp].split("\\")
        problem = problem[len(problem)-1].split('.')
        data = open(folder_path + '//' + problem[0] + '_' + str(number_islands) + '_Islands.txt', 'a')

        data.write('{} {} {} {} {} {} {} {} {}'.format(
            population_size, generations, backpack_capacity,
            max_weight, number_islands, crossover_probability,
            mutation_probability, migration_probability, file_path[kp]))
        print(file_path[kp])
        for Rmigration in range(len(migration_probability)):
            if Rmigration == 0:
                size = population_size[0]
            else:
                size = population_size[1]
            for run in range(run_times):
                # Creates four empty Knapsacks to later store the best Knapsack of each island
                best_Knapsack = [Knapsack(max_weight, backpack_capacity)] * number_islands
                data.write('\n{} {}'.format(run + 1, migration_probability[Rmigration]))
                GA.geneticAlgorithm(num_tournament, num_parents_to_select, individuals_to_exchange,
                                    number_islands, list_items, population, size,
                                    generations, crossover_probability, mutation_probability,
                                    migration_probability[Rmigration], backpack_capacity, max_weight,
                                    best_Knapsack, table, data)
        table.write('\n')
        data.close()
    table.close()
