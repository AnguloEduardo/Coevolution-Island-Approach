# Libraries
# =======================
import os
import GA as ga
from items import Items
from knapsack import Knapsack

# Variables
num_tournament = 5
num_parents_to_select = 2
individuals_to_exchange = 5
number_islands = 4
run_times = 30
population_size = 300
generations = 50
# [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
crossover_probability = [0.5, 0.6, 0.7, 0.8]
# [0.01, 0.05, 0.07, 0.08, 0.10, 0.11, 0.12, 0.15, 0.18, 0.20]
mutation_probability = [0.10, 0.11, 0.12, 0.15]
migration_probability = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]

# Path to instances
path = os.getcwd() + '\\Instances KP\\'
path_solution = os.getcwd() + '\\experiments\\'
folder_name = str(population_size) + '-' + str(generations) + '-' + str(number_islands) + \
              '-' + str(crossover_probability) + \
              '-' + str(mutation_probability) + '\\'
folder_path = os.path.join(path_solution, folder_name)
os.chdir(path)
file_path = []
# Iterate over all the files in the directory
for file in os.listdir():
   if file.endswith('.txt'):
      # Create the filepath of particular file
      file_path.append(f"{path}\\{file}")

if __name__ == '__main__':
    table = open(path_solution + folder_name + 'table.txt', 'a')
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
        # from the second row and forth, the first element represent the weight
        # the second element represent the profit
        backpack_capacity = int(problemCharacteristics[0])  # Number of items in the problem
        max_weight = int(problemCharacteristics[1])         # Maximum weight for the backpack to carry

        # Creation of item's characteristics with the information from the .txt file
        for idx in range(backpack_capacity):
            instanceItem = instance.readline().rstrip("\n")
            instanceItem = instanceItem.split(", ")
            list_items.append(Items(idx, int(instanceItem[0]), float(instanceItem[1])))
        instance.close()

        # Creates four empty Knapsacks to later store the best Knapsack of each island
        best_Knapsack = [Knapsack(max_weight, backpack_capacity)] * number_islands

        # Opening text file to save the data of each run
        problem = file_path[kp].split("\\")
        problem = problem[len(problem)-1].split('.')
        data = open(path_solution + folder_name + problem[0] +'_' +
                    str(number_islands) + '_Islands.txt', 'a')

        data.write('{} {} {} {} {} {} {} {} {}'.format(
            population_size, generations, backpack_capacity,
            max_weight, number_islands, crossover_probability,
            mutation_probability, migration_probability, file_path[kp]))
        
        for Rmigration in range(len(migration_probability)):
            for run in range(run_times):
                data.write('\n{} {}'.format(run + 1, migration_probability[Rmigration]))
                ga.geneticAlgorithm(num_tournament, num_parents_to_select, individuals_to_exchange,
                                    number_islands,run_times, list_items, population, population_size,
                                    generations, crossover_probability,mutation_probability,
                                    migration_probability[Rmigration], backpack_capacity, max_weight,
                                    best_Knapsack, table, data)
        data.close()
    table.close()
