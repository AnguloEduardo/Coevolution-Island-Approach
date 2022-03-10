# Libraries
# =======================
import random
from tqdm import tqdm
from numpy.random import randint
from numpy import concatenate
from items import Items
from knapsack import Knapsack

# Variables
num_tournament = 3
num_parents_to_select = 2
individuals_to_exchange = 5
number_islands = 0
file_name = 'ks_300_0'
run_times = 30
# List with the items of the problem
list_items = []
# Creates a list of lists to save the different populations from the islands
population = [[]] * number_islands
# Initialization of population size, generations, crossover and mutation probabilities
# for the four different islands
population_size = 500
generations = 500
crossover_probability = [0.3, 0.5, 0.7, 1.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
mutation_probability = [0.05, 0.10, 0.15, 0.20, 0.01, 0.07, 0.08, 0.12, 0.18, 0.11]


# Generate population
def generate_population():
    new_population = []
    for i in range(population_size):
        # Initialize Random population
        items = list_items.copy()
        individual = Knapsack(max_weight, backpack_capacity)
        index = randint(len(items))
        while individual.canPack(items[index]):
            individual.pack(items.pop(index))
            index = randint(len(items))
        new_population.append(individual)
    return new_population


def crossover_island_1(parentA, parentB):
    if random.random() <= crossover_probability[number_islands - 4]:
        offspring_a = concatenate((parentA[:backpack_capacity // 2], parentB[backpack_capacity // 2:])).tolist()
        offspring_b = concatenate((parentA[backpack_capacity // 2:], parentB[:backpack_capacity // 2])).tolist()
    else:
        offspring_a = parentA.copy()
        offspring_b = parentB.copy()
    return offspring_a, offspring_b


def crossover_island_2(parentA, parentB):
    if random.random() <= crossover_probability[number_islands - 3]:
        index = randint(1, len(parentA)-1)
        offspring_a = concatenate((parentA[:index], parentB[index:])).tolist()
        offspring_b = concatenate((parentB[:index], parentA[index:])).tolist()
    else:
        offspring_a = parentA.copy()
        offspring_b = parentB.copy()
    return offspring_a, offspring_b


def crossover_island_3(parentA, parentB):
    if random.random() <= crossover_probability[number_islands - 2]:
        index_1, index_2 = randint(1, len(parentA)-2, 2)
        if index_1 > index_2:
            temp = index_1
            index_1 = index_2
            index_2 = temp
        offspring_a = concatenate((parentA[:index_1], parentB[index_1:index_2], parentA[index_2:])).tolist()
        offspring_b = concatenate((parentB[:index_1], parentA[index_1:index_2], parentB[index_2:])).tolist()
    else:
        offspring_a = parentA.copy()
        offspring_b = parentB.copy()
    return offspring_a, offspring_b


def crossover_island_4(parentA, parentB):
    if random.random() <= crossover_probability[number_islands - 1]:
        for x in range(len(parentA)-1):
            if parentA[x] != parentB[x]:
                if random.random() <= 0.5:
                    temp = parentA[x]
                    parentA[x] = parentB[x]
                    parentB[x] = temp
        offspring_a = parentA.copy()
        offspring_b = parentB.copy()
    else:
        offspring_a = parentA.copy()
        offspring_b = parentB.copy()
    return offspring_a, offspring_b


# Crossover operator
# =======================
def combine(parentA, parentB, island):
    if island == 0 or island == 4 or island == 8:
        offspring_a, offspring_b = crossover_island_1(parentA, parentB)
    elif island == 1 or island == 5 or island == 9:
        offspring_a, offspring_b = crossover_island_2(parentA, parentB)
    elif island == 2 or island == 6:
        offspring_a, offspring_b = crossover_island_3(parentA, parentB)
    else:
        offspring_a, offspring_b = crossover_island_4(parentA, parentB)
    return offspring_a, offspring_b


# Mutation operator
# =======================
def mutate(individual, mRate):
    if random.random() <= mRate:
        i, j = random.sample(range(backpack_capacity), 2)
        individual.chromosome[i] = 0 if individual.chromosome[i] == 1 else 1
        individual.chromosome[j] = 0 if individual.chromosome[j] == 1 else 1
        return individual, True
    return individual, False


def calculate_weight_value(chromosome):
    weight, value = float(max_weight), 0
    for x, gene in enumerate(chromosome):
        if gene == 1:
            if list_items[x].getWeight() <= weight:
                weight -= list_items[x].getWeight()
                value += list_items[x].getValue()
            else:
                chromosome[x] = 0
    return weight, value, chromosome


# Tournament selection
# =======================
def select(island, tSize, numParents):
    parent_a = Knapsack(max_weight, backpack_capacity)
    parent_b = Knapsack(max_weight, backpack_capacity)
    for x in range(numParents):
        island_population = population[island].copy()
        winner = randint(len(island_population)-1)
        rival = randint(len(island_population)-1)
        individualWinner = island_population.pop(winner)
        individualRival = island_population.pop(rival)
        for i in range(tSize):
            if individualRival.getValue() > individualWinner.getValue():  # It is searching for the Knapsack
                winner = rival                                            # with the highest value
            rival = randint(len(island_population)-1)
            individualRival = island_population.pop(rival)
        if x == 0:
            parent_a = population[island][winner]
        else:
            parent_b = population[island][winner]
    return parent_a, parent_b


def sort_population():
    # Sorting Knapsacks by they value
    for i in range(number_islands):
        population[i].sort(key=lambda x: x.value, reverse=True)
        # Keeping track of the best solution found on each island
        if population[i][0].getValue() > best_Knapsack[i].getValue():
            best_Knapsack[i] = population[i][0]


def migrate():
    # Exchanging the best five Knapsacks solutions of each island to another
    temp_knapsack = []
    for index_1 in range(individuals_to_exchange):
        temp_knapsack.append(population[0][index_1])
    for index_1 in range(number_islands - 1):
        for index_2 in range(individuals_to_exchange):
            population[index_1][index_2] = population[index_1 + 1][index_2]
    for index_1 in range(individuals_to_exchange):
        population[number_islands - 1][index_1] = temp_knapsack[index_1]


# Genetic algorithm
# =======================
def geneticAlgorithm():
    global best_Knapsack
    # Running the genetic algorithm 30 times
    # to have a sample of how the algorithm
    # is behaving
    for run in range(run_times):
        data.write('\n\nRun number: {}'.format(run + 1))
        # Random generating the populations of the islands
        for x in range(number_islands):
            population[x] = generate_population()

        # Runs the evolutionary process
        for i in tqdm(range(generations)):
            for island in range(number_islands):
                # Crossover
                new_population = []
                for _ in range(population_size // 2):
                    parent_a, parent_b = select(island, num_tournament, num_parents_to_select)
                    offspring_a, offspring_b = combine(parent_a.getChromosome().copy(), parent_b.getChromosome().copy(), island)
                    weight, value, offspring_a = calculate_weight_value(offspring_a.copy())
                    if isinstance(offspring_a[0], float):
                        offspring_a = [int(x) for x in offspring_a]
                    child_a = Knapsack(weight, value, offspring_a)
                    weight, value, offspring_b = calculate_weight_value(offspring_b.copy())
                    if isinstance(offspring_b[0], float):
                        offspring_b = [int(x) for x in offspring_b]
                    child_b = Knapsack(weight, value, offspring_b)
                    new_population.extend([child_a, child_b])
                population[island] = new_population

                # Mutation
                for index in range(population_size):
                    individual, boolean = mutate(population[island][index], mutation_probability[island])
                    if boolean:
                        weight, value, chromosome = calculate_weight_value(individual.chromosome.copy())
                        population[island][index].chromosome = chromosome
                        population[island][index].value = value
                        population[island][index].totalWeight = weight

            # Printing useful information
            if (i % 100 == 0 or i == generations - 1) and i != 0:
                sort_population()
                if number_islands > 1 and i != generations - 1:
                    data.write('\n\nMigrating individuals to other islands')
                    migrate()
                data.write('\n\nCurrent generation...: {}'.format(i))
                for y in range(number_islands):
                    data.write('\nBest solution found so far in island {}: Weight left: {} Value: {}'.format(y + 1,
                                      best_Knapsack[y].getTotalWeight(), best_Knapsack[y].getValue()))
        best = 0
        backpack = []
        for z in range(number_islands):
            if best_Knapsack[z].getValue() > best_Knapsack[best].getValue(): best = z
            data.write('\n\nSolution found in island {}:'.format(z + 1))
            data.write('\nWeight left: {}'.format(best_Knapsack[z].getTotalWeight()))
            data.write('\nValue: {}'.format(best_Knapsack[z].getValue()))
            for x, gen in enumerate(best_Knapsack[z].getChromosome()):
                if gen == 1:
                    backpack.append(x)
            best_Knapsack[z].chromosome = backpack
            backpack = []
            data.write('\nBackpack configuration: {}'.format(best_Knapsack[z].getChromosome()))

        data_simple.write('\n\nRun number: {}'.format(run + 1))
        data_simple.write('\nBest general solution:')
        data_simple.write('\nWeight left: {}'.format(best_Knapsack[best].getTotalWeight()))
        data_simple.write('\nValue: {}'.format(best_Knapsack[best].getValue()))
        data_simple.write('\nBackpack configuration: {}'.format(best_Knapsack[best].getChromosome()))

        # Creates four empty Knapsacks to later store the best Knapsack of each island
        best_Knapsack = [Knapsack(max_weight, backpack_capacity)] * number_islands

if __name__ == '__main__':
    # Reading files with the instance problem
    instance = open('Instances KP\\' + file_name + '.txt', 'r')
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

    for i in range(9):
        number_islands = i + 2
        # Creates a list of lists to save the different populations from the islands
        population = [[]] * number_islands
        if number_islands != 4:
            # Creates four empty Knapsacks to later store the best Knapsack of each island
            best_Knapsack = [Knapsack(max_weight, backpack_capacity)] * number_islands

            # Opening text file to save the data of each run
            data = open('experiments\\' + file_name + '\\' + file_name + '_' + str(number_islands) + '_Islands.txt', 'a')
            data_simple = open(
                'experiments\\' + file_name + '\\' + file_name + '_' + str(number_islands) + '_Islands_simple.txt', 'a')

            data.write('\n\n-------------------------------------------------------')
            data.write('\n-------------------------------------------------------')
            data.write('\n\n---Generated Parameters---')
            data.write('\nPopulation size per island: {}'.format(population_size))
            data.write('\nNumber of generations: {}'.format(generations))
            data.write('\nCrossover probability: {}'.format(crossover_probability))
            data.write('\nMutation probability: {}'.format(mutation_probability))
            data.write('\nBackpack capacity: {}'.format(backpack_capacity))
            data.write('\nMax backpack weight: {}'.format(max_weight))
            data.write('\nInstance used: {}'.format(file_name))
            data.write('\nNumber of islands: {}'.format(number_islands))
            geneticAlgorithm()
            data.close()
            data_simple.close()
