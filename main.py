# Libraries
# =======================
import random
from tqdm import tqdm
from numpy.random import randint
from numpy import concatenate
from items import Items
from knapsack import Knapsack

# List with the items of the problem
list_items = []
# Creates a list of lists to save the different populations from the islands
population = []
# Lists to save data from each run
solutions = []
# Variables
num_tournament = 3
num_parents_to_select = 2


# Generate population
def generate_population(size, capacity, weight):
    new_population = []
    for i in range(size):
        # Initialize Random population
        items = list_items.copy()
        individual = Knapsack(weight, capacity)
        index = randint(len(items))
        while individual.canPack(items[index]):
            individual.pack(items.pop(index))
            index = randint(len(items))
        new_population.append(individual)
    return new_population


# Crossover operator
# =======================
def combine(parentA, parentB, cRate):
    if random.random() < cRate:
        offspring_a = concatenate((parentA[:backpack_capacity // 2], parentB[backpack_capacity // 2:])).tolist()
        offspring_b = concatenate((parentA[backpack_capacity // 2:], parentB[:backpack_capacity // 2])).tolist()
    else:
        offspring_a = parentA.copy()
        offspring_b = parentB.copy()
    return offspring_a, offspring_b


# Mutation operator
# =======================
def mutate(individual, mRate):
    if random.random() < mRate:
        i, j = random.sample(range(backpack_capacity), 2)
        individual.chromosome[i] = 0 if individual.chromosome[i] == 0 else 1
        individual.chromosome[j] = 0 if individual.chromosome[j] == 0 else 1
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


def migrate(islands):
    # Sorting Knapsacks by they value
    for i in range(number_islands):
        islands[i].sort(key=lambda x: x.value, reverse=True)
    # Exchanging the best four Knapsacks solutions of each island to another
    Knapsack1, Knapsack2, Knapsack3, Knapsack4 = islands[0][0], islands[0][1], islands[0][2], islands[0][3]
    islands[0][0], islands[0][1], islands[0][2], islands[0][3] = islands[1][0], islands[1][1], islands[1][2], islands[1][3]
    islands[1][0], islands[1][1], islands[1][2], islands[1][3] = islands[2][0], islands[2][1], islands[2][2], islands[2][3]
    islands[2][0], islands[2][1], islands[2][2], islands[2][3] = islands[3][0], islands[3][1], islands[3][2], islands[3][3]
    islands[3][0], islands[3][1], islands[3][2], islands[3][3] = Knapsack1, Knapsack2, Knapsack3, Knapsack4


# Genetic algorithm
# =======================
def geneticAlgorithm(pSize, gens, cRate, mRate, numberItems, weight):
    # Random generating the populations of the islands
    for _ in range(number_islands):
        population.append(generate_population(pSize, numberItems, weight))

    # Runs the evolutionary process
    for i in tqdm(range(gens)):
        for island in range(number_islands):
            # Crossover
            new_population = []
            for _ in range(pSize // 2):
                parent_a, parent_b = select(island, num_tournament, num_parents_to_select)
                offspring_a, offspring_b = combine(parent_a.getChromosome(), parent_b.getChromosome(), cRate[island])
                weight, value, offspring_a = calculate_weight_value(offspring_a)
                child_a = Knapsack(weight, value, offspring_a)
                weight, value, offspring_b = calculate_weight_value(offspring_b)
                child_b = Knapsack(weight, value, offspring_b)
                new_population.extend([child_a, child_b])
            population[island] = new_population

            # Mutation
            for index in range(pSize):
                individual, boolean = mutate(population[island][index], mRate[island])
                if boolean:
                    weight, value, chromosome = calculate_weight_value(individual.chromosome)
                    population[island][index].chromosome = chromosome
                    population[island][index].value = value
                    population[island][index].totalWeight = weight

            # Keeps a record of the best Knapsack found so far in each island
            for individual in range(pSize):
                if population[island][individual].getTotalWeight() >= 0:
                    if population[island][individual].getValue() > best_Knapsack[island].getValue():
                        best_Knapsack[island] = population[island][individual]

        # Printing useful information
        if (i % 100 == 0 or i == gens - 1) and i != 0:
            if number_islands > 1 and i != gens - 1:
                print('\nMigrating individuals to other islands')
                migrate(population)
            print('\nCurrent generation...: {}'.format(i + 1))
            for y in range(number_islands):
                print('Best solution so far in island {}: {}'.format(y + 1, best_Knapsack[y].getValue()))
    best = 0
    for z in range(number_islands):
        if best_Knapsack[z].getValue() > best_Knapsack[best].getValue(): best = z
        print('\nSolution found in island {}:'.format(z + 1))
        print('Weight left: {}'.format(best_Knapsack[z].getTotalWeight()))
        print('Value: {}'.format(best_Knapsack[z].getValue()))
        print('Backpack configuration: {}'.format(best_Knapsack[z].getChromosome()))

    print('\nBest general solution:')
    print('Weight left: {}'.format(best_Knapsack[best].getTotalWeight()))
    print('Value: {}'.format(best_Knapsack[best].getValue()))
    print('Backpack configuration: {}'.format(best_Knapsack[best].getChromosome()))


if __name__ == '__main__':
    # Number of islands or subpopulations
    number_islands = 4

    # Reading files with the instance problem
    file_name = '\ks_10000_0.txt'
    instance = open('Instances KP' + file_name, 'r')
    problemCharacteristics = instance.readline().rstrip("\n")
    problemCharacteristics = problemCharacteristics.split(", ")

    # Initialization of population size, generations, crossover and mutation probabilities
    # for the four different islands
    population_size = 2000 // 4
    generations = 500
    crossover_probability = [0.3, 0.5, 0.7, 1.0]
    mutation_probability = [0.05, 0.10, 0.15, 0.20]

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

    # Creates four empty Knapsacks to later store the best Knapsack of each island
    best_Knapsack = [Knapsack(max_weight, backpack_capacity),
                     Knapsack(max_weight, backpack_capacity),
                     Knapsack(max_weight, backpack_capacity),
                     Knapsack(max_weight, backpack_capacity)]

    print('\n\n---Generated Parameters---')
    print('Population size per island: {}'.format(population_size))
    print('Number of generations: {}'.format(generations))
    print('Crossover probability: {}'.format(crossover_probability))
    print('Mutation probability: {}'.format(mutation_probability))
    print('Backpack capacity: {}'.format(backpack_capacity))
    print('Max backpack weight: {}'.format(max_weight))
    print('Instance used: {}'.format(file_name))
    print('Number of islands: {}'.format(number_islands))
    geneticAlgorithm(population_size, generations, crossover_probability,
                     mutation_probability, backpack_capacity, max_weight)
