# Libraries
# =======================
import random
import numpy
from tqdm import tqdm
from numpy.random import randint
from numpy import concatenate
import multiprocessing as mp
from items import Items
from knapsack import Knapsack

# List with the items of the problem
list_items = []
# Creates a list of lists to save the different populations from the islands
population = []
# Lists to save data from each run
solution_1 = [None] * 7
solution_2 = [None] * 7
solution_3 = [None] * 7
solution_4 = [None] * 7
current_generation = [None] * 7


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
        offspring_a = concatenate((parentA[:int(backpack_capacity / 2)], parentB[int(backpack_capacity / 2):]))
        offspring_b = concatenate((parentA[int(backpack_capacity / 2):], parentB[:int(backpack_capacity / 2)]))
    else:
        offspring_a = numpy.copy(parentA)
        offspring_b = numpy.copy(parentB)
    return offspring_a, offspring_b


# Mutation operator
# =======================
def mutate(chromosome, mRate, backpack_capacity):
    if random.random() < mRate:
        i, j = random.sample(range(backpack_capacity - 1), 2)
        if chromosome[i] == 1:
            chromosome[i] = 0
        else:
            chromosome[i] = 1
        if chromosome[j] == 1:
            chromosome[j] = 0
        else:
            chromosome[j] = 1
    return chromosome


def calculate_weight_value(chromosome):
    weight, value = 0, max_weight
    for x, gene in enumerate(chromosome):
        if chromosome.all():
            weight += list_items[x].weight
            value += list_items[x].value
        else:
            if gene == 1:
                weight -= list_items[x].weight
                value += list_items[x].value
    return weight, value


# Evaluation function
# =======================
def calculate_fitness(island_population):
    for i in range(len(island_population)):
        weight, value = calculate_weight_value(island_population[i].getChromosome())
        island_population[i].modValWeight(weight, value)
    return island_population


# Tournament selection
# =======================
def select(island, tSize):
    winner = randint(len(population[island]))
    rival = randint(len(population[island]))
    individualWinner = population[island][winner]
    individualRival = population[island][rival]
    for i in range(tSize):
        if individualRival.getValue() > individualWinner.getValue():  # It is searching for the Knapsack
            winner = rival  # with the highest value
    return population[island][winner]


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
        # Crossover
        for island in range(number_islands):
            k = 0
            new_population = []
            for j in range(pSize // 2):
                parent_a = select(island, 3)
                parent_b = select(island, 3)
                offspring1, offspring2 = combine(parent_a.getChromosome(), parent_b.getChromosome(), cRate[island])
                child1 = Knapsack(weight, numberItems)
                child2 = Knapsack(weight, numberItems)
                child1.modChromosome(offspring1)
                child2.modChromosome(offspring2)
                new_population.append(child1)
                new_population.append(child2)
                k = k + 2
            population[island] = new_population
        # Mutation
        for island in range(number_islands):
            for index in range(pSize):
                chromosome = mutate(population[island][index].getChromosome(), mRate[island], numberItems)
                population[island][index].modChromosome(chromosome)
            population[island] = calculate_fitness(population[island])
        # Keeps a record of the best Knapsack found so far in each island
        for x in range(number_islands):
            for z in range(pSize - 1):
                if population[x][z].getTotalWeight() <= max_weight:
                    if population[x][z].getValue() > best_Knapsack[x].getValue():
                        best_Knapsack[x] = population[x][z]
        if i % 100 == 0 and number_islands > 1 and i != 0:
            print('\nMigrating individuals to other islands')
            migrate(population)
        # Printing useful information
        if (i % 100 == 0 or i == gens - 1) and i != 0:
            print('\nCurrent generation...: {}'.format(i))
            current_generation[i // 100] = i
            for y in range(number_islands):
                print('Best solution so far in island {}: {}'.format(y, best_Knapsack[y].getValue()))
                if y == 0:
                    solution_1[i // 100] = best_Knapsack[y].getValue()
                if y == 1:
                    solution_2[i // 100] = best_Knapsack[y].getValue()
                if y == 2:
                    solution_3[i // 100] = best_Knapsack[y].getValue()
                if y == 3:
                    solution_4[i // 100] = best_Knapsack[y].getValue()
    best = 0
    for z in range(number_islands):
        if best_Knapsack[z].getValue() > best_Knapsack[best].getValue():
            best = z
        print('\nSolution found in island {}:'.format(z + 1))
        print('Weight: {}'.format(best_Knapsack[z].getTotalWeight()))
        print('Value: {}'.format(best_Knapsack[z].getValue()))
        print('Backpack configuration: {}'.format(best_Knapsack[z].getChromosome()))

    print('\nBest general solution:')
    print('Weight: {}'.format(best_Knapsack[best].getTotalWeight()))
    print('Value: {}'.format(best_Knapsack[best].getValue()))
    print('Backpack configuration: {}'.format(best_Knapsack[best].getChromosome()))


if __name__ == '__main__':
    # Number of islands or subpopulations
    number_islands = 4

    # Reading files with the instance problem
    file_name = '\ks_19_0.txt'
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
                     Knapsack(max_weight, backpack_capacity),
                     None, None, None]

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
