# Libraries
# =======================
import random
import numpy
import collections
from tqdm import tqdm
from numpy.random import randint
from numpy import concatenate

Item = collections.namedtuple("backpack", "weight value")
Individual = collections.namedtuple('population', 'chromosome weight value')


# Generate population
def generate_population(size, capacity):
    new_population = []
    for i in range(size):
        # Initialize Random population
        new_population.append(Individual(
            chromosome=randint(2, size=(1, capacity))[0],
            weight=-1,
            value=-1))
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
        i, j = random.sample(range(0, backpack_capacity - 1), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome


def calculate_weight_value(chromosome, list_of_items):
    weight, value = 0, 0.0
    for x, gene in enumerate(chromosome):
        if chromosome.all():
            weight += list_of_items[x].weight
            value += list_of_items[x].value
        else:
            if gene == 1:
                weight += list_of_items[x].weight
                value += list_of_items[x].value
    return weight, value


# Evaluation function
# =======================
def calculate_fitness(island_population, item, mRate, numberItems):
    for i in range(len(island_population)):
        weight, value = calculate_weight_value(island_population[i].chromosome, item)
        island_population[i] = island_population[i]._replace(weight=weight, value=value)
    return island_population


# Tournament selection
# =======================
def select(population, tSize):
    winner = numpy.random.randint(0, len(population))
    while population[winner].weight > max_weight:
        winner = numpy.random.randint(0, len(population))
    for i in range(tSize - 1):
        rival = numpy.random.randint(0, len(population))
        while population[rival].weight > max_weight:
            rival = numpy.random.randint(0, len(population))
        if population[rival].value > population[winner].value:  # It is searching for the individual
            winner = rival  # with the highest value
    return population[winner]


# Exchanging individuals between islands
def get_value(chromosome):
    return chromosome.value


def exchange(islands):
    # Sorting individuals by they value
    for i in range(0, 4):
        islands[i].sort(key=get_value, reverse=True)
    # Exchanging the best two individuals of each island to another
    individual1, individual2 = islands[0][0], islands[0][1]
    islands[0][0], islands[0][1] = islands[1][0], islands[1][1]
    islands[1][0], islands[1][1] = islands[2][0], islands[2][1]
    islands[2][0], islands[2][1] = islands[3][0], islands[3][1]
    islands[3][0], islands[3][1] = individual1, individual2


# Genetic algorithm
# =======================
def geneticAlgorithm(pSize, gens, cRate, mRate, numberItems, list_items):
    # Creates an array to save the different populations from the islands
    population = [[None], [None], [None], [None]]
    # Creates four empty individuals to later store the best individual of each island
    best_individual = [Individual(chromosome=[None], weight=-1, value=-1),
                       Individual(chromosome=[None], weight=-1, value=-1),
                       Individual(chromosome=[None], weight=-1, value=-1),
                       Individual(chromosome=[None], weight=-1, value=-1)]
    # Random generating the populations of the islands
    for index in range(0, 4):
        population[index] = generate_population(pSize, numberItems)
        population[index] = calculate_fitness(population[index], list_items, mRate[index], numberItems)
    # Runs the evolutionary process
    for i in tqdm(range(gens)):
        # Crossover
        for island in range(0, 4):
            k = 0
            new_population = []
            for j in range(pSize // 2):
                parent_a = select(population[island], 3)
                parent_b = select(population[island], 3)
                offspring1, offspring2 = combine(parent_a.chromosome, parent_b.chromosome, cRate[island])
                new_population.append(Individual(chromosome=offspring1, weight=-1, value=-1))
                new_population.append(Individual(chromosome=offspring2, weight=-1, value=-1))
                k = k + 2
            population[island] = new_population
        # Mutation
        for island in range(0, 4):
            for index in range(pSize):
                chromosome = mutate(population[island][index].chromosome, mRate[island], numberItems)
                population[island][index] = population[island][index]._replace(chromosome=chromosome)
            population[island] = calculate_fitness(population[island], list_items, mRate[island], numberItems)
        # Keeps a record of the best individual found so far in each island
        for x in range(0, 4):
            for z in range(0, pSize - 1):
                if population[x][z].weight <= max_weight:
                    if population[x][z].value > best_individual[x].value:
                        best_individual[x] = population[x][z]
        # Printing useful information
        if i % 100 == 0:
            if i != 0:
                print('\nExchanging individuals to other islands')
                exchange(population)
            print('\nCurrent generation...: {}'.format(i))
            for y in range(0, 4):
                print('Best solution so far in island {}: {}'.format(y, best_individual[y].value))
    for z in range(0, 4):
        best = 0
        if best_individual[z].value > best_individual[best].value:
            best = z
        print('\nSolution found in island {}:'.format(z + 1))
        print('Weight: {}'.format(best_individual[z].weight))
        print('Value: {}'.format(best_individual[z].value))
        print('Backpack configuration: {}'.format(best_individual[z].chromosome))

    print('\nBest general solution:')
    print('Weight: {}'.format(best_individual[best].weight))
    print('Value: {}'.format(best_individual[best].value))
    print('Backpack configuration: {}'.format(best_individual[best].chromosome))


if __name__ == '__main__':
    # Reading files with the instance problem
    file_name = '\ks_50_0.txt'
    instance = open('Instances KP' + file_name, 'r')
    itemsCapacity = instance.readline().rstrip("\n")
    itemsCapacity = itemsCapacity.split(", ")
    # Initialization of population size, generations, crossover and mutation probabilities
    # for the four different islands
    population_size = 150
    generations = 500
    crossover_probability = [0.3, 0.5, 0.7, 1.0]
    mutation_probability = [0.05, 0.10, 0.15, 0.20]

    # This information needs to be taken from the .txt files
    # First element in the first row indicates the number of items
    # second element of the first row indicates the backpack capacity
    # from the second row and forth, the first element represent the weight
    # the second element represent the profit
    backpack_capacity = int(itemsCapacity[0])  # Number of items in the problem
    max_weight = int(itemsCapacity[1])
    items = []
    # Creation of item's characteristics with the information form the .txt file
    for idx in range(backpack_capacity):
        instanceItem = instance.readline().rstrip("\n")
        instanceItem = instanceItem.split(", ")
        items.append(Item(weight=int(instanceItem[0]),
                          value=float(instanceItem[1])))

    print('\n\n---Generated Parameters---')
    print('Population size: {}'.format(population_size))
    print('Number of generations: {}'.format(generations))
    print('Crossover probability: {}'.format(crossover_probability))
    print('Mutation probability: {}'.format(mutation_probability))
    print('Backpack capacity: {}'.format(backpack_capacity))
    print('Max backpack weight: {}'.format(max_weight))
    print('Instance used: {}'.format(file_name))
    geneticAlgorithm(population_size, generations, crossover_probability,
                     mutation_probability, backpack_capacity, items)
