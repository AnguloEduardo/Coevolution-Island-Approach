# Libraries
from tqdm import tqdm
from numpy.random import randint
from numpy import concatenate
from knapsack import Knapsack
import random

# Generate population
def generate_population(size, list_items, max_weight, capacity):
    new_population = []
    for i in range(size):
        # Initialize Random population
        items = list_items.copy()
        individual = Knapsack(max_weight, capacity)
        index = randint(len(items) - 1)
        while individual.canPack(items[index]):
            individual.pack(items.pop(index))
            index = randint(len(items) - 1)
        new_population.append(individual)
    return new_population


def crossover_1(parentA, parentB, crossover, islands, capacity):
    if random.random() <= crossover[islands - 4]:
        offspring_a = concatenate((parentA[:capacity // 2], parentB[capacity // 2:])).tolist()
        offspring_b = concatenate((parentA[capacity // 2:], parentB[:capacity // 2])).tolist()
    else:
        offspring_a = parentA.copy()
        offspring_b = parentB.copy()
    return offspring_a, offspring_b


def crossover_2(parentA, parentB, crossover, islands):
    if random.random() <= crossover[islands - 3]:
        index = randint(1, len(parentA) - 1)
        offspring_a = concatenate((parentA[:index], parentB[index:])).tolist()
        offspring_b = concatenate((parentB[:index], parentA[index:])).tolist()
    else:
        offspring_a = parentA.copy()
        offspring_b = parentB.copy()
    return offspring_a, offspring_b


def crossover_3(parentA, parentB, crossover, islands):
    if random.random() <= crossover[islands - 2]:
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


def crossover_4(parentA, parentB, crossover, islands):
    if random.random() <= crossover[islands - 1]:
        for x in range(len(parentA)):
            if parentA[x] != parentB[x]:
                if random.random() <= 0.5:
                    temp = int(parentA[x])
                    parentA[x] = int(parentB[x])
                    parentB[x] = int(temp)
        offspring_a = parentA.copy()
        offspring_b = parentB.copy()
    else:
        offspring_a = parentA.copy()
        offspring_b = parentB.copy()
    return offspring_a, offspring_b


# Crossover operator
# =======================
def combine(parentA, parentB, island, crossover, islands, capacity):
    if island == 0:
        offspring_a, offspring_b = crossover_1(parentA.getChromosome(), parentB.getChromosome(), crossover, islands, capacity)
    elif island == 1:
        offspring_a, offspring_b = crossover_2(parentA.getChromosome(), parentB.getChromosome(), crossover, islands)
    elif island == 2:
        offspring_a, offspring_b = crossover_3(parentA.getChromosome(), parentB.getChromosome(), crossover, islands)
    else:
        offspring_a, offspring_b = crossover_4(parentA.getChromosome(), parentB.getChromosome(), crossover, islands)
    offspring_a = [int(x) for x in offspring_a]
    offspring_b = [int(x) for x in offspring_a]
    return offspring_a, offspring_b


# Mutation operator
# =======================
def mutate(individual, mRate, capacity):
    if random.random() <= mRate:
        i, j = random.sample(range(capacity), 2)
        individual.chromosome[i] = int(0) if individual.chromosome[i] == 1 else int(1)
        individual.chromosome[j] = int(0) if individual.chromosome[j] == 1 else int(1)
        return individual, True
    return individual, False


def fitness(chromosome_a, chromosome_b, list_items, max_weight):
    weight_a, value_a = 0.0, 0
    weight_b, value_b = 0.0, 0
    for x in range(len(chromosome_a)):
        if chromosome_a[x] == 1:
            value_a += list_items[x].getValue()
            weight_a += list_items[x].getWeight()
        if chromosome_b[x] == 1:
            value_b += list_items[x].getValue()
            weight_b += list_items[x].getWeight()
    if weight_a > max_weight:
        value_a = 0
    if weight_b > max_weight:
        value_b = 0
    return weight_a, value_a, weight_b, value_b


# Tournament selection
# =======================
def select(island, tSize, numParents, max_weight, capacity,population):
    parent_a = Knapsack(max_weight, capacity)
    parent_b = Knapsack(max_weight, capacity)
    for x in range(numParents):
        island_population = population[island].copy()
        winner = randint(len(island_population) - 1)
        rival = randint(len(island_population) - 1)
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


def sort_population(islands, population, best):
    # Sorting Knapsacks by they value
    population_copy = population.copy()
    for i in range(islands):
        population_copy[i].sort(key=lambda x: x.value, reverse=True)
        # Keeping track of the best solution found on each island
        if population_copy[i][0].getValue() > best[i].getValue():
            best[i] = population_copy[i][0]
    return population_copy, best


def migrate(exchange, population, sorted_population, islands):
    # Exchanging the best 'exchange' Knapsacks solutions of each island to another
    temp_knapsack = []
    for index_1 in range(exchange):
        temp_knapsack.append(sorted_population[0][index_1])
    for index_1 in range(islands - 1):
        index = random.sample(range(len(population[0])), exchange)
        for index_2 in range(exchange):
            population[index_1][index[index_2]] = sorted_population[index_1 + 1][index_2]
    index = random.sample(range(len(population[0])), exchange)
    for index_1 in range(exchange):
        population[islands - 1][index[index_1]] = temp_knapsack[index_1]
    return population


# Genetic algorithm
# =======================
# (num_tournament, num_parents_to_select, individuals_to_exchange, number_islands,
# run_times, list_items, population, population_size, generations, crossover_probability,
# mutation_probability, backpack_capacity, max_weight, best_Knapsack, table, data)
def geneticAlgorithm(tournament, parents, exchange, islands, list_items, population, size, generations, crossover,
                     mutation, migration, capacity, max_weight, best, table, data):
    # Random generating the populations of the islands
    # Need to parallelize this loop
    for x in range(islands):
        population[x] = generate_population(size, list_items, max_weight, capacity)
    # Runs the evolutionary process
    for i in tqdm(range(generations)):
        # Need to parallelize this loop
        for island in range(islands):
            # Crossover
            new_population = []
            for _ in range(size // 2):
                parent_a, parent_b = select(island, tournament, parents, max_weight, capacity, population)
                offspring_a, offspring_b = combine(parent_a, parent_b, island, crossover, islands, capacity)
                weight_a, value_a, weight_b, value_b  = fitness(offspring_a, offspring_b, list_items, max_weight)
                child_a = Knapsack(weight_a, value_a, offspring_a)
                child_b = Knapsack(weight_b, value_b, offspring_b)
                new_population.extend([child_a, child_b])
            population[island] = new_population

            # Mutation
            for index in range(size):
                individual, boolean = mutate(population[island][index], mutation[island], capacity)
                if boolean:
                    weight, value, _, _ = fitness(individual.getChromosome(), individual.getChromosome(), list_items, max_weight)
                    population[island][index].chromosome = individual.getChromosome().copy()
                    population[island][index].value = value
                    population[island][index].totalWeight = weight

        sorted_population, best = sort_population(islands, population, best)
        rate = random.random() <= migration
        if islands > 1 and i != generations - 1 and rate:
            population = migrate(exchange, population, sorted_population, islands)
        data.write('\n1') if rate else data.write('\n0')
        data.write(' {}'.format(i))
        for y in range(islands):
            data.write(' {} {} {}'.format(best[y].getValue(), best[y].getTotalWeight(), best[y].getChromosome()))
    solution = 0
    backpack = []
    data.write('\n')
    for z in range(islands):
        if best[z].getValue() > best[solution].getValue(): solution = z
        for x, gen in enumerate(best[z].getChromosome()):
            if gen == 1:
                backpack.append(x)
        best[z].chromosome = backpack
        backpack = []
        data.write('{} {} {} '.format(best[z].getValue(), best[z].getTotalWeight(), best[z].getChromosome()))
    data.write('\n{} {} {}'.format(best[solution].getValue(), best[solution].getTotalWeight(), best[solution].getChromosome()))

    if migration == 0.0:
        for z in range(islands):
            table.write('{} '.format(best[z].getValue()))
            # table.write('{} {} '.format(best[z].getValue(), best[z].getTotalWeight()))
            # for _, gene in enumerate(best[z].getChromosome()):
            #     table.write('{} '.format(gene))
    else:
        table.write('{} '.format(best[solution].getValue()))
        # table.write('{} {} '.format(best[solution].getValue(), best[solution].getTotalWeight()))
        # for _, gene in enumerate(best[solution].getChromosome()):
        #     table.write('{} '.format(gene))