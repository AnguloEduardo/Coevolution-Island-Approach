# Libraries
# =======================
import random
import numpy
import collections
import pandas as pd
from tqdm import tqdm
from numpy.random import randint
from numpy import concatenate
from csv import writer

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
        if chromosome[i] == 1:
            chromosome[i] = 0
        else:
            chromosome[i] = 1
        if chromosome[j] == 1:
            chromosome[j] = 0
        else:
            chromosome[j] = 1
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


def to_ceros(chromosome):
    for idx, gene in enumerate(chromosome):
        chromosome[idx] = 0
    chromosome[0] = 1
    value = items[0].value
    weight = items[0].weight
    return value, weight, chromosome


# Tournament selection
# =======================
def select(population, tSize):
    k = 0
    winner = numpy.random.randint(0, len(population))
    for k in range(len(population)):
        if population[winner].weight > max_weight:
            chromosome = mutate(population[winner].chromosome, 1.0, backpack_capacity)
            population[winner] = population[winner]._replace(chromosome=chromosome)
            winner = numpy.random.randint(0, len(population))
    if k == len(population) - 1:
        value, weight, chromosome = to_ceros(population[winner].chromosome)
        individual = Individual(chromosome=chromosome, weight=weight, value=value)
    else:
        for i in range(tSize - 1):
            k = 0
            rival = numpy.random.randint(0, len(population))
            for k in range(len(population)):
                if population[winner].weight > max_weight:
                    chromosome = mutate(population[rival].chromosome, 1.0, backpack_capacity)
                    population[rival] = population[rival]._replace(chromosome=chromosome)
                    rival = numpy.random.randint(0, len(population))
            if k == len(population) - 1:
                value, weight, chromosome = to_ceros(population[winner].chromosome)
                population[winner] = population[winner]._replace(chromosome=chromosome, weight=weight, value=value)
            if population[rival].value > population[winner].value:  # It is searching for the individual
                winner = rival  # with the highest value
        return population[winner]
    return individual


# Exchanging individuals between islands
def get_value(chromosome):
    return chromosome.value


def migrate(islands):
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
        if i % 100 == 0 or i == gens - 1:
            if i != 0:
                print('\nMigrating individuals to other islands')
                migrate(population)
            print('\nCurrent generation...: {}'.format(i))
            current_generation[i // 100] = i
            for y in range(0, 4):
                print('Best solution so far in island {}: {}'.format(y, best_individual[y].value))
                if y == 0:
                    solution_1[i // 100] = best_individual[y].value
                if y == 1:
                    solution_2[i // 100] = best_individual[y].value
                if y == 2:
                    solution_3[i // 100] = best_individual[y].value
                if y == 3:
                    solution_4[i // 100] = best_individual[y].value
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


def generate_csv():
    df = pd.DataFrame({
        'Parameters': ['Population size', 'Number of generations', 'Crossover probability', 'Mutation probability',
                       'Backpack capacity', 'Maximum backpack weight', 'Instance used'],
        'Data': [population_size, generations, crossover_probability, mutation_probability, backpack_capacity,
                 max_weight, file_name],
        'Current generation': current_generation,
        'Best solution\nfound so far\nin island 1': solution_1,
        'Best solution\nfound so far\nin island 2': solution_2,
        'Best solution\nfound so far\nin island 3': solution_3,
        'Best solution\nfound so far\nin island 4': solution_4,
        'Best solution\nin island 1': ['Weight', 'Value', 'Backpack configuration', None, None, None, None],
        'Results 1': [best_individual[0].weight, best_individual[0].value, best_individual[0].chromosome,
                      None, None, None, None],
        'Best solution\nin island 2': ['Weight', 'Value', 'Backpack configuration', None, None, None, None],
        'Results 2': [best_individual[1].weight, best_individual[1].value, best_individual[1].chromosome,
                      None, None, None, None],
        'Best solution\nin island 3': ['Weight', 'Value', 'Backpack configuration', None, None, None, None],
        'Results 3': [best_individual[2].weight, best_individual[2].value, best_individual[2].chromosome,
                      None, None, None, None],
        'Best solution\nin island 4': ['Weight', 'Value', 'Backpack configuration', None, None, None, None],
        'Results 4': [best_individual[3].weight, best_individual[3].value, best_individual[3].chromosome,
                      None, None, None, None],
    })
    # Saving the data frame into a .csv file
    df.to_csv('coevolution.csv', index=False, encoding='utf_8_sig')


def write_data(data):
    with open('coevolution.csv', 'a', newline='') as f_object:
        # Pass the CSV  file object to the writer() function
        writer_object = writer(f_object)
        # Result - a writer object
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow(data)
        # Close the file object
        f_object.close()


def write_to_csv():
    empty = [None, None, None, None, None, None, None]
    list_data_1 = ['Population size', population_size, current_generation[0], solution_1[0], solution_2[0],
                   solution_3[0], solution_4[0], 'Weight', best_individual[0].weight, 'Weight',
                   best_individual[1].weight, 'Weight', best_individual[2].weight, 'Weight', best_individual[3].weight]
    list_data_2 = ['Number of generations', generations, current_generation[1], solution_1[1], solution_2[1],
                   solution_3[1], solution_4[1], 'Value', best_individual[0].value, 'Value', best_individual[1].value,
                   'Value', best_individual[2].value, 'Value', best_individual[3].value]
    list_data_3 = ['Crossover probability', crossover_probability, current_generation[2], solution_1[2], solution_2[2],
                   solution_3[2], solution_4[2], 'Backpack configuration', best_individual[0].chromosome,
                   'Backpack configuration', best_individual[1].chromosome, 'Backpack configuration',
                   best_individual[2].chromosome, 'Backpack configuration', best_individual[3].chromosome]
    list_data_4 = ['Mutation probability', mutation_probability, current_generation[3], solution_1[3], solution_2[3],
                   solution_3[3], solution_4[3], None, None, None, None, None, None, None, None]
    list_data_5 = ['Backpack capacity', backpack_capacity, current_generation[4], solution_1[4], solution_2[4],
                   solution_3[4], solution_4[4], None, None, None, None, None, None, None, None]
    list_data_6 = ['Maximum backpack weight', max_weight, current_generation[5], solution_1[5], solution_2[5],
                   solution_3[5], solution_4[5], None, None, None, None, None, None, None, None]
    list_data_7 = ['Instance used', file_name, current_generation[6], solution_1[6], solution_2[6],
                   solution_3[6], solution_4[6], None, None, None, None, None, None, None, None]
    write_data(empty)
    write_data(list_data_1)
    write_data(list_data_2)
    write_data(list_data_3)
    write_data(list_data_4)
    write_data(list_data_5)
    write_data(list_data_6)
    write_data(list_data_7)


if __name__ == '__main__':
    # Creates an array to save the different populations from the islands
    population = [[None], [None], [None], [None]]
    # Creates four empty individuals to later store the best individual of each island
    best_individual = [Individual(chromosome=[None], weight=-1, value=-1),
                       Individual(chromosome=[None], weight=-1, value=-1),
                       Individual(chromosome=[None], weight=-1, value=-1),
                       Individual(chromosome=[None], weight=-1, value=-1),
                       None, None, None]
    # Reading files with the instance problem
    file_name = '\ks_30_0.txt'
    instance = open('Instances KP' + file_name, 'r')
    itemsCapacity = instance.readline().rstrip("\n")
    itemsCapacity = itemsCapacity.split(", ")
    # Initialization of population size, generations, crossover and mutation probabilities
    # for the four different islands
    population_size = 500
    generations = 500
    crossover_probability = [0.3, 0.5, 0.7, 1.0]
    mutation_probability = [0.05, 0.10, 0.15, 0.20]
    # Lists to save data from each run
    solution_1 = [None] * 7
    solution_2 = [None] * 7
    solution_3 = [None] * 7
    solution_4 = [None] * 7
    current_generation = [None] * 7

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
    # Uncomment this line if the .csv file does not exist
    # generate_csv()
    write_to_csv()
