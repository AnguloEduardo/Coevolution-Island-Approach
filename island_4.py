# Libraries
# =======================
import random
from tqdm import tqdm
from numpy.random import randint
from items import Items
from knapsack import Knapsack

# List with the items of the problem
list_items = []
# Creates a list to save the population of the island
population = []
# Variables
num_tournament = 3
num_parents_to_select = 2
file_name = 'ks_500_0'
run_times = 30
# Initialization of population size, generations, crossover and mutation probabilities
# for the four different islands
population_size = 500
generations = 500
crossover_probability = 1.0
mutation_probability = 0.20


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


# Crossover operator
# =======================
def crossover(parentA, parentB):
    if random.random() <= crossover_probability:
        for x in range(len(parentA) - 1):
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


# Mutation operator
# =======================
def mutate(individual):
    if random.random() <= mutation_probability:
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
def select(tSize, numParents):
    parent_a = Knapsack(max_weight, backpack_capacity)
    parent_b = Knapsack(max_weight, backpack_capacity)
    for x in range(numParents):
        island_population = population.copy()
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
            parent_a = population[winner]
        else:
            parent_b = population[winner]
    return parent_a, parent_b


def sort_population():
    global best_Knapsack
    # Sorting Knapsacks by they value
    population.sort(key=lambda x: x.value, reverse=True)
    if population[0].getValue() > best_Knapsack.getValue():
        best_Knapsack = population[0]


# Genetic algorithm
# =======================
def geneticAlgorithm():
    global population
    global best_Knapsack
    # Running the genetic algorithm 30 times
    # to have a sample of how the algorithm
    # is behaving
    for run in range(run_times):
        data.write('\n\n Run number: {}'.format(run + 1))
        # Random generating the populations of the islands
        population = generate_population()

        # Runs the evolutionary process
        for i in tqdm(range(generations)):
            # Crossover
            new_population = []
            for _ in range(population_size // 2):
                parent_a, parent_b = select(num_tournament, num_parents_to_select)
                offspring_a, offspring_b = crossover(parent_a.getChromosome().copy(), parent_b.getChromosome().copy())
                weight, value, offspring_a = calculate_weight_value(offspring_a.copy())
                child_a = Knapsack(weight, value, offspring_a)
                weight, value, offspring_b = calculate_weight_value(offspring_b.copy())
                child_b = Knapsack(weight, value, offspring_b)
                new_population.extend([child_a, child_b])
            population = new_population

            # Mutation
            for index in range(population_size):
                individual, boolean = mutate(population[index])
                if boolean:
                    weight, value, chromosome = calculate_weight_value(individual.chromosome.copy())
                    population[index].chromosome = chromosome
                    population[index].value = value
                    population[index].totalWeight = weight

            # Printing useful information
            if (i % 100 == 0 or i == generations - 1) and i != 0:
                sort_population()
                data.write('\n\nCurrent generation...: {}'.format(i))
                data.write('\nBest solution found so far in island: Weight left: {} Value: {}'.format(
                                      population[0].getTotalWeight(), population[0].getValue()))

        backpack = []
        data.write('\n\nSolution found in island:')
        data.write('\nWeight left: {}'.format(best_Knapsack.getTotalWeight()))
        data.write('\nValue: {}'.format(best_Knapsack.getValue()))
        for x, gen in enumerate(best_Knapsack.getChromosome()):
            if gen == 1:
                backpack.append(x)
        best_Knapsack.chromosome = backpack
        data.write('\nBackpack configuration: {}'.format(best_Knapsack.getChromosome()))

        data_simple.write('\n\nRun number: {}'.format(run + 1))
        data_simple.write('\nBest general solution:')
        data_simple.write('\nWeight left: {}'.format(best_Knapsack.getTotalWeight()))
        data_simple.write('\nValue: {}'.format(best_Knapsack.getValue()))
        data_simple.write('\nBackpack configuration: {}'.format(best_Knapsack.getChromosome()))

        # Creates empty Knapsack to later store the best Knapsack of the island
        best_Knapsack = Knapsack(max_weight, backpack_capacity)

if __name__ == '__main__':
    # Reading files with the instance problem
    instance = open('Instances KP\\' + file_name + '.txt', 'r')
    problemCharacteristics = instance.readline().rstrip("\n")
    problemCharacteristics = problemCharacteristics.split(", ")

    # Opening text file to save the data of each run
    data = open('experiments\\' + file_name + '\\' + file_name + '_Island_4.txt', 'a')
    data_simple = open('experiments\\' + file_name + '\\' + file_name + '_Island_4_simple.txt', 'a')

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
    best_Knapsack = Knapsack(max_weight, backpack_capacity)

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
    geneticAlgorithm()
    data.close()
    data_simple.close()