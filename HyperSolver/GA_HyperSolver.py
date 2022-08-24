# Libraries
from tqdm import tqdm
from numpy.random import randint
from numpy import concatenate
from knapsack_HyperSolver import Knapsack
from items_HyperSolver import Items
from HyperHeuristic import HyperHeuristic
import random


# Crossover operator
# =======================
def crossover(parentA, parentB, crossover_prob, len_features, len_heuristics, n_rules):
    if random.random() <= crossover_prob:
        conditions_A = []
        conditions_B = []
        offspring_a = []
        offspring_b = []
        child_a_temp = []
        child_b_temp = []
        for x in range(len_features):
            lst_A = []
            lst_B = []
            for parameter in parentA:
                lst_A.append(parameter[x])
            for parameter in parentB:
                lst_B.append(parameter[x])
            conditions_A.append(list(lst_A))
            conditions_B.append(list(lst_B))
        for x in range(len_features):
            param_feature_a = conditions_A[x]
            param_feature_b = conditions_B[x]
            for y in range(len_heuristics):
                split = len(param_feature_a) // len_heuristics
                index = split // 2
                if y == 0:
                    rule_a = param_feature_a[:split]
                    rule_b = param_feature_b[:split]
                else:
                    rule_a = param_feature_a[split:]
                    rule_b = param_feature_b[split:]
                child_a = concatenate((rule_a[:index], rule_b[index:])).tolist()
                child_b = concatenate((rule_b[index:], rule_a[:index])).tolist()
                child_a_temp = concatenate((child_a_temp, child_a)).tolist()
                child_b_temp = concatenate((child_b_temp, child_b)).tolist()
        for x in range(n_rules):
            temp_a = []
            temp_b = []
            for y in range(len_features):
                temp_a.append(child_a_temp[3 * x + y])
                temp_b.append(child_b_temp[3 * x + y])
            offspring_a.append(temp_a)
            offspring_b.append(temp_b)
    else:
        offspring_a = parentA.copy()
        offspring_b = parentB.copy()
    return offspring_a, offspring_b


# Mutation operator
# =======================
def mutate(individual, mRate, nrules, len_features):
    if random.random() <= mRate:
        i, j = random.sample(range(nrules), 2)
        rule_i, rule_j = individual.conditions[i], individual.conditions[j]
        for x in range(len_features):
            if random.random() <= 0.50:
                rule_i[x], rule_j[x] = rule_i[x] + 0.015, rule_j[x] + 0.015
            else:
                rule_i[x], rule_j[x] = rule_i[x] - 0.015, rule_j[x] - 0.015
        return individual, True
    return individual, False


# Tournament selection
# =======================
# Return type: HyperHeuristic object
def select(island, tSize, numParents, max_weight, capacity, population):
    parent_a = Knapsack(max_weight, capacity)
    parent_b = Knapsack(max_weight, capacity)
    for x in range(numParents):
        island_population = population[island].copy()
        winner = randint(len(island_population) - 1)
        rival = randint(len(island_population) - 1)
        individualWinner = island_population.pop(winner)
        individualRival = island_population.pop(rival)
        for i in range(tSize):
            # It is searching for the Knapsack
            if individualRival.individual.getValue() > individualWinner.individual.getValue():
                winner = rival  # with the highest value
            rival = randint(len(island_population) - 1)
            individualRival = island_population.pop(rival)
        if x == 0:
            parent_a = population[island][winner]
        else:
            parent_b = population[island][winner]
    return parent_a, parent_b


def sort_population(islands, population):
    # Sorting Knapsacks by they value
    population_copy = population.copy()
    for i in range(islands):
        population_copy[i].sort(key=lambda x: x.individual.getValue(), reverse=True)
    return population_copy


def migrate(exchange, population, sorted_population, islands):
    # Exchanging the best 'exchange' hyper heuristics of each island to another
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
def geneticAlgorithm(tournament, parents, exchange, number_islands, size, generations,
                     crossover_prob, mutation, migration, features, heuristics,
                     nbRules, file_path, folder_path, run_times):
    # Creates a list of lists to save the different populations from the islands
    population = [[]] * number_islands
    sub_population = []
    for x in range(number_islands):
        for y in range(size):
            sub_population.append(HyperHeuristic(features, heuristics, nbRules))
        population[x] = sub_population.copy()
        sub_population = []
    for kp in range(len(file_path)):
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
            size, generations, backpack_capacity,
            max_weight, number_islands, crossover_prob,
            mutation, migration, file_path[kp]))
        print(file_path[kp])
        for migration_prob in range(len(migration)):
            for run in range(run_times):
                data.write('\n{} {}'.format(run + 1, migration))

                for x in range(number_islands):
                    for y in range(size):
                        population[x][y].individual = Knapsack(max_weight, backpack_capacity)

                # Runs the evolutionary process
                for i in tqdm(range(generations)):
                    # Need to parallelize this loop
                    for island in range(number_islands):
                        # Crossover
                        new_population = []
                        for _ in range(size // 2):
                            parent_a, parent_b = select(island, tournament, parents, max_weight, backpack_capacity,
                                                        population)
                            offspring_a, offspring_b = crossover(parent_a.get_conditions(), parent_b.get_conditions(),
                                                                 crossover_prob[island], len(features), len(heuristics),
                                                                 nbRules)
                            child_a = HyperHeuristic(features, heuristics, parent_a.actions, offspring_a, max_weight,
                                                     backpack_capacity)
                            child_b = HyperHeuristic(features, heuristics, parent_b.actions, offspring_b, max_weight,
                                                     backpack_capacity)
                            child_a.evaluate(list_items, max_weight)
                            child_b.evaluate(list_items, max_weight)
                            new_population.extend([child_a, child_b])
                        population[island] = new_population

                        # Mutation
                        for index in range(size):
                            individual, boolean = mutate(population[island][index], mutation[island], nbRules,
                                                         len(features))
                            if boolean:
                                individual.evaluate(list_items, max_weight)

                    sorted_population = sort_population(number_islands, population)
                    rate = random.random() <= migration_prob
                    if number_islands > 1 and i != generations - 1 and rate:
                        population = migrate(exchange, population, sorted_population, number_islands)

        for x in range(number_islands):
            for y in range(size):
                data.write('\n{}'.format(population[x][y]))

