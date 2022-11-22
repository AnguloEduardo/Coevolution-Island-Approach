# Libraries
from tqdm import tqdm
from numpy.random import randint
from numpy import concatenate
from knapsack_HyperSolver import Knapsack
from knapsack_HyperSolver import read_instance
from HyperHeuristic import HyperHeuristic
from HyperHeuristic import random_generate_individual
import random


# Crossover operator
# =======================
def crossover(parents, crossover_prob, features, heuristics, n_rules, len_pool):
    if random.random() <= crossover_prob:
        conditions_A = []
        conditions_B = []
        child_a_temp = []
        child_b_temp = []
        for x in range(len(features)):
            lst_A = []
            lst_B = []
            for parameter in parents[0].get_conditions():
                lst_A.append(parameter[x])
            for parameter in parents[1].get_conditions():
                lst_B.append(parameter[x])
            conditions_A.append(list(lst_A))
            conditions_B.append(list(lst_B))
        for x in range(len(features)):
            param_feature_a = conditions_A[x]
            param_feature_b = conditions_B[x]
            for y in range(len(heuristics)):
                split = len(param_feature_a) // len(heuristics)
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
        conditions_A, conditions_B = [], []
        for x in range(n_rules):
            temp_a = []
            temp_b = []
            for y in range(len(features)):
                temp_a.append(child_a_temp[3 * x + y])
                temp_b.append(child_b_temp[3 * x + y])
            conditions_A.append(temp_a)
            conditions_B.append(temp_b)
        offspring_a = HyperHeuristic(features, heuristics, parents[0].get_actions(), conditions_A, len_pool)
        offspring_b = HyperHeuristic(features, heuristics, parents[1].get_actions(), conditions_B, len_pool)
    else:
        offspring_a = parents[0]
        offspring_b = parents[1]
    return offspring_a, offspring_b


# Mutation operator
# =======================
def mutate(population, mRate, nrules, len_features, size):
    if random.random() <= mRate:
        for index in range(size):
            i, j = random.sample(range(nrules), 2)
            rule_i, rule_j = population[index].get_conditions()[i], population[index].get_conditions()[j]
            for x in range(len_features):
                if random.random() <= 0.50:
                    rule_i[x], rule_j[x] = rule_i[x] + 0.015, rule_j[x] + 0.015
                else:
                    rule_i[x], rule_j[x] = rule_i[x] - 0.015, rule_j[x] - 0.015
            population[index].get_conditions()[i], population[index].get_conditions()[j] = rule_i, rule_j
    return population


# Tournament selection
# =======================
# Return type: HyperHeuristic object
def select(island, tSize, numParents, population):
    parents = [None] * numParents
    for x in range(numParents):
        island_population = population[island].copy()
        winner = randint(len(island_population) - 1)
        rival = randint(len(island_population) - 1)
        individualWinner = island_population.pop(winner)
        individualRival = island_population.pop(rival)

        for i in range(tSize):
            if individualRival.get_fitness() > individualWinner.get_fitness():
                winner = rival
            rival = randint(len(island_population) - 1)
            individualRival = island_population.pop(rival)

        parents[x] = population[island][winner]

    return parents


def sort_population(islands, population):
    # Sorting Knapsacks by they value
    population_copy = population.copy()
    for i in range(islands):
        population_copy[i].sort(key=lambda x: x.get_fitness(), reverse=True)
    return population_copy


def migrate(exchange, population, sorted_population, islands):
    # Exchanging the best 'exchange' hyper heuristics of each island to another
    temp_hh = []
    for index_1 in range(exchange):
        temp_hh.append(sorted_population[0][index_1])
    for index_1 in range(islands - 1):
        index = random.sample(range(len(population[0])), exchange)
        for index_2 in range(exchange):
            population[index_1][index[index_2]] = sorted_population[index_1 + 1][index_2]
    index = random.sample(range(len(population[0])), exchange)
    for index_1 in range(exchange):
        population[islands - 1][index[index_1]] = temp_hh[index_1]
    return population


def fitness(population, num_islands, problem_pool):
    for island in range(num_islands):
        for individuals in population[island]:
            individuals.evaluate(problem_pool)
    return population


# Genetic algorithm
# =======================
# (num_tournament, num_parents_to_select, individuals_to_exchange, number_islands,
# run_times, list_items, population, population_size, generations, crossover_probability,
# mutation_probability, backpack_capacity, max_weight, best_Knapsack, table, data)
def genetic_algorithm(tournament, num_parents, exchange, number_islands, size, generations,
                      crossover_prob, mutation, migration, features, heuristics,
                      nbRules, problem_pool, folder_path, run_times):
    # Creates a list of lists to save the different populations from the islands
    population = [[]] * number_islands
    sub_population = []

    # Creation of the population
    for x in range(number_islands):
        for y in range(size):
            sub_population.append(HyperHeuristic(features, heuristics, nbRules, len(problem_pool)))
        population[x] = sub_population.copy()
        sub_population = []

    # Fitness
    population = fitness(population, number_islands, problem_pool)

    # Runs the evolutionary process
    for i in tqdm(range(generations)):
        for _ in range(run_times):
            for island in range(number_islands):
                # Crossover
                new_population = []
                for _ in range(size // 2):
                    parents = select(island, tournament, num_parents, population)
                    offspring_a, offspring_b = crossover(parents, crossover_prob[island], features, heuristics, nbRules,
                                                         len(problem_pool))
                    new_population.extend([offspring_a, offspring_b])
                population[island] = new_population

                # Mutation
                population[island] = mutate(population[island], mutation[island], nbRules, len(features), size)

            # Fitness
            population = fitness(population, number_islands, problem_pool)

            # Migration
            sorted_population = sort_population(number_islands, population)
            rate = random.random() <= migration[0]
            if number_islands > 1 and i != generations - 1 and rate:
                population = migrate(exchange, population, sorted_population, number_islands)

    best_hh = open(folder_path + '\\hh_results.txt', 'a')
    for x in range(number_islands):
        for y in range(size):
            best_hh.write("{}".format(population[x][y]))
    best_hh.close()
