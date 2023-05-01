# Libraries
from tqdm import tqdm
from numpy.random import randint
from HyperHeuristic import HyperHeuristic
import numpy as np
import random


# Crossover operator
# =======================
def simulated_binary_crossover(parents, lb, ub, pc, nc, crossover_prob, features, heuristics, len_pool):
    if np.random.random() <= crossover_prob:
        parentA = np.array(parents[0].get_conditions())
        parentB = np.array(parents[1].get_conditions())
        n = len(parentA)
        beta = np.zeros((n, len(parentA[0])))
        mu = np.random.rand(n, len(parentA[0]))
        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (nc + 1))
        beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (nc + 1))
        beta *= ((-1) ** randint(0, 2, (n, len(parentA[0]))))
        beta[np.random.rand(n, len(parentA[0])) <= 0.5] = 1
        beta[np.tile(np.random.rand() > pc, (len(parentA[0]), n))[0]] = 1
        offspringA_temp = (parentA + parentB) / 2 + beta * (parentA - parentB) / 2
        offspringB_temp = (parentA + parentB) / 2 - beta * (parentA - parentB) / 2
        offspringA_temp = np.minimum(np.maximum(offspringA_temp, lb), ub)
        offspringB_temp = np.minimum(np.maximum(offspringB_temp, lb), ub)
        offspringA = HyperHeuristic(features, heuristics, parents[0].get_actions(), offspringA_temp.tolist(), len_pool)
        offspringB = HyperHeuristic(features, heuristics, parents[0].get_actions(), offspringB_temp.tolist(), len_pool)
    else:
        offspringA = parents[0]
        offspringB = parents[1]
    return offspringA, offspringB


def polynomial_mutation(population, lb, ub, nm, mRate, size, len_rules):
    if np.random.random() <= mRate:
        for index in range(size):
            for x in range(len_rules):
                rule = np.array(population[index].get_conditions()[x])
                n = len(rule)
                pm = 1 / n
                mutate = np.random.rand(n) <= pm
                mu = np.random.rand(n)
                temp = mutate & (mu <= 0.5)
                rule[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (
                        1 - (rule[temp] - lb[temp]) / (ub[temp] - lb[temp])) ** (nm + 1)) ** (1 / (nm + 1)) - 1)
                temp = mutate & (mu > 0.5)
                rule[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (
                        1 - (ub[temp] - rule[temp]) / (ub[temp] - lb[temp])) ** (nm + 1)) ** (1 / (nm + 1)))
                rule = np.minimum(np.maximum(rule, lb), ub)
                population[index].get_conditions()[x] = rule.tolist()
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
                      crossover_prob, pm, migration, features, heuristics,
                      nbRules, problem_pool, folder_path, run_times, lb_num, ub_num, nm, pc, nc):
    # Creates a list of lists to save the different populations from the islands
    population = [[]] * number_islands
    # Upper and Lower bounds for polynomial mutation
    lb = np.array([lb_num] * len(features))
    ub = np.array([ub_num] * len(features))

    # Creation of the initial population
    for x in range(number_islands):
        population[x] = [HyperHeuristic(features, heuristics, nbRules, len(problem_pool), lb_num, ub_num) for _ in range(size)]

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
                    offspring_a, offspring_b = simulated_binary_crossover(parents, lb, ub, pc, nc, crossover_prob[island],
                                                                          features, heuristics, len(problem_pool))
                    new_population.extend([offspring_a, offspring_b])
                population[island] = new_population

                # Mutation
                population[island] = polynomial_mutation(population[island], lb, ub, nm, pm[island], size, nbRules)

            # Fitness
            population = fitness(population, number_islands, problem_pool)

            # Migration
            sorted_population = sort_population(number_islands, population)
            rate = np.random.random() <= migration[0]
            if number_islands > 1 and i != generations - 1 and rate:
                population = migrate(exchange, population, sorted_population, number_islands)

    hh = open(folder_path + '\\hh.txt', 'a')
    for x in range(number_islands):
        for y in range(size):
            hh.write("{}".format(population[x][y]))
    hh.close()

    best_hh_candidates = list()
    best_fitness, index = 0, 0
    sort_final_population = sort_population(number_islands, population)
    best_hh_candidates.append(sort_final_population[0][0])
    best_hh_candidates.append(sort_final_population[1][0])
    best_hh_candidates.append(sort_final_population[2][0])
    best_hh_candidates.append(sort_final_population[3][0])
    for x in range(4):
        if best_hh_candidates[x].get_fitness() > best_fitness:
            index = x
    best_hh = open(folder_path + '\\best_hh.txt', 'a')
    best_hh.write("{}".format(best_hh_candidates[index]))
