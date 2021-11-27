# Final Exam Intelligent Systems
# Eduardo Angulo Martínez A00820188
# Gerardo Uriel Monroy Vazquez A01372286
# Gabriel Ichcanziho Perez Landa A01651212

# Libraries
# =======================
import random
import numpy
import matplotlib.pyplot
from tqdm import tqdm


# Individual initialization
# =======================
def createIndividual():
    x = list(range(1, 26))  # numeros del 1 al 25 sin repetidos
    random.shuffle(x)  # los revuelvo / permutaciones, secuencia de 25 números para el magic square
    return x


# Crossover operator
# =======================
def combine(parentA, parentB, cRate):
    # REPLACE with the actual code to combine two individuals
    if random.random() < cRate:
        offspringA, offspringB = PartialMappingCrossOver(parentA, parentB)
    else:
        offspringA = numpy.copy(parentA)
        offspringB = numpy.copy(parentB)
    return offspringA, offspringB


def PartialMappingCrossOver(parent1, parent2):
    firstCrossPoint = numpy.random.randint(0, len(parent1) - 2)
    secondCrossPoint = numpy.random.randint(firstCrossPoint + 1, len(parent1) - 1)
    parent1MiddleCross = parent1[firstCrossPoint:secondCrossPoint]
    parent2MiddleCross = parent2[firstCrossPoint:secondCrossPoint]
    temp_child1 = parent1[:firstCrossPoint] + parent2MiddleCross + parent1[secondCrossPoint:]
    temp_child2 = parent2[:firstCrossPoint] + parent1MiddleCross + parent2[secondCrossPoint:]
    relations = []
    for i in range(len(parent1MiddleCross)):
        relations.append([parent2MiddleCross[i], parent1MiddleCross[i]])

    def recursion1(temp_child, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross):
        child = numpy.array([0 for i in range(len(parent1))])
        for i, j in enumerate(temp_child[:firstCrossPoint]):
            c = 0
            for x in relations:
                if j == x[0]:
                    child[i] = x[1]
                    c = 1
                    break
            if c == 0:
                child[i] = j
        j = 0
        for i in range(firstCrossPoint, secondCrossPoint):
            child[i] = parent2MiddleCross[j]
            j += 1

        for i, j in enumerate(temp_child[secondCrossPoint:]):
            c = 0
            for x in relations:
                if j == x[0]:
                    child[i + secondCrossPoint] = x[1]
                    c = 1
                    break
            if c == 0:
                child[i + secondCrossPoint] = j
        child_unique = numpy.unique(child)
        if len(child) > len(child_unique):
            child = recursion1(child, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross)
        return (child)

    def recursion2(temp_child, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross):
        child = numpy.array([0 for i in range(len(parent1))])
        for i, j in enumerate(temp_child[:firstCrossPoint]):
            c = 0
            for x in relations:
                if j == x[1]:
                    child[i] = x[0]
                    c = 1
                    break
            if c == 0:
                child[i] = j
        j = 0
        for i in range(firstCrossPoint, secondCrossPoint):
            child[i] = parent1MiddleCross[j]
            j += 1

        for i, j in enumerate(temp_child[secondCrossPoint:]):
            c = 0
            for x in relations:
                if j == x[1]:
                    child[i + secondCrossPoint] = x[0]
                    c = 1
                    break
            if c == 0:
                child[i + secondCrossPoint] = j
        child_unique = numpy.unique(child)
        if len(child) > len(child_unique):
            child = recursion2(child, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross)
        return (child)

    child1 = recursion1(temp_child1, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross)
    child2 = recursion2(temp_child2, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross)

    return list(child1), list(child2)


# Mutation operator
# =======================
def mutate(individual, mRate):
    if random.random() < mRate:
        i, j = random.sample(range(0, 25), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual


# Evaluation function
# =======================
def evaluate(individual):
    cost = 0
    # Sum of rows
    cost += abs(65 - sum(individual[0:5]))
    cost += abs(65 - sum(individual[5:10]))
    cost += abs(65 - sum(individual[10:15]))
    cost += abs(65 - sum(individual[15:20]))
    cost += abs(65 - sum(individual[20:25]))
    # Sum of columns
    cost += abs(65 - individual[0] - individual[5] - individual[10] - individual[15] - individual[20])
    cost += abs(65 - individual[1] - individual[6] - individual[11] - individual[16] - individual[21])
    cost += abs(65 - individual[2] - individual[7] - individual[12] - individual[17] - individual[22])
    cost += abs(65 - individual[3] - individual[8] - individual[13] - individual[18] - individual[23])
    cost += abs(65 - individual[4] - individual[9] - individual[14] - individual[19] - individual[24])
    # Sum of diagonals
    cost += abs(65 - individual[0] - individual[6] - individual[12] - individual[18] - individual[24])
    cost += abs(65 - individual[4] - individual[8] - individual[12] - individual[16] - individual[20])

    return cost


# Tournament selection
# =======================
def select(population, evaluation, tSize):
    winner = numpy.random.randint(0, len(population))
    for i in range(tSize - 1):
        rival = numpy.random.randint(0, len(population))
        if evaluation[rival] < evaluation[winner]:  # esta buscando la que minimice la evaluación, con menor costo
            winner = rival
    return population[winner]


# Neighbor generator
# =======================
# Receives a board and generates a neighbor. The neighbor is generated by swapping
# two randomly chosen numbers on the board.
def neighbor(mutated):
    # REPLACE with the actual code to generate the neighbor
    newMutated = mutated.copy()
    index_a = random.randint(0, 24)
    index_b = random.randint(0, 24)
    newMutated[index_a], newMutated[index_b] = newMutated[index_b], newMutated[index_a]
    return newMutated


# Local search
def optimize(individual, evaluateFunction, iterations):
    # Iterates to optimize the best solution found
    for i in range(iterations):
        # Generates a neighbor of x
        neighbor_a = neighbor(individual)
        # If the cost of y is smaller than the cost of x, we replace x with y
        if evaluateFunction(neighbor_a) < evaluateFunction(individual):
            individual = neighbor_a
    # Returns the best solution found
    return individual


# Genetic algorithm
# =======================
def geneticAlgorithm(pSize, gens, cRate, mRate):
    # Creates the initial population
    population = [None] * pSize
    evaluation = [None] * pSize
    for i in range(pSize):
        population[i] = createIndividual()
        evaluation[i] = evaluate(population[i])
    # Keeps a record of the best individual found so far
    index = 0
    for i in range(1, pSize):
        if evaluation[i] < evaluation[index]:
            index = i
    bestIndividual = population[index]
    bestEvaluation = evaluation[index]
    # Keeps the information for plotting the performance of the algorithm
    best = [0] * gens
    avg = [0] * gens
    # Runs the evolutionary process
    for i in tqdm(range(gens)):
        k = 0
        newPopulation = [None] * pSize
        # Crossover
        for j in range(pSize // 2):
            parentA = select(population, evaluation, 3)
            parentB = select(population, evaluation, 3)
            offspring1, offspring2 = combine(parentA, parentB, cRate)
            newPopulation[k] = offspring1
            newPopulation[k + 1] = offspring2
            k = k + 2
        population = newPopulation
        # Mutation
        for j in range(pSize):
            population[j] = mutate(population[j], mRate)
            population[j] = optimize(population[j], evaluate, 500)
            evaluation[j] = evaluate(population[j])
            # Keeps a record of the best individual found so far
            if evaluation[j] < bestEvaluation:
                bestEvaluation = evaluation[j]
                bestIndividual = population[j]
            best[i] = bestEvaluation
            avg[i] = numpy.average(evaluation)
        if bestEvaluation == 0:
            break
    matplotlib.pyplot.plot(range(gens), best, label="Best")
    matplotlib.pyplot.plot(range(gens), avg, label="Average")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title("GA Run")
    matplotlib.pyplot.show()
    # Returns the best individual found so far
    return bestIndividual, bestEvaluation


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Tests
    # =======================
    solution, evaluation = geneticAlgorithm(100, 50, 1.0, 0.05)
    print("Solution found:")
    print(solution[:5])
    print(solution[5:10])
    print(solution[10:15])
    print(solution[15:20])
    print(solution[20:])
    print("Error: ", evaluation)
