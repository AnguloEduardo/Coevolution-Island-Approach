# Coevolution island approach
# Eduardo Angulo Martínez A00820188

# Libraries
# =======================
import random
import numpy
import matplotlib.pyplot


# Individual initialization
# =======================
def createIndividual():
    x = list(range(1, 17))  # numeros del 1 al 16 sin repetidos
    random.shuffle(x)  # los revuelvo / permutaciones, secuencia de 16 números para el magic square
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
    # REPLACE with the actual code to mutate an individual
    if random.random() < mRate:
        i, j = random.sample(range(0, 16), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual


# Evaluation function
# =======================
def evaluate(individual):
    # REPLACE with the actual code to evaluate an individual
    cost = 0
    cost += abs(34 - sum(individual[0:4]))
    cost += abs(34 - sum(individual[4:8]))
    cost += abs(34 - sum(individual[8:12]))
    cost += abs(34 - sum(individual[12:16]))

    cost += abs(34 - individual[0] - individual[4] - individual[8] - individual[12])
    cost += abs(34 - individual[1] - individual[5] - individual[9] - individual[13])
    cost += abs(34 - individual[2] - individual[6] - individual[10] - individual[14])
    cost += abs(34 - individual[3] - individual[7] - individual[11] - individual[15])

    cost += abs(34 - individual[0] - individual[5] - individual[10] - individual[15])
    cost += abs(34 - individual[3] - individual[6] - individual[9] - individual[12])

    return cost


# Torunament selection
# =======================
def select(population, evaluation, tSize):
    winner = numpy.random.randint(0, len(population))
    for i in range(tSize - 1):
        rival = numpy.random.randint(0, len(population))
        if (evaluation[rival] < evaluation[winner]):  # esta buscando la que minimice la evaluación, con menor costo
            winner = rival
    return population[winner]


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
    index = 0;
    for i in range(1, pSize):
        if (evaluation[i] < evaluation[index]):
            index = i;
    bestIndividual = population[index]
    bestEvaluation = evaluation[index]
    # Keeps the information for plotting the performance of the algorithm
    best = [0] * gens
    avg = [0] * gens
    # Runs the evolutionary process
    for i in range(gens):
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
            evaluation[j] = evaluate(population[j])
            # Keeps a record of the best individual found so far
            if (evaluation[j] < bestEvaluation):
                bestEvaluation = evaluation[j]
                bestIndividual = population[j]
            best[i] = bestEvaluation
            avg[i] = numpy.average(evaluation)
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
    solution, eval = geneticAlgorithm(100, 250, 1.0, 0.25)
    print(solution)
    print(eval)
