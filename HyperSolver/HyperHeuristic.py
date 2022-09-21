import random
from numpy.random import randint
import math
from knapsack_HyperSolver import Knapsack


def generate_individual(list_items, max_weight, capacity):
    # Initialize Random population
    items = list_items.copy()
    individual = Knapsack(max_weight, capacity)
    index = randint(len(items) - 1)
    while individual.canPack(items[index]):
        individual.pack(items.pop(index))
        index = randint(len(items) - 1)
    return individual


def split(a, n):
    x = []
    k, m = divmod(a, n)
    for i in range(n):
        x.append(int(i * k + min(i, m)))
    return x


class HyperHeuristic:
    # Constructor first case
    #   0. features = A list with the names of the features to be used by this hyper-heuristic
    #   1. heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
    #   2. nbRules = The number of rules to be contained in this hyper-heuristic

    # Constructor second case
    #   0. features = A list with the names of the features to be used by this hyper-heuristic
    #   1. heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
    #   2. actions = A list indicating which heuristic triggers with each rule
    #   3. conditions = A list of rules composed of features
    #   4. max_weight
    #   5. capacity

    def __init__(self, *args):
        if len(args) == 3:
            self.features = args[0].copy()
            self.heuristics = args[1].copy()
            self.individual = None
            self.actions = []
            self.conditions = []
            for i in range(args[2]):
                # Initializing conditions randomly
                self.conditions.append([0] * len(args[0]))
                for x in range(len(args[0])):
                    self.conditions[i][x] = random.random()

                # Adding the heuristics to use (in this case we use 4 heuristics)
                # MAX_PROFIT, MAX_PROFIT/WEIGHT, MIN_WEIGHT, MARKOVITZ
                # We divide the number of rules into four spaces. If the number of rules
                # is a multiple of four, the spaces will have the same length. In the case
                # of an odd number, some spaces will be smaller than the others.
                ranges = split(args[2], 4)
                if i < ranges[1]:
                    self.actions.append(args[1][0])
                elif i < ranges[2]:
                    self.actions.append(args[1][1])
                elif i < ranges[3]:
                    self.actions.append(args[1][2])
                else:
                    self.actions.append(args[1][3])
        elif len(args) == 6:
            self.features = args[0].copy()
            self.heuristics = args[1].copy()
            self.actions = args[2].copy()
            self.conditions = args[3].copy()
            self.individual = Knapsack(args[4], args[5])

        elif len(args) == 4:
            self.features = args[0].copy()
            self.heuristics = args[1].copy()
            self.actions = args[2].copy()
            self.conditions = args[3].copy()
            self.individual = None

    def get_conditions(self):
        return self.conditions

    # Returns the next heuristic to use
    def next_heuristic(self, list_items, weight):
        min_distance = float("inf")
        index = -1
        state = []
        for i in range(len(self.features)):
            state.append(self.individual.get_feature(self.features[i], weight, list_items))
        for i in range(len(self.conditions)):
            distance = self.__distance(self.conditions[i], state)
            if distance < min_distance:
                min_distance = distance
                index = i
        heuristic = self.actions[index]
        return heuristic

    # Returns the string representation of this dummy hyper-heuristic
    def __str__(self):
        text = "Features:\n\t" + str(self.features) + "\nHeuristics:\n\t" + str(self.heuristics) + "\nRules:\n"
        for i in range(len(self.conditions)):
            text += "\t" + str(self.conditions[i]) + " => " + self.actions[i] + "\n"
        return text

    # Returns the Euclidean distance between two vectors
    def __distance(self, vector_a, vector_b):
        distance = 0
        for i in range(len(vector_a)):
            distance += (vector_a[i] - vector_b[i]) ** 2
        distance = math.sqrt(distance)
        return distance

    def evaluate(self, list_items, weight):
        item = self.individual.solve(self.next_heuristic(list_items, weight), list_items)
        while item is not None:
            self.individual.pack(item)
            item = self.individual.solve(self.next_heuristic(list_items, weight), list_items)
