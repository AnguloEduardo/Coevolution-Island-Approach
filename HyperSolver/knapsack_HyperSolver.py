import sys
from items_HyperSolver import Items


def read_instance(file_path):
    total_value = 0
    total_weight = 0
    # List with the items of the problem
    list_items = []
    # Reading files with the instance problem
    instance = open(file_path, 'r')
    problemCharacteristics = instance.readline().rstrip("\n")
    problemCharacteristics = problemCharacteristics.split(", ")

    # This information needs to be taken from the .kp files
    # First element in the first row indicates the number of items
    # second element of the first row indicates the backpack capacity
    # from the second row and forth, the first element represent the profit
    # the second element represent the weight
    backpack_capacity = int(problemCharacteristics[0])  # Number of items in the problem
    max_weight = float(problemCharacteristics[1])  # Maximum weight for the backpack to carry

    # Creation of item's characteristics with the information from the .txt file
    for idx in range(backpack_capacity):
        instanceItem = instance.readline().rstrip("\n")
        instanceItem = instanceItem.split(", ")
        list_items.append(Items(idx, float(instanceItem[1]), int(instanceItem[0])))
    instance.close()

    for item in list_items:
        total_value += item.get_value()
        total_weight += item.get_weight()

    return list_items, max_weight, backpack_capacity, total_value, total_weight


class Knapsack:
    # Creates a new instance of Knapsack
    # First argument: Weight
    # Second argument: Length of chromosome
    def __init__(self, *args):
        if len(args) == 2:
            self.totalWeight = float(args[0])
            self.chromosome = [0] * args[1]
            self.value = int(0)

        # First argument: Weight
        # Second argument: Value
        # Third argument: Chromosome
        elif len(args) == 3:
            self.totalWeight = float(args[0])
            self.value = int(args[1])
            self.chromosome = args[2]

        # No arguments
        elif len(args) == 0:
            self.totalWeight = float(0)
            self.value = int(0)
            self.chromosome = []

    # Creates a new instance of Knapsack with predefined parameters
    def knapsack(self, value, weight, chromosome):
        self.totalWeight = float(weight)
        self.value = int(value)
        self.chromosome = chromosome

    # Returns the current totalWeight
    def get_total_weight(self):
        return self.totalWeight

    # Returns the current value
    def get_value(self):
        return self.value

    # Returns the chromosome
    def get_chromosome(self):
        return self.chromosome

    # Revises if the item provided can be packed in this knapsack
    def can_pack(self, item):
        return item.get_weight() <= self.totalWeight and self.chromosome[item.get_id()] == 0

    # Packs and item into this knapsack
    def pack(self, item):
        self.chromosome[item.get_id()] = 1
        self.totalWeight -= float(item.get_weight())
        self.value += int(item.get_value())

    # Returns the value of the feature provided as argument
    # feature = A string with the name of one available feature
    def get_feature(self, feature, total_weight, total_value, list_items):
        value = 0
        if feature == 'WEIGHT':
            value = self.totalWeight / total_weight
        elif feature == 'ITEMS_IN_KNAPSACK':
            count = 0
            for _, gene in enumerate(self.chromosome):
                if gene == 1:
                    count += 1
            value = count / len(list_items)
        elif feature == 'ITEMS_OUT_KNAPSACK':
            count = 0
            for _, gene in enumerate(self.chromosome):
                if gene == 0:
                    count += 1
            value = count / len(list_items)
        elif feature == 'TOTAL_WEIGHT_LEFT':
            weight_left = 0
            for x, gene in enumerate(self.chromosome):
                if gene == 0:
                    weight_left += list_items[x].get_weight()
            value = weight_left / total_weight
        elif feature == 'TOTAL_VALUE_LEFT':
            value_left = 0
            for x, gene in enumerate(self.chromosome):
                if gene == 0:
                    value_left += list_items[x].get_value()
            value = value_left / total_value
        return value

    def solve(self, heuristic, problem):
        selected = None
        # Max Profit
        if heuristic == 'MAXP':
            problem.get_items().sort(key=lambda x: x.get_value(), reverse=True)
            for idx, item in enumerate(problem.get_items()):
                if self.can_pack(item):
                    selected = item
                    break
            return selected

        # Max Profit/Weight
        elif heuristic == 'MAXPW':
            problem.get_items().sort(key=lambda x: x.get_value_per_weight(), reverse=True)
            for idx, item in enumerate(problem.get_items()):
                if self.can_pack(item):
                    selected = item
                    break
            return selected

        # Min Weight
        elif heuristic == 'MINW':
            problem.get_items().sort(key=lambda x: x.get_weight(), reverse=False)
            for idx, item in enumerate(problem.get_items()):
                if self.can_pack(item):
                    selected = item
                    break
            return selected

        # Mark
        elif heuristic == 'MARK':
            problem.get_items().sort(key=lambda x: x.get_value() * x.get_weight(), reverse=False)
            for idx, item in enumerate(problem.get_items()):
                if self.can_pack(item):
                    selected = item
                    break
            return selected

        # Default
        elif heuristic == 'DEF':
            for idx, item in enumerate(problem.get_items()):
                if self.can_pack(item):
                    selected = item
                    break
            return selected
