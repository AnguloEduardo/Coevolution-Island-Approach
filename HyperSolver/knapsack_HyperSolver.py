import sys


class Knapsack:
    # Creates a new instance of Knapsack
    def __init__(self, *args): # First argument: Weight. Second argument: Length of chromosome
        if len(args) == 2:
            self.totalWeight = float(args[0])
            self.chromosome = [0] * args[1]
            self.value = int(0)
        elif len(args) == 3:    # First argument: Weight. Second argument: Value
            self.totalWeight = float(args[0])   # Third argument: Chromosome
            self.value = args[1]
            self.chromosome = args[2]
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
    def getTotalWeight(self):
        return self.totalWeight

    # Returns the current value
    def getValue(self):
        return self.value

    # Returns the chromosome
    def getChromosome(self):
        return self.chromosome

    # Revises if the item provided can be packed in this knapsack
    def canPack(self, item):
        return item.getWeight() <= self.getTotalWeight() and self.chromosome[item.getID()] == 0

    # Packs and item into this knapsack
    def pack(self, item):
        self.chromosome[item.getID()] = 1
        self.totalWeight -= float(item.getWeight())
        self.value += int(item.getValue())

    # Returns the value of the feature provided as argument
    # feature = A string with the name of one available feature
    def get_feature(self, feature, weight, list_items):
        value = 0
        if feature == 'WEIGHT':
            value = self.totalWeight * weight / 100
        elif feature == 'ITEMS_IN_KNAPSACK':
            count = 0
            for _, gene in enumerate(self.chromosome):
                if gene == 1:
                    count += 1
            value = count * len(list_items) / 100
        elif feature == 'ITEMS_OUT_KNAPSACK':
            count = 0
            for _, gene in enumerate(self.chromosome):
                if gene == 0:
                    count += 1
            value = count * len(list_items) / 100
        elif feature == 'TOTAL_WEIGHT_LEFT':
            weight_left = 0
            for x, gene in enumerate(self.chromosome):
                if gene == 0:
                    weight_left += list_items[x].getWeight()
            value = weight_left
        elif feature == 'TOTAL_VALUE_LEFT':
            value_left = 0
            for x, gene in enumerate(self.chromosome):
                if gene == 0:
                    value_left += list_items[x].getValue()
            value = value_left
        return value

    def solve(self, heuristic, list_items):
        selected = None
        # Max Profit
        if heuristic == 'MAXP':
            value = -sys.float_info.max
            for idx, item in enumerate(list_items):
                if self.canPack(item) and item.getValue() > value:
                    selected = item
                    value = selected.getValue()
            return selected
        # Max Profit/Weight
        elif heuristic == 'MAXPW':
            value = -sys.float_info.max
            for idx, item in enumerate(list_items):
                if self.canPack(item) and item.getvaluePerWeight() > value:
                    selected = item
                    value = selected.getvaluePerWeight()
            return selected
        # Min Weight
        elif heuristic == 'MINW':
            value = sys.float_info.max
            for idx, item in enumerate(list_items):
                if self.canPack(item) and item.getWeight() < value:
                    selected = item
                    value = selected.getWeight()
            return selected
        # Mark
        elif heuristic == 'MARK':
            value = -sys.float_info.max
            for idx, item in enumerate(list_items):
                if self.canPack(item) and item.getValue() * item.getWeight() > value:
                    selected = item
                    value = item.getValue() * item.getWeight()
            return selected
