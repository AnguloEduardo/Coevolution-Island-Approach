class Knapsack:
    # Creates a new instance of Knapsack
    def __init__(self, totalWeight, len_chromosome):
        self.value = int(0)
        self.totalWeight = float(totalWeight)
        self.chromosome = [0] * len_chromosome

    # Creates a new instance of Knapsack from an existing instance
    def knapsack(self, knapsack):
        self.totalWeight = knapsack.totalWeight
        self.value = knapsack.value
        self.chromosome = knapsack.chromosome

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
        return item.getWeight() <= self.getTotalWeight()

    # Packs and item into this knapsack
    def pack(self, item):
        self.chromosome[item.getID()] = 1
        self.totalWeight -= float(item.getWeight())
        self.value += int(item.getValue())

    # Modifies the chromosome
    def modChromosome(self, chromosome):
        self.chromosome = chromosome

    # Modifies value and total weight
    def modValWeight(self, value, weight):
        self.value = int(value)
        self.totalWeight = float(weight)
