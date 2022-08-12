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
        return item.getWeight() <= self.getTotalWeight()

    # Packs and item into this knapsack
    def pack(self, item):
        self.chromosome[item.getID()] = 1
        self.totalWeight -= float(item.getWeight())
        self.value += int(item.getValue())
