from items import Items


class Knapsack:
    # Creates a new instance of Knapsack
    def __init__(self, capacity, len_chromosome):
        self.capacity = capacity
        value = 0
        chromosome = [None * len_chromosome]

    # Creates a new instance of Knapsack from an existing instance
    def knapsack(self, knapsack):
        self.capacity = knapsack.capacity
        self.value = knapsack.value
        self.chromosome = knapsack.chromosome

    # Returns the current capacity
    def getCapacity(self):
        return self.capacity

    # Returns the current value
    def getValue(self):
        return self.value

    # Returns the number of chromosome
    def getItems(self):
        return len(self.chromosome)

    # Revises if the item provided can be packed in this knapsack
    def canPack(self, item):
        return item.getWeight() <= self.getCapacity()

    # Packs and item into this knapsack
    def pack(self, item):
        if item.getWeight() <= self.getCapacity():
            self.chromosome.append(item)
            self.capacity -= item.getWeight()
            self.value += item.getValue()
            return True
        return False
